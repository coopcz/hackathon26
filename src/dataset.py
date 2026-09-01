"""
Build the ESP32 training table from labelled dashboard exports in `data/`.

One CSV = one 30-second recording = one label.  This module turns a directory of
them into (X, y, condition, group) where `group` is the recording a window came
from, because THAT is the unit the train/test split has to respect.

THREE DECISIONS HERE MATTER MORE THAN THE MODEL
-----------------------------------------------

1. SUBCARRIERS ARE PINNED, NOT AUTO-DETECTED.
   `load_esp32_csi_csv` defaults to picking usable subcarriers empirically per
   file.  That is right for inspecting one capture and wrong for building a
   dataset: if one recording happens to have a subcarrier that never went
   non-zero, its feature vector is computed over a different set of channels
   than every other recording, and column 3 of the matrix stops meaning one
   thing.  Training pins the documented map (`valid_subcarriers="reference"`)
   so every row is comparable, and any file that cannot use it is excluded
   loudly rather than silently mis-sliced.

2. WINDOWS OVERLAP, AND THAT IS SAFE *ONLY* BECAUSE OF `group`.
   30 s at ~60 Hz is ~1800 packets = 11 non-overlapping 2.56 s windows.  Across
   40 recordings that is ~440 rows, which is thin.  At 75% overlap the same data
   yields ~49 windows per recording, ~2000 rows.  Two overlapping windows share
   packets, so this would be blatant leakage under a random split -- but every
   evaluation in train_esp32.py splits by RECORDING, and overlapping windows
   never cross a recording boundary.  The cost is that rows within a recording
   are correlated, so the effective sample size is smaller than the row count;
   the honest CV score already accounts for that.

3. BAD CAPTURES ARE EXCLUDED, NOT AVERAGED IN.
   A wandering AGC produces an amplitude step that is indistinguishable from a
   person walking past, and a lossy link breaks the uniform-sampling assumption
   the spectral features rest on.  Training on those recordings does not add
   data, it adds a second, wrong definition of "motion".  `quality_report()`
   flags them and `build(exclude_bad=True)` drops them.
"""

import glob
import os
import re

import numpy as np

from .esp_csi import load_esp32_csi_csv
from .features import FEATURE_NAMES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Two sources, both first-class:
#   recordings/*.jsonl  the dashboard's NATIVE output, written automatically
#   data/*.csv          the manual "Export CSV" of the same sessions
# The JSONL is a strict superset of the CSV, so reading it directly removes the
# per-file export click entirely. A session present in both is loaded once.
SOURCE_DIRS = [os.path.join(_ROOT, "recordings"), os.path.join(_ROOT, "data")]
DATA_DIR = None  # None = scan SOURCE_DIRS
CACHE = os.path.join(_ROOT, "artifacts", "esp32_features.npz")

AWAY, HOME = 0, 1

# Window length in SECONDS (not packets), so a 60 Hz and a 100 Hz capture cover
# the same span of real time.  2.56 s matches the Intel baseline.
WINDOW_SECONDS = 2.56
OVERLAP = 0.75

# --- capture-quality gates -------------------------------------------------
# Thresholds mirror the OPEN items reported by verify_esp32_assumptions().
MAX_AGC_STD = 2.0        # above this the AGC is wandering -> fake motion
MAX_JITTER = 0.5         # sd/mean of packet interval; above this sampling is not uniform
MAX_BAD_ROW_FRAC = 0.05  # malformed serial rows
MAX_ZERO_CSI_FRAC = 0.20 # packets whose CSI is entirely zero
MIN_WINDOWS = 3          # a recording contributing fewer rows than this is noise


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# The operator types the label into the dashboard, so casing and wording drift.
# Normalise on meaning, never on an exact string match.

_EMPTY_WORDS = ("empty", "vacant", "unoccupied", "nobody", "no_one", "noone",
                "away", "none", "absent")
_STILL_WORDS = ("still", "static", "stationary", "sitting", "seated", "sit",
                "idle", "not_moving", "no_motion", "resting")
_MOVING_WORDS = ("moving", "move", "walking", "walk", "motion", "active", "pacing")
_OCCUPIED_WORDS = ("occupied", "present", "person", "people", "home", "human")

CONDITIONS = ("empty", "occupied_still", "occupied_moving", "occupied")


def normalize_label(raw):
    """Map a free-typed label onto one of CONDITIONS, or None if unrecognised.

    Order matters: 'unoccupied' contains 'occupied', and 'not_moving' contains
    'moving', so the more specific test has to run first in both cases.
    """
    if not raw:
        return None
    t = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if not t:
        return None
    if any(w in t for w in _EMPTY_WORDS):
        return "empty"
    if any(w in t for w in _OCCUPIED_WORDS):
        if any(w in t for w in _STILL_WORDS):
            return "occupied_still"
        if any(w in t for w in _MOVING_WORDS):
            return "occupied_moving"
        return "occupied"
    # a bare "still"/"moving" with no occupancy word still means somebody is there
    if any(w in t for w in _STILL_WORDS):
        return "occupied_still"
    if any(w in t for w in _MOVING_WORDS):
        return "occupied_moving"
    return None


def label_of(path, session_label):
    """Prefer the label stored inside the CSV; fall back to the filename."""
    return normalize_label(session_label) or normalize_label(os.path.basename(path))


def to_binary(condition):
    """AWAY = nobody in the room. HOME = anybody, moving or not."""
    return AWAY if condition == "empty" else HOME


# ---------------------------------------------------------------------------
# Per-recording quality
# ---------------------------------------------------------------------------

def _zero_csi_fraction(amp):
    """Fraction of packets whose entire CSI vector is zero.

    The first hardware capture had 9.9% of these. A zero packet is not a quiet
    room, it is a dropped measurement, and it drags every variance feature in
    whatever window contains it.
    """
    return float((amp.reshape(len(amp), -1).sum(axis=1) == 0).mean()) if len(amp) else 1.0


def _quality(out, amp):
    d = out["diagnostics"]
    seen = max(1, d["n_rows_seen"])
    q = {
        "fs": out["fs"],
        "duration_s": d["duration_s"],
        "n_windows": len(out["X"]),
        "agc_gain_std": d["agc_gain_std"],
        "jitter": d["packet_interval_jitter"],
        "bad_row_frac": d["n_rows_bad"] / seen,
        "zero_csi_frac": _zero_csi_fraction(amp),
        "rssi_mean": d["rssi_mean"],
        "rssi_std": d["rssi_std"],
    }
    problems = []
    if np.isfinite(q["agc_gain_std"]) and q["agc_gain_std"] > MAX_AGC_STD:
        problems.append(f"AGC wandering (sd={q['agc_gain_std']:.1f}) - set CONFIG_FORCE_GAIN 1")
    if np.isfinite(q["jitter"]) and q["jitter"] > MAX_JITTER:
        problems.append(f"lossy link (jitter={q['jitter']:.2f}) - spectral features unreliable")
    if q["bad_row_frac"] > MAX_BAD_ROW_FRAC:
        problems.append(f"{q['bad_row_frac']:.0%} malformed rows")
    if q["zero_csi_frac"] > MAX_ZERO_CSI_FRAC:
        problems.append(f"{q['zero_csi_frac']:.0%} all-zero CSI packets")
    if q["n_windows"] < MIN_WINDOWS:
        problems.append(f"only {q['n_windows']} windows")
    q["problems"] = problems
    q["usable"] = not problems
    return q


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(data_dir=DATA_DIR, window_seconds=WINDOW_SECONDS, overlap=OVERLAP,
          exclude_bad=True, force=False, cache=CACHE, verbose=True):
    """Load every labelled CSV in `data_dir` into one feature table.

    Returns a dict with:
      X          (n_windows, 16)  features, columns == FEATURE_NAMES
      y          (n_windows,)     0 = AWAY, 1 = HOME
      condition  (n_windows,)     'empty' / 'occupied_still' / 'occupied_moving'
      group      (n_windows,)     source recording -- the split unit
      order      (n_windows,)     recording index in filename order, for a drift split
      fs         (n_windows,)     packet rate of the source recording
      quality    {filename: {...}} per-recording QC, including skipped files
    """
    if cache and os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        if (float(z["window_seconds"]) == window_seconds
                and float(z["overlap"]) == overlap
                and bool(z["exclude_bad"]) == exclude_bad):
            out = {k: z[k] for k in ("X", "y", "condition", "group", "order", "fs")}
            out["quality"] = z["quality"].item()
            return out

    dirs = [data_dir] if data_dir else SOURCE_DIRS
    paths, seen_stems = [], set()
    for d in dirs:
        for ext in ("*.jsonl", "*.csv"):        # JSONL first: it is the native form
            for p in sorted(glob.glob(os.path.join(d, ext))):
                stem = os.path.splitext(os.path.basename(p))[0]
                if stem in seen_stems:          # same session exported to both forms
                    continue
                seen_stems.add(stem)
                paths.append(p)
    paths.sort(key=lambda p: os.path.basename(p))
    if not paths:
        raise FileNotFoundError(
            f"no recordings found in {', '.join(dirs)}. Record a session in the "
            f"dashboard (it writes recordings/*.jsonl automatically), or export a "
            f"session to CSV and drop it in data/.")

    X, y, condition, group, order, fsv = [], [], [], [], [], []
    quality, skipped = {}, []

    for i, path in enumerate(paths):
        name = os.path.basename(path)
        try:
            out = load_esp32_csi_csv(path, valid_subcarriers="reference",
                                     window_seconds=window_seconds, overlap=overlap,
                                     verbose=False)
        except (ValueError, OSError) as exc:
            quality[name] = {"usable": False, "problems": [f"unreadable: {exc}"]}
            skipped.append(name)
            continue

        cond = label_of(path, out["session_label"])
        q = _quality(out, out["amp"])
        q["condition"] = cond
        q["label_source"] = "session_label" if normalize_label(out["session_label"]) else "filename"
        quality[name] = q

        if cond is None:
            q["usable"] = False
            q["problems"] = q.get("problems", []) + [
                f"unrecognised label {out['session_label']!r}; rename the file or "
                f"add a keyword from {CONDITIONS}"]
        if not q["usable"] and exclude_bad:
            skipped.append(name)
            continue

        n = len(out["X"])
        X.append(out["X"])
        y.append(np.full(n, to_binary(cond)))
        condition.append(np.full(n, cond))
        group.append(np.full(n, name))
        order.append(np.full(n, i))
        fsv.append(np.full(n, out["fs"]))

    if verbose:
        print_quality(quality, exclude_bad)

    if not X:
        raise ValueError(
            f"every recording in {data_dir} was excluded. Quality report:\n" +
            "\n".join(f"  {k}: {'; '.join(v.get('problems', []))}" for k, v in quality.items()))

    res = {
        "X": np.vstack(X), "y": np.concatenate(y),
        "condition": np.concatenate(condition), "group": np.concatenate(group),
        "order": np.concatenate(order), "fs": np.concatenate(fsv),
        "quality": quality,
    }

    if verbose:
        print(f"\n  {res['X'].shape[0]} windows x {len(FEATURE_NAMES)} features "
              f"from {len(np.unique(res['group']))} recordings "
              f"({len(skipped)} excluded)")

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(
            cache, **{k: v for k, v in res.items() if k != "quality"},
            quality=np.array(quality, dtype=object),
            feature_names=np.array(FEATURE_NAMES),
            window_seconds=window_seconds, overlap=overlap, exclude_bad=exclude_bad)
    return res


def print_quality(quality, exclude_bad=True):
    """One line per recording, so a bad capture is visible before it is trained on."""
    print(f"  {'recording':<44} {'label':<16} {'win':>4} {'Hz':>6} {'agc':>6} "
          f"{'jit':>6} {'0CSI':>6}  status")
    print("  " + "-" * 104)
    for name, q in quality.items():
        if "fs" not in q:
            print(f"  {name[:43]:<44} {'-':<16} {'-':>4} {'-':>6} {'-':>6} {'-':>6} {'-':>6}  "
                  f"SKIP: {'; '.join(q['problems'])}")
            continue
        verdict = "EXCLUDED" if exclude_bad else "FLAGGED (kept)"
        status = "ok" if q["usable"] else f"{verdict}: " + "; ".join(q["problems"])
        print(f"  {name[:43]:<44} {str(q.get('condition')):<16} {q['n_windows']:>4} "
              f"{q['fs']:>6.1f} {q['agc_gain_std']:>6.1f} {q['jitter']:>6.2f} "
              f"{q['zero_csi_frac']:>6.1%}  {status}")


def summarize(ds):
    """Class balance by condition and by recording -- read this before training."""
    cond, group, y = ds["condition"], ds["group"], ds["y"]
    lines = ["  windows per condition:"]
    for c in CONDITIONS:
        m = cond == c
        if m.any():
            lines.append(f"    {c:<18} {int(m.sum()):5d} windows "
                         f"across {len(np.unique(group[m])):3d} recordings")
    n_away, n_home = int((y == AWAY).sum()), int((y == HOME).sum())
    lines.append(f"  binary balance: AWAY {n_away} / HOME {n_home} "
                 f"(1:{n_home / max(1, n_away):.1f})")
    return "\n".join(lines)


def separation(ds, top=8):
    """How far apart are empty and occupied, per feature? (Cohen's d)

    This is the "is there any signal at all" check, and it is worth running after
    the first few recordings of EACH class rather than at the end of the session.
    Cohen's d is the gap between the two class means measured in standard
    deviations, so it does not care about units:

        |d| < 0.5   the classes overlap almost completely
        |d| ~ 0.8   a conventionally "large" effect
        |d| > 1.5   the feature separates them on its own

    If every feature comes back near zero, no model will rescue it and the
    problem is physical -- board placement, a wandering AGC, or a room where the
    "empty" recordings were not actually empty. Find that out at recording 6,
    not recording 60.
    """
    X, y = ds["X"], ds["y"]
    if len(np.unique(y)) < 2:
        return "  separation: needs both empty and occupied recordings."
    a, h = X[y == AWAY], X[y == HOME]
    pooled = np.sqrt((a.var(axis=0) + h.var(axis=0)) / 2)
    d = np.divide(h.mean(axis=0) - a.mean(axis=0), pooled,
                  out=np.zeros(X.shape[1]), where=pooled > 0)

    lines = ["  class separation, occupied vs empty (Cohen's d):"]
    for i in np.argsort(-np.abs(d))[:top]:
        lines.append(f"    {FEATURE_NAMES[i]:<18} empty={a[:, i].mean():10.4f}  "
                     f"occupied={h[:, i].mean():10.4f}  d={d[i]:+6.2f} "
                     f"{'#' * min(int(abs(d[i]) * 6), 30)}")
    best = float(np.abs(d).max())
    if best < 0.5:
        lines.append(f"\n    NO USABLE SIGNAL YET (best |d| = {best:.2f}). Every feature")
        lines.append(f"    overlaps between empty and occupied. Check, in this order:")
        lines.append(f"      1. was the room genuinely empty during the empty recordings?")
        lines.append(f"      2. is the AGC stable? a wandering gain adds fake motion to both")
        lines.append(f"      3. are the boards placed so a person sits BETWEEN them?")
    elif best < 1.0:
        lines.append(f"\n    Weak but present (best |d| = {best:.2f}). A model can work with")
        lines.append(f"    this, but more recordings and cleaner captures will matter a lot.")
    else:
        lines.append(f"\n    Strong separation (best |d| = {best:.2f}). Worth training on.")
    return "\n".join(lines)


if __name__ == "__main__":
    ds = build(force=True)
    print()
    print(summarize(ds))
    print()
    print(separation(ds))
