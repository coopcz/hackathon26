"""
Manual labelling for live ESP32 capture sessions.

We have no door sensor, so ground truth comes from a person with a text file.
That is a perfectly good label source -- it is what the WiFi-CrowdCounting
dataset did too, just encoded in directory names (`0p`..`8p`) instead of
timestamps -- but it moves the labelling problem from "parse a filename" to
"align two clocks", which is where the care has to go.


THE LOG FORMAT
--------------
One CSV, written by hand (or by `tools/mark.py`) during the capture:

    timestamp,event,n_people,note
    2026-08-31T19:05:00,capture_start,0,room empty AC off
    2026-08-31T19:12:30,entered,1,walked in sat on couch
    2026-08-31T19:20:05,left,0,stepped out
    2026-08-31T19:24:40,entered,2,two of us walking around
    2026-08-31T19:31:00,capture_end,0,

  timestamp : ISO-8601 (`2026-08-31T19:05:00`), a bare wall clock (`19:05:00`),
              or plain seconds since capture start (`0`, `450.0`).  All three
              work; do not mix ISO and bare-seconds in one file.
  event     : capture_start | entered | left | set | capture_end
  n_people  : the occupancy AFTER this event.  This is what is actually used.
              `entered`/`left` without a count fall back to +1/-1, but writing
              the absolute count is more robust -- one missed line then costs
              you one interval instead of corrupting the whole rest of the file.
  note      : free text, ignored.


CLOCK ALIGNMENT -- the one thing that can silently ruin the labels
------------------------------------------------------------------
`local_timestamp` in a csi_recv capture is microseconds since the ESP32 booted.
It has no relationship to wall-clock time.  So the join is anchored on the
`capture_start` line: that instant is defined to be the timestamp of the FIRST
CSI packet in the file (t = 0 on the loader's relative clock).  Everything else
in the log is expressed as an offset from it.

That anchoring is only as good as the operator's stopwatch.  A 20 s error in
`capture_start` mislabels 20 s of every transition in the session.  Two defences:

  * GUARD_SECONDS -- windows that overlap a transition, plus a margin either
    side, are dropped rather than labelled.  This is the same discipline as the
    Intel pipeline, where a window never spans two occupancy conditions.
  * `clock_offset` -- if you discover afterwards that the log ran N seconds
    fast, pass `clock_offset=-N` instead of re-typing the file.

Output is byte-identical in structure to what `src/build_dataset.py` caches, so
anything that consumes the Intel training set consumes this unchanged:
    X (n,16) float, y (n,) 0=AWAY/1=HOME, room (n,), n_people (n,), session (n,),
    feature_names (16,)
"""

import csv
import os
import re
from datetime import datetime, timedelta

import numpy as np

from .esp_csi import load_esp32_csi_csv
from .features import FEATURE_NAMES

AWAY, HOME = 0, 1

# Drop windows within this many seconds of an occupancy change.  A person does
# not teleport: they are half-in the room while the door swings, and the
# operator's keystroke lands somewhere in that interval.  2 s is a little under
# one 2.56 s window, so a transition costs at most two windows either side.
GUARD_SECONDS = 2.0

VALID_EVENTS = {"capture_start", "capture_end", "entered", "left", "set"}
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?")
_CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$")


class ManualLogError(ValueError):
    """The manual log is malformed in a way that would produce wrong labels."""


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_time(raw):
    """Return (kind, value): ('abs', datetime) or ('rel', seconds)."""
    s = raw.strip()
    if _ISO_RE.match(s):
        return "abs", datetime.fromisoformat(s.replace(" ", "T"))
    if _CLOCK_RE.match(s):
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.append(0.0)
        return "clock", timedelta(hours=parts[0], minutes=parts[1],
                                  seconds=parts[2]).total_seconds()
    return "rel", float(s)


def read_manual_log(path):
    """Parse the operator's log into ordered (t_seconds_from_start, n_people, event).

    Raises ManualLogError rather than guessing whenever the file is ambiguous --
    a silently mislabelled training set is far more expensive than a stack trace
    at 11pm.
    """
    with open(path, newline="") as fh:
        # '#' comment lines are stripped before the CSV reader sees them, so the
        # template can explain itself without the explanations parsing as events.
        kept = [(n, ln) for n, ln in enumerate(fh, start=1)
                if not ln.lstrip().startswith("#") and ln.strip()]
    linenos = [n for n, _ in kept[1:]]          # data rows only; [0] is the header
    rows = [(lineno, r) for lineno, r in zip(linenos, csv.DictReader(ln for _, ln in kept))
            if r.get("event") and r["event"].strip()]
    if not rows:
        raise ManualLogError(f"{path}: no event rows "
                             f"(expected a header 'timestamp,event,n_people,note')")

    parsed, kinds = [], set()
    for i, r in rows:
        event = r["event"].strip().lower()
        if event not in VALID_EVENTS:
            raise ManualLogError(f"{path} line {i}: unknown event {event!r}; "
                                 f"expected one of {sorted(VALID_EVENTS)}")
        try:
            kind, value = _parse_time(r["timestamp"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ManualLogError(f"{path} line {i}: unparseable timestamp "
                                 f"{r.get('timestamp')!r} ({exc})") from None
        kinds.add(kind)
        n_raw = (r.get("n_people") or "").strip()
        n = int(n_raw) if n_raw else None
        parsed.append({"kind": kind, "value": value, "event": event,
                       "n_people": n, "line": i})

    if len(kinds) > 1:
        raise ManualLogError(f"{path}: mixes timestamp styles {sorted(kinds)}. "
                             f"Use ISO-8601 throughout, or seconds throughout.")
    kind = kinds.pop()

    starts = [p for p in parsed if p["event"] == "capture_start"]
    if len(starts) != 1:
        raise ManualLogError(f"{path}: expected exactly one 'capture_start' row, "
                             f"found {len(starts)}. Without it the log cannot be "
                             f"aligned to the CSI clock.")
    origin = starts[0]["value"]

    events = []
    for p in parsed:
        if kind == "abs":
            t = (p["value"] - origin).total_seconds()
        else:
            t = float(p["value"]) - float(origin)
        events.append({"t": t, "event": p["event"],
                       "n_people": p["n_people"], "line": p["line"]})
    events.sort(key=lambda e: e["t"])

    # Resolve the occupancy step function.
    n = 0
    for e in events:
        if e["event"] == "capture_start":
            n = e["n_people"] if e["n_people"] is not None else 0
        elif e["event"] in ("entered", "set"):
            n = e["n_people"] if e["n_people"] is not None else n + 1
        elif e["event"] == "left":
            n = e["n_people"] if e["n_people"] is not None else n - 1
        # capture_end carries the occupancy forward unchanged
        if n < 0:
            raise ManualLogError(f"{path} line {e['line']}: occupancy went negative. "
                                 f"A 'left' is missing its matching 'entered', or an "
                                 f"n_people column is wrong.")
        e["occupancy"] = n

    end = [e for e in events if e["event"] == "capture_end"]
    return {"events": events, "t_end": end[-1]["t"] if end else None, "style": kind}


def occupancy_intervals(log):
    """Collapse the event list into [(t_start, t_end, n_people), ...] segments."""
    events = log["events"]
    segments = []
    for e, nxt in zip(events, events[1:] + [None]):
        t1 = nxt["t"] if nxt is not None else (log["t_end"] if log["t_end"] is not None
                                               else float("inf"))
        if t1 > e["t"]:
            segments.append((e["t"], t1, e["occupancy"]))
    # merge adjacent segments with the same occupancy so a stray duplicate event
    # does not manufacture a spurious transition (and a spurious guard band)
    merged = []
    for s in segments:
        if merged and merged[-1][2] == s[2] and abs(merged[-1][1] - s[0]) < 1e-9:
            merged[-1] = (merged[-1][0], s[1], s[2])
        else:
            merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# The connector
# ---------------------------------------------------------------------------

def label_from_manual_log(csi_capture_file, manual_log_file, site=None,
                          session=None, guard_seconds=GUARD_SECONDS,
                          clock_offset=0.0, out_npz=None, verbose=True,
                          **loader_kwargs):
    """Join a real esp-csi capture to a hand-written occupancy log.

    Produces exactly the labelled training table `src/build_dataset.py` produces
    for the Intel data, so it drops straight into `build_calibrated()` /
    `train_production_model()` with no adapter.

    csi_capture_file : the CSV written by csi_data_read_parse.py -s
    manual_log_file  : the operator's log (see module docstring)
    site             : site/room id, the calibration unit. Defaults to the
                       capture's basename.  Every window from one capture shares
                       it, because per-site baselines are fitted per site.
    guard_seconds    : windows within this margin of an occupancy change are
                       discarded rather than labelled (see module docstring).
    clock_offset     : seconds to add to every log time, to correct a stopwatch
                       that ran fast (negative) or slow (positive).
    out_npz          : if given, save the table there in build_dataset's format.

    Returns dict with X, y, room, n_people, session, feature_names, plus the
    loader output under "capture" and a "report" of what was kept and dropped.
    """
    out = load_esp32_csi_csv(csi_capture_file, verbose=verbose, **loader_kwargs)
    log = read_manual_log(manual_log_file)
    segments = [(a + clock_offset, b + clock_offset, n)
                for a, b, n in occupancy_intervals(log)]
    if not segments:
        raise ManualLogError(f"{manual_log_file}: no occupancy intervals; the log "
                             f"needs at least a capture_start and a capture_end.")

    t0, t1 = out["window_t0"], out["window_t1"]
    n_windows = len(t0)
    if n_windows == 0:
        raise ValueError(f"{csi_capture_file}: capture is shorter than one "
                         f"{out['window']}-packet window; record for longer.")

    # transition instants (interior segment boundaries only)
    transitions = [end for (_, end, _) in segments[:-1]]

    keep, occ = [], []
    dropped_guard = dropped_uncovered = dropped_straddle = 0
    for i in range(n_windows):
        a, b = float(t0[i]), float(t1[i])
        if any(a < x + guard_seconds and b > x - guard_seconds for x in transitions):
            dropped_guard += 1
            continue
        covering = [n for (s, e, n) in segments if s <= a and b <= e]
        if not covering:
            # either outside capture_start..capture_end, or spanning a boundary
            # that the guard band did not already catch
            overlapping = [n for (s, e, n) in segments if e > a and s < b]
            if len(overlapping) > 1:
                dropped_straddle += 1
            else:
                dropped_uncovered += 1
            continue
        keep.append(i)
        occ.append(covering[0])

    keep = np.asarray(keep, dtype=int)
    occ = np.asarray(occ, dtype=int)
    X = out["X"][keep] if len(keep) else np.empty((0, len(FEATURE_NAMES)))
    # Same binary convention as build_dataset.py: 0 people = AWAY, >=1 = HOME.
    y = (occ > 0).astype(int) if len(keep) else np.empty(0, dtype=int)

    site = site or os.path.splitext(os.path.basename(csi_capture_file))[0]
    session = session or site
    room_arr = np.full(len(keep), site)
    session_arr = np.full(len(keep), session)

    report = {
        "n_windows_total": n_windows,
        "n_labelled": int(len(keep)),
        "n_dropped_guard": dropped_guard,
        "n_dropped_straddle": dropped_straddle,
        "n_dropped_outside_capture": dropped_uncovered,
        "n_away": int((y == 0).sum()) if len(keep) else 0,
        "n_home": int((y == 1).sum()) if len(keep) else 0,
        "occupancy_levels": {int(k): int(v) for k, v in
                             zip(*np.unique(occ, return_counts=True))} if len(keep) else {},
        "segments": segments,
        "window_seconds": out["window"] / out["fs"],
        "fs": out["fs"],
    }

    if verbose:
        print(f"  manual labels: {report['n_labelled']}/{n_windows} windows kept "
              f"({report['n_away']} AWAY / {report['n_home']} HOME), "
              f"dropped {dropped_guard} in guard bands, {dropped_straddle} straddling, "
              f"{dropped_uncovered} outside the logged capture")
        for s, e, n in segments:
            print(f"    {s:8.1f}s - {e:8.1f}s  {n} people "
                  f"-> {'HOME' if n > 0 else 'AWAY'}")
        if report["n_away"] == 0 or report["n_home"] == 0:
            print("  [warn] only one class survived. A capture with no empty-room "
                  "stretch cannot calibrate (fit_site_baseline needs a quiet 5th "
                  "percentile) and cannot train.")

    result = {"X": X, "y": y, "room": room_arr, "n_people": occ,
              "session": session_arr, "feature_names": np.array(FEATURE_NAMES),
              "capture": out, "report": report}

    if out_npz:
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
        np.savez_compressed(out_npz, X=X, y=y, room=room_arr, n_people=occ,
                            session=session_arr,
                            feature_names=np.array(FEATURE_NAMES))
        if verbose:
            print(f"  wrote {out_npz}")
    return result


def merge_sessions(results):
    """Stack several label_from_manual_log() results into one training table."""
    if not results:
        raise ValueError("nothing to merge")
    return {
        "X": np.vstack([r["X"] for r in results]),
        "y": np.concatenate([r["y"] for r in results]),
        "room": np.concatenate([r["room"] for r in results]),
        "n_people": np.concatenate([r["n_people"] for r in results]),
        "session": np.concatenate([r["session"] for r in results]),
        "feature_names": np.array(FEATURE_NAMES),
    }


MANUAL_LOG_TEMPLATE = """timestamp,event,n_people,note
# Start this file at the SAME MOMENT you start csi_data_read_parse.py.
# timestamp: ISO-8601 (2026-08-31T19:05:00) or seconds from start (0, 450).
# event: capture_start | entered | left | set | capture_end
# n_people: occupancy AFTER the event. Always write it if you can.
0,capture_start,0,room empty - leave it empty for 5 min to calibrate
300,entered,1,one person walks in and sits still
600,set,1,same person now moving around the room
900,left,0,room empty again
1200,entered,2,two people moving
1500,left,0,everyone out
1800,capture_end,0,
"""


def write_manual_log_template(path):
    """Drop a ready-to-edit log template next to the capture."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(MANUAL_LOG_TEMPLATE)
    return path
