"""Turn the downloaded EHUNAM subset into the same feature table as the Intel data."""

import os
import glob
import numpy as np

from .features import windows_from_arrays, FEATURE_NAMES, WINDOW
from .ehunam import load_measurement, UnsupportedBandwidth, PRESENCE_GROUPS, fetch_summary

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT_DIR = os.path.join(_ROOT, "data", "ehunam", "mat")
CACHE = os.path.join(_ROOT, "artifacts", "ehunam_features.npz")

# Same 2.56 s window as the Intel baseline (128 packets at 50 Hz).  EHUNAM's
# captures do not all land at 50 Hz -- the FTP recordings run at 10-20 Hz, which
# is already below the target so decimation leaves them alone -- so the window is
# defined in SECONDS and converted per file.  A window then covers the same span
# of real time everywhere, which is what makes the features comparable.
TARGET_FS = 50.0
WINDOW_SECONDS = WINDOW / TARGET_FS      # 2.56 s
MIN_WINDOW_PACKETS = 24


def build(force=False, verbose=True):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    summary = fetch_summary()
    X, group, app, env, date, setid, npeople, nmach, fname = [], [], [], [], [], [], [], [], []
    skipped = 0

    for p in sorted(glob.glob(os.path.join(MAT_DIR, "*.mat"))):
        base = os.path.basename(p)
        meta_s = summary.get(base, {})
        if meta_s.get("BW") != "20":
            skipped += 1
            continue
        try:
            amp, ph, fs, meta = load_measurement(p, target_fs=TARGET_FS)
        except UnsupportedBandwidth:
            skipped += 1
            continue

        w = max(MIN_WINDOW_PACKETS, int(round(fs * WINDOW_SECONDS)))
        W = windows_from_arrays(amp, ph, window=w, stride=w, fs=fs)
        if len(W) == 0:
            skipped += 1
            continue

        a = meta_s.get("Application", meta["app"])
        X.append(W)
        n = len(W)
        group += [PRESENCE_GROUPS.get(a, "?")] * n
        app += [a] * n
        env += [meta_s.get("Enviroment", meta["environment"])] * n
        date += [meta_s.get("Date", "")] * n
        setid += [meta_s.get("Set", "")] * n
        npeople += [int(meta_s.get("N_People") or 0)] * n
        nmach += [len(meta_s.get("Machine") or "")] * n
        fname += [base] * n

    out = {
        "X": np.vstack(X), "group": np.array(group), "app": np.array(app),
        "env": np.array(env), "date": np.array(date), "setid": np.array(setid),
        "n_people": np.array(npeople), "n_machines": np.array(nmach),
        "src_file": np.array(fname), "feature_names": np.array(FEATURE_NAMES),
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    if verbose:
        print(f"  {out['X'].shape[0]} windows from {len(set(fname))} files "
              f"({skipped} files skipped)")
    return out


if __name__ == "__main__":
    d = build(force=True)
    import collections
    print("\nfeature matrix", d["X"].shape)
    print("groups     :", dict(collections.Counter(d["group"])))
    print("apps       :", dict(collections.Counter(d["app"])))
    print("environments:", dict(collections.Counter(d["env"])))
