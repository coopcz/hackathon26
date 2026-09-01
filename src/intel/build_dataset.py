"""Parse every capture into a windowed feature table and cache it to disk."""

import os
import numpy as np

from ..features import windows_from_arrays, FEATURE_NAMES
from .csi_reader import file_to_arrays

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "WiFi-CrowdCounting")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "artifacts", "features.npz")


def build(force=False):
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return z["X"], z["y"], z["room"], z["n_people"], z["session"]

    X, y, room, n_people, session = [], [], [], [], []
    for r in sorted(os.listdir(DATA_ROOT)):
        rdir = os.path.join(DATA_ROOT, r)
        if not os.path.isdir(rdir):
            continue
        for f in sorted(os.listdir(rdir)):
            if not f.endswith("p"):
                continue
            n = int(f[:-1])
            amp, ph = file_to_arrays(os.path.join(rdir, f))
            W = windows_from_arrays(amp, ph)
            X.append(W)
            # binary presence label: 0 people = AWAY, >=1 person = HOME
            y.append(np.full(len(W), 1 if n > 0 else 0))
            room.append(np.full(len(W), r))
            n_people.append(np.full(len(W), n))
            session.append(np.full(len(W), f"{r}/{f}"))
            print(f"  {r}/{f}: {len(amp)} packets -> {len(W)} windows")

    X = np.vstack(X)
    y = np.concatenate(y)
    room = np.concatenate(room)
    n_people = np.concatenate(n_people)
    session = np.concatenate(session)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, room=room, n_people=n_people, session=session,
                        feature_names=np.array(FEATURE_NAMES))
    return X, y, room, n_people, session


if __name__ == "__main__":
    X, y, room, n_people, session = build(force=True)
    print(f"\nfeature matrix {X.shape}, labels {np.bincount(y)}")
