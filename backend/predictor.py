"""
Live occupancy inference on the packet stream.

Mirrors the offline path in src/train_esp32.py exactly, because a training/serving
skew here would be invisible: the model would keep returning confident numbers
that simply mean something different from what it learned.  Three places that
skew could creep in, and what stops it:

  SUBCARRIERS  The model was trained on the pinned HT40 map, not on whichever
               subcarriers happen to be non-zero right now.  The map ships inside
               the bundle and is applied here verbatim.

  PHASE        backend/features.py computes a per-packet phase with plain
               np.angle for the charts.  The model was trained on phase that is
               DETRENDED ACROSS THE WHOLE WINDOW (_sanitize_phase) -- raw phase
               carries a large random per-packet slope that swamps the signal.
               So the window is rebuilt from raw_csi here rather than reusing the
               packet's `phase` field.

  CALIBRATION  Features are divided by the site baseline that was fitted at
               training time and stored in the bundle.  Without it the numbers
               are in units the model never saw.

Returns None until a full window has arrived, and whenever no model is loaded.
It never guesses: no model means no verdict.
"""

import logging
import threading
from collections import deque

import joblib
import numpy as np

from src.features import _sanitize_phase, window_features
from src.pipeline import AWAY, HOME, LABEL_NAMES, should_run_ac
from src.train_esp32 import apply_calibration

log = logging.getLogger(__name__)

# Predict about twice a second rather than on every packet. The window is 2.56 s
# of context, so consecutive packets carry almost identical information and
# scoring all ~88/s would burn CPU for no extra signal.
PREDICT_EVERY = 40


class Predictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.lock = threading.Lock()
        self.bundle = None
        self.error = None
        self.window = deque()
        self.smooth = deque()
        self.n_seen = 0
        self.latest = None
        self.load()

    # -- model -----------------------------------------------------------
    def load(self):
        """(Re)load the trained bundle. A missing model is a normal state, not
        an error: the dashboard is useful for collecting data before one exists."""
        with self.lock:
            self.bundle, self.error = None, None
            try:
                b = joblib.load(self.model_path)
            except FileNotFoundError:
                return False
            except Exception as exc:                      # corrupt / version skew
                self.error = f"could not load model: {exc}"
                log.warning(self.error)
                return False
            self.bundle = b
            self.window = deque(maxlen=int(b["window_packets"]))
            self.smooth = deque(maxlen=int(b["smoothing_windows"]))
            self.n_seen = 0
            self.latest = None
            log.info("loaded model: %s / %s, %d-packet window, threshold %.2f",
                     b["feature_set"], b["model"].__class__.__name__,
                     b["window_packets"], b["threshold"])
            return True

    def reset(self):
        """Drop accumulated context -- on connect, disconnect, or replay start."""
        with self.lock:
            self.window.clear()
            self.smooth.clear()
            self.n_seen = 0
            self.latest = None

    # -- inference -------------------------------------------------------
    def append(self, packet):
        """Feed one packet. Returns a verdict dict, or None if there is nothing
        new to say yet."""
        b = self.bundle
        if b is None:
            return None
        with self.lock:
            self.window.append(packet.raw_csi)
            self.n_seen += 1
            if len(self.window) < self.window.maxlen or self.n_seen % PREDICT_EVERY:
                return None
            raw = np.asarray(self.window, dtype=np.float32)
        try:
            return self._score(raw, b)
        except Exception as exc:                          # never kill the serial thread
            log.warning("prediction failed: %s", exc)
            return None

    def _score(self, raw, b):
        # esp-csi serialises [imag, real, imag, real, ...] int8 pairs
        imag, real = raw[:, 0::2], raw[:, 1::2]
        csi = (real + 1j * imag)[:, :, None]              # (T, n_sub, n_rx=1)
        sub = b["subcarriers"]
        if csi.shape[1] <= max(sub):
            raise ValueError(f"packet has {csi.shape[1]} subcarriers, model expects "
                             f"the {len(sub)}-wide HT40 map")
        csi = csi[:, sub, :]

        amp = np.abs(csi).astype(np.float32)
        ph = _sanitize_phase(np.angle(csi)).astype(np.float32)
        feats = window_features(amp, ph, fs=b["fs"])
        Xc, _ = apply_calibration(feats[None, :], b["baseline"], b["feature_set"])

        p_home = float(b["model"].predict_proba(Xc)[0, HOME])
        self.smooth.append(p_home)
        p_home = float(np.mean(self.smooth))
        p_away = 1.0 - p_home

        # The tuned threshold already encodes the comfort-vs-energy asymmetry, so
        # it is the only gate: AWAY has to clear it, HOME is the fail-safe default.
        pred = AWAY if p_away >= b["threshold"] else HOME
        confidence = p_away if pred == AWAY else p_home
        run_ac, reason = should_run_ac(pred, confidence, threshold=b["threshold"])

        verdict = {
            "presence": LABEL_NAMES[pred],
            "confidence": confidence,
            "p_home": p_home,
            "run_ac": run_ac,
            "reason": reason,
            "window_packets": int(self.window.maxlen),
            "smoothing_windows": int(self.smooth.maxlen),
        }
        self.latest = verdict
        return verdict

    # -- status ----------------------------------------------------------
    def status(self):
        b = self.bundle
        if b is None:
            return {"loaded": False, "error": self.error,
                    "message": "No trained model. Run: python -m src.train_esp32"}
        return {
            "loaded": True,
            "error": self.error,
            "feature_set": b["feature_set"],
            "model": b["model"].__class__.__name__,
            "threshold": b["threshold"],
            "window_packets": int(b["window_packets"]),
            "smoothing_windows": int(b["smoothing_windows"]),
            "trained_at": b.get("trained_at"),
            "n_recordings": b.get("n_recordings"),
            "n_windows": b.get("n_windows"),
            "cv": b.get("cv"),
            "deployable": b.get("deployable"),
            "buffer": len(self.window),
            "latest": self.latest,
        }
