"""
Deployment-facing inference pipeline for CSI presence detection.

Three pieces:
  predict_presence()    - CSI window  -> (presence, confidence)
  should_run_ac()       - prediction  -> HVAC command
  load_esp32_csi_csv()  - our own ESP32-CSI-Tool capture -> the SAME feature space

The whole point of the design is that the ESP32 path and the Intel-5300 path
converge on `window_features()`.  Nothing downstream of that function knows or
cares which radio produced the data.
"""

import csv
import os
import numpy as np

from .features import (window_features, windows_from_arrays, _sanitize_phase,
                       FEATURE_NAMES, SCALE_FREE_FEATURES, WINDOW,
                       fit_site_baseline, apply_site_baseline)

AWAY, HOME = 0, 1
LABEL_NAMES = {AWAY: "AWAY", HOME: "HOME"}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_presence(model, csi_window, baseline=None, fs=50.0):
    """Predict occupancy from one window of CSI.

    model       : classifier trained on calibrated scale-free features
    csi_window  : (amplitude, phase) arrays each (n_packets, n_subcarriers, n_rx),
                  or a precomputed length-16 feature vector
    baseline    : this site's quiet baseline from fit_site_baseline().  Required
                  for a calibrated model -- without it the features are in raw
                  units the model was never trained on.
    fs          : packet rate of the capture, in Hz

    Returns (presence_int, confidence_float) where confidence is the model's
    probability for the class it actually predicted.
    """
    if isinstance(csi_window, tuple):
        amp, ph = csi_window
        feats = window_features(np.asarray(amp), np.asarray(ph), fs=fs)
    else:
        feats = np.asarray(csi_window, dtype=float).ravel()

    if feats.shape[0] == len(FEATURE_NAMES):
        if baseline is None:
            raise ValueError(
                "A site baseline is required. Fit one with fit_site_baseline() on "
                "unlabelled windows from this install before predicting.")
        x = apply_site_baseline(feats[None, :], FEATURE_NAMES, baseline)
    elif feats.shape[0] == len(SCALE_FREE_FEATURES):
        x = feats[None, :]  # already calibrated
    else:
        raise ValueError(f"expected {len(FEATURE_NAMES)} or {len(SCALE_FREE_FEATURES)} "
                         f"features, got {feats.shape[0]}")

    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))
    return pred, float(proba[pred])


def should_run_ac(prediction, confidence, threshold=0.7):
    """Turn a prediction into an HVAC command.

    The asymmetry here is deliberate and is a comfort-vs-energy trade-off.
    Switching the AC off wrongly is a comfort failure the occupant notices
    immediately; leaving it on wrongly costs a few cents.  So AWAY only wins when
    the model is confident: a low-confidence AWAY keeps the AC running.

    Returns (run_ac: bool, reason: str).
    """
    if confidence < threshold:
        return True, f"low confidence ({confidence:.2f} < {threshold:.2f}) - failing safe, AC ON"
    if prediction == HOME:
        return True, f"occupied (confidence {confidence:.2f}) - AC ON"
    return False, f"empty (confidence {confidence:.2f}) - AC OFF, saving energy"


# ---------------------------------------------------------------------------
# ESP32-CSI-Tool ingestion
# ---------------------------------------------------------------------------
#
# Target format: StevenMHernandez/ESP32-CSI-Tool, which prints one line per
# received packet over serial.  Written from the tool's documented output; every
# point where we guessed is tagged ASSUMPTION and is cheap to correct once a real
# capture exists.  Run `verify_esp32_assumptions()` on the first real file to get
# a checklist of which ones held.

ESP32_EXPECTED_COLUMNS = [
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", "CSI_DATA",
]

# ASSUMPTION 1: the CSI payload lives in a column named "CSI_DATA".  Some builds
# name it "data".  Both are accepted below.
CSI_COLUMN_CANDIDATES = ("CSI_DATA", "data", "csi_data")

# ASSUMPTION 2: ESP-IDF hands back int8 pairs in (imaginary, real) order -- this
# is the documented ESP-IDF wifi_csi_info layout and is the opposite of what most
# people assume.  If presence detection works but phase features look like noise,
# flip this first.
ESP32_IMAG_FIRST = True

# ASSUMPTION 3: HT20 capture -> 64 raw subcarriers (len == 128 bytes).  In HT20
# the LLTF occupies indices 0-63 with DC at 0 and guard bands at the edges, so
# only these carry usable energy.  A HT40 capture (len 256/384) needs a different
# map; we detect the length and warn rather than silently mis-slice.
ESP32_HT20_VALID_SUBCARRIERS = list(range(6, 32)) + list(range(33, 59))

# ASSUMPTION 4: single RX antenna, so n_rx = 1.  The Intel 5300 data had 3.  Our
# features average over the RX axis, so this changes their variance but not their
# meaning -- expect the ESP32's per-window estimates to be noisier.

# ASSUMPTION 5: local_timestamp is in microseconds since boot.  Used only to
# derive the true packet rate, which the spectral feature needs.
ESP32_TIMESTAMP_UNITS_PER_SEC = 1_000_000


def _parse_csi_blob(blob):
    """Parse the '[1 -2 3 ...]' bracketed int list the tool prints.

    ASSUMPTION 7: values are whitespace-separated.  Comma-separated is handled too,
    since some forks of the tool emit that instead.
    """
    return np.array(blob.strip().strip("[]").replace(",", " ").split(), dtype=np.float32)


def load_esp32_csi_csv(filepath, window=None, stride=None, valid_subcarriers=None,
                       imag_first=ESP32_IMAG_FIRST, verbose=True):
    """Load an ESP32-CSI-Tool CSV into the exact feature space the model expects.

    Returns dict with:
      X    : (n_windows, 16) feature matrix, same columns as FEATURE_NAMES
      fs   : measured packet rate in Hz
      amp, phase : the raw arrays, for plotting or debugging
    """
    rows, ts = [], []
    with open(filepath, newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        has_header = any(c in sample for c in CSI_COLUMN_CANDIDATES)
        if has_header:
            reader = csv.DictReader(fh)
            csi_col = next((c for c in CSI_COLUMN_CANDIDATES if c in (reader.fieldnames or [])), None)
            if csi_col is None:
                raise ValueError(f"no CSI column found; saw {reader.fieldnames}")
            for rec in reader:
                blob = rec.get(csi_col) or ""
                if "[" not in blob:
                    continue
                rows.append(blob)
                try:
                    ts.append(float(rec.get("local_timestamp") or 0))
                except ValueError:
                    ts.append(0.0)
        else:
            # ASSUMPTION 6: headerless capture -> CSI blob is the last bracketed
            # field on the line, timestamp at the documented column index.
            for line in fh:
                if "[" not in line:
                    continue
                blob = line[line.index("["):line.rindex("]") + 1]
                rows.append(blob)
                parts = line.split(",")
                idx = ESP32_EXPECTED_COLUMNS.index("local_timestamp")
                try:
                    ts.append(float(parts[idx]))
                except (ValueError, IndexError):
                    ts.append(0.0)

    if not rows:
        raise ValueError(f"no CSI rows parsed from {filepath}")

    raw = [_parse_csi_blob(b) for b in rows]
    n_vals = int(np.median([len(r) for r in raw]))
    raw = np.array([r for r in raw if len(r) == n_vals], dtype=np.float32)
    n_sub_total = n_vals // 2

    if valid_subcarriers is None:
        if n_sub_total == 64:
            valid_subcarriers = ESP32_HT20_VALID_SUBCARRIERS
        else:
            # ASSUMPTION 3 did not hold -- keep everything with real energy rather
            # than guessing a subcarrier map for a bandwidth we have not seen.
            valid_subcarriers = None
            if verbose:
                print(f"  [warn] {n_sub_total} subcarriers, not the expected 64 (HT20). "
                      f"Using all non-zero subcarriers; check ASSUMPTION 3.")

    if imag_first:
        imag, real = raw[:, 0::2], raw[:, 1::2]
    else:
        real, imag = raw[:, 0::2], raw[:, 1::2]
    csi = (real + 1j * imag)[:, :, None]  # (n_pkt, n_sub, n_rx=1)

    if valid_subcarriers is not None:
        csi = csi[:, valid_subcarriers, :]
    else:
        keep = np.abs(csi[:, :, 0]).mean(axis=0) > 0
        csi = csi[:, keep, :]

    amp = np.abs(csi).astype(np.float32)
    phase = _sanitize_phase(np.angle(csi)).astype(np.float32)

    tsa = np.asarray(ts[:len(csi)], dtype=np.float64)
    span = tsa[-1] - tsa[0] if len(tsa) > 1 else 0.0
    fs = (len(tsa) - 1) / (span / ESP32_TIMESTAMP_UNITS_PER_SEC) if span > 0 else 50.0
    if not np.isfinite(fs) or not (1.0 < fs < 5000.0):
        if verbose:
            print(f"  [warn] implausible packet rate from timestamps; defaulting to 50 Hz. "
                  f"Check ASSUMPTION 5.")
        fs = 50.0

    # Window length is defined in SECONDS, not packets, so a faster ESP32 capture
    # still yields physically comparable windows to the 2.56 s Intel windows.
    if window is None:
        window = max(32, int(round(fs * (WINDOW / 50.0))))
    if stride is None:
        stride = window

    X = windows_from_arrays(amp, phase, window=window, stride=stride, fs=fs)
    if verbose:
        print(f"  loaded {len(csi)} packets, {csi.shape[1]} usable subcarriers, "
              f"{fs:.1f} Hz -> {len(X)} windows of {window} packets ({window/fs:.2f} s)")
    return {"X": X, "fs": fs, "amp": amp, "phase": phase,
            "window": window, "n_subcarriers": csi.shape[1]}


def verify_esp32_assumptions(filepath):
    """Print a checklist against a real capture. Run this the day the data lands."""
    print(f"Checking ESP32 assumptions against {filepath}")
    with open(filepath) as fh:
        header = fh.readline().strip()
        first = fh.readline().strip()
    print(f"  header: {header[:160]}")
    col = next((c for c in CSI_COLUMN_CANDIDATES if c in header), None)
    print(f"  [1] CSI column name .......... {'OK: ' + col if col else 'FAILED - inspect header'}")
    if "[" in first:
        blob = first[first.index("["):first.rindex("]") + 1]
        vals = _parse_csi_blob(blob)
        print(f"  [3] payload length ........... {len(vals)} ints = {len(vals)//2} subcarriers "
              f"({'OK, HT20' if len(vals) == 128 else 'UNEXPECTED - set valid_subcarriers'})")
    out = load_esp32_csi_csv(filepath, verbose=True)
    print(f"  [5] packet rate .............. {out['fs']:.1f} Hz")
    print(f"  [2] imag/real order .......... not auto-detectable; if phase_* features "
          f"look like noise, flip ESP32_IMAG_FIRST")
    return out


def make_synthetic_esp32_csv(path, n_packets=1200, fs=100.0, occupied=True, seed=0):
    """Emit a file in the assumed ESP32-CSI-Tool format so the loader is testable today.

    This is a FORMAT fixture, not physics.  It exists to prove the ESP32 code path
    runs end-to-end; it must not be used to claim anything about accuracy.
    """
    rng = np.random.default_rng(seed)
    n_sub = 64
    base = np.zeros(n_sub)
    base[ESP32_HT20_VALID_SUBCARRIERS] = 20 + 8 * np.sin(
        np.linspace(0, 3.5, len(ESP32_HT20_VALID_SUBCARRIERS)))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ESP32_EXPECTED_COLUMNS)
        # occupied -> slow correlated fading; empty -> static channel + thermal noise
        walk = np.cumsum(rng.normal(0, 0.35, n_packets)) if occupied else np.zeros(n_packets)
        for i in range(n_packets):
            gain = 1.0 + 0.30 * np.sin(walk[i]) if occupied else 1.0
            mag = base * gain + rng.normal(0, 0.45, n_sub)
            phi = rng.normal(0, 0.30 if occupied else 0.03, n_sub) + np.linspace(0, 2.0, n_sub)
            mag[base == 0] = 0.0
            re = np.round(mag * np.cos(phi)).astype(int)
            im = np.round(mag * np.sin(phi)).astype(int)
            inter = np.empty(2 * n_sub, dtype=int)
            inter[0::2], inter[1::2] = (im, re) if ESP32_IMAG_FIRST else (re, im)
            row = ["CSI_DATA", "AP", "aa:bb:cc:dd:ee:ff", -42, 11, 1, 7, 1, 1, 1, 0, 0, 0, 1,
                   -94, 0, 6, 0, int(i * ESP32_TIMESTAMP_UNITS_PER_SEC / fs), 0, 100, 0, 0, 0,
                   2 * n_sub, "[" + " ".join(map(str, inter)) + "]"]
            w.writerow(row)
    return path
