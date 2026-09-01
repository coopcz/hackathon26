"""
Deployment-facing inference pipeline for CSI presence detection.

Three pieces:
  predict_presence()    - CSI window  -> (presence, confidence)
  should_run_ac()       - prediction  -> HVAC command
  load_esp32_csi_csv()  - an esp-csi capture -> the SAME feature space (src/esp_csi.py)

The whole point of the design is that the ESP32 path and the Intel-5300 path
converge on `window_features()`.  Nothing downstream of that function knows or
cares which radio produced the data.
"""

import numpy as np

from .features import (window_features, FEATURE_NAMES, SCALE_FREE_FEATURES,
                       apply_site_baseline)

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
# ESP32 ingestion
# ---------------------------------------------------------------------------
# Lives in src/esp_csi.py now that the format is known rather than guessed.  It
# targets Espressif's own esp-csi `csi_recv` output (both the 15-column
# C5/C6/C61 schema and the 25-column schema the older parts print), and still
# reads the third-party ESP32-CSI-Tool format this repo originally assumed.
# Re-exported here so `from src.pipeline import load_esp32_csi_csv` keeps working.

from .esp_csi import (  # noqa: E402,F401
    load_esp32_csi_csv,
    read_esp_csi_rows,
    verify_esp32_assumptions,
    ESP_CSI_C6_COLUMNS,
    ESP_CSI_CLASSIC_COLUMNS,
    ESP32_CSI_TOOL_COLUMNS,
    ESP32_EXPECTED_COLUMNS,
    ESP_CSI_HT40_VALID_SUBCARRIERS,
    ESP_CSI_HT20_VALID_SUBCARRIERS,
    ESP32_HT20_VALID_SUBCARRIERS,
    ESP32_IMAG_FIRST,
    ESP32_TIMESTAMP_UNITS_PER_SEC,
)
