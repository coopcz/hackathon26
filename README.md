# WiFi CSI Presence Detection — Hackathon 2026 (BYU)

Binary home/away detection from WiFi Channel State Information, for automatic HVAC control.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.build_dataset   # parse 26 captures -> feature cache (~21 s)
.venv/bin/python main.py                # full evaluation + demo
```

## Result in one line

**94.4% AWAY recall / 98.9% accuracy on a room the model has never seen** — but only
after per-site calibration. Without it, cross-room AWAY recall is 8.7%.

| Evaluation | Accuracy | AWAY recall | Trust it? |
|---|---|---|---|
| Random 80/20 split | 99.7% | 100% | ❌ session leakage — inflated |
| Leave-one-room-out, absolute features | 89.1% | **8.7%** | ✅ honest, and it fails |
| Leave-one-room-out, calibrated | 98.9% | **94.4%** | ✅ honest, and it works |

Validated against a **second dataset on different hardware** (EHUNAM, Broadcom
BCM43455 / Raspberry Pi). That validation tempers the headline — see
`validate_ehunam.py`:

| Question | Answer |
|---|---|
| Do running appliances read as a person? | **7.8–25.8% false HOME** (industrial machines 39.1%); genuinely empty rooms **0.0%** |
| Does an Intel-trained model work on Broadcom data, no retraining? | macro F1 **0.72** calibrated vs **0.45** uncalibrated |
| Does it hold across days in one room? | 90.3% acc / **57.4%** AWAY recall, vs 98.1% / 100% same-day |
| Does it hold across environments with only ONE training site? | **No** — calibration does not rescue it |

## Layout

| File | What |
|---|---|
| `src/csi_reader.py` | NumPy port of the Intel 5300 CSI Tool reader (upstream is MATLAB-only) |
| `src/features.py` | CSI → 16 features; why each one, and per-site baseline calibration |
| `src/build_dataset.py` | Parse all 26 captures → cached feature matrix |
| `src/train.py` | Both evaluation protocols, metrics, feature importance |
| `src/pipeline.py` | `predict_presence` / `should_run_ac` / `load_esp32_csi_csv` |
| `src/ehunam.py` | Range-extracts single files from EHUNAM's 77.7 GB zip; nexmon loader |
| `src/build_ehunam.py` | EHUNAM → the same feature table |
| `main.py` | End-to-end run |
| `validate_ehunam.py` | Cross-dataset / cross-hardware validation |

## Dataset

[RadioPoints/Device-free_RF_Human_Sensing_Datasets](https://github.com/RadioPoints/Device-free_RF_Human_Sensing_Datasets)
— WiFi-CrowdCounting (Di Domenico et al., Tor Vergata). Intel IWL-5300, 3 rooms ×
occupancy 0–8, 229,837 packets → 1,784 windows. `0p` = AWAY, `1p`–`8p` = HOME.
Not committed; `git clone` it into `data/`.

**Validation set:** [EHUNAM](https://doi.org/10.6084/m9.figshare.28541225) (Diaz et al.,
*Scientific Data* 2025). Published as one 77.7 GB zip with no per-file download, so
`src/ehunam.py` parses the zip index from the archive's last 4 MB and range-fetches only
the members it needs — 248 files / 2.4 GB instead of 77.7 GB. Members are Deflate64, which
Python's `zipfile` cannot read; extraction shells out to `unzip`.

```bash
.venv/bin/python -m src.build_ehunam   # downloads on first run
.venv/bin/python validate_ehunam.py
```

## Plugging in ESP32 data

```python
from src.pipeline import load_esp32_csi_csv, predict_presence, should_run_ac
from src.features import fit_site_baseline, FEATURE_NAMES

out  = load_esp32_csi_csv("my_capture.csv")
base = fit_site_baseline(out["X"], FEATURE_NAMES)      # calibrate to this house
pred, conf = predict_presence(model, out["X"][0], baseline=base, fs=out["fs"])
run_ac, why = should_run_ac(pred, conf)
```

Run `verify_esp32_assumptions(path)` on the first real capture — it checks the
7 documented ASSUMPTIONs in `src/pipeline.py` and reports which held.

**The feature pipeline transfers to ESP32. The fitted model does not** — 1×1 antenna
and 52 usable subcarriers vs the Intel's 3×2 and 90 channels. Retrain on ESP32 data
using this same code.
