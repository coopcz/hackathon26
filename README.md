# WiFi CSI Presence Detection

This project uses Wi-Fi Channel State Information (CSI) from two ESP32-C6 boards
to decide whether a room is occupied. The eventual goal is to turn the AC off
when nobody is home.

## Live CSI collection dashboard

The repository now includes a local FastAPI dashboard for connecting directly
to the ESP32-C6 receiver, visualizing real CSI, labeling timed sessions, and
exporting lossless JSONL or CSV datasets. It never substitutes generated values
for missing board data. See [`docs/WIFI_CSI_LAB.md`](docs/WIFI_CSI_LAB.md) for
setup, data-integrity guarantees, and recording details.

### Record CSI locally

#### 1. Prepare the boards

- Flash one ESP32-C6 with Espressif's `csi_send` example (TX).
- Flash the other ESP32-C6 with Espressif's `csi_recv` example (RX).
- Power the TX board and connect the RX board to the Mac by USB.
- Confirm that the RX serial output contains lines beginning with `CSI_DATA,`.

Only the RX board connects to the dashboard. The TX board runs independently.
The backend uses 921600 baud and owns the serial connection, so close any serial
monitor that already has the RX port open.

#### 2. Install the local app

From the repository root on the Mac:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

This setup is required only once. On later runs, reuse the existing `.venv`.

#### 3. Start the dashboard

```bash
.venv/bin/python -m uvicorn backend.main:app \
  --host 127.0.0.1 \
  --port 8000
```

Keep that terminal open. Open <http://127.0.0.1:8000> in a browser.

#### 4. Connect the receiver

1. Click **Refresh ports**.
2. Choose the RX port, normally `/dev/cu.usbmodem...` on macOS.
3. Click **Connect**.
4. Confirm that the status says **Connected** and the packet count increases.
5. Check that RSSI, channel, CSI length, and the plots update from received data.

If two USB modem ports are listed, try one at a time. The RX is the port whose
packet count increases with valid CSI. Do not connect both ports in the app.

#### 5. Record a labeled session

1. Select one label: `empty`, `occupied_still`, or `occupied_moving`.
2. Add notes describing the person, position, room, door state, board placement,
   and anything moving in the environment.
3. Click **Start recording**.
4. Move to the intended test position during the 10-second countdown.
5. The app records for 30 seconds and then stops automatically.

The backend excludes every packet received during the countdown. Once the
countdown finishes, every accepted packet is saved for exactly 30 seconds. The
backend then closes the file automatically, so walking back to the computer
after completion does not contaminate an empty-room trial. **Cancel / Stop** is
still available to discard the countdown or end a recording early.

#### 6. See the live verdict

Once a model exists (`artifacts/esp32_model.joblib`), the top of the dashboard
shows the room's current state, updated about twice a second:

- **Room** — HOME or AWAY
- **Confidence** — the model's probability for the class it chose
- **Air conditioning** — AC ON / AC OFF, with the reason spelled out
- **p(occupied) timeline** — the probability trace against the tuned decision
  line, so the decision *margin* is visible rather than just the verdict

The line under it names the model, the window size, the threshold, how much data
it was trained on, and its cross-validated recall. That cross-validated number is
the honest one; the on-screen confidence is a single window's opinion.

The predictor rebuilds the window from `raw_csi` and re-derives phase across the
whole window, exactly as training does. It never falls back to a guess: no model
means no verdict.

#### 7. Replay a recording (no board needed)

The **Replay a recording** panel feeds a saved session's packets through the live
model at their original rate. One button per condition replays the first matching
recording. This is how to demo or sanity-check the model without standing in the
room with the hardware.

Replayed values are read verbatim off disk — nothing is generated — and a replay
is never written to a recording, so it cannot contaminate a labelled session. The
card shows a **REPLAY** badge whenever the verdict came from a replay rather than
a live board.

#### 8. Find or export the data

Native recordings are stored as ignored JSONL files in:

```text
recordings/<timestamp>_<label>.jsonl
```

To export a session:

1. Find it under **Previous recordings**.
2. Select the filename.
3. Click **Export CSV**.

The CSV preserves `raw_csi`, `amplitude`, `phase`, and `features` as JSON inside
properly quoted cells. It also includes the board metadata, session label,
notes, timestamps, and motion score. The JSONL and CSV arrays contain real
accepted board values; visualization smoothing and normalization are never
written into the dataset.

#### Troubleshooting

| Problem | Check |
|---|---|
| No `/dev/cu.usbmodem...` port | Reseat the RX USB cable, try a data-capable cable/port, then click **Refresh ports**. |
| Port is busy or permission is denied | Close ESP-IDF Monitor, `screen`, Arduino Serial Monitor, and any other program using the port. |
| Connected but packet count stays at zero | Verify the selected board is RX, its firmware is running, TX is powered, baud is 921600, and serial output begins with `CSI_DATA,`. |
| Rejected count increases | Inspect the backend terminal; malformed rows and CSI length mismatches are rejected instead of saved. |
| Browser opens but the app does not load | Confirm the Uvicorn terminal is still running and use exactly `http://127.0.0.1:8000`. |
| Charts appear stale | Confirm packets are increasing, then refresh the browser; recording happens in the backend independently of chart refresh rate. |

Stop the server with `Control-C` in its terminal. Existing recordings remain in
`recordings/` and are intentionally excluded from Git commits.

## Results on our own hardware

First real training run, 1 September 2026. 30 recordings of 30 seconds each
(10 `empty`, 10 `occupied_still`, 10 `occupied_moving`), boards untouched
throughout, `CONFIG_FORCE_GAIN` enabled. 28 recordings passed the quality gate.

Cross-validation **grouped by recording** — no window in a test fold shares a
recording with any training window:

| Metric | Value |
|---|---:|
| Accuracy | 96.5% |
| **AWAY recall** (correctly detects an empty room) | **93.5%** |
| **HOME recall** (leaves the AC on for someone who is home) | **98.2%** |
| Macro F1 | 0.962 |

Per condition:

| Condition | Correct |
|---|---:|
| `occupied_moving` | 99.7% |
| `occupied_still` | 96.7% |
| `empty` | 93.5% |

Winning configuration: the 7 **scale-free** features, baseline-calibrated,
logistic regression, no smoothing, AWAY threshold 0.60. Notably the scale-free
model beat the raw 16-feature model (93.5% vs 90.0% AWAY recall) — the features
that carry an absolute amplitude scale were not just unnecessary, they were
worse. `occupied_still` was expected to be the hard case and was not.

### Ruling out the obvious confound

All 10 `empty` recordings were made first (01:13–01:24) and every occupied
recording after (01:28–01:57), so "empty" was perfectly confounded with "early in
the session". A channel that merely drifted over the hour would produce this
same result. Two checks:

- **Boundary test** — evaluating only on recordings either side of the
  changeover, 4–7 minutes apart: 88.1% accuracy, 100% HOME recall. Still clearly
  separated.
- **Drift test** — train a model to distinguish the first five `empty`
  recordings from the last five. It scored **0.406, below chance**. The channel
  was stable across the block, so drift cannot explain the result.

### What is still untested

There is no `empty` recording from late in the session, so the drift check in
`train_esp32.py` cannot run and "does it still work an hour later, or after the
boards are bumped" is unanswered. The cheapest fix is five more `empty`
recordings at the end of the next session, and interleaving conditions
(`empty`, occupied, `empty`, ...) from then on.

## Notes from the first (failed) capture

The very first capture, `data/firstdata-*.csv`, is kept as a negative example.
It is excluded automatically on quality: the AGC was wandering (sd 11.8),
9.9% of packets carried all-zero CSI, and the packet interval jitter was 8.59
against 0.16 in the good captures. Setting `CONFIG_FORCE_GAIN=1` in the receiver
firmware is what fixed it — the second batch reports an AGC standard deviation
of exactly 0.0.

## Run an ESP32 CSV

Install the Python packages once:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then check a capture:

```bash
.venv/bin/python -m src.esp_csi data/your_capture.csv
```

For the first file:

```bash
.venv/bin/python -m src.esp_csi \
  data/firstdata-20260831T231802.310639Z_occupied_still.csv
```

This command checks the CSV, packet loss, timing, gain stability, CSI length,
and usable subcarriers. It also converts the raw packets into the 16 features
used by the model. It does not train a new model.

If Python reports that NumPy has the wrong CPU architecture on an Apple Silicon
Mac, run the same command with `arch -arm64` at the front.

The loader can also be used from Python:

```python
from src.esp_csi import load_esp32_csi_csv

capture = load_esp32_csi_csv("data/your_capture.csv")

print(capture["X"].shape)       # one row per analysis window, 16 features
print(capture["diagnostics"])  # packet rate, RSSI, gain, bad rows, and gaps
```

## Repository layout

```text
backend/            FastAPI collection server -- serial reader, recorder, CSV export
frontend/           the dashboard served at http://127.0.0.1:8000
src/
  esp_csi.py        ESP32 CSV -> feature space, plus per-capture quality checks
  features.py       raw CSI -> the 16 features; per-site baseline calibration
  dataset.py        data/*.csv -> labelled training table (X, y, condition, group)
  train_esp32.py    evaluate honestly, fit, and save the deployed model
  train.py          shared estimators, metrics, split helpers
  pipeline.py       inference + AC ON/OFF decision
  intel/            reference experiment on the Intel dataset (not the deployed system)
docs/               board setup, capture protocol, dashboard guarantees
data/               your exported recordings
artifacts/          build products: feature cache, trained model
recordings/         raw JSONL written by the dashboard
```

## The training loop

Everything from here is one cycle: record, check, train, read the report.

### Demo-room workflow (train first, prove second)

The fitted ESP32 model is site-specific. Moving the boards or changing rooms
invalidates its quiet baseline, so the verdict shown while collecting in a new
room is not evidence. Ground truth is the label selected by the operator.

The dashboard now manages this workflow directly. Create a room profile, then
use **Room setup** to connect hardware, collect room-owned training and holdout
recordings, train, validate, and activate that room's model. The files under
`rooms/<room-id>/` are generated working data and model versions; raw captures
remain in `recordings/`. A validated model must be explicitly activated before
it can replace the live model. The command-line flow below remains available for
inspection and recovery.

1. Put the boards in their final positions and restart the backend. The port
   picker now shows physical USB serial devices only. Wait for the **Stream**
   badge; hover it for rate, jitter, RSSI, and gain warnings.
2. Record one genuinely empty 30-second file. Under **Previous recordings**,
   select it and click **Check training quality**. Do not collect a batch until
   `usable` is `true`.
3. Mark the start of a clean training batch:

   ```bash
   mkdir -p demo_room_data
   touch demo_room_data/.start
   ```

4. Without moving the boards, record ten interleaved cycles of `empty`,
   `occupied_still`, and `occupied_moving`. Every file must contain one pure
   condition for all 30 seconds.
5. Copy only this batch and train it:

   ```bash
   find recordings -maxdepth 1 -name '*.jsonl' \
     -newer demo_room_data/.start \
     -exec cp {} demo_room_data/ \;

   arch -arm64 .venv/bin/python -m src.train_esp32 \
     --data-dir demo_room_data --force
   ```

   Bad captures are excluded. A model that misses the HOME-safety floor or
   almost never says AWAY is not saved. When a model is saved, the old bundle is
   backed up as `artifacts/esp32_model.previous.joblib` and the replacement is
   atomic.
6. Reload the model, then create a new marker before collecting holdout files:

   ```bash
   curl -X POST http://127.0.0.1:8000/api/model/reload
   mkdir -p demo_holdout
   touch demo_holdout/.start
   ```

7. Record at least three new files per condition. Copy them to `demo_holdout/`
   with the same `find ... -newer ... -exec cp` pattern, then evaluate the exact
   serving path:

   ```bash
   arch -arm64 .venv/bin/python -m src.evaluate_live demo_holdout
   ```

   This fails if a capture is low quality, a recording has the wrong majority,
   condition accuracy misses its safety target, or the wrong state persists for
   more than three seconds.

Only after those holdouts pass is the live entrance/exit demonstration evidence:
an actually empty room should settle to AWAY, entry should settle to HOME, a
still person should remain HOME, and complete exit should return to AWAY. The
transition takes roughly one full 2.56-second window.

### 1. Record and export

Use the dashboard (above). One label per 30-second recording. Export each to
`data/` with the label in the filename:

```text
data/apt_empty_01.csv
data/apt_occupied_still_01.csv
data/apt_occupied_moving_01.csv
```

The `session_label` column inside the CSV is authoritative; the filename is a
fallback and a convenience for humans. Labels are matched on **meaning**, not on
an exact string, so `Empty`, `occupied - still` and `Occupied Moving` all work.

### 2. Check each capture before trusting it

```bash
.venv/bin/python -m src.esp_csi data/apt_empty_01.csv
```

A recording is automatically **excluded from training** if it has a wandering
AGC, a lossy link, too many malformed rows, or too many all-zero CSI packets.
Those are not noisy data, they are a second and wrong definition of "motion".
Fix the capture rather than training around it.

### 3. Build the table

```bash
.venv/bin/python -m src.dataset
```

Prints one line per recording — label, windows, packet rate, AGC spread, jitter,
zero-CSI fraction, and whether it was kept — then the class balance.

### 4. Train and evaluate

```bash
.venv/bin/python -m src.train_esp32
```

This searches feature sets x baseline modes x models x smoothing x decision
threshold, scores every combination with cross-validation **grouped by
recording**, prints the table, and saves the winner to
`artifacts/esp32_model.joblib`.

It refuses to save a model that is not deployable — one that would switch the AC
off on somebody who is home, or one that never says AWAY at all and therefore
saves nothing. `--save-anyway` overrides that; `--no-save` reports only.

Useful flags:

| Flag | Effect |
|---|---|
| `--overlap 0.9` | More training rows from the same recordings. Safe: splits are by recording. |
| `--window-seconds 5` | Longer windows. Steadier variance estimates, slower to react. |
| `--augment-intel` | Add the calibrated Intel windows to training folds. Tests whether scale-free calibration really does transfer across radios. |
| `--include-bad` | Train on quality-flagged recordings too. Contaminates the result; for debugging only. |
| `--force` | Rebuild the feature cache after adding recordings. |

### Proving it isn't guessing or memorizing order

```bash
.venv/bin/python -m src.prove
```

Three checks against the exact model in `artifacts/esp32_model.joblib`:

| Check | Answers | Result |
|---|---|---:|
| Label permutation (200 shuffles) | "Is it just guessing accurately?" | real 93.5% vs. chance 1.5% avg, p=0.0000 |
| Order-memorization (per condition) | "Is it remembering recording order?" | not distinguishable from chance, p=0.14–0.36 |
| Blind holdout (7 recordings, never tuned on) | "Does it generalize?" | 100% accuracy |

Saves two charts to `artifacts/proof/` for slides. Full writeup, exact wording
for each objection, and what *not* to claim: [`docs/PROOF.md`](docs/PROOF.md).

### What to read in the report

- **AWAY recall** is the number that matters. Accuracy is nearly meaningless
  here because it moves with the class balance.
- **HOME recall** is a constraint, not a metric to trade: below 98% the model
  turns the AC off on people who are home.
- **Accuracy per condition** tells you where you are losing. `occupied_still` is
  the hard case — someone sitting perfectly still barely modulates the channel.
- **The drift check** trains on the earliest recordings and tests on the latest.
  A large drop there means the channel moved during the session, and the
  baseline needs re-fitting periodically at deploy time.

## What to collect next

Fix the connection before collecting training data:

1. Set `CONFIG_FORCE_GAIN=1` in the receiver firmware and reflash it.
2. Move the boards or change the Wi-Fi channel until long gaps and zero-CSI
   packets stop appearing.
3. Record a 30-second test and run `python -m src.esp_csi` on it. It must come
   back with a stable AGC and low jitter before anything else is worth recording.

Then, with the boards left in exactly the same positions, aim for **at least 10
recordings of each condition** and interleave them (`empty`, occupied, `empty`,
...) rather than doing all of one and then all of the other — that is what makes
the drift check meaningful. See [`docs/ESP32_SETUP.md`](docs/ESP32_SETUP.md) for
the full protocol.

## The Intel reference experiment

`src/intel/` is not the deployed system. It is the experiment on the
WiFi-CrowdCounting dataset (Intel IWL-5300 hardware) that produced the one
design decision everything else rests on.

```bash
.venv/bin/python -m src.intel.build_dataset   # parse the .dat captures once
.venv/bin/python -m src.intel.demo            # the full write-up
```

| Evaluation | Accuracy | AWAY recall |
|---|---:|---:|
| Random 80/20 split over windows | 99.7% | — (inflated by session leakage) |
| Leave-one-room-out, raw features | 89.1% | **8.7%** |
| Leave-one-room-out, calibrated | 98.9% | **94.4%** |

The 8.7% is the finding. Trained on two rooms and tested on a third, the model
missed almost every empty room — it would leave the AC running in an empty
house, which is the exact failure the product exists to prevent. Accuracy still
read 89% only because 89% of windows were occupied.

The fix was to stop using absolute feature values and express each one relative
to that site's own quiet baseline. This needs no labels at the new site, only
that the room is empty some of the time — and it is also why an ESP32 model is
plausible at all: "5x my own noise floor" means the same thing on a $5 radio as
on an Intel 5300.

Those numbers are **not** performance numbers for the ESP32 boards. The feature
pipeline transfers; the fitted model does not (1x1 antenna and 114 HT40
subcarriers versus 3x2 and 90 channels). Retraining is `src/train_esp32.py`.
