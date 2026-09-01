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

#### 6. Find or export the data

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

## What happened with our first ESP32 capture

The file we tested was:

```text
data/firstdata-20260831T231802.310639Z_occupied_still.csv
```

We were inside the room for the entire recording. The correct answer for every
part of this file is therefore **HOME**.

The program successfully read all 2,007 CSI packets and turned them into 13
short analysis windows. When we forced those windows through the old model, it
returned:

| Result | Windows |
|---|---:|
| HOME | 10 |
| AWAY | 3 |

The three AWAY results do **not** mean the room became empty. They were wrong.
This is not a real accuracy test because:

- the current model was trained on Intel Wi-Fi hardware, not our ESP32 boards;
- this capture only contains one condition: occupied and still;
- we do not have an empty-room ESP32 recording to use as the room baseline;
- the radio signal and packet timing changed heavily during the recording.

The capture still helped. It proved that our CSV export contains valid raw CSI,
the loader understands it, and the feature code runs on real ESP32-C6 data.
It is useful for checking the data pipeline, but it is not enough to train or
judge the occupancy detector.

The main capture problems were:

- 199 packets (9.9%) contained all-zero CSI;
- 1,025 packet IDs were missing;
- one gap lasted 6.26 seconds;
- the receiver gain kept changing and was maxed out for about half the capture;
- the signal dropped from about -65 dBm to around -90 dBm.

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

## Files that matter

| File | Why it matters |
|---|---|
| `data/*.csv` | Put ESP32 capture files here. |
| `src/esp_csi.py` | Reads and checks ESP32 CSV files. Start here for a new capture. |
| `src/features.py` | Converts raw CSI into the 16 values used by the model. |
| `src/pipeline.py` | Turns model output into HOME/AWAY and AC ON/OFF decisions. |
| `src/train.py` | Trains and evaluates the occupancy model. |
| `src/manual_label.py` | Combines a capture with a log of when people entered or left. |
| `docs/ESP32_SETUP.md` | Board setup and the full data-collection checklist. |
| `esp32_dry_run.py` | Tests the ESP32 code using fake data. |
| `main.py` | Runs the older Intel-dataset demo. It does not take an ESP32 CSV. |

## What to collect next

Fix the connection before collecting training data:

1. Set `CONFIG_FORCE_GAIN=1` in the receiver firmware and reflash it.
2. Move the boards or change the Wi-Fi channel until long gaps and zero-CSI
   packets stop appearing.
3. Record a two-minute test and run `python -m src.esp_csi` on it.

Once that test looks clean, record both classes with the boards left in the
same positions:

- room empty;
- one person sitting still;
- one person moving around.

We need both empty and occupied data before we can fit the room baseline,
retrain on ESP32 measurements, and report meaningful accuracy. See
[`docs/ESP32_SETUP.md`](docs/ESP32_SETUP.md) for the longer collection plan.

## Older model results

The model currently in the project was built from the WiFi-CrowdCounting
dataset recorded with Intel IWL-5300 hardware. After room calibration it reached
98.9% accuracy and 94.4% AWAY recall when testing on a room it had not seen.

Those numbers are evidence that the feature approach can work. They are not
performance numbers for our ESP32 boards. The ESP32 model needs to be retrained
with clean ESP32 empty and occupied recordings.

Commands for the older dataset work:

```bash
.venv/bin/python -m src.build_dataset
.venv/bin/python main.py
.venv/bin/python validate_ehunam.py
```
