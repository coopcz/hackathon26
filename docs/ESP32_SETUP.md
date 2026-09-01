# ESP32-C6 CSI capture — flashing, capture, and session protocol

Everything here is for **two ESP32-C6 boards**, one running `csi_send`, one running
`csi_recv`, from [espressif/esp-csi](https://github.com/espressif/esp-csi)
`examples/get-started`. Hand this page to whoever does the physical setup.

Two things to know before you start, because both bite:

1. **The two boards must be at least 1 metre apart.** This is esp-csi's own
   instruction (`examples/get-started/README.md`). Closer than that and the direct
   path saturates the receiver — you measure the AGC, not the room.
2. **`csi_data_read_parse.py` writes a header that lies.** It always writes the
   25-column header, even though a C6 prints 15-column rows. Our loader ignores
   the header entirely and detects the schema from the row field count, so this
   costs you nothing — but do not be alarmed by it, and do not "fix" the CSV by
   hand.

---

## 1. Prerequisites

```bash
# ESP-IDF v5.4 or later (the C6 CSI API needs it)
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git ~/esp/esp-idf
~/esp/esp-idf/install.sh esp32c6
. ~/esp/esp-idf/export.sh          # run this in EVERY new shell

git clone https://github.com/espressif/esp-csi.git ~/esp/esp-csi
```

Find the two serial ports and write down which is which — you will need them all night:

```bash
ls /dev/cu.usbmodem* /dev/cu.usbserial* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

On macOS a C6 devkit usually appears as `/dev/cu.usbmodem*`. Plug in **one board at a
time** and note the port, otherwise you will flash the wrong one.

## 2. Flash the transmitter (`csi_send`)

```bash
cd ~/esp/esp-csi/examples/get-started/csi_send
idf.py set-target esp32c6
idf.py flash -b 921600 -p /dev/cu.usbmodemTX monitor
```

Expected: it starts printing send counts. It transmits ESP-NOW broadcasts at
`CONFIG_SEND_FREQUENCY = 100` Hz on channel 11 with source MAC
`1a:00:00:00:00:00`. Leave it running; it needs no PC after flashing, so once it
works you can move it to a USB power brick.

Press `Ctrl-]` to exit the monitor.

If you see `<ESP_ERR_ESPNOW_NO_MEM> ESP-NOW send error` repeatedly, channel 11 is
congested. Change `CONFIG_LESS_INTERFERENCE_CHANNEL` in **both** `app_main.c`
files to the same quieter channel and reflash both boards.

## 3. Flash the receiver (`csi_recv`)

```bash
cd ~/esp/esp-csi/examples/get-started/csi_recv
idf.py set-target esp32c6
idf.py flash -b 921600 -p /dev/cu.usbmodemRX monitor
```

Expected within a second or two:

```
I (xxx) csi_recv: ================ CSI RECV ================
type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data
CSI_DATA,7,1a:00:00:00:00:00,-23,11,-96,32,4,11,372852,47,0,256,0,"[0,0,0,...]"
```

Sanity-check that first row before going further:

| Field | Expect | If it's wrong |
|---|---|---|
| `mac` | `1a:00:00:00:00:00` | receiver is hearing something other than our TX; `csi_recv` filters on this MAC, so if you changed it, change it in both |
| `rssi` | roughly **−25 to −45** dBm at 1–3 m | closer than −20 → move them apart; below −70 → move them closer or use the external antenna |
| `len` | **256** (HT40) | 128 means it negotiated HT20; fine, but note it, the subcarrier count changes |
| `noise_floor` | around −92 to −96 | much higher → a noisy RF environment |
| rows appearing | ~100/s | far fewer → packet loss; see step 6 |

**`Ctrl-]` to exit the monitor. `csi_data_read_parse.py` cannot open the port while
`idf.py monitor` holds it.** This is the single most common way to lose the first
ten minutes of a capture session.

## 4. Capture to a file

```bash
cd ~/esp/esp-csi/examples/get-started/tools
pip install -r requirements.txt        # PyQt5, pyqtgraph, pyserial, pandas, scipy, statsmodels

python csi_data_read_parse.py \
    -p /dev/cu.usbmodemRX \
    -s ~/captures/session1.csv \
    -l ~/captures/session1_log.txt
```

- `-p` receiver serial port
- `-s` **the CSV we consume** — one row per packet
- `-l` a separate text log for non-CSI serial chatter and rejected rows. Check it
  afterwards: lots of `element number is not equal` / `data is incomplete` means
  the PC could not keep up with the serial stream.

**Baud rate is 921600, 8N1, hardcoded** in `csi_data_read_parse.py`
(`serial.Serial(port=port, baudrate=921600, bytesize=8, parity='N', stopbits=1)`).
Do not change it on one side only. `idf.py flash -b 921600` is the *flashing* baud
and is unrelated — it just makes flashing faster.

### ⚠ The serial link is the tightest constraint in this whole setup

Measured on a representative HT40 row: **~814 bytes per packet**. At `csi_send`'s
100 Hz that is **~80 KB/s**, and 921600 baud 8N1 carries **~90 KB/s**.

> **1.13× headroom.** Any burst, any longer-than-average row, and rows are dropped.

This is exactly the symptom esp-csi's own A&Q describes (`element number is not
equal`, `data is incomplete`) and its recommended fix is "advance the baud rate of
the serial port". Pick one of these *before* the real session:

- **Use the C6's native USB port** (USB Serial/JTAG, usually the port labelled
  `USB`, appearing as `/dev/cu.usbmodem*`). Over native USB CDC the baud rate is
  nominal — throughput is USB-limited, ~1 MB/s, and the constraint disappears.
  **This is the easy fix and probably the one you want.** If your `/dev` entry is a
  `usbserial`/`SLAB`/`CP210x`/`CH34x` name you are on the UART bridge and the 90 KB/s
  ceiling is real.
- **Raise the baud rate on both sides.** `idf.py menuconfig` →
  *Component config → ESP System Settings → Channel for console output → Custom
  baudrate* → `2000000`, reflash `csi_recv`, then edit `baudrate=921600` to
  `baudrate=2000000` in `csi_data_read_parse.py`. Both sides or neither.
- **Halve the packet rate.** Set `CONFIG_SEND_FREQUENCY` to `50` in
  `csi_send/main/app_main.c`. 50 Hz still gives 128 packets per 2.56 s window,
  which is exactly what the Intel training data had.
- **Drop to HT20** (`len=128`, ~470 bytes/row, 46 KB/s — comfortable), at the cost
  of half the subcarriers.

Whichever you choose, the `[OPEN 4]` line of the verify step tells you afterwards
whether it worked: it reports the delivered rate and the interval jitter.

The script opens a PyQt window with live subcarrier plots. That window is the
reason for most dropped rows (upstream's own A&Q says so), but it is also the
fastest way to confirm CSI is actually moving when someone waves an arm. Keep it
open for the first minute, then, if the drop rate is bad, restart the capture with
the window minimised.

**Headless alternative** if PyQt is a problem tonight — this loses the plots but
produces a file our loader reads identically:

```bash
python - <<'EOF'
import csv, serial
CSI = "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data"
s = serial.Serial("/dev/cu.usbmodemRX", 921600, timeout=1)
with open("session1.csv", "w", newline="") as fh:
    fh.write(CSI + "\n"); fh.flush()
    while True:
        line = s.readline().decode(errors="replace").strip()
        if line.startswith("CSI_DATA,"):
            fh.write(line + "\n"); fh.flush()
EOF
```

## 5. Verify the capture before you trust it

Back in this repo, on the very first file:

```bash
.venv/bin/python -m src.esp_csi ~/captures/session1.csv
```

This prints the assumption checklist. The `[RESOLVED]` lines are confirmed against
esp-csi's source and should just be true. **Read the `[OPEN 1..5]` lines** — those
are the only things that could not be settled without hardware:

| Check | Good | Bad, and what to do |
|---|---|---|
| `OPEN 1` subcarrier map | "matches the documented HT40 map" (114 of 128) | "DIFFERS" → the empirical set is used anyway; tell the software side the numbers |
| `OPEN 2` first_word_invalid | near 0% | >50% → rerun the loader with `drop_first_word=True` |
| `OPEN 3` gain stability | agc/fft sd ≈ 0 | sd > 2 → AGC is wandering and will look like motion. Set `#define CONFIG_FORCE_GAIN 1` in `csi_recv/main/app_main.c` and reflash |
| `OPEN 4` packet rate | ~100 Hz, jitter < 0.5 | low rate or high jitter → lossy link; move the boards, change channel |
| `OPEN 5` mixed bandwidths | 0 rows dropped | many dropped → the link is flapping HT20/HT40. Set `acquire_csi_ht20 = false` in `csi_recv` to pin HT40 and reflash |

Do a **2-minute test capture and run this check before the real session.** Finding
a wandering AGC after 40 minutes of careful labelling is the worst outcome available.

## 6. Physical setup

- Boards **≥ 1 m apart**, ideally 2–4 m, with the sensing area between them.
- **Use the external IPEX antenna** if the devkit has the connector. esp-csi notes
  the PCB antenna is directional and gets interfered with by the board itself.
- Both boards on stable surfaces — a board that shifts is indistinguishable from a
  person moving. Tape them down.
- No one else in the space, and no other moving objects (a fan, a laptop screen
  being opened, a pet). esp-csi's note 2 is blunt about this.
- Same physical placement for **every** session. Move the boards and the per-site
  calibration baseline is invalidated; you would have to recalibrate.
- Keep the two USB cables out of the direct path if you can.

---

# The capture session — checklist for the real run

## Before you press record

- [ ] `csi_send` powered and transmitting (its LED / monitor output confirms).
- [ ] `csi_recv` flashed, `idf.py monitor` **closed**.
- [ ] 2-minute test capture done and `python -m src.esp_csi` run on it; all five
      `[OPEN]` lines acceptable.
- [ ] Boards taped in final position, ≥ 1 m apart, external antennas on.
- [ ] Phone/laptop clock visible so you can write timestamps without guessing.
- [ ] `manual_log_TEMPLATE.csv` copied to `session1_log.csv` and open in an editor.
- [ ] Disk space: ~80 KB/s at 100 Hz HT40 (~814 bytes/row). A 40-minute session is
      roughly **200 MB**. Trivial, but check the drive isn't already full.
- [ ] Serial throughput decision made (native USB port, or raised baud, or 50 Hz —
      see the warning in step 4). Confirmed on the test capture.

## What to record

Every dashboard recording is exactly 30 seconds, so a "block" below means that
many separate recordings with the same label. Match the categories to what the
Intel training set already contains, so results are comparable rather than a
separate universe:

| # | Scenario | Dashboard label | Recordings | Maps to |
|---|---|---|---|---|
| 1 | **Empty room** — nobody inside, doors shut, you outside | `empty` | **20** | `0p` → AWAY |
| 2 | **One person static** — seated, reading or on a laptop, not pacing | `occupied_still` | 10 | `1p` → HOME |
| 3 | **One person moving** — walking the room continuously | `occupied_moving` | 10 | `1p` → HOME |
| 4 | **Empty again** | `empty` | 10 | AWAY — the repeat is the point, see below |
| 5 | **Two people** — both moving | `occupied_moving` | 10 | `2p` → HOME |
| 6 | **Three or more**, if you can find the bodies | `occupied_moving` | 10 | `3p`+ → HOME |
| 7 | **Empty, final** | `empty` | 10 | AWAY |

**Total ≈ 80 recordings, ≈ 40 minutes of recorded time** plus countdowns. If you
only have half that, cut scenarios 5–6, never the empty stretches.

Why the empty room appears three times and gets the most recordings:

- `fit_site_baseline()` takes the **5th percentile** of this site's own windows as
  the quiet reference. If the room is never actually empty, the baseline is fitted
  to an occupied channel and every subsequent prediction is calibrated against the
  wrong floor. This is the failure mode the whole cross-room evaluation was about.
- Repeating it at the start, middle and end is what tells you afterwards whether the
  channel *drifted* during the session — if the three empty blocks do not look alike,
  something moved, and that is worth knowing before you train on it.
- 20 recordings at ~60 Hz with 2.56 s windows is ~220 empty windows, a solid
  baseline.

## Labelling discipline — this is the ground truth

There is no door sensor. **The label you pick in the dashboard before each
recording is the label**, and it applies to the whole 30-second file.

The dashboard enforces the protocol so nobody has to read a clock:

1. Pick one label: `empty`, `occupied_still`, or `occupied_moving`.
2. Write notes: who, where they sat, room, door state, board placement, anything
   else in the room that moves.
3. Press **Start recording**, then walk to the test position during the
   10-second countdown. Packets received during the countdown are discarded.
4. The backend records for exactly 30 seconds and stops itself, so walking back
   to the laptop afterwards cannot contaminate an `empty` trial.

Checklist:

- [ ] One condition per recording. Never change what the room is doing halfway
      through — start a second recording instead.
- [ ] Take many short recordings rather than a few long ones. Each 30 s file is
      ~11 windows, and per-file grouping is what makes an honest
      leave-one-recording-out split possible.
- [ ] Re-record `empty` at the start, middle and end of the session. If the three
      do not look alike, the channel drifted and you want to know that before
      training on it.
- [ ] Don't stand next to the boards while an `empty` recording runs. You are a
      reflector too, even from the next room.
- [ ] Put anything unplanned in the notes: a pet, a neighbour walking past the
      wall, the HVAC cycling on. It is the only explanation you will have later
      for a weird window.

### Naming exported files

Export each session to CSV from **Previous recordings** and drop it in `data/`.
Keep the label in the filename so the dataset builder can read it without opening
the file:

```text
data/<site>_<label>_<n>.csv

data/apt_empty_01.csv
data/apt_occupied_still_01.csv
data/apt_occupied_moving_01.csv
```

The label is also stored inside the CSV in the `session_label` column, which is
the authoritative copy; the filename is for humans.

## Immediately after the session

```bash
# Did the capture survive?
.venv/bin/python -m src.esp_csi data/apt_empty_01.csv
```

Check the report before you trust the file:

- `packet rate` should be steady and the interval jitter low. A lossy link
  destroys the spectral features.
- `gain stability` should be flat. If AGC is wandering, set `CONFIG_FORCE_GAIN 1`
  in the receiver firmware and reflash — a gain step looks exactly like motion.
- `first_word_invalid` and malformed-row counts should be near zero.
- `link quality` RSSI should not be collapsing over the recording.

- [ ] Back up the CSVs **before** anyone touches the boards again.
- [ ] Keep the serial text log too; it is the only record of dropped rows.

**One class is not enough.** A session with no `empty` recordings cannot
calibrate a site baseline and cannot train.


## Retraining

The feature pipeline transfers to the ESP32. **The Intel-fitted model does not** —
1×1 antenna and 114 HT40 subcarriers versus the Intel 5300's 3×2 and 90 channels.
Retraining is two commands:

```bash
.venv/bin/python -m src.dataset        # inspect what your recordings look like
.venv/bin/python -m src.train_esp32    # evaluate, fit, save
```

`src/train_esp32.py` searches feature sets × baseline modes × models × smoothing
× decision threshold and scores each with cross-validation **grouped by
recording**, so no window in a test fold shares a recording with a training
window. It refuses to save a model that would switch the AC off on somebody who
is home, or one that never says AWAY and therefore saves nothing.

Three things to know before reading its output:

- **AWAY recall is the number.** Accuracy tracks the class balance and will look
  respectable even when the model is useless.
- **`occupied_still` is the hard condition.** A person sitting perfectly still
  barely modulates the channel; the phase features are what separates them from
  an empty room, and they need a stable AGC to work.
- **The drift check** trains on your earliest recordings and tests on the latest.
  If it drops sharply, the channel moved during the session — interleave the
  conditions while recording and re-fit the baseline periodically at deploy time.

To use a saved model on a new capture:

```python
import joblib, numpy as np
from src.esp_csi import load_esp32_csi_csv
from src.train_esp32 import apply_calibration
from src.pipeline import should_run_ac, LABEL_NAMES

b = joblib.load("artifacts/esp32_model.joblib")
out = load_esp32_csi_csv("data/new_capture.csv", valid_subcarriers="reference",
                         window_seconds=b["window_seconds"], verbose=False)
Xc, _ = apply_calibration(out["X"], b["baseline"], b["feature_set"])
p_home = b["model"].predict_proba(Xc)[:, 1]
pred = np.where((1 - p_home) >= b["threshold"], 0, 1)
print([LABEL_NAMES[i] for i in pred])
```

The baseline ships inside the bundle. A model without its site baseline is
unusable — the features would be in units it was never trained on.
