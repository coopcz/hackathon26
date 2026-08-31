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

Match the categories to what both training datasets already contain, so results are
comparable rather than a separate universe:

| # | Scenario | Duration | Maps to |
|---|---|---|---|
| 1 | **Empty room** — nobody inside, doors shut, you outside | **10 min** | `0p` (WiFi-CrowdCounting), `E` (EHUNAM) → AWAY |
| 2 | **One person static** — seated, reading or on a laptop, not pacing | 5 min | `1p`, EHUNAM `PC` → HOME |
| 3 | **One person moving** — walking the room continuously | 5 min | `1p`, EHUNAM `HAR` → HOME |
| 4 | **Empty again** | 5 min | AWAY — the repeat is the point, see below |
| 5 | **Two people** — both moving | 5 min | `2p` → HOME |
| 6 | **Three or more**, if you can find the bodies | 5 min | `3p`+ → HOME |
| 7 | **Empty, final** | 5 min | AWAY |

**Total ≈ 40 minutes.** If you only have 20, cut scenarios 5–6, never the empty
stretches.

Why the empty room appears three times and gets the longest single block:

- `fit_site_baseline()` takes the **5th percentile** of this site's own windows as
  the quiet reference. If the room is never actually empty, the baseline is fitted
  to an occupied channel and every subsequent prediction is calibrated against the
  wrong floor. This is the failure mode the whole cross-room evaluation was about.
- Repeating it at the start, middle and end is what tells you afterwards whether the
  channel *drifted* during the session — if the three empty blocks do not look alike,
  something moved, and that is worth knowing before you train on it.
- 10 minutes at 100 Hz with 2.56 s windows is ~230 empty windows, a solid baseline.

**One class is not enough.** A capture with no empty stretch cannot calibrate and
cannot train; `label_from_manual_log` warns loudly if only one class survives.

## Logging discipline — this is the ground truth

There is no door sensor. The log file **is** the labels.

- [ ] Write the `capture_start` line **at the same moment** you start
      `csi_data_read_parse.py`. Everything else is measured from it. A 20-second
      error here mislabels 20 seconds around every transition.
- [ ] Log a line **the moment** someone crosses the threshold, not once they have
      sat down. If you are unsure to within a few seconds, say so in the `note`
      column — a window near a transition is dropped anyway (2 s guard band either
      side), so an honest "±5s" is far better than a confident wrong number.
- [ ] Always write the **absolute** `n_people` after the event, not just
      entered/left. One missed line then costs one interval instead of corrupting
      everything after it.
- [ ] Log a `set` event whenever the *behaviour* changes without the count changing
      (static → moving). The occupancy label does not change, but the note tells
      you later which windows were which.
- [ ] Note anything unplanned in the `note` column and log it as an event: a pet, a
      neighbour walking past the wall, someone briefly opening the door, the HVAC
      cycling on. Anything that moved is a candidate explanation for a weird window.
- [ ] `capture_end` before you stop the capture script.
- [ ] Don't hover next to the boards while logging. You are a reflector too.

### Easiest way: `tools/mark.py`

In a second terminal, started at the same moment as the capture:

```bash
.venv/bin/python tools/mark.py ~/captures/session1_log.csv
```

Then just press a key and Enter as things happen — it stamps the time for you, so
nobody is reading a clock while someone walks through the door:

```
> i person A sat on the couch     # someone entered   (count + 1)
> 2 B joined, both walking        # set the count to exactly 2
> o                               # someone left      (count - 1)
> n neighbour walked past the wall   # note, count unchanged
> q                               # writes capture_end and exits
```

It writes seconds-since-start, flushes and fsyncs every line, and produces exactly
the format below. Stdlib only, nothing to install.

### Or write it by hand

Full spec in `src/manual_label.py`:

```csv
timestamp,event,n_people,note
2026-08-31T19:05:00,capture_start,0,empty - calibration block
2026-08-31T19:15:00,entered,1,person A walks in, sits on couch
2026-08-31T19:20:00,set,1,person A now walking around
2026-08-31T19:25:00,left,0,room empty again
2026-08-31T19:30:00,entered,2,A and B both moving
2026-08-31T19:35:00,left,0,everyone out
2026-08-31T19:40:00,capture_end,0,
```

`timestamp` may be ISO-8601 as above, a bare `19:05:00` clock, or plain seconds from
the start (`0`, `600`, `900`). Pick one style and use it for the whole file — mixing
them is rejected rather than guessed at.

## Immediately after the session

```bash
# 1. Did the capture survive?
.venv/bin/python -m src.esp_csi ~/captures/session1.csv

# 2. Join it to the log and produce the labelled training table
.venv/bin/python - <<'EOF'
from src.manual_label import label_from_manual_log
res = label_from_manual_log(
    "~/captures/session1.csv",
    "~/captures/session1_log.csv",
    site="byu_apt",                      # the calibration unit; keep it stable
    session="session1",
    out_npz="artifacts/esp32_session1.npz")
print(res["report"])
EOF
```

Check the printed report: the occupancy step function it echoes back should match
what you remember happening. If a segment looks shifted, the `capture_start` anchor
was off — fix it without retyping the log by passing `clock_offset=-12.0` (seconds).

The `.npz` it writes has exactly the keys `src/build_dataset.py` produces
(`X`, `y`, `room`, `n_people`, `session`, `feature_names`), so it feeds
`build_calibrated()` and `train_production_model()` unchanged.

- [ ] Back up both the CSV and the log **before** anyone touches the boards again.
- [ ] Keep the `-l` text log too; it is the only record of dropped rows.

## Retraining

The feature pipeline transfers to the ESP32. **The Intel-fitted model does not** —
1×1 antenna and 114 HT40 subcarriers versus the Intel 5300's 3×2 and 90 channels.
Retrain on the ESP32 table using the same code:

```python
import numpy as np
from src.features import FEATURE_NAMES
from src.train import build_calibrated, train_production_model

z = np.load("artifacts/esp32_session1.npz", allow_pickle=True)
Xc = build_calibrated(z["X"], z["y"], z["room"], FEATURE_NAMES)
model = train_production_model(Xc, z["y"])
```

With several sessions, `src.manual_label.merge_sessions([...])` stacks them, and
`session` / `room` give you the grouping columns for an honest leave-one-session-out
or leave-one-site-out split. A random split across windows from one continuous
recording will read ~99% and mean nothing — that is the session-leakage trap
documented in the README.
