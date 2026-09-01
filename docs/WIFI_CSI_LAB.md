# WiFi CSI Lab

Local FastAPI dashboard for collecting real CSI emitted by Espressif's
`csi_recv` on an ESP32-C6. The application contains no sample-data generator
and never substitutes synthetic values for missing board data.

## Run

For the complete board-to-CSV walkthrough and troubleshooting table, see
[`README.md`](../README.md#record-csi-locally).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, select the receiver's `/dev/cu.*` port, and
connect. The backend exclusively owns the serial port at 921600 baud.

## Data integrity

- Only lines beginning `CSI_DATA,` can produce packets.
- Rows with malformed metadata, invalid JSON I/Q arrays, odd I/Q counts, or a
  declared/actual length mismatch are rejected and logged.
- Complex CSI is derived exactly as `real + 1j * imag` from `[imag, real, ...]`.
- Motion score and amplitude statistics are deterministic derivatives of
  consecutive received frames. The first frame has no motion score and remains
  `null`.
- JSONL recordings store every accepted row's original integer I/Q array,
  board metadata, timestamps, amplitude, phase, and derived features.
- Normalization, range selection, smoothing, and zero filtering affect charts
  only and never saved data.
- A disconnected or silent board produces blank values and an explicit no-data
  message, never fake telemetry.

Recordings are written to `recordings/`. Each line is one accepted packet and
is directly loadable with Python/pandas.

Saved sessions can also be exported from the dashboard as CSV. Array fields
(`raw_csi`, `amplitude`, and `phase`) are JSON-encoded inside correctly quoted
CSV cells, preserving every value and the original I/Q ordering.

Recording starts use a backend-enforced 10-second setup exclusion window. CSI
packets received during the countdown are displayed live but are not written to
the session, preventing the operator leaving the computer from contaminating
the labeled sample.
