"""
Ingestion for Espressif **esp-csi** captures (github.com/espressif/esp-csi).

GROUND TRUTH
------------
Everything below was read off the upstream sources, not guessed:

  * `examples/get-started/csi_recv/main/app_main.c`  -- the `ets_printf` calls in
    `wifi_csi_rx_cb()` that emit the header line and one line per packet.
  * `examples/get-started/tools/csi_data_read_parse.py` -- the capture tool that
    writes the CSV we actually load.
  * `examples/get-started/README.md` -- documents the field order and, critically,
    states "for each subcarrier, the imaginary part is stored first, followed by
    the real part", with LTFs ordered LLTF, HT-LTF, STBC-HT-LTF.

`csi_recv` prints TWO different schemas depending on the target:

  15 columns (ESP32-C5 / C6 / C61 -- our boards):
    type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,
    local_timestamp,sig_len,rx_state,len,first_word,data

  25 columns (ESP32 / S2 / S3 / C3):
    type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
    aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,
    secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data

`type` is the literal string `CSI_DATA` on every data row.  `data` is a
double-quoted, comma-separated bracketed list of int8s; `len` is the number of
INTS in it, so subcarriers = len/2.

  UPSTREAM BUG WE MUST TOLERATE
  -----------------------------
  `csi_data_read_parse.py` writes the 25-column header unconditionally
  (`SubThread.__init__` -> `writerow(DATA_COLUMNS_NAMES)`), even on a C6 whose
  rows carry 15 fields.  A real C6 capture therefore has a header that LIES.
  So we key schema detection off the *field count of the data rows*, never the
  header, and a `csv.DictReader` must not be used on these files.

Also supported:

  * the 29-column session export produced by the collection UI (label and
    receive timestamps followed by the original esp-csi metadata, raw CSI,
    and precomputed display values); and
  * the older third-party StevenMHernandez/ESP32-CSI-Tool format (26 columns,
    space-separated blob) that this repo originally targeted.

The session export is normalized back to the original `raw_csi`,
`esp_timestamp`, and `declared_len` fields.  Its precomputed amplitude, phase,
and feature columns are intentionally ignored so this pipeline recomputes its
own features from the raw radio samples.
"""

import csv
import os
import json
import re
import sys

import numpy as np

from .features import (window_features, windows_from_arrays, _sanitize_phase,
                       FEATURE_NAMES, WINDOW)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# csi_recv on ESP32-C5 / C6 / C61.  NOTE: the firmware header calls the 12th
# column `rx_format` while csi_data_read_parse.py calls it `rx_state`; same field.
ESP_CSI_C6_COLUMNS = [
    "type", "seq", "mac", "rssi", "rate", "noise_floor", "fft_gain", "agc_gain",
    "channel", "local_timestamp", "sig_len", "rx_state", "len", "first_word", "data",
]

# csi_recv on ESP32 / S2 / S3 / C3.
ESP_CSI_CLASSIC_COLUMNS = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "len", "first_word", "data",
]

# Legacy: StevenMHernandez/ESP32-CSI-Tool, the format the first draft targeted.
ESP32_CSI_TOOL_COLUMNS = [
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
    "ant", "sig_len", "rx_state", "real_time_set", "real_timestamp", "len", "CSI_DATA",
]

SCHEMAS = {
    "esp-csi/c5c6c61": ESP_CSI_C6_COLUMNS,
    "esp-csi/classic": ESP_CSI_CLASSIC_COLUMNS,
    "esp32-csi-tool": ESP32_CSI_TOOL_COLUMNS,
}
# field count -> schema name.  15/25/26 are mutually exclusive, so one row is
# enough to identify the producer.
_BY_WIDTH = {len(cols): name for name, cols in SCHEMAS.items()}

# Backwards compatibility with the pre-esp-csi draft.
ESP32_EXPECTED_COLUMNS = ESP32_CSI_TOOL_COLUMNS
CSI_COLUMN_CANDIDATES = ("data", "CSI_DATA", "csi_data")

# Collector/session-export headers can grow as display-only columns are added,
# so detect this format by its required source fields rather than exact width.
SESSION_EXPORT_REQUIRED_COLUMNS = {
    "session_label", "esp_timestamp", "declared_len", "raw_csi",
}

# ---------------------------------------------------------------------------
# What we still cannot know without hardware  -- see docs/ESP32_SETUP.md
# ---------------------------------------------------------------------------
#
# RESOLVED (was ASSUMPTION 1): the CSI column is named `data` in esp-csi and is
#   always the LAST field on the row.  We take it positionally, so the name --
#   and the lying header -- cannot hurt us.
#
# RESOLVED (was ASSUMPTION 2): int8 pairs really are (imaginary, real).  The
#   esp-csi README states it outright and csi_data_read_parse.py builds
#   `complex(raw[2i+1], raw[2i])`.  Kept as a flag only so it can be flipped in
#   an experiment, not because it is in doubt.
ESP_CSI_IMAG_FIRST = True
ESP32_IMAG_FIRST = ESP_CSI_IMAG_FIRST  # legacy alias

# RESOLVED (was ASSUMPTION 5): `local_timestamp` is `rx_ctrl->timestamp`, the
#   receive time in MICROSECONDS.  New wrinkle the first draft missed: it is a
#   uint32, so it WRAPS every 2**32 us = 4294.97 s = 71.6 minutes.  Any capture
#   longer than ~71 min contains a backwards jump; _unwrap_timestamps fixes it.
ESP32_TIMESTAMP_UNITS_PER_SEC = 1_000_000
_TIMESTAMP_MODULUS = 2 ** 32

# RESOLVED (was ASSUMPTION 4): n_rx = 1.  Every chip in the ESP32 family has a
#   single RX chain, so the (n_pkt, n_sub, n_rx) array always has n_rx == 1.
#   Expect noisier per-window statistics than the Intel 5300's 3 chains.

# RESOLVED (was ASSUMPTION 7): the blob is COMMA-separated inside `"[...]"`.
#   The legacy ESP32-CSI-Tool used spaces; both are accepted.

# RESOLVED (was ASSUMPTION 3, partially): subcarrier count is not a guess -- the
#   `len` column states it, and subcarriers = len/2.  With csi_recv's shipped
#   config (HT40, MCS0_LGI, channel 11) a C6 emits len=256 -> 128 subcarriers.
#   The map below was derived from the real example capture printed in esp-csi's
#   own get-started README, whose hard zeros sit at subcarrier indices 0-5,
#   63-65 and 123-127 -- i.e. a centred HT40 layout (index i is subcarrier
#   i-64) with the -64..-59 and +59..+63 guard bands and DC+/-1 nulled.
ESP_CSI_HT40_VALID_SUBCARRIERS = list(range(6, 63)) + list(range(66, 123))   # 114
# HT20 (len=128 -> 64 subcarriers) on the same centring: guards -32..-29 and
# +29..+31, DC at index 32.
ESP_CSI_HT20_VALID_SUBCARRIERS = list(range(4, 32)) + list(range(33, 61))    # 56
# The legacy ESP32-CSI-Tool map this repo shipped with (52 subcarriers), kept so
# old fixtures still load identically.
ESP32_HT20_VALID_SUBCARRIERS = list(range(6, 32)) + list(range(33, 59))      # 52

REFERENCE_SUBCARRIER_MAPS = {
    128: ("HT40", ESP_CSI_HT40_VALID_SUBCARRIERS),
    64: ("HT20", ESP_CSI_HT20_VALID_SUBCARRIERS),
}

# ---------------------------------------------------------------------------
# OPEN ASSUMPTIONS -- cannot be settled without a real capture in hand.
# These are the first things to check the moment hardware data exists; every one
# of them is reported on by verify_esp32_assumptions().
# ---------------------------------------------------------------------------
#
# OPEN 1 -- SUBCARRIER MAP.  We know the guard-band layout from ONE example
#   packet in the upstream README.  One packet cannot distinguish a structural
#   guard band from a subcarrier that merely happened to round to (0,0).  So the
#   default is `valid_subcarriers="auto"`: keep subcarriers that are non-zero in
#   at least a few percent of packets, and WARN if that empirical set disagrees
#   with REFERENCE_SUBCARRIER_MAPS.  Check the warning on the first real file.
AUTO_SUBCARRIER_NONZERO_FRACTION = 0.05

# OPEN 2 -- first_word_invalid.  csi_recv forwards `info->first_word_invalid`.
#   On some targets the first 4 bytes (= first 2 subcarriers) of the buffer are
#   garbage when this flag is 1.  We do NOT drop them by default because on a
#   centred layout those indices are guard band anyway and get dropped by the
#   subcarrier map. If a real capture shows first_word=1 AND subcarriers 0-1
#   carrying energy, set drop_first_word=True.
#
# OPEN 3 -- AGC / FFT gain stability.  csi_recv already multiplies the raw CSI
#   by `compensate_gain` from esp_csi_gain_ctrl before printing, and freezes a
#   gain baseline after 100 packets.  If that compensation works, amplitude is
#   comparable packet-to-packet and our variance features are honest.  If it does
#   NOT, an AGC step will look exactly like a person walking past.  We report the
#   spread of agc_gain/fft_gain over the capture so this is visible; a capture
#   with a wandering AGC needs CONFIG_FORCE_GAIN=1 in the firmware.
#
# OPEN 4 -- PACKET RATE STABILITY.  csi_send sends at CONFIG_SEND_FREQUENCY=100 Hz,
#   but ESP-NOW drops packets under channel congestion, so the DELIVERED rate is
#   unknown until measured.  We derive fs from the timestamps and report the
#   jitter; a heavily-dropping link makes the spectral features (dop_ratio,
#   acf_lag) unreliable because they assume uniform sampling.
#
# OPEN 5 -- SUBCARRIER COUNT UNDER HT40.  csi_recv sets HT40 and enables both
#   acquire_csi_ht20 and acquire_csi_ht40, so a single capture may contain a MIX
#   of 128-int (HT20) and 256-int (HT40) rows.  We keep only the modal length and
#   report how many rows were discarded; if the split is near 50/50 the capture
#   should be redone with one bandwidth pinned.

_CSI_LINE_RE = re.compile(r"^CSI_DATA,")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_csi_blob(blob):
    """Parse the bracketed int list. esp-csi uses commas, ESP32-CSI-Tool spaces.

    JSONL recordings hand us a real list, which needs no parsing at all.
    """
    if isinstance(blob, (list, tuple, np.ndarray)):
        return np.asarray(blob, dtype=np.float32)
    return np.array(blob.strip().strip('"').strip("[]").replace(",", " ").split(),
                    dtype=np.float32)


def _unwrap_timestamps(ts_us):
    """Undo the uint32 microsecond wrap in `local_timestamp`.

    rx_ctrl->timestamp is a 32-bit microsecond counter, so it rolls over every
    71.6 minutes.  Left alone, a single rollover in a long capture makes the
    measured span negative and the derived packet rate nonsense.
    """
    ts = np.asarray(ts_us, dtype=np.float64)
    # The counter is UNSIGNED 32-bit, but it reaches us through a signed int, so
    # everything past 2**31 us (~35.8 min of board uptime) arrives negative.
    # Left alone, the first negative value makes the measured span nonsense and
    # the derived packet rate with it.
    ts = np.where(ts < 0, ts + _TIMESTAMP_MODULUS, ts)
    if len(ts) < 2:
        return ts
    # a genuine backwards step of more than half the modulus is a wrap, not jitter
    wraps = np.cumsum(np.diff(ts) < -(_TIMESTAMP_MODULUS / 2))
    return ts + np.concatenate([[0.0], wraps]) * _TIMESTAMP_MODULUS


JSONL_REQUIRED_KEYS = {"raw_csi", "esp_timestamp", "declared_len"}


def _read_jsonl_rows(filepath):
    """Read a dashboard JSONL recording: one JSON object per accepted packet.

    This is the recorder's NATIVE output and is a strict superset of the CSV
    export -- same raw integer I/Q array, same board metadata, same session
    label -- so training reads it directly and nobody has to click Export CSV
    thirty times.  A truncated final line (power lost mid-write) is counted as
    malformed rather than aborting the file.
    """
    blobs, meta, bad, seen = [], [], 0, 0
    with open(filepath, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            raw = rec.get("raw_csi")
            if not isinstance(raw, list) or not raw or not JSONL_REQUIRED_KEYS <= rec.keys():
                bad += 1
                continue
            # normalise onto the same field names the CSV schemas use
            rec["data"] = raw
            rec["len"] = rec.get("declared_len", "")
            rec["local_timestamp"] = rec.get("esp_timestamp", "")
            blobs.append(raw)
            meta.append(rec)
    return "esp-csi/jsonl-recording", sorted(JSONL_REQUIRED_KEYS), blobs, meta, seen, bad


def _pack_rows(filepath, schema_name, cols, blobs, meta, seen, bad):
    """Shared tail: parallel per-packet arrays from whichever reader ran."""
    if not blobs:
        raise ValueError(f"no CSI packets parsed from {filepath} "
                         f"({seen} candidate rows seen, {bad} rejected)")

    def col(name, dtype=float, default=np.nan):
        vals = []
        for r in meta:
            try:
                vals.append(dtype(r[name]))
            except (KeyError, TypeError, ValueError):
                vals.append(default)
        return np.array(vals)

    return {
        "schema": schema_name,
        "columns": cols,
        "blobs": blobs,
        # session exports and JSONL carry the operator's label; raw dumps do not
        "session_label": next((r["session_label"] for r in meta if r.get("session_label")), ""),
        "session_notes": next((r["session_notes"] for r in meta if r.get("session_notes")), ""),
        "mac": np.array([r.get("mac", "") for r in meta]),
        "rssi": col("rssi"),
        "noise_floor": col("noise_floor"),
        "agc_gain": col("agc_gain"),
        "fft_gain": col("fft_gain"),
        "channel": col("channel"),
        "len": col("len"),
        "first_word": col("first_word"),
        "ts_us_raw": col("local_timestamp"),
        "n_rows_seen": seen,
        "n_rows_bad": bad,
    }


def read_esp_csi_rows(filepath):
    """Read one capture file into parallel per-packet arrays.

    Tolerates: the lying 25-column header the capture tool writes on a C6,
    interleaved ESP_LOG lines from `idf.py monitor`, truncated final rows,
    headerless raw serial dumps, and the labelled collector/session export.
    Raw esp-csi schema is decided by the field count of the data rows, never by
    the header.  Session exports are identified by required header names.

    Returns a dict of numpy arrays plus `schema`, `n_rows_seen`, `n_rows_bad`.
    """
    if str(filepath).endswith(".jsonl"):
        return _pack_rows(filepath, *_read_jsonl_rows(filepath))

    blobs, meta, bad, seen = [], [], 0, 0
    schema_name, cols = None, None
    session_export_cols = None

    with open(filepath, newline="", errors="replace") as fh:
        for row in csv.reader(fh):
            if not row:
                continue

            # The collection UI stores the untouched raw CSI alongside labels
            # and display-only derived arrays.  Normalize only the source
            # fields; downstream feature extraction remains exactly the same.
            header_names = {name.strip() for name in row}
            if (session_export_cols is None
                    and SESSION_EXPORT_REQUIRED_COLUMNS.issubset(header_names)):
                session_export_cols = [name.strip() for name in row]
                schema_name = "esp-csi/session-export"
                cols = session_export_cols
                continue
            if session_export_cols is not None:
                seen += 1
                if len(row) != len(session_export_cols):
                    bad += 1
                    continue
                rec = dict(zip(session_export_cols, row))
                blob = rec.get("raw_csi", "")
                if "[" not in blob:
                    bad += 1
                    continue
                rec["data"] = blob
                rec["len"] = rec.get("declared_len", "")
                rec["local_timestamp"] = rec.get("esp_timestamp", "")
                blobs.append(blob)
                meta.append(rec)
                continue

            if row[0].strip() != "CSI_DATA":
                # header line, ESP_LOG chatter, or a blank -- not a packet
                if row and any(c in row[0] for c in ("type,", "type")) and len(row) > 5:
                    pass  # header; ignored on purpose (see UPSTREAM BUG note)
                continue
            seen += 1
            if schema_name is None:
                schema_name = _BY_WIDTH.get(len(row))
                if schema_name is None:
                    raise ValueError(
                        f"{filepath}: CSI_DATA row has {len(row)} fields; expected one of "
                        f"{sorted(_BY_WIDTH)} (esp-csi C5/C6/C61=15, esp-csi classic=25, "
                        f"ESP32-CSI-Tool=26). Row starts: {','.join(row[:6])}")
                cols = SCHEMAS[schema_name]
            if len(row) != len(cols):
                bad += 1          # truncated / interleaved line
                continue
            rec = dict(zip(cols, row))
            blob = rec.get("data") or rec.get("CSI_DATA") or ""
            if "[" not in blob:
                bad += 1
                continue
            blobs.append(blob)
            meta.append(rec)

    return _pack_rows(filepath, schema_name, cols, blobs, meta, seen, bad)


def _auto_valid_subcarriers(csi, verbose=True):
    """Pick usable subcarriers empirically (OPEN 1).

    A guard-band / DC subcarrier is *exactly* (0,0) in every packet, because the
    radio never transmitted there.  A real subcarrier is essentially never zero
    twice in a row.  So "non-zero in more than a few percent of packets" is a
    clean structural separator, and unlike a hard-coded index map it cannot
    silently mis-slice a bandwidth we have not seen.
    """
    nonzero_frac = (np.abs(csi[:, :, 0]) > 0).mean(axis=0)
    keep = np.where(nonzero_frac > AUTO_SUBCARRIER_NONZERO_FRACTION)[0]
    n_sub_total = csi.shape[1]

    ref = REFERENCE_SUBCARRIER_MAPS.get(n_sub_total)
    if ref is not None and verbose:
        name, expected = ref
        if list(keep) != expected:
            extra = sorted(set(keep) - set(expected))
            missing = sorted(set(expected) - set(keep))
            print(f"  [OPEN 1] empirical subcarrier set ({len(keep)}) differs from the "
                  f"documented {name} map ({len(expected)}). "
                  f"unexpected-active={extra[:8]}{'...' if len(extra) > 8 else ''} "
                  f"unexpectedly-dead={missing[:8]}{'...' if len(missing) > 8 else ''}. "
                  f"Trusting the data; update ESP_CSI_{name}_VALID_SUBCARRIERS if this "
                  f"holds across captures.")
    return list(keep)


def load_esp32_csi_csv(filepath, window=None, stride=None, valid_subcarriers="auto",
                       window_seconds=None, overlap=0.0,
                       imag_first=ESP_CSI_IMAG_FIRST, drop_first_word=False,
                       verbose=True):
    """Load an esp-csi (or legacy ESP32-CSI-Tool) capture into our feature space.

    valid_subcarriers : "auto" (default, empirical -- see OPEN 1), an explicit
                        list of indices, or None to keep every subcarrier.

    Returns a dict with:
      X            : (n_windows, 16) features, columns == FEATURE_NAMES
      fs           : measured packet rate in Hz
      amp, phase   : (n_pkt, n_sub, 1) raw arrays, for plotting/debugging
      t            : (n_pkt,) capture-relative time in SECONDS -- what the manual
                     labelling connector joins against
      window_t0/t1 : (n_windows,) start/end time of each window, in the same
                     seconds-since-capture-start clock
      schema, n_subcarriers, window, diagnostics
    """
    rec = read_esp_csi_rows(filepath)

    raw_list = [_parse_csi_blob(b) for b in rec["blobs"]]
    lengths = np.array([len(r) for r in raw_list])
    n_vals = int(np.bincount(lengths).argmax())          # modal length (OPEN 5)
    keep_row = lengths == n_vals
    n_mixed = int((~keep_row).sum())
    raw = np.stack([r for r, k in zip(raw_list, keep_row) if k]).astype(np.float32)

    # cross-check the `len` column against what we actually parsed
    declared = rec["len"][keep_row]
    len_mismatch = int(np.sum(np.isfinite(declared) & (declared != n_vals)))

    n_sub_total = n_vals // 2
    if imag_first:
        imag, real = raw[:, 0::2], raw[:, 1::2]
    else:
        real, imag = raw[:, 0::2], raw[:, 1::2]
    csi = (real + 1j * imag)[:, :, None]                 # (n_pkt, n_sub, n_rx=1)

    if drop_first_word:                                  # OPEN 2
        csi[:, :2, :] = 0

    if valid_subcarriers == "reference":
        # Training REQUIRES every recording to use an identical subcarrier set --
        # "auto" is per-file and a single dead subcarrier in one capture silently
        # shifts every column of the feature vector.  Pin the documented map.
        ref = REFERENCE_SUBCARRIER_MAPS.get(n_sub_total)
        if ref is None:
            raise ValueError(
                f"{filepath}: no documented subcarrier map for {n_sub_total} raw "
                f"subcarriers (known: {sorted(REFERENCE_SUBCARRIER_MAPS)}). "
                f"Pass an explicit list or 'auto'.")
        valid_subcarriers = ref[1]
    elif valid_subcarriers == "auto":
        valid_subcarriers = _auto_valid_subcarriers(csi, verbose=verbose)
    if valid_subcarriers is not None:
        csi = csi[:, list(valid_subcarriers), :]

    amp = np.abs(csi).astype(np.float32)
    phase = _sanitize_phase(np.angle(csi)).astype(np.float32)

    # --- timing (OPEN 4) -------------------------------------------------
    ts = _unwrap_timestamps(rec["ts_us_raw"][keep_row])
    t = (ts - ts[0]) / ESP32_TIMESTAMP_UNITS_PER_SEC if len(ts) else np.zeros(len(csi))
    span = float(t[-1]) if len(t) > 1 else 0.0
    fs = (len(t) - 1) / span if span > 0 else 50.0
    dt = np.diff(t) if len(t) > 1 else np.array([1 / 50.0])
    jitter = float(np.std(dt) / np.mean(dt)) if np.mean(dt) > 0 else float("nan")
    fs_ok = np.isfinite(fs) and 1.0 < fs < 5000.0
    if not fs_ok:
        if verbose:
            print(f"  [warn] implausible packet rate ({fs:.3g} Hz) from local_timestamp; "
                  f"defaulting to 50 Hz. Timestamps may be broken.")
        fs = 50.0
        t = np.arange(len(csi)) / fs

    # Window length is defined in SECONDS so a 100 Hz ESP32 capture yields
    # physically comparable windows to the 2.56 s Intel windows.
    if window is None:
        window = max(32, int(round(fs * (window_seconds or (WINDOW / 50.0)))))
    if stride is None:
        # overlap>0 multiplies the number of training rows.  It is only safe when
        # the train/test split is BY RECORDING -- two overlapping windows share
        # packets, so splitting by window would put the same packets on both sides.
        stride = max(1, int(round(window * (1.0 - overlap))))

    X = windows_from_arrays(amp, phase, window=window, stride=stride, fs=fs)
    starts = np.arange(0, len(amp) - window + 1, stride)
    window_t0 = t[starts] if len(starts) else np.empty(0)
    window_t1 = t[starts + window - 1] if len(starts) else np.empty(0)

    def _stat(fn, name):
        """nan-safe summary: a column absent from this schema is all-NaN, and
        np.nanstd of an all-NaN slice warns rather than just returning NaN."""
        v = rec[name][keep_row]
        v = v[np.isfinite(v)]
        return float(fn(v)) if len(v) else float("nan")

    diagnostics = {
        "n_rows_seen": rec["n_rows_seen"],
        "n_rows_bad": rec["n_rows_bad"],
        "n_mixed_length_rows": n_mixed,
        "len_column_mismatches": len_mismatch,
        "n_subcarriers_raw": n_sub_total,
        "rssi_mean": _stat(np.mean, "rssi"),
        "rssi_std": _stat(np.std, "rssi"),
        "noise_floor_mean": _stat(np.mean, "noise_floor"),
        "agc_gain_std": _stat(np.std, "agc_gain"),
        "fft_gain_std": _stat(np.std, "fft_gain"),
        "first_word_invalid_frac": _stat(lambda v: np.mean(v != 0), "first_word"),
        "packet_interval_jitter": jitter,
        "duration_s": span,
    }

    if verbose:
        print(f"  loaded {len(csi)} packets ({rec['schema']}), {csi.shape[1]} usable "
              f"of {n_sub_total} subcarriers, {fs:.1f} Hz over {span:.1f} s "
              f"-> {len(X)} windows of {window} packets ({window / fs:.2f} s)")

    return {"X": X, "fs": fs, "amp": amp, "phase": phase, "t": t,
            "window_t0": window_t0, "window_t1": window_t1,
            "window": window, "stride": stride, "schema": rec["schema"],
            "session_label": rec["session_label"], "session_notes": rec["session_notes"],
            "subcarriers": list(valid_subcarriers) if valid_subcarriers is not None else None,
            "n_subcarriers": csi.shape[1], "rssi": rec["rssi"][keep_row],
            "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# Assumption checklist
# ---------------------------------------------------------------------------

def verify_esp32_assumptions(filepath, verbose=True):
    """Print a checklist against a capture. Run this on the FIRST real hardware file.

    Everything marked RESOLVED is confirmed against upstream source; the OPEN
    items are the ones only real data can settle.
    """
    print(f"Checking esp-csi assumptions against {filepath}")
    rec = read_esp_csi_rows(filepath)
    schema_source = ("header field names" if rec["schema"] == "esp-csi/session-export"
                     else "row field count")
    print(f"  schema detected .............. {rec['schema']} "
          f"({len(rec['columns'])} columns, from {schema_source})")
    with open(filepath, errors="replace") as fh:
        header = fh.readline().strip()
    hdr_n = len(header.split(","))
    if header.startswith("type") and hdr_n != len(rec["columns"]):
        print(f"  header row ................... LIES ({hdr_n} names vs "
              f"{len(rec['columns'])} fields) - known csi_data_read_parse.py bug, "
              f"ignored by design")
    out = load_esp32_csi_csv(filepath, verbose=verbose)
    d = out["diagnostics"]

    csi_location = ("raw_csi field in session export" if rec["schema"] == "esp-csi/session-export"
                    else "positional (last field), name-independent")
    print(f"  [RESOLVED] CSI column ........ {csi_location}")
    print(f"  [RESOLVED] imag-first order .. per esp-csi README + parse tool")
    print(f"  [RESOLVED] n_rx = 1 .......... ESP32 family has one RX chain")
    print(f"  [RESOLVED] timestamp units ... microseconds, uint32, wrap-unwrapped")
    ref = REFERENCE_SUBCARRIER_MAPS.get(d["n_subcarriers_raw"])
    if ref is None:
        verdict = "no documented map for this width - inspect manually"
    elif out["n_subcarriers"] == len(ref[1]):
        verdict = f"matches the documented {ref[0]} map"
    else:
        verdict = f"DIFFERS from the documented {ref[0]} map ({len(ref[1])})"
    print(f"  [OPEN 1] subcarrier map ...... {d['n_subcarriers_raw']} raw -> "
          f"{out['n_subcarriers']} kept ({verdict})")
    print(f"  [OPEN 2] first_word_invalid .. {d['first_word_invalid_frac']:.1%} of packets"
          f"{'  <-- consider drop_first_word=True' if d['first_word_invalid_frac'] > 0.5 else ''}")
    if not np.isfinite(d["agc_gain_std"]):
        gain = "not reported by this schema"
    elif d["agc_gain_std"] > 2:
        gain = (f"agc sd={d['agc_gain_std']:.2f}, fft sd={d['fft_gain_std']:.2f}"
                f"  <-- AGC is wandering; set CONFIG_FORCE_GAIN 1 and reflash")
    else:
        gain = f"agc sd={d['agc_gain_std']:.2f}, fft sd={d['fft_gain_std']:.2f}  (stable)"
    print(f"  [OPEN 3] gain stability ...... {gain}")
    print(f"  [OPEN 4] packet rate ......... {out['fs']:.1f} Hz over {d['duration_s']:.1f} s, "
          f"interval jitter {d['packet_interval_jitter']:.2f}"
          f"{'  <-- lossy link, spectral features degraded' if d['packet_interval_jitter'] > 0.5 else '  (uniform)'}")
    print(f"  [OPEN 5] mixed bandwidths .... {d['n_mixed_length_rows']} rows dropped for "
          f"non-modal CSI length"
          f"{'  <-- pin one bandwidth and recapture' if d['n_mixed_length_rows'] > 0.1 * d['n_rows_seen'] else ''}")
    print(f"  link quality ................. RSSI {d['rssi_mean']:.1f} +/- {d['rssi_std']:.1f} dBm, "
          f"noise floor {d['noise_floor_mean']:.1f} dBm")
    print(f"  parse health ................. {d['n_rows_bad']} malformed of "
          f"{d['n_rows_seen']} CSI rows, {d['len_column_mismatches']} len-column mismatches")
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        verify_esp32_assumptions(target)
    else:
        print(__doc__)
