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
    """Parse the bracketed int list. esp-csi uses commas, ESP32-CSI-Tool spaces."""
    return np.array(blob.strip().strip('"').strip("[]").replace(",", " ").split(),
                    dtype=np.float32)


def _unwrap_timestamps(ts_us):
    """Undo the uint32 microsecond wrap in `local_timestamp`.

    rx_ctrl->timestamp is a 32-bit microsecond counter, so it rolls over every
    71.6 minutes.  Left alone, a single rollover in a long capture makes the
    measured span negative and the derived packet rate nonsense.
    """
    ts = np.asarray(ts_us, dtype=np.float64)
    if len(ts) < 2:
        return ts
    # a genuine backwards step of more than half the modulus is a wrap, not jitter
    wraps = np.cumsum(np.diff(ts) < -(_TIMESTAMP_MODULUS / 2))
    return ts + np.concatenate([[0.0], wraps]) * _TIMESTAMP_MODULUS


def read_esp_csi_rows(filepath):
    """Read one capture file into parallel per-packet arrays.

    Tolerates: the lying 25-column header the capture tool writes on a C6,
    interleaved ESP_LOG lines from `idf.py monitor`, truncated final rows,
    headerless raw serial dumps, and the labelled collector/session export.
    Raw esp-csi schema is decided by the field count of the data rows, never by
    the header.  Session exports are identified by required header names.

    Returns a dict of numpy arrays plus `schema`, `n_rows_seen`, `n_rows_bad`.
    """
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

    if not blobs:
        raise ValueError(f"no CSI_DATA rows parsed from {filepath} "
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

    if valid_subcarriers == "auto":
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
        window = max(32, int(round(fs * (WINDOW / 50.0))))
    if stride is None:
        stride = window

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


# ---------------------------------------------------------------------------
# Synthetic fixture in the EXACT esp-csi format
# ---------------------------------------------------------------------------

def make_esp_csi_fixture(path, n_packets=1500, fs=100.0, occupied=True, seed=0,
                         schema="esp-csi/c5c6c61", buggy_header=True,
                         n_people=1):
    """Write a file byte-compatible with what csi_data_read_parse.py saves.

    This is a FORMAT fixture, not physics.  It proves the ingestion path runs
    end to end before hardware exists; it must never be used to claim anything
    about accuracy.

    buggy_header : reproduce the upstream bug where the tool writes the
                   25-column header even for 15-column C6 rows.  Default True,
                   because that is what a real capture will look like.
    """
    rng = np.random.default_rng(seed)
    cols = SCHEMAS[schema]
    c6 = schema == "esp-csi/c5c6c61"

    # HT40 on a C6: 128 subcarriers, of which 114 carry energy.
    n_sub = 128 if c6 else 64
    valid = (ESP_CSI_HT40_VALID_SUBCARRIERS if n_sub == 128
             else ESP_CSI_HT20_VALID_SUBCARRIERS)

    # A plausible static multipath signature: frequency-selective fading across
    # the band, amplitudes in the int8 range the firmware actually prints
    # (the upstream example row spans roughly -31..+19 per component).
    base = np.zeros(n_sub)
    k = np.linspace(0, 1, len(valid))
    base[valid] = 18 + 7 * np.sin(2 * np.pi * 1.7 * k) + 3 * np.cos(2 * np.pi * 4.3 * k)

    # Occupied: a body is a slow, correlated modulator of the multipath sum, and
    # it perturbs subcarriers unequally.  Empty: a static channel plus thermal
    # noise, which is broadband and decorrelates instantly.  This is exactly the
    # contrast features.py is built to measure.
    if occupied:
        walk = np.cumsum(rng.normal(0, 0.28, n_packets))
        sc_weight = 0.5 + rng.random(n_sub)
        amp_noise, ph_noise = 0.55, 0.16
    else:
        walk = np.zeros(n_packets)
        sc_weight = np.zeros(n_sub)
        amp_noise, ph_noise = 0.45, 0.03

    ramp = np.linspace(-2.0, 2.0, n_sub)  # the per-packet linear phase slope
    ts0 = int(rng.integers(0, 2 ** 31))   # boot clock starts wherever it likes
    mac = "1a:00:00:00:00:00"             # CONFIG_CSI_SEND_MAC from csi_send

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ESP_CSI_CLASSIC_COLUMNS if (buggy_header and c6) else cols)
        for i in range(n_packets):
            gain = 1.0 + 0.30 * sc_weight * np.sin(walk[i]) if occupied else 1.0
            mag = base * gain + rng.normal(0, amp_noise, n_sub)
            phi = (ramp + rng.normal(0, 0.9)          # per-packet CFO/STO slope
                   + rng.normal(0, ph_noise, n_sub)
                   + (0.8 * sc_weight * np.sin(walk[i]) if occupied else 0.0))
            mag[base == 0] = 0.0                       # guard bands are hard zeros
            re = np.clip(np.round(mag * np.cos(phi)), -128, 127).astype(int)
            im = np.clip(np.round(mag * np.sin(phi)), -128, 127).astype(int)
            re[base == 0] = 0
            im[base == 0] = 0
            inter = np.empty(2 * n_sub, dtype=int)
            inter[0::2], inter[1::2] = (im, re) if ESP_CSI_IMAG_FIRST else (re, im)
            blob = "[" + ",".join(map(str, inter)) + "]"

            # 100 Hz nominal with realistic jitter, plus the odd ESP-NOW drop
            ts = ts0 + int(i * ESP32_TIMESTAMP_UNITS_PER_SEC / fs
                           + rng.normal(0, 250)) % _TIMESTAMP_MODULUS
            rssi = int(np.round(rng.normal(-28 if occupied else -27, 1.5)))
            if c6:
                row = ["CSI_DATA", i, mac, rssi, 11, -96,
                       32, 4, 11, ts, 47, 0, 2 * n_sub, 0, blob]
            else:
                row = ["CSI_DATA", i, mac, rssi, 11, 1, 6, 1, 0, 1, 0, 1, 0, 0,
                       -93, 0, 11, 2, ts, 0, 67, 0, 2 * n_sub, 0, blob]
            w.writerow(row)
    return path


def make_synthetic_esp32_csv(path, n_packets=1200, fs=100.0, occupied=True, seed=0):
    """Back-compat shim: the old name now emits the real esp-csi C6 format."""
    return make_esp_csi_fixture(path, n_packets=n_packets, fs=fs,
                                occupied=occupied, seed=seed)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        verify_esp32_assumptions(target)
    else:
        print(__doc__)
