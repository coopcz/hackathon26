"""
Selective access to the EHUNAM dataset (Diaz et al., Scientific Data 2025).

  https://doi.org/10.6084/m9.figshare.28541225

WHY THIS FILE EXISTS
--------------------
EHUNAM is published as a single 77.7 GB zip.  Figshare exposes no per-file
download, which normally rules it out for a short project.  But the S3 endpoint
behind the download URL honours HTTP Range requests, and a zip stores its index
(the "central directory") at the END of the archive.  So we can:

  1. fetch the last ~4 MB and parse the index of all 2,401 members,
  2. range-fetch only the members we actually want,
  3. rebuild each one as a standalone one-entry zip and extract it.

Total transferred for a useful working subset: a few GB instead of 77.7.

ONE TOOLING WRINKLE: every member is stored with compression method 9
(Deflate64).  Python's `zipfile` does not implement it and raises
NotImplementedError.  macOS/Info-ZIP `unzip` does implement it, so extraction
shells out.  (`bsdtar` and `ditto` both fail on method 9 -- do not substitute.)

WHY THIS DATASET, GIVEN WE ALREADY HAVE A BASELINE
--------------------------------------------------
It is not a replacement for WiFi-CrowdCounting; it is the validation set that
the Intel data cannot provide:

  * Hardware is a Broadcom BCM43455C0 (Raspberry Pi + nexmon): 64 subcarriers,
    HT20, single RX per capture.  An ESP32 is 64 subcarriers, HT20, single RX.
    That geometry match makes it a far better ESP32 proxy than the Intel 5300's
    30-subcarrier 3x2 MIMO.
  * 8 environments over 23 days, with repeat visits -- so it tests both
    cross-environment and cross-DAY drift.  Our 3-room dataset tests neither.
  * 168 `MAR` measurements are machines running with NO people present.  That is
    a direct false-positive test for the HVAC use case: does a running appliance
    read as a human?
"""

import os
import re
import struct
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FIGSHARE_ARTICLE = 28541225
ZIP_URL = "https://ndownloader.figshare.com/files/52814369"
# Authoritative total from the Content-Range header; the zip index is keyed to it.
ZIP_TOTAL = 77732935390

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "data", "ehunam")
INVENTORY = os.path.join(CACHE_DIR, "inventory.tsv")

# Filename schema, from the dataset's own documentation:
#   Campaign_Set_Receiver_Application_People_Activity_Machine_Status_Sequence
# e.g. MC1_01A_1_E_#_#_#_#_01.mat  -> campaign MC1, set 01A, receiver 1,
#      application E (empty), no people, no activity, no machine, no status, seq 01
FIELDS = ["campaign", "setid", "rx", "app", "people", "activity", "machine", "status", "seq"]

# Application codes actually present in the archive (counts verified against the index):
#   E     369  empty room, no people AND no machines
#   PC    937  people counting
#   HAR   663  human activity recognition
#   MAR   168  machine activity, NO people      <- our false-positive probe
#   PCMAR 177  people and machines together
#   MR     87  home-appliance recognition


# ---------------------------------------------------------------------------
# 1. the index
# ---------------------------------------------------------------------------

def _curl_range(url, start, end, dest, timeout=300):
    """Range-GET via curl. curl -L re-sends the Range header across figshare's
    redirect to a presigned S3 URL, which expires in 10 s -- so every request
    must start from the figshare URL rather than reusing a signed one."""
    subprocess.run(["curl", "-sSL", "--max-time", str(timeout), "-r",
                    f"{start}-{end}", url, "-o", dest], check=True)


def _zip64_fixup(extra, usz, csz, off):
    """Replace saturated 32-bit values with the ZIP64 extra field's 64-bit ones."""
    MAX32 = 0xFFFFFFFF
    if MAX32 not in (usz, csz, off):
        return usz, csz, off
    i = 0
    while i + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[i:i + 4])
        body = extra[i + 4:i + 4 + size]
        if tag == 0x0001:
            vals, k = [], 0
            # order is fixed: uncompressed, compressed, local-header offset, disk
            for saturated in (usz == MAX32, csz == MAX32, off == MAX32):
                if saturated and k + 8 <= len(body):
                    vals.append(struct.unpack("<Q", body[k:k + 8])[0])
                    k += 8
                else:
                    vals.append(None)
            usz = vals[0] if vals[0] is not None else usz
            csz = vals[1] if vals[1] is not None else csz
            off = vals[2] if vals[2] is not None else off
            break
        i += 4 + size
    return usz, csz, off


def fetch_inventory(force=False, tail_bytes=4 << 20):
    """Parse all 2,401 member records out of the zip's central directory.

    Downloads ~4 MB, not 77.7 GB.
    """
    if os.path.exists(INVENTORY) and not force:
        return read_inventory()

    os.makedirs(CACHE_DIR, exist_ok=True)
    tail = os.path.join(CACHE_DIR, "_tail.bin")
    _curl_range(ZIP_URL, ZIP_TOTAL - tail_bytes, ZIP_TOTAL - 1, tail)
    buf = open(tail, "rb").read()
    base = ZIP_TOTAL - len(buf)

    # The classic EOCD caps cd_offset at 0xFFFFFFFF for archives >4 GB, so the
    # real offset lives in the ZIP64 EOCD record.
    j = buf.rfind(b"PK\x06\x06")
    if j < 0:
        raise RuntimeError("no ZIP64 end-of-central-directory found")
    z = struct.unpack("<IQHHIIQQQQ", buf[j:j + 56])
    cd_total, cd_size, cd_off = z[7], z[8], z[9]
    if cd_off < base:
        raise RuntimeError("central directory not inside fetched tail; raise tail_bytes")

    cd = buf[cd_off - base: cd_off - base + cd_size]
    recs, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        hdr = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        meth, csz, usz, nlen, elen, clen, off = hdr[4], hdr[8], hdr[9], hdr[10], hdr[11], hdr[12], hdr[16]
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen: p + 46 + nlen + elen]
        # In a ZIP64 archive any 32-bit field that overflows is stored as all-ones
        # and its real value moves into the 0x0001 extra field, present in that
        # fixed order and ONLY for the fields that actually overflowed.  Every
        # member here sits past the 4 GB mark, so the local-header offset is the
        # one that matters -- read it straight from the 32-bit field and you seek
        # to garbage.
        usz, csz, off = _zip64_fixup(extra, usz, csz, off)
        recs.append({"name": name, "csz": csz, "usz": usz, "offset": off, "method": meth})
        p += 46 + nlen + elen + clen
    if len(recs) != cd_total:
        raise RuntimeError(f"parsed {len(recs)} entries, index claims {cd_total}")

    os.remove(tail)
    with open(INVENTORY, "w") as fh:
        fh.write("\t".join(["name", "csz", "usz", "offset", "method"]) + "\n")
        for r in recs:
            fh.write(f"{r['name']}\t{r['csz']}\t{r['usz']}\t{r['offset']}\t{r['method']}\n")
    return read_inventory()


SUMMARY_URL = "https://ndownloader.figshare.com/files/52834283"   # Summary.xlsx, 268 KB
SUMMARY_XLSX = os.path.join(CACHE_DIR, "Summary.xlsx")


def fetch_summary(force=False):
    """Per-measurement metadata for all 2,401 files, without touching the 77.7 GB zip.

    Figshare publishes Summary.xlsx as a separate 268 KB file.  It carries BW,
    Enviroment, NIC, Traffic, Date and N_CSI (the packet count) per measurement,
    which is what makes principled selection possible: bandwidth and environment
    are NOT encoded in the filename, so without this you would have to download
    blind and discard afterwards.

    Parsed straight from the xlsx XML to avoid adding an openpyxl dependency.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(SUMMARY_XLSX) or force:
        subprocess.run(["curl", "-sSL", "--max-time", "120", SUMMARY_URL,
                        "-o", SUMMARY_XLSX], check=True)

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(SUMMARY_XLSX)
    shared = ["".join(t.text or "" for t in si.iter(ns + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        vals = []
        for c in row.iter(ns + "c"):
            v = c.find(ns + "v")
            if v is None:
                vals.append("")
            elif c.get("t") == "s":
                vals.append(shared[int(v.text)])
            else:
                vals.append(v.text)
        rows.append(vals)

    head = rows[0]
    return {r[0]: dict(zip(head, r)) for r in rows[1:] if r}


def _parse_name(name):
    parts = re.sub(r"\.mat$", "", name).split("_")
    if len(parts) != len(FIELDS):
        return {}
    d = dict(zip(FIELDS, parts))
    # People are encoded as concatenated per-person letters ('#' when nobody);
    # the count of letters is the occupancy.
    d["n_people"] = 0 if d["people"] == "#" else len(d["people"])
    return d


def read_inventory():
    out = []
    with open(INVENTORY) as fh:
        next(fh)
        for line in fh:
            name, csz, usz, off, meth = line.rstrip("\n").split("\t")
            rec = {"name": name, "csz": int(csz), "usz": int(usz),
                   "offset": int(off), "method": int(meth)}
            rec.update(_parse_name(name))
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# 2. selective extraction
# ---------------------------------------------------------------------------

def _rebuild_single_entry_zip(member_bytes, rec, dest_zip):
    """Wrap one range-fetched member in a synthetic one-entry zip.

    The fetched bytes start with the member's local file header, which already
    carries the name, CRC and sizes; we only have to append a central directory
    and an EOCD so a normal unzip will accept the file.
    """
    if member_bytes[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{rec['name']}: expected local header, got {member_bytes[:4]!r}")
    nlen, elen = struct.unpack("<HH", member_bytes[26:30])
    data_start = 30 + nlen + elen
    local = member_bytes[:data_start + rec["csz"]]
    name = member_bytes[30:30 + nlen]
    mtime, mdate = struct.unpack("<HH", member_bytes[10:14])
    crc = struct.unpack("<I", member_bytes[14:18])[0]
    cdir = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 0, rec["method"],
                       mtime, mdate, crc, rec["csz"], rec["usz"],
                       nlen, 0, 0, 0, 0, 0, 0) + name
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(cdir), len(local), 0)
    with open(dest_zip, "wb") as fh:
        fh.write(local + cdir + eocd)


def extract_member(rec, dest_dir, overwrite=False):
    """Range-fetch one member and extract it. Returns the local .mat path."""
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, rec["name"])
    if os.path.exists(out_path) and not overwrite:
        return out_path

    tmp_zip = out_path + ".part.zip"
    start = rec["offset"]
    # 30-byte local header + name + extra field; 1 KB slack covers any extra field.
    end = start + 30 + 1024 + rec["csz"]
    _curl_range(ZIP_URL, start, min(end, ZIP_TOTAL - 1), tmp_zip)
    raw = open(tmp_zip, "rb").read()
    _rebuild_single_entry_zip(raw, rec, tmp_zip)

    # Method 9 (Deflate64): Info-ZIP `unzip` handles it, bsdtar/ditto do not.
    res = subprocess.run(["unzip", "-o", "-q", tmp_zip, "-d", dest_dir],
                         capture_output=True, text=True)
    os.remove(tmp_zip)
    if not os.path.exists(out_path):
        raise RuntimeError(f"extract failed for {rec['name']}: {res.stderr.strip()[:300]}")
    return out_path


def extract_many(recs, dest_dir, workers=8, progress=True):
    """Fetch many members in parallel. Returns (paths, failures)."""
    paths, failures = [], []

    def one(r):
        try:
            return ("ok", extract_member(r, dest_dir))
        except Exception as e:                      # keep going; report at the end
            return ("err", f"{r['name']}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (kind, val) in enumerate(ex.map(one, recs), 1):
            (paths if kind == "ok" else failures).append(val)
            if progress and i % 10 == 0:
                print(f"    {i}/{len(recs)} ...", flush=True)
    return paths, failures


# ---------------------------------------------------------------------------
# 3. loading a measurement into our feature pipeline's input format
# ---------------------------------------------------------------------------

# 802.11n HT20 with a 64-point FFT, in fftshifted order (index 32 == DC):
# occupied subcarriers are -28..-1 and +1..+28, i.e. 4..31 and 33..60 = 56,
# which matches the `Occupied_SC` field the files themselves report.
HT20_OCCUPIED = list(range(4, 32)) + list(range(33, 61))

# We default to a narrower slice than that, for two reasons:
#  1. nexmon on the BCM43455 returns unreliable values at the band edges -- in
#     the first file inspected, subcarriers 60/61 read ~10277 and ~35650 against
#     a typical ~300, which would dominate any variance feature.
#  2. It is exactly the range used for ESP32 captures in pipeline.py, so EHUNAM
#     and our own hardware end up with identical subcarrier geometry (52) and the
#     cross-hardware comparison is like-for-like rather than approximate.
ESP32_ALIGNED = list(range(6, 32)) + list(range(33, 59))


class UnsupportedBandwidth(ValueError):
    """Raised for captures whose subcarrier layout we deliberately do not handle."""


def _default_subcarriers(width, edge_guard=True):
    """Pick the subcarrier slice for a capture, by its array width.

    The campaigns are not uniform: MC1/MC3 ship the raw 64-bin FFT (nulls
    included), while MC2 ships 56 bins with the nulls already stripped.  The 80
    MHz captures (256/241/208 bins) are a different channel geometry entirely and
    are rejected rather than silently mixed in -- an ESP32 is a 20 MHz radio, so
    including 80 MHz data would compare across two different physical channels.
    """
    if width == 64:
        return ESP32_ALIGNED if edge_guard else HT20_OCCUPIED
    if width == 56:
        # already occupied-only; trim 2 per edge to land on the same 52-subcarrier
        # geometry as the 64-bin captures and as our ESP32 loader
        return list(range(2, 28)) + list(range(28, 54)) if edge_guard else list(range(56))
    raise UnsupportedBandwidth(
        f"{width} subcarriers: 80 MHz capture, not comparable to 20 MHz ESP32 data")


def load_measurement(path, subcarriers=None, edge_guard=True, target_fs=50.0):
    """Load one EHUNAM .mat into (amp, phase, fs, meta).

    Returns amp/phase shaped (n_packets, n_subcarriers, 1) -- the trailing axis
    is the RX chain, so the arrays drop straight into window_features() exactly
    like Intel or ESP32 data.

    target_fs: decimate to approximately this packet rate (None to keep the
    original).  This is not cosmetic.  EHUNAM captures run anywhere from ~250 Hz
    (FTP) to ~4300 Hz (HT sounding), while the Intel baseline is 50 Hz and an
    ESP32 is typically 50-100 Hz.  Two features are defined against the sample
    rate, and although both are now defined against absolute time (see
    features.py) a 4 kHz stream still spends almost all its samples measuring
    thermal noise rather than motion -- so
    comparing a 4300 Hz capture against a 50 Hz one measures the packet rate,
    not the room.  Decimating to a common rate makes the numbers mean the same
    thing.  Human motion is well under 10 Hz, so 50 Hz keeps it comfortably
    inside Nyquist.
    """
    import scipy.io as sio
    m = sio.loadmat(path)

    csi = np.asarray(m["CSI"])                       # (n_pkt, n_sub) complex
    if subcarriers is None:
        subcarriers = _default_subcarriers(csi.shape[1], edge_guard)
    csi = csi[:, subcarriers][:, :, None]            # -> (n_pkt, n_sub, 1)

    ts = np.asarray(m["TimeStamp"], dtype=np.float64).ravel()
    span = float(ts[-1] - ts[0]) if ts.size > 1 else 0.0
    # TimeStamp is in SECONDS since the start of the measurement (verified: a
    # 603-packet file spans 60.29, i.e. the declared 60 s measurement window).
    fs = (len(ts) - 1) / span if span > 0 else float("nan")

    from .features import _sanitize_phase
    amp = np.abs(csi).astype(np.float32)
    phase = _sanitize_phase(np.angle(csi)).astype(np.float32)

    if target_fs and np.isfinite(fs) and fs > target_fs * 1.5:
        step = int(round(fs / target_fs))
        # Downsampling must happen AFTER |CSI|, never on the complex CSI.
        # Consecutive packets carry independent random carrier-phase offsets, so
        # low-pass filtering the complex values makes them interfere
        # destructively and the resulting "amplitude" is an artefact -- measured
        # on this data it drove the coefficient of variation to ~0.52 for empty
        # AND occupied rooms alike, erasing the signal completely.
        #
        # A boxcar mean over `step` packets of the amplitude is a clean
        # anti-alias filter plus decimation in one step, and it is immune to the
        # phase problem because amplitude is already phase-invariant.
        n = (len(amp) // step) * step
        amp = amp[:n].reshape(-1, step, amp.shape[1], amp.shape[2]).mean(axis=1)
        phase = phase[:n].reshape(-1, step, phase.shape[1], phase.shape[2]).mean(axis=1)
        ts = ts[:n:step][:amp.shape[0]]
        fs = fs / step

    def scalar(key, default=None):
        v = m.get(key)
        if v is None or np.size(v) == 0:
            return default
        v = np.asarray(v).ravel()[0]
        return v.item() if hasattr(v, "item") else v

    meta = {
        "app": str(scalar("Application", "")),
        "environment": str(scalar("Enviroment", "")),   # sic -- the dataset's spelling
        "nic": str(scalar("NIC", "")),
        "traffic": str(scalar("Traffic", "")),
        "bw": scalar("BW"), "band": scalar("Band"), "channel": scalar("Channel"),
        "rx": scalar("Rx"), "n_rx": scalar("N_Rx"),
        "occupied_sc": scalar("Occupied_SC"), "subcarriers": scalar("Subcarriers"),
        "t_meas": scalar("T_Meas"), "n_packets": int(amp.shape[0]),
        "fs": fs, "set": str(scalar("Set", "")),
        "n_people": _parse_name(os.path.basename(path)).get("n_people", 0),
    }

    return amp, phase, fs, meta


# ---------------------------------------------------------------------------
# 4. choosing what to download
# ---------------------------------------------------------------------------

# Presence label mapping for the HVAC task.  The three-way split matters more
# than a binary one, because the interesting failure is in the middle group.
PRESENCE_GROUPS = {
    "E":     "away",            # no people, no machines -- the true negative
    "PC":    "home",            # 1-4 people
    "HAR":   "home",            # 1 person performing activities
    "MR":    "away_machines",   # 0 people, 1-4 HOME APPLIANCES running
    "MAR":   "away_machines",   # 0 people, industrial machines running
    # PC+MAR (people AND machines) is excluded: it is "home" but confounded, and
    # we have enough clean "home" data without it.
}


def select_subset(summary, inv, bw="20", rx="1", apps=None, max_total_gb=2.5):
    """Choose which members to download, using the summary metadata.

    Three constraints, each load-bearing:

    * bw="20" -- an ESP32 is a 20 MHz radio.  The 80 MHz captures use a different
      subcarrier geometry (256 bins) and mixing them in would mean comparing
      across two different physical channels.  This alone removes ~75% of the
      archive, and it CANNOT be done from filenames -- bandwidth only appears in
      the summary, which is why a naive filename-based pick wastes the download.
    * rx="1" -- each event is captured simultaneously by 3-4 receivers.  Taking
      all of them would inflate the sample with near-duplicate views of the same
      60 seconds, the same session-leakage trap as the Intel baseline.
    * apps -- keep the clean presence classes plus the machine-only classes.
    """
    if apps is None:
        apps = tuple(PRESENCE_GROUPS)
    by_name = {r["name"]: r for r in inv}

    picked = []
    for name, meta in summary.items():
        if meta.get("BW") != bw or meta.get("Rx") != rx:
            continue
        if meta.get("Application") not in apps:
            continue
        rec = by_name.get(name)
        if rec is None:
            continue
        rec = dict(rec)
        rec["meta"] = meta
        rec["group"] = PRESENCE_GROUPS[meta["Application"]]
        picked.append(rec)

    # If the pick overruns the budget, drop the largest files first -- they are
    # the highest-packet-rate captures, and we already have plenty of packets.
    picked.sort(key=lambda r: r["csz"])
    total, keep = 0, []
    for r in picked:
        if total + r["csz"] > max_total_gb * 1e9:
            break
        keep.append(r)
        total += r["csz"]
    return keep
