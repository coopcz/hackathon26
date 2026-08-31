"""
Parser for Intel IWL5300 "Linux 802.11n CSI Tool" .dat log files.

The Di Domenico WiFi-CrowdCounting dataset ships raw binary logs produced by
dhalperi's csitool.  Upstream only provides a MATLAB reader (read_bf_file.m),
so this is a direct NumPy port of read_bf_file.m + read_bfee.c + get_scaled_csi.m.

File layout: a flat concatenation of records, each of which is
    uint16 field_len   (big-endian)   -- bytes that follow, including the code byte
    uint8  code                       -- 187 (0xBB) means "beamforming / CSI report"
    uint8  payload[field_len - 1]

For code==187 the payload is a bfee_notif struct:
    off  0..3   timestamp_low   uint32 LE  (microsecond clock on the NIC)
    off  4..5   bfee_count      uint16 LE
    off  6..7   reserved
    off  8      Nrx             number of receive antennas
    off  9      Ntx             number of transmit antennas / spatial streams
    off 10..12  rssi_a/b/c      per-antenna RSSI in dB above an AGC-dependent floor
    off 13      noise           int8, noise floor in dBm (-127 == "not measured")
    off 14      agc             AGC gain in dB
    off 15      antenna_sel     2 bits per RX chain: the permutation applied by the NIC
    off 16..17  len             uint16 LE, length of the packed CSI blob
    off 18..19  fake_rate_n_flags
    off 20..     packed CSI: 30 subcarriers x (Nrx*Ntx) streams x (int8 real, int8 imag),
                 bit-packed with a 3-bit skip before each subcarrier block.
"""

import numpy as np

BFEE_CODE = 187
N_SUBCARRIERS = 30


def _as_int8(v):
    """Reinterpret the low 8 bits of an int as a signed char."""
    v &= 0xFF
    return v - 256 if v > 127 else v


def _db(x):
    return 10.0 * np.log10(x)


def _dbinv(x):
    return 10.0 ** (x / 10.0)


def _total_rss(rssi_a, rssi_b, rssi_c, agc):
    """Total received signal strength in dBm, summed over the active RX chains.

    The per-antenna rssi_* fields are relative to an AGC-dependent reference; the
    -44 and -agc terms convert them back to absolute dBm (from get_total_rss.m).
    """
    mag = 0.0
    if rssi_a != 0:
        mag += _dbinv(rssi_a)
    if rssi_b != 0:
        mag += _dbinv(rssi_b)
    if rssi_c != 0:
        mag += _dbinv(rssi_c)
    if mag == 0.0:
        return np.nan
    return _db(mag) - 44.0 - agc


def _unpack_csi(payload, nrx, ntx):
    """Bit-unpack the 30 x Nrx x Ntx complex CSI matrix.

    Each subcarrier block is preceded by a 3-bit gap, so after the first
    subcarrier the int8 samples are no longer byte-aligned and have to be
    reassembled from two adjacent bytes.  This mirrors read_bfee.c exactly.
    """
    csi = np.zeros((N_SUBCARRIERS, nrx, ntx), dtype=np.complex64)
    n_streams = nrx * ntx
    idx = 0
    for sub in range(N_SUBCARRIERS):
        idx += 3
        remainder = idx % 8
        for s in range(n_streams):
            byte = idx >> 3
            real = (payload[byte] >> remainder) | (payload[byte + 1] << (8 - remainder))
            imag = (payload[byte + 1] >> remainder) | (payload[byte + 2] << (8 - remainder))
            csi[sub, s % nrx, s // nrx] = complex(_as_int8(real), _as_int8(imag))
            idx += 16
    return csi


def get_scaled_csi(entry):
    """Convert raw CSI to an SNR-normalised scale (port of get_scaled_csi.m).

    Raw CSI comes out of the NIC in arbitrary units that shift with the AGC
    setting, so two packets recorded seconds apart are not directly comparable.
    Rescaling against measured RSSI and the noise floor puts every packet on a
    common physical footing, which matters here because our features are
    amplitude statistics compared across a whole recording session.
    """
    csi = entry["csi"]
    nrx, ntx = entry["Nrx"], entry["Ntx"]

    csi_pwr = np.sum(np.abs(csi) ** 2)
    rssi_pwr = _dbinv(_total_rss(entry["rssi_a"], entry["rssi_b"], entry["rssi_c"], entry["agc"]))
    if not np.isfinite(rssi_pwr) or csi_pwr == 0:
        return csi
    scale = rssi_pwr / (csi_pwr / N_SUBCARRIERS)

    noise_db = -92.0 if entry["noise"] == -127 else float(entry["noise"])
    thermal_noise_pwr = _dbinv(noise_db)
    # each additional spatial stream adds one more quantisation-noise contribution
    quant_error_pwr = scale * (nrx * ntx)
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    ret = csi * np.sqrt(scale / total_noise_pwr)
    if ntx == 2:
        ret = ret * np.sqrt(2.0)
    elif ntx == 3:
        # 4.5 dB is the power split the NIC applies across 3 streams
        ret = ret * np.sqrt(_dbinv(4.5))
    return ret


def read_bf_file(path, max_packets=None):
    """Yield parsed CSI records from an Intel 5300 .dat log."""
    with open(path, "rb") as fh:
        buf = fh.read()

    entries = []
    pos = 0
    n = len(buf)
    while pos + 3 <= n:
        field_len = (buf[pos] << 8) | buf[pos + 1]
        code = buf[pos + 2]
        pos += 3
        payload_len = field_len - 1
        if payload_len < 0 or pos + payload_len > n:
            break
        if code != BFEE_CODE:
            pos += payload_len
            continue

        p = buf[pos:pos + payload_len]
        pos += payload_len

        nrx = p[8]
        ntx = p[9]
        clen = p[16] | (p[17] << 8)
        expected = (30 * (nrx * ntx * 8 * 2 + 3) + 7) // 8
        if clen != expected or len(p) < 20 + clen:
            continue  # corrupt record, skip

        entry = {
            "timestamp_low": int.from_bytes(p[0:4], "little"),
            "Nrx": nrx,
            "Ntx": ntx,
            "rssi_a": p[10],
            "rssi_b": p[11],
            "rssi_c": p[12],
            "noise": _as_int8(p[13]),
            "agc": p[14],
            "perm": [(p[15] & 0x3), ((p[15] >> 2) & 0x3), ((p[15] >> 4) & 0x3)],
            "csi": _unpack_csi(np.frombuffer(p[20:20 + clen + 2] + b"\x00\x00", dtype=np.uint8).astype(np.int32), nrx, ntx),
        }
        # undo the antenna permutation the NIC applied, so RX chain order is stable
        perm = entry["perm"]
        if nrx == 3 and sorted(perm) == [0, 1, 2]:
            reordered = np.zeros_like(entry["csi"])
            reordered[:, perm, :] = entry["csi"][:, [0, 1, 2], :]
            entry["csi"] = reordered

        entries.append(entry)
        if max_packets is not None and len(entries) >= max_packets:
            break
    return entries
