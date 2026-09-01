import csv
import json
from datetime import datetime, timezone

import numpy as np

from .models import CSIPacket


class CSIParseError(ValueError):
    pass


def parse_csi_line(line: str) -> CSIPacket:
    if not line.startswith("CSI_DATA,"):
        raise CSIParseError("not a CSI_DATA row")
    try:
        fields = next(csv.reader([line], skipinitialspace=True))
    except csv.Error as exc:
        raise CSIParseError(f"invalid CSV: {exc}") from exc
    if len(fields) < 15:
        raise CSIParseError(f"expected at least 15 fields, received {len(fields)}")
    # The CSI array is the final field. Joining protects against firmware variants
    # whose array was not quoted correctly while remaining valid JSON.
    data_text = ",".join(fields[14:]).strip()
    try:
        raw = json.loads(data_text)
        if not isinstance(raw, list) or any(type(x) is not int for x in raw):
            raise TypeError("CSI data is not an integer array")
        if not raw or len(raw) % 2:
            raise ValueError("CSI array must contain a non-zero even number of I/Q values")
        declared_len = int(fields[12])
        # Espressif's len is the count of serialized I/Q integers. Reject truncation.
        if declared_len > 0 and declared_len != len(raw):
            raise ValueError(f"declared CSI length {declared_len} != received {len(raw)}")
        imag = np.asarray(raw[0::2], dtype=np.float64)
        real = np.asarray(raw[1::2], dtype=np.float64)
        csi = real + 1j * imag
        return CSIPacket(
            packet_id=fields[1], mac=fields[2], rssi=int(fields[3]),
            rate=int(fields[4]), noise_floor=int(fields[5]), fft_gain=int(fields[6]),
            agc_gain=int(fields[7]), channel=int(fields[8]), esp_timestamp=int(fields[9]),
            sig_len=int(fields[10]), rx_state=int(fields[11]), declared_len=declared_len,
            first_word=int(fields[13]), raw_csi=raw, amplitude=np.abs(csi).tolist(),
            phase=np.angle(csi).tolist(), received_at=datetime.now(timezone.utc).isoformat(),
        )
    except (ValueError, TypeError, json.JSONDecodeError, IndexError) as exc:
        raise CSIParseError(str(exc)) from exc
