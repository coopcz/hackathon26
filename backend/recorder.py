import csv
import io
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import CSIPacket

LABELS = {"empty", "occupied_still", "occupied_moving"}


class Recorder:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.file = None
        self.info = None
        self.record_after = None

    def start(self, label: str, notes: str, delay_seconds: int = 10) -> dict:
        if label not in LABELS:
            raise ValueError("invalid recording label")
        if delay_seconds < 0 or delay_seconds > 60:
            raise ValueError("recording delay must be between 0 and 60 seconds")
        with self.lock:
            if self.info:
                raise RuntimeError("a recording is already active")
            now = datetime.now(timezone.utc)
            starts_at = now + timedelta(seconds=delay_seconds)
            safe_label = re.sub(r"[^a-z0-9_-]", "_", label)
            path = self.directory / f"{starts_at.strftime('%Y%m%dT%H%M%S.%fZ')}_{safe_label}.jsonl"
            self.info = {"filename": path.name, "label": label, "notes": notes,
                         "requested_at": now.isoformat(), "started_at": starts_at.isoformat(),
                         "packet_count": 0, "delay_seconds": delay_seconds}
            self.record_after = time.monotonic() + delay_seconds
            return self._status_unlocked()

    def append(self, packet: CSIPacket) -> None:
        with self.lock:
            if not self.info or time.monotonic() < self.record_after:
                return
            if not self.file:
                path = self.directory / self.info["filename"]
                self.file = path.open("x", encoding="utf-8")
            row = packet.live_dict() | {
                "session_label": self.info["label"], "session_notes": self.info["notes"],
                "session_timestamp": self.info["started_at"], "raw_csi": packet.raw_csi,
                "rate": packet.rate, "sig_len": packet.sig_len, "rx_state": packet.rx_state,
                "declared_len": packet.declared_len, "first_word": packet.first_word,
            }
            self.file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.file.flush()
            self.info["packet_count"] += 1

    def stop(self) -> dict | None:
        with self.lock:
            if not self.info:
                return None
            if self.file:
                self.file.close()
            self.file = None
            result, self.info = self.info, None
            result["cancelled_during_countdown"] = time.monotonic() < self.record_after
            self.record_after = None
            return result

    def status(self) -> dict:
        with self.lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict:
        if not self.info:
            return {"active": False, "state": "idle"}
        remaining = max(0.0, self.record_after - time.monotonic())
        return {
            "active": True,
            "state": "countdown" if remaining > 0 else "recording",
            "countdown_remaining": remaining,
            **self.info,
        }

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            try:
                with path.open(encoding="utf-8") as handle:
                    first = json.loads(handle.readline())
                result.append({"filename": path.name, "size_bytes": path.stat().st_size,
                               "label": first.get("session_label"), "notes": first.get("session_notes"),
                               "started_at": first.get("session_timestamp")})
            except (OSError, json.JSONDecodeError):
                result.append({"filename": path.name, "error": "unreadable recording"})
        return result

    def load(self, filename: str, limit: int = 5000) -> dict:
        path = self._recording_path(filename)
        packets = []
        with path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    break
                packets.append(json.loads(line))
        return {"filename": filename, "packets": packets, "truncated": len(packets) == limit}

    def _recording_path(self, filename: str) -> Path:
        path = (self.directory / filename).resolve()
        if path.parent != self.directory.resolve() or path.suffix != ".jsonl" or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def csv_rows(self, filename: str):
        """Stream one lossless CSV row per recorded packet.

        Array/object values remain valid JSON inside their CSV cells. csv.writer
        handles the quoting, so commas in CSI arrays cannot shift columns.
        """
        path = self._recording_path(filename)
        fields = [
            "session_label", "session_notes", "session_timestamp", "received_at",
            "packet_id", "esp_timestamp", "mac", "rssi", "rate", "noise_floor",
            "fft_gain", "agc_gain", "channel", "sig_len", "rx_state",
            "declared_len", "first_word", "csi_length", "raw_csi_length",
            "motion_score", "mean_amplitude", "std_amplitude", "median_amplitude",
            "variance_amplitude", "frame_difference", "raw_csi", "amplitude",
            "phase", "features",
        ]
        def generate():
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            yield buffer.getvalue()
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    features = row.get("features") or {}
                    output = row | {
                        "mean_amplitude": features.get("mean_amplitude"),
                        "std_amplitude": features.get("std_amplitude"),
                        "median_amplitude": features.get("median_amplitude"),
                        "variance_amplitude": features.get("variance_amplitude"),
                        "frame_difference": features.get("frame_difference"),
                    }
                    for key in ("raw_csi", "amplitude", "phase", "features"):
                        output[key] = json.dumps(row.get(key), separators=(",", ":"))
                    buffer.seek(0)
                    buffer.truncate(0)
                    writer.writerow(output)
                    yield buffer.getvalue()
        return generate()
