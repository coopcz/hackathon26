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
        self.record_until = None
        self.last_result = None

    def start(self, label: str, notes: str, delay_seconds: int = 10,
              duration_seconds: int = 30, room_id: str | None = None,
              split: str = "training") -> dict:
        if label not in LABELS:
            raise ValueError("invalid recording label")
        if delay_seconds < 0 or delay_seconds > 60:
            raise ValueError("recording delay must be between 0 and 60 seconds")
        if duration_seconds < 1 or duration_seconds > 3600:
            raise ValueError("recording duration must be between 1 and 3600 seconds")
        with self.lock:
            if self.info:
                raise RuntimeError("a recording is already active")
            now = datetime.now(timezone.utc)
            starts_at = now + timedelta(seconds=delay_seconds)
            ends_at = starts_at + timedelta(seconds=duration_seconds)
            safe_label = re.sub(r"[^a-z0-9_-]", "_", label)
            path = self.directory / f"{starts_at.strftime('%Y%m%dT%H%M%S.%fZ')}_{safe_label}.jsonl"
            self.info = {"filename": path.name, "label": label, "notes": notes,
                         "room_id": room_id, "split": split,
                         "requested_at": now.isoformat(), "started_at": starts_at.isoformat(),
                         "ends_at": ends_at.isoformat(), "packet_count": 0,
                         "delay_seconds": delay_seconds, "duration_seconds": duration_seconds}
            self.record_after = time.monotonic() + delay_seconds
            self.record_until = self.record_after + duration_seconds
            self.last_result = None
            return self._status_unlocked()

    def append(self, packet: CSIPacket) -> None:
        with self.lock:
            now = time.monotonic()
            if not self.info or now < self.record_after:
                return
            if now >= self.record_until:
                self._finish_unlocked("automatic")
                return
            if not self.file:
                path = self.directory / self.info["filename"]
                self.file = path.open("x", encoding="utf-8")
            row = packet.live_dict() | {
                "session_label": self.info["label"], "session_notes": self.info["notes"],
                "session_room_id": self.info.get("room_id"),
                "session_split": self.info.get("split"),
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
            reason = "cancelled" if time.monotonic() < self.record_after else "manual"
            return self._finish_unlocked(reason)

    def status(self) -> dict:
        with self.lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict:
        if self.info and time.monotonic() >= self.record_until:
            self._finish_unlocked("automatic")
        if not self.info:
            return {"active": False, "state": "idle", "last_result": self.last_result}
        remaining = max(0.0, self.record_after - time.monotonic())
        recording_remaining = max(0.0, self.record_until - time.monotonic())
        return {
            "active": True,
            "state": "countdown" if remaining > 0 else "recording",
            "countdown_remaining": remaining,
            "recording_remaining": recording_remaining,
            **self.info,
        }

    def _finish_unlocked(self, reason: str) -> dict:
        if self.file:
            self.file.close()
        self.file = None
        result, self.info = self.info, None
        result["stop_reason"] = reason
        result["cancelled_during_countdown"] = reason == "cancelled"
        result["stopped_at"] = datetime.now(timezone.utc).isoformat()
        self.record_after = None
        self.record_until = None
        self.last_result = dict(result)
        return result

    def list(self) -> list[dict]:
        result = []
        for path in sorted(self.directory.glob("*.jsonl"), reverse=True):
            try:
                with path.open(encoding="utf-8") as handle:
                    first = json.loads(handle.readline())
                result.append({"filename": path.name, "size_bytes": path.stat().st_size,
                               "label": first.get("session_label"), "notes": first.get("session_notes"),
                               "room_id": first.get("session_room_id"),
                               "split": first.get("session_split", "training"),
                               "started_at": first.get("session_timestamp")})
            except (OSError, json.JSONDecodeError):
                result.append({"filename": path.name, "error": "unreadable recording"})
        return result

    def load(self, filename: str, limit: int = 5000) -> dict:
        path = self.recording_path(filename)
        packets = []
        with path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    break
                packets.append(json.loads(line))
        return {"filename": filename, "packets": packets, "truncated": len(packets) == limit}

    def recording_path(self, filename: str) -> Path:
        path = (self.directory / filename).resolve()
        if path.parent != self.directory.resolve() or path.suffix != ".jsonl" or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def csv_rows(self, filename: str):
        """Stream one lossless CSV row per recorded packet.

        Array/object values remain valid JSON inside their CSV cells. csv.writer
        handles the quoting, so commas in CSI arrays cannot shift columns.
        """
        path = self.recording_path(filename)
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
