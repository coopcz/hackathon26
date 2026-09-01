"""
Replay a saved recording through the live prediction path.

WHY THIS EXISTS: the model can only be demonstrated when a board is streaming,
which makes it impossible to show or regression-test the inference path without
standing in a room with the hardware.  Replay feeds the packets of a recording
that already happened through the exact same `on_packet` callback the serial
reader uses, so what you see is the real model scoring real measurements.

WHAT IT IS NOT: a data generator.  Every value replayed was recorded from the
board and is read verbatim off disk -- nothing is synthesised, interpolated or
smoothed.  Replay never writes to the recorder, so it cannot contaminate a
session, and the UI labels it REPLAY so it is never mistaken for live telemetry.
"""

import json
import logging
import threading
import time
from datetime import datetime

from .models import CSIPacket

log = logging.getLogger(__name__)

MAX_GAP_SECONDS = 0.5   # a long gap in the source is not worth re-living


def _packet_from(record):
    return CSIPacket(
        packet_id=str(record.get("packet_id", "")), mac=record.get("mac", ""),
        rssi=int(record["rssi"]), rate=int(record.get("rate", 0)),
        noise_floor=int(record.get("noise_floor", 0)),
        fft_gain=int(record.get("fft_gain", 0)), agc_gain=int(record.get("agc_gain", 0)),
        channel=int(record.get("channel", 0)), esp_timestamp=int(record["esp_timestamp"]),
        sig_len=int(record.get("sig_len", 0)), rx_state=int(record.get("rx_state", 0)),
        declared_len=int(record["declared_len"]), first_word=int(record.get("first_word", 0)),
        raw_csi=record["raw_csi"], amplitude=record["amplitude"], phase=record["phase"],
        received_at=record.get("received_at", ""),
        motion_score=record.get("motion_score"), features=record.get("features") or {},
    )


class Replayer:
    def __init__(self, directory, on_packet):
        self.directory = directory
        self.on_packet = on_packet
        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.info = None

    def available(self):
        return sorted(
            ({"filename": p.name,
              "label": next((part for part in p.stem.split("_", 1)[1:]), "unknown")}
             for p in self.directory.glob("*.jsonl")),
            key=lambda r: r["filename"])

    def start(self, filename, speed=1.0):
        path = (self.directory / filename).resolve()
        if self.directory.resolve() not in path.parents or not path.exists():
            raise FileNotFoundError("recording not found")
        if speed <= 0 or speed > 20:
            raise ValueError("speed must be between 0 and 20")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("a replay is already running")
            self.stop_event.clear()
            self.info = {"filename": path.name, "speed": speed, "packet_count": 0,
                         "label": path.stem.split("_", 1)[-1], "finished": False}
            self.thread = threading.Thread(target=self._run, args=(path, speed),
                                           name="csi-replay", daemon=True)
            self.thread.start()
            return dict(self.info)

    def stop(self):
        self.stop_event.set()
        t = self.thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        with self.lock:
            if self.info:
                self.info["finished"] = True
            return dict(self.info) if self.info else {"active": False}

    def _run(self, path, speed):
        previous = None
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if self.stop_event.is_set():
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        packet = _packet_from(record)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue          # same rejection policy as the live parser

                    # re-live the original inter-packet timing so the verdict
                    # updates at the rate it would on a live board
                    stamp = record.get("received_at")
                    if previous and stamp:
                        try:
                            gap = (datetime.fromisoformat(stamp)
                                   - datetime.fromisoformat(previous)).total_seconds()
                            if 0 < gap < MAX_GAP_SECONDS:
                                time.sleep(gap / speed)
                        except ValueError:
                            pass
                    previous = stamp or previous

                    with self.lock:
                        if self.info:
                            self.info["packet_count"] += 1
                    self.on_packet(packet)
        except OSError as exc:
            log.warning("replay failed: %s", exc)
        finally:
            with self.lock:
                if self.info:
                    self.info["finished"] = True

    def status(self):
        with self.lock:
            active = bool(self.thread and self.thread.is_alive()
                          and not self.stop_event.is_set())
            if not self.info:
                return {"active": False}
            return {"active": active, **self.info}
