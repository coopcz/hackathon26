import logging
import threading
import time
from collections.abc import Callable

import serial
from serial.tools import list_ports

from .csi_parser import CSIParseError, parse_csi_line
from .features import add_features
from .models import CSIPacket

log = logging.getLogger(__name__)


def available_ports() -> list[dict]:
    return [{"device": p.device, "description": p.description}
            for p in list_ports.comports() if p.device.startswith("/dev/cu.")]


class SerialReader:
    def __init__(self, on_packet: Callable[[CSIPacket], None]):
        self.on_packet = on_packet
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.serial = None
        self.port = None
        self.packet_count = 0
        self.rejected_count = 0
        self.error = None
        self.started_at = None
        self.last_packet_at = None

    def connect(self, port: str) -> None:
        if not port.startswith("/dev/cu."):
            raise ValueError("port must be a macOS /dev/cu.* device")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("serial reader is already connected")
            self.serial = serial.Serial(port, 921600, timeout=0.25)
            self.port, self.packet_count, self.rejected_count, self.error = port, 0, 0, None
            self.started_at = time.monotonic()
            self.last_packet_at = None
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, name="csi-serial-reader", daemon=True)
            self.thread.start()

    def _run(self) -> None:
        previous = None
        try:
            while not self.stop_event.is_set():
                raw = self.serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("CSI_DATA,"):
                    continue
                try:
                    packet = parse_csi_line(line)
                    add_features(packet, previous)
                    previous = packet.amplitude
                    self.packet_count += 1
                    self.last_packet_at = time.monotonic()
                    self.on_packet(packet)
                except CSIParseError as exc:
                    self.rejected_count += 1
                    log.warning("Rejected CSI row: %s", exc)
        except (serial.SerialException, OSError) as exc:
            self.error = str(exc)
            log.exception("Serial reader stopped")
        finally:
            if self.serial and self.serial.is_open:
                self.serial.close()

    def disconnect(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self.lock:
            if self.serial and self.serial.is_open:
                self.serial.close()
            self.thread, self.serial, self.port = None, None, None

    def status(self) -> dict:
        connected = bool(self.thread and self.thread.is_alive() and not self.error)
        elapsed = max(time.monotonic() - self.started_at, 0.001) if self.started_at else 0
        age = time.monotonic() - self.last_packet_at if self.last_packet_at else None
        return {"connected": connected, "port": self.port, "packet_count": self.packet_count,
                "rejected_count": self.rejected_count,
                "packets_per_second": self.packet_count / elapsed if connected else 0,
                "seconds_since_last_packet": age, "error": self.error}
