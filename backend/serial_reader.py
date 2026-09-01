import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable

import serial
from serial.tools import list_ports

from .csi_parser import CSIParseError, parse_csi_line
from .features import add_features
from .models import CSIPacket

log = logging.getLogger(__name__)


def is_csi_serial_port(device: str) -> bool:
    """Only offer physical USB serial devices, never macOS pseudo-ports."""
    name = device.lower()
    return (name.startswith("/dev/cu.")
            and any(token in name for token in
                    ("usbmodem", "usbserial", "wchusbserial", "slab_usb", "uart"))
            and "bluetooth" not in name
            and "debug-console" not in name)


def available_ports() -> list[dict]:
    return [{"device": p.device, "description": p.description}
            for p in list_ports.comports() if is_csi_serial_port(p.device)]


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
        self.recent = deque(maxlen=500)

    def connect(self, port: str) -> None:
        if not is_csi_serial_port(port):
            raise ValueError("choose a physical USB serial port (normally /dev/cu.usbmodem*)")
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("serial reader is already connected")
            self.serial = serial.Serial(port, 921600, timeout=0.25)
            self.port, self.packet_count, self.rejected_count, self.error = port, 0, 0, None
            self.started_at = time.monotonic()
            self.last_packet_at = None
            self.recent.clear()
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
                    self.recent.append((self.last_packet_at, packet.esp_timestamp,
                                        packet.rssi, packet.agc_gain, packet.fft_gain,
                                        packet.declared_len))
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
                "seconds_since_last_packet": age, "quality": self._recent_quality(),
                "error": self.error}

    def _recent_quality(self) -> dict:
        """A fast live warning, not a substitute for per-recording quality QA."""
        rows = list(self.recent)
        if len(rows) < 2:
            return {"ready": False, "healthy": False, "problems": ["waiting for CSI packets"]}
        host_t = [r[0] for r in rows]
        intervals = [b - a for a, b in zip(host_t, host_t[1:]) if b > a]
        mean_dt = statistics.fmean(intervals) if intervals else 0.0
        jitter = statistics.pstdev(intervals) / mean_dt if len(intervals) > 1 and mean_dt else None
        span = host_t[-1] - host_t[0]
        rate = (len(rows) - 1) / span if span > 0 else 0.0
        rssi = statistics.fmean(r[2] for r in rows)
        agc_sd = statistics.pstdev(r[3] for r in rows)
        fft_sd = statistics.pstdev(r[4] for r in rows)
        modal_len = statistics.mode(r[5] for r in rows)
        problems = []
        if len(rows) < 50:
            problems.append("warming up")
        if rate < 20:
            problems.append(f"low packet rate ({rate:.1f}/s)")
        if jitter is not None and jitter > 0.75:
            problems.append(f"irregular delivery (jitter {jitter:.2f})")
        if agc_sd > 2:
            problems.append(f"wandering AGC (sd {agc_sd:.1f})")
        if rssi < -80:
            problems.append(f"weak signal ({rssi:.0f} dBm)")
        if modal_len != 256:
            problems.append(f"CSI length {modal_len}, expected HT40 length 256")
        return {
            "ready": len(rows) >= 50,
            "healthy": len(rows) >= 50 and not problems,
            "sample_count": len(rows), "recent_packets_per_second": rate,
            "delivery_jitter": jitter, "rssi_mean": rssi,
            "agc_gain_std": agc_sd, "fft_gain_std": fft_sd,
            "modal_declared_len": modal_len, "problems": problems,
        }
