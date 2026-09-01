from dataclasses import dataclass, field
from typing import Any


@dataclass
class CSIPacket:
    packet_id: str
    mac: str
    rssi: int
    rate: int
    noise_floor: int
    fft_gain: int
    agc_gain: int
    channel: int
    esp_timestamp: int
    sig_len: int
    rx_state: int
    declared_len: int
    first_word: int
    raw_csi: list[int]
    amplitude: list[float]
    phase: list[float]
    received_at: str
    motion_score: float | None = None
    features: dict[str, float] = field(default_factory=dict)

    def live_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id, "mac": self.mac, "rssi": self.rssi,
            "noise_floor": self.noise_floor, "fft_gain": self.fft_gain,
            "agc_gain": self.agc_gain, "channel": self.channel,
            "esp_timestamp": self.esp_timestamp, "csi_length": len(self.amplitude),
            "raw_csi_length": len(self.raw_csi), "amplitude": self.amplitude,
            "phase": self.phase, "motion_score": self.motion_score,
            "features": self.features, "received_at": self.received_at,
        }
