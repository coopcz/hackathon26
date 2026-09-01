import numpy as np

from .models import CSIPacket


def add_features(packet: CSIPacket, previous_amplitude: list[float] | None) -> None:
    amplitude = np.asarray(packet.amplitude, dtype=float)
    motion = None
    if previous_amplitude is not None and len(previous_amplitude) == len(amplitude):
        motion = float(np.mean(np.abs(amplitude - np.asarray(previous_amplitude))))
    packet.motion_score = motion
    packet.features = {
        "mean_amplitude": float(np.mean(amplitude)),
        "std_amplitude": float(np.std(amplitude)),
        "median_amplitude": float(np.median(amplitude)),
        "variance_amplitude": float(np.var(amplitude)),
    }
    if motion is not None:
        packet.features["frame_difference"] = motion
