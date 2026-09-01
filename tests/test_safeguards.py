import tempfile
import time
import unittest
from pathlib import Path

from backend.serial_reader import SerialReader, is_csi_serial_port
from backend.recorder import Recorder
from backend.room_manager import RoomManager
from src.dataset import source_manifest
from src.evaluate_live import longest_run


class PortFilterTests(unittest.TestCase):
    def test_accepts_usb_serial_ports(self):
        self.assertTrue(is_csi_serial_port("/dev/cu.usbmodem2101"))
        self.assertTrue(is_csi_serial_port("/dev/cu.usbserial-0001"))

    def test_rejects_macos_pseudo_ports(self):
        self.assertFalse(is_csi_serial_port("/dev/cu.Bluetooth-Incoming-Port"))
        self.assertFalse(is_csi_serial_port("/dev/cu.debug-console"))


class LiveQualityTests(unittest.TestCase):
    def test_warns_on_weak_signal(self):
        reader = SerialReader(lambda packet: None)
        start = time.monotonic()
        for i in range(60):
            reader.recent.append((start + i * 0.02, i * 20_000, -90, 70, 2, 256))
        quality = reader._recent_quality()
        self.assertTrue(quality["ready"])
        self.assertFalse(quality["healthy"])
        self.assertIn("weak signal", " ".join(quality["problems"]))


class CacheIdentityTests(unittest.TestCase):
    def test_manifest_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text("one")
            first = source_manifest([path])
            path.write_text("two-two")
            second = source_manifest([path])
            self.assertNotEqual(first, second)


class HoldoutHelpersTests(unittest.TestCase):
    def test_longest_run(self):
        self.assertEqual(longest_run(["HOME", "AWAY", "AWAY", "HOME"], "AWAY"), 2)


class RoomWorkflowTests(unittest.TestCase):
    class PredictorStub:
        error = None
        loads = 0

        def load(self):
            self.loads += 1
            return True

    def test_room_profile_persists_and_counts_scoped_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = Recorder(root / "recordings")
            manager = RoomManager(root, recorder, self.PredictorStub())
            room = manager.create("Presentation Room", "8 feet apart")
            self.assertEqual(room["id"], "presentation-room")
            self.assertEqual(room["placement"], "8 feet apart")
            reloaded = RoomManager(root, recorder, self.PredictorStub())
            self.assertEqual(reloaded.get(room["id"])["counts"]["training"]["empty"], 0)

    def test_recording_status_keeps_room_and_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory))
            status = recorder.start("empty", "quiet", delay_seconds=10,
                                    room_id="bedroom", split="holdout")
            self.assertEqual(status["room_id"], "bedroom")
            self.assertEqual(status["split"], "holdout")
            recorder.stop()

    def test_unvalidated_model_can_preview_without_becoming_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = Recorder(root / "recordings")
            predictor = self.PredictorStub()
            manager = RoomManager(root, recorder, predictor)
            room = manager.create("Lab")
            model = root / "rooms" / room["id"] / "models" / "model.joblib"
            model.write_bytes(b"model")
            manager.profiles[room["id"]]["latest_model"] = str(model.relative_to(root))
            manager._save()

            preview = manager.preview(room["id"])

            self.assertTrue(preview["active"])
            self.assertTrue(preview["preview"])
            self.assertFalse(preview["validated"])
            self.assertEqual((root / "artifacts" / "esp32_model.joblib").read_bytes(),
                             b"model")


if __name__ == "__main__":
    unittest.main()
