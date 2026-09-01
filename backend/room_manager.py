"""Persistent room profiles and room-scoped model workflow jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:48] or "room"


class RoomManager:
    def __init__(self, root: Path, recordings, predictor):
        self.root = root
        self.recordings = recordings
        self.predictor = predictor
        self.directory = root / "rooms"
        self.index_path = self.directory / "profiles.json"
        self.active_model = root / "artifacts" / "esp32_model.joblib"
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profiles = self._load()

    def _load(self) -> dict[str, dict]:
        try:
            rows = json.loads(self.index_path.read_text(encoding="utf-8"))
            return {row["id"]: row for row in rows}
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            return {}

    def _save(self) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(list(self.profiles.values()), indent=2), encoding="utf-8")
        os.replace(temporary, self.index_path)

    def _recordings_for(self, room_id: str, split: str | None = None) -> list[dict]:
        rows = [row for row in self.recordings.list() if row.get("room_id") == room_id]
        return [row for row in rows if split is None or row.get("split") == split]

    def _public(self, room: dict) -> dict:
        counts = {split: {label: 0 for label in ("empty", "occupied_still", "occupied_moving")}
                  for split in ("training", "holdout")}
        for row in self._recordings_for(room["id"]):
            split, label = row.get("split"), row.get("label")
            if split in counts and label in counts[split]:
                counts[split][label] += 1
        latest = room.get("latest_model")
        return {**room, "counts": counts,
                "model_ready": bool(latest and (self.root / latest).is_file()),
                "job": self.jobs.get(room["id"])}

    def list(self) -> list[dict]:
        with self.lock:
            return [self._public(room) for room in self.profiles.values()]

    def get(self, room_id: str) -> dict:
        with self.lock:
            if room_id not in self.profiles:
                raise KeyError(room_id)
            return self._public(self.profiles[room_id])

    def create(self, name: str, placement: str = "") -> dict:
        with self.lock:
            base = _slug(name)
            room_id = base
            while room_id in self.profiles:
                room_id = f"{base}-{uuid.uuid4().hex[:4]}"
            room = {"id": room_id, "name": name.strip(), "placement": placement.strip(),
                    "created_at": _now(), "latest_model": None, "trained_at": None,
                    "validated": False, "validated_at": None, "validation_report": None,
                    "active": False}
            self.profiles[room_id] = room
            (self.directory / room_id / "models").mkdir(parents=True, exist_ok=True)
            self._save()
            return self._public(room)

    def update(self, room_id: str, *, name: str | None = None,
               placement: str | None = None) -> dict:
        with self.lock:
            room = self.profiles[room_id]
            if name is not None:
                room["name"] = name.strip()
            if placement is not None:
                room["placement"] = placement.strip()
            self._save()
            return self._public(room)

    def start_job(self, room_id: str, kind: str) -> dict:
        with self.lock:
            room = self.profiles[room_id]
            current = self.jobs.get(room_id)
            if current and current["status"] == "running":
                raise RuntimeError("a room workflow job is already running")
            split = "training" if kind == "train" else "holdout"
            counts = self._public(room)["counts"][split]
            minimum = 3
            missing = [label for label, count in counts.items() if count < minimum]
            if missing:
                names = ", ".join(label.replace("_", " ") for label in missing)
                raise RuntimeError(f"record at least {minimum} {split} sessions for: {names}")
            job = {"kind": kind, "status": "running", "started_at": _now(), "output": ""}
            self.jobs[room_id] = job
            thread = threading.Thread(target=self._run_job, args=(room_id, kind), daemon=True)
            thread.start()
            return dict(job)

    def _materialize(self, room_id: str, split: str) -> Path:
        target = self.directory / room_id / "datasets" / split
        target.mkdir(parents=True, exist_ok=True)
        for old in target.glob("*.jsonl"):
            old.unlink()
        for row in self._recordings_for(room_id, split):
            source = self.recordings.recording_path(row["filename"])
            destination = target / source.name
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        return target

    def _run_job(self, room_id: str, kind: str) -> None:
        try:
            if kind == "train":
                data = self._materialize(room_id, "training")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                model = self.directory / room_id / "models" / f"model-{stamp}.joblib"
                command = [sys.executable, "-m", "src.train_esp32", "--data-dir", str(data),
                           "--force", "--output", str(model)]
            elif kind == "validate":
                data = self._materialize(room_id, "holdout")
                with self.lock:
                    latest = self.profiles[room_id].get("latest_model")
                if not latest:
                    raise RuntimeError("train a room model before validation")
                model = self.root / latest
                command = [sys.executable, "-m", "src.evaluate_live", str(data),
                           "--model", str(model)]
            else:
                raise RuntimeError(f"unknown job: {kind}")

            result = subprocess.run(command, cwd=self.root, text=True, capture_output=True,
                                    timeout=1800, check=False)
            output = (result.stdout + "\n" + result.stderr).strip()[-16000:]
            if result.returncode:
                raise RuntimeError(output or f"{kind} exited {result.returncode}")
            if kind == "train" and not model.is_file():
                raise RuntimeError(output + "\nModel was not saved because it did not meet deployment gates.")

            with self.lock:
                room = self.profiles[room_id]
                if kind == "train":
                    room["latest_model"] = str(model.relative_to(self.root))
                    room["trained_at"] = _now()
                    room["validated"] = False
                    room["validated_at"] = None
                    room["validation_report"] = None
                else:
                    room["validated"] = True
                    room["validated_at"] = _now()
                    room["validation_report"] = output
                self.jobs[room_id] = {**self.jobs[room_id], "status": "complete",
                                      "finished_at": _now(), "output": output}
                self._save()
        except Exception as exc:
            with self.lock:
                self.jobs[room_id] = {**self.jobs.get(room_id, {"kind": kind}),
                                      "status": "failed", "finished_at": _now(),
                                      "output": str(exc)[-16000:]}

    def activate(self, room_id: str) -> dict:
        with self.lock:
            room = self.profiles[room_id]
            if not room.get("validated"):
                raise RuntimeError("validate this model before activating it")
            return self._serve_model(room_id, preview=False)

    def preview(self, room_id: str) -> dict:
        """Serve a trained model for screen-only experimentation.

        Preview deliberately does not change validation state.  It exists so a
        team can observe live predictions and collect more labelled recordings
        before the model is good enough to deploy.
        """
        with self.lock:
            return self._serve_model(room_id, preview=True)

    def _serve_model(self, room_id: str, *, preview: bool) -> dict:
        room = self.profiles[room_id]
        if not preview and not room.get("validated"):
            raise RuntimeError("validate this model before activating it")
        if not room.get("latest_model"):
            raise RuntimeError("train a room model before starting live preview")
        source = self.root / room["latest_model"]
        if not source.is_file():
            raise RuntimeError("the room model file is missing")
        self.active_model.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.active_model.with_suffix(".joblib.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, self.active_model)
        if not self.predictor.load():
            raise RuntimeError(self.predictor.error or "room model could not be loaded")
        for profile in self.profiles.values():
            profile["active"] = profile["id"] == room_id
            profile["preview"] = bool(preview and profile["id"] == room_id)
        if preview:
            room["preview_started_at"] = _now()
        else:
            room["activated_at"] = _now()
        self._save()
        return self._public(room)
