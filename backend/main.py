import asyncio
import logging
import math
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import CSIPacket
from .predictor import Predictor
from .recorder import Recorder
from .replay import Replayer
from .room_manager import RoomManager
from .serial_reader import SerialReader, available_ports
from src.dataset import _quality
from src.esp_csi import load_esp32_csi_csv

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="RoomSense")
recorder = Recorder(ROOT / "recordings")
predictor = Predictor(ROOT / "artifacts" / "esp32_model.joblib")
rooms = RoomManager(ROOT, recorder, predictor)
clients: set[WebSocket] = set()
loop: asyncio.AbstractEventLoop | None = None


async def broadcast(kind: str, data: dict) -> None:
    dead = []
    for ws in tuple(clients):
        try:
            await ws.send_json({"type": kind, "data": data})
        except Exception:
            dead.append(ws)
    clients.difference_update(dead)


def _emit(kind: str, data: dict) -> None:
    if loop:
        asyncio.run_coroutine_threadsafe(broadcast(kind, data), loop)


def _score_and_emit(packet: CSIPacket, source: str) -> None:
    """Shared tail for live and replayed packets: chart update, then verdict."""
    _emit("packet", {**packet.live_dict(), "source": source})
    verdict = predictor.append(packet)
    if verdict:
        _emit("prediction", {**verdict, "source": source})


def on_packet(packet: CSIPacket) -> None:
    recorder.append(packet)
    _score_and_emit(packet, "live")


def on_replay_packet(packet: CSIPacket) -> None:
    # deliberately does NOT touch the recorder: a replay must never be able to
    # write itself into a labelled session
    _score_and_emit(packet, "replay")


reader = SerialReader(on_packet)
replayer = Replayer(ROOT / "recordings", on_replay_packet)


class ConnectBody(BaseModel):
    port: str


class ReplayBody(BaseModel):
    filename: str
    speed: float = Field(default=1.0, gt=0, le=20)


class RecordBody(BaseModel):
    label: str
    notes: str = Field(default="", max_length=2000)
    delay_seconds: int = Field(default=10, ge=0, le=60)
    duration_seconds: int = Field(default=30, ge=1, le=3600)
    room_id: str | None = None
    split: str = "training"


class RoomBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    placement: str = Field(default="", max_length=500)


class RoomUpdateBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    placement: str | None = Field(default=None, max_length=500)


@app.on_event("startup")
async def startup() -> None:
    global loop
    loop = asyncio.get_running_loop()


@app.on_event("shutdown")
def shutdown() -> None:
    reader.disconnect()
    replayer.stop()
    recorder.stop()


@app.get("/api/ports")
def ports(): return available_ports()


@app.get("/api/status")
def status():
    return {"serial": reader.status(), "recording": recorder.status(),
            "prediction": predictor.status(), "replay": replayer.status(),
            "active_room": next((room for room in rooms.list() if room.get("active")), None)}


@app.get("/api/rooms")
def list_rooms(): return rooms.list()


@app.post("/api/rooms")
def create_room(body: RoomBody):
    return rooms.create(body.name, body.placement)


@app.patch("/api/rooms/{room_id}")
def update_room(room_id: str, body: RoomUpdateBody):
    try: return rooms.update(room_id, name=body.name, placement=body.placement)
    except KeyError as exc: raise HTTPException(404, "room not found") from exc


@app.post("/api/rooms/{room_id}/train")
def train_room(room_id: str):
    try: return rooms.start_job(room_id, "train")
    except KeyError as exc: raise HTTPException(404, "room not found") from exc
    except RuntimeError as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/rooms/{room_id}/validate")
def validate_room(room_id: str):
    try: return rooms.start_job(room_id, "validate")
    except KeyError as exc: raise HTTPException(404, "room not found") from exc
    except RuntimeError as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/rooms/{room_id}/activate")
def activate_room(room_id: str):
    try: return rooms.activate(room_id)
    except KeyError as exc: raise HTTPException(404, "room not found") from exc
    except RuntimeError as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/rooms/{room_id}/preview")
def preview_room(room_id: str):
    try: return rooms.preview(room_id)
    except KeyError as exc: raise HTTPException(404, "room not found") from exc
    except RuntimeError as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/model/reload")
def reload_model():
    predictor.load()
    return predictor.status()


@app.get("/api/replay/recordings")
def replay_recordings(): return replayer.available()


@app.post("/api/replay/start")
def replay_start(body: ReplayBody):
    if reader.status().get("connected"):
        raise HTTPException(400, "disconnect the board before replaying a recording")
    try:
        predictor.reset()
        return replayer.start(body.filename, body.speed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/replay/stop")
def replay_stop(): return replayer.stop()


@app.post("/api/connect")
def connect(body: ConnectBody):
    if replayer.status().get("active"):
        raise HTTPException(400, "stop the replay before connecting to a board")
    try:
        predictor.reset()
        reader.connect(body.port)
        return reader.status()
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/disconnect")
def disconnect():
    reader.disconnect()
    return reader.status()


@app.post("/api/recordings/start")
def start_recording(body: RecordBody):
    serial = reader.status()
    if not serial.get("connected"):
        raise HTTPException(400, "connect the ESP32 receiver before recording")
    if serial.get("packet_count", 0) < 20:
        raise HTTPException(400, "wait for at least 20 CSI packets before recording")
    age = serial.get("seconds_since_last_packet")
    if age is None or age > 1.5:
        raise HTTPException(400, "CSI stream is stale; reconnect the receiver")
    if body.split not in {"training", "holdout"}:
        raise HTTPException(400, "recording split must be training or holdout")
    if body.room_id:
        try: rooms.get(body.room_id)
        except KeyError as exc: raise HTTPException(404, "room not found") from exc
    try: return recorder.start(body.label, body.notes, body.delay_seconds,
                               body.duration_seconds, body.room_id, body.split)
    except (ValueError, RuntimeError, OSError) as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/recordings/stop")
def stop_recording(): return recorder.stop() or {"active": False}


@app.get("/api/recordings")
def recordings(): return recorder.list()


@app.get("/api/recordings/{filename}")
def recording(filename: str):
    try: return recorder.load(filename)
    except FileNotFoundError as exc: raise HTTPException(404, "recording not found") from exc


@app.get("/api/recordings/{filename}/quality")
def recording_quality(filename: str):
    try:
        path = recorder.recording_path(filename)
        out = load_esp32_csi_csv(path, valid_subcarriers="reference", verbose=False)
        q = _quality(out, out["amp"])
        q["filename"] = filename
        return {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in q.items()}
    except FileNotFoundError as exc:
        raise HTTPException(404, "recording not found") from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(400, f"quality check failed: {exc}") from exc


@app.get("/api/recordings/{filename}/csv")
def recording_csv(filename: str):
    try:
        rows = recorder.csv_rows(filename)
        csv_name = f"{Path(filename).stem}.csv"
        return StreamingResponse(
            rows,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{csv_name}"'},
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, "recording not found") from exc


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: clients.discard(ws)


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def index(): return FileResponse(ROOT / "frontend" / "index.html")
