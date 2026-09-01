import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import CSIPacket
from .predictor import Predictor
from .recorder import Recorder
from .replay import Replayer
from .serial_reader import SerialReader, available_ports

logging.basicConfig(level=logging.INFO)
ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="WiFi CSI Lab")
recorder = Recorder(ROOT / "recordings")
predictor = Predictor(ROOT / "artifacts" / "esp32_model.joblib")
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
            "prediction": predictor.status(), "replay": replayer.status()}


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
    try: return recorder.start(body.label, body.notes, body.delay_seconds, body.duration_seconds)
    except (ValueError, RuntimeError, OSError) as exc: raise HTTPException(400, str(exc)) from exc


@app.post("/api/recordings/stop")
def stop_recording(): return recorder.stop() or {"active": False}


@app.get("/api/recordings")
def recordings(): return recorder.list()


@app.get("/api/recordings/{filename}")
def recording(filename: str):
    try: return recorder.load(filename)
    except FileNotFoundError as exc: raise HTTPException(404, "recording not found") from exc


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
