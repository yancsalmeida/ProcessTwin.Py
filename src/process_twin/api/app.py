import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from process_twin.core.models import (
    PlantTelemetry,
    ActuatorCommand,
    AlarmEvent,
)
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.db.storage import TelemetryDatabase

# Global engine instances
DB_PATH = os.getenv("PROCESS_TWIN_DB", "processtwin.db")
db = TelemetryDatabase(DB_PATH)
twin = PlantDigitalTwin()
active_websockets: Set[WebSocket] = set()
background_task: asyncio.Task = None


async def telemetry_background_loop() -> None:
    """Continuous 5 Hz simulation loop broadcasting to WebSockets and persisting to SQL."""
    step_count = 0
    while True:
        try:
            frame = twin.step(dt_seconds=0.2)
            step_count += 1

            # Broadcast to all connected WebSocket clients
            if active_websockets:
                payload = frame.model_dump_json()
                disconnected: Set[WebSocket] = set()
                for ws in active_websockets:
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        disconnected.add(ws)
                active_websockets.difference_update(disconnected)

            # Persist to database at 1 Hz (every 5 steps) or immediately upon alarm/interlock
            if step_count % 5 == 0 or frame.interlock_tripped:
                await db.insert_telemetry(frame)
                for alarm in twin.active_alarms:
                    await db.insert_alarm(alarm)

            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ERROR in simulation loop]: {e}")
            await asyncio.sleep(0.5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task
    await db.init_db()
    background_task = asyncio.create_task(telemetry_background_loop())
    yield
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="ProcessTwin.Py — Industrial Process Digital Twin API",
    description="Real-time telemetry streaming, safety interlocks, and digital twin simulation using FastAPI & Asyncio.",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", response_class=HTMLResponse)
async def get_dashboard() -> str:
    """Serves the live interactive SCADA operator dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>ProcessTwin.Py Backend Running</h1><p>Visit <a href='/docs'>/docs</a> for Swagger API.</p>"


@app.get("/api/telemetry/current", response_model=PlantTelemetry)
async def get_current_telemetry() -> PlantTelemetry:
    """Returns the latest measured plant sensor telemetry frame."""
    return twin.step(dt_seconds=0.0)


@app.get("/api/telemetry/history")
async def get_telemetry_history(limit: int = 60):
    """Retrieves chronological historical telemetry frames from SQLite."""
    return await db.get_recent_telemetry(limit=min(limit, 300))


@app.post("/api/actuators")
async def override_actuators(command: ActuatorCommand) -> dict:
    """Applies operator control overrides (cooling valve, emergency feed shutoff)."""
    twin.apply_actuator_command(command)
    return {"status": "success", "command_applied": command.model_dump(exclude_none=True)}


@app.post("/api/simulation/reset")
async def reset_simulation() -> dict:
    """Resets the digital twin to initial steady-state conditions (85°C, 5.0m level, 45% cooling)."""
    twin.reset()
    return {"status": "success", "message": "Simulation reset to initial steady-state conditions."}


@app.get("/api/alarms")
async def get_alarms() -> list:
    """Returns list of currently active alarms and safety trips."""
    return [a.model_dump() for a in twin.active_alarms]


@app.post("/api/alarms/{alarm_id}/acknowledge")
async def acknowledge_alarm(alarm_id: str) -> dict:
    """Acknowledges an annunciated alarm event."""
    found = twin.acknowledge_alarm(alarm_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Alarm '{alarm_id}' not found.")
    return {"status": "acknowledged", "alarm_id": alarm_id}


@app.websocket("/ws/telemetry")
async def websocket_telemetry_feed(websocket: WebSocket) -> None:
    """Real-time 5 Hz WebSocket feed streaming JSON telemetry frames to SCADA clients."""
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            # Keep connection alive & listen for potential client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.discard(websocket)
