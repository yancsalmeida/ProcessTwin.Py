import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from process_twin.core.models import (
    PlantTelemetry,
    ActuatorCommand,
    AlarmEvent,
    NmpcToggleCommand,
    NmpcSetpointsCommand,
)
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.telemetry_bridge import IndustrialTelemetryBridge, BridgeStatus
from process_twin.core.chemflow_bridge import ChemFlowThermodynamicsBridge, ChemFlowStatus
from process_twin.db.storage import TelemetryDatabase

# Global engine instances
DB_PATH = os.getenv("PROCESS_TWIN_DB", "processtwin.db")
db = TelemetryDatabase(DB_PATH)
twin = PlantDigitalTwin()
bridge = IndustrialTelemetryBridge()
chemflow_bridge = ChemFlowThermodynamicsBridge()
active_websockets: Set[WebSocket] = set()
background_task: asyncio.Task = None


async def telemetry_background_loop() -> None:
    """Continuous 5 Hz simulation loop broadcasting to WebSockets, persisting to SQLite and forwarding to .NET."""
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

            # Persist to local database at 1 Hz (every 5 steps) or immediately upon alarm/interlock
            if step_count % 5 == 0 or frame.interlock_tripped:
                # Query ChemFlow.NET real-fluid thermodynamics
                if chemflow_bridge.enabled:
                    flash_data = await chemflow_bridge.calculate_flash(
                        temperature_c=twin.reactor_temp_c,
                        pressure_bar=frame.reactor_pressure_bar
                    )
                    twin.update_thermodynamics(flash_data)

                await db.insert_telemetry(frame)
                for alarm in twin.active_alarms:
                    await db.insert_alarm(alarm)

                # Forward real-time stream to .NET IndustrialTelemetry.Engine
                if bridge.enabled:
                    if frame.interlock_tripped and bridge.active_batch_id:
                        await bridge.abort_batch(reason="Emergency SIS interlock tripped on ProcessTwin.")
                    else:
                        await bridge.send_frame(frame)

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
    # Check .NET IndustrialTelemetry.Engine & ChemFlow.NET connectivity on startup
    await bridge.check_health()
    await chemflow_bridge.check_health()
    background_task = asyncio.create_task(telemetry_background_loop())
    yield
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
    await bridge.close()
    await chemflow_bridge.close()


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


@app.get("/api/ekf/diagnostics")
async def get_ekf_diagnostics() -> dict:
    """Returns current Extended Kalman Filter (EKF) internal states, covariance, gains, and NIS metrics."""
    if twin.latest_ekf is None:
        twin.step(dt_seconds=0.0)

    ekf = twin.latest_ekf
    return {
        "status": "operational",
        "estimate": {
            "ca_mol_per_l": ekf.ca_mol_per_l if ekf else None,
            "temp_celsius": ekf.temp_celsius if ekf else None,
            "temp_kelvin": ekf.temp_kelvin if ekf else None,
            "conversion_pct": ekf.conversion_pct if ekf else None,
            "reaction_rate": ekf.reaction_rate if ekf else None,
        },
        "diagnostics": {
            "temp_residual_c": ekf.temp_residual if ekf else None,
            "nis": ekf.nis if ekf else None,
            "sensor_healthy": (ekf.nis < 6.63) if ekf else True,
            "p_ca": ekf.p_ca if ekf else None,
            "p_temp": ekf.p_temp if ekf else None,
            "p_ca_temp": ekf.p_ca_temp if ekf else None,
            "kalman_gain_ca": ekf.kalman_gain_ca if ekf else None,
            "kalman_gain_temp": ekf.kalman_gain_temp if ekf else None,
        }
    }


# --- Non-Linear Model Predictive Control (NMPC) Endpoints ---

@app.get("/api/nmpc/diagnostics")
async def get_nmpc_diagnostics() -> dict:
    """Returns NMPC controller configuration, internal status, predicted trajectories, and solver performance."""
    return twin.get_nmpc_diagnostics()


@app.post("/api/nmpc/toggle")
async def toggle_nmpc(cmd: Optional[NmpcToggleCommand] = None) -> dict:
    """Enables or disables NMPC closed-loop cooling valve control."""
    if cmd is not None:
        twin.enable_nmpc(
            enabled=cmd.enabled,
            temp_setpoint_c=cmd.temp_setpoint_c,
            ca_setpoint=cmd.ca_setpoint,
        )
    else:
        twin.enable_nmpc(enabled=not twin.nmpc_enabled)

    return {
        "status": "success",
        "nmpc_enabled": twin.nmpc_enabled,
        "temp_setpoint_c": twin.nmpc.config.temp_setpoint_c,
        "ca_setpoint": twin.nmpc.config.ca_setpoint,
    }


@app.post("/api/nmpc/setpoint")
async def set_nmpc_setpoints(cmd: NmpcSetpointsCommand) -> dict:
    """Updates NMPC target reactor temperature (°C) and/or concentration setpoints."""
    twin.update_nmpc_setpoints(
        temp_setpoint_c=cmd.temp_setpoint_c,
        ca_setpoint=cmd.ca_setpoint,
    )
    return {
        "status": "success",
        "temp_setpoint_c": twin.nmpc.config.temp_setpoint_c,
        "ca_setpoint": twin.nmpc.config.ca_setpoint,
    }


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


# --- IndustrialTelemetry.Engine (.NET 8) Bridge Endpoints ---

@app.get("/api/bridge/status", response_model=BridgeStatus)
async def get_bridge_status() -> BridgeStatus:
    """Returns real-time connectivity, active batch metadata, and transmission metrics for the .NET engine bridge."""
    await bridge.check_health()
    return bridge.get_status()


@app.post("/api/bridge/toggle")
async def toggle_bridge() -> dict:
    """Enables or disables continuous telemetry streaming to the .NET engine."""
    bridge.enabled = not bridge.enabled
    return {"status": "success", "enabled": bridge.enabled}


@app.post("/api/bridge/batch/new")
async def start_new_bridge_batch() -> dict:
    """Forces the creation and start of a fresh batch session in IndustrialTelemetry.Engine."""
    bridge.active_batch_id = None
    bridge.active_batch_code = None
    batch_id = await bridge.ensure_active_batch()
    if not batch_id:
        raise HTTPException(status_code=502, detail=f"Failed to start batch in .NET Engine: {bridge.last_error}")
    return {
        "status": "success",
        "batch_id": batch_id,
        "batch_code": bridge.active_batch_code
    }


@app.post("/api/bridge/batch/complete")
async def complete_bridge_batch() -> dict:
    """Completes the current batch run in .NET Engine and returns the compliance analytics summary."""
    summary = await bridge.complete_batch()
    if not summary:
        raise HTTPException(status_code=400, detail="No active batch to complete or request failed.")
    return {"status": "success", "summary": summary}


@app.get("/api/bridge/batch/summary")
async def get_bridge_batch_summary(batch_id: Optional[str] = None) -> dict:
    """Fetches batch analytics summary from .NET Engine."""
    summary = await bridge.get_summary(batch_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Batch summary not available.")
    return summary


@app.get("/api/thermo/current")
async def get_current_thermodynamics() -> dict:
    """Returns the current real-fluid thermodynamic state calculated via ChemFlow.NET Peng-Robinson EOS."""
    return {
        "status": chemflow_bridge.get_status().model_dump(),
        "latest_flash": {
            "vapor_fraction_beta": twin.latest_beta,
            "thermo_phase": twin.latest_phase,
            "thermo_model": twin.latest_thermo_model,
            "molar_enthalpy_j_mol": twin.latest_molar_enthalpy,
        }
    }


@app.get("/api/thermo/status", response_model=ChemFlowStatus)
async def get_thermo_status() -> ChemFlowStatus:
    """Returns the connectivity and health status of the ChemFlow.NET thermodynamics bridge."""
    await chemflow_bridge.check_health()
    return chemflow_bridge.get_status()


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
