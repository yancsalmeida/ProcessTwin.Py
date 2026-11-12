"""
Unit Tests for IndustrialGateway and FastAPI Industrial Endpoints.
Validates:
1. Unified lifecycle management of OPC UA and Modbus TCP servers.
2. Status reporting via get_status() and /api/industrial/status endpoint.
3. Start/stop toggling of individual industrial protocols.
4. Seamless telemetry synchronization across dual industrial stacks.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.models import PlantTelemetry, IndustrialGatewayStatus
from process_twin.industrial.gateway import IndustrialGateway
from process_twin.api.app import app


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_industrial_gateway_lifecycle():
    twin = PlantDigitalTwin()
    gw = IndustrialGateway(
        twin_ref=twin,
        opcua_endpoint="opc.tcp://127.0.0.1:48402/processtwin/gw_test/",
        modbus_host="127.0.0.1",
        modbus_port=50216,
    )

    status_initial = gw.get_status()
    assert isinstance(status_initial, IndustrialGatewayStatus)
    assert status_initial.opcua_running is False
    assert status_initial.modbus_running is False

    # Start gateway
    await gw.start()
    status_running = gw.get_status()
    assert status_running.opcua_running is True
    assert status_running.modbus_running is True
    assert status_running.opcua_tags_count >= 30

    # Sync telemetry frame
    frame = twin.step(dt_seconds=0.2)
    await gw.sync_telemetry(frame)
    await gw.check_client_writes()
    assert gw.sync_count == 1
    assert gw.last_sync_utc is not None

    # Stop gateway
    await gw.stop()
    status_stopped = gw.get_status()
    assert status_stopped.opcua_running is False
    assert status_stopped.modbus_running is False


@pytest.mark.asyncio
async def test_industrial_endpoints(async_client):
    # 1. GET /api/industrial/status
    resp = await async_client.get("/api/industrial/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "opcua_enabled" in data
    assert "modbus_enabled" in data
    assert "opcua_endpoint" in data
    assert "modbus_port" in data

    # 2. Toggle Modbus stop and start
    stop_resp = await async_client.post("/api/industrial/modbus/stop")
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "success"

    start_resp = await async_client.post("/api/industrial/modbus/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "success"
