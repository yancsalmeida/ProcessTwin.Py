import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from process_twin.api.app import app, twin


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_get_dashboard(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert "ProcessTwin" in response.text
    assert "TIT-101-EST" in response.text
    assert "AT-101-EST" in response.text
    assert "chartCanvas" in response.text
    assert "chart.umd.min.js" in response.text

    static_resp = await async_client.get("/static/chart.umd.min.js")
    assert static_resp.status_code == 200
    assert len(static_resp.content) > 100000


async def test_get_current_telemetry(async_client):
    response = await async_client.get("/api/telemetry/current")
    assert response.status_code == 200
    data = response.json()
    assert "tank_level_m" in data
    assert "reactor_temp_c" in data
    assert "status" in data
    assert data["tank_level_m"] >= 0.0


async def test_override_actuators(async_client):
    payload = {
        "cooling_valve_pct": 82.5,
        "feed_valve_open": True,
        "target_feed_flow_m3_h": 28.0,
    }
    response = await async_client.post("/api/actuators", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["command_applied"]["cooling_valve_pct"] == 82.5

    # Verify state updated in twin
    assert twin.cooling_valve_pct == 82.5
    assert twin.feed_flow_m3_h == 28.0


async def test_alarms_endpoints(async_client):
    # Fetch alarms
    response = await async_client.get("/api/alarms")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    # Acknowledge non-existent alarm returns 404
    resp_404 = await async_client.post("/api/alarms/fake-id-999/acknowledge")
    assert resp_404.status_code == 404


async def test_reset_simulation_endpoint(async_client):
    # Alter twin state
    twin.tank_level_m = 9.9
    twin.reactor_temp_c = 199.0

    response = await async_client.post("/api/simulation/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    # Verify state reset
    assert twin.tank_level_m == 5.0
    assert twin.reactor_temp_c == 85.0
    assert twin.cooling_valve_pct == 45.0


def test_websocket_telemetry_stream():
    # TestClient synchronous context for WebSocket
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry") as ws:
        # Trigger one simulation step to push frame
        frame = twin.step(dt_seconds=0.2)
        # Verify websocket connection established
        assert ws is not None
