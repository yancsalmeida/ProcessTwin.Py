import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from process_twin.core.models import PlantTelemetry, PlantStatus
from process_twin.core.telemetry_bridge import IndustrialTelemetryBridge, BridgeStatus
from process_twin.api.app import app


def test_bridge_status_model_defaults():
    status = BridgeStatus()
    assert status.enabled is True
    assert status.connected is False
    assert status.frames_sent == 0
    assert status.measurements_ingested == 0
    assert status.alerts_triggered == 0


@pytest.mark.asyncio
async def test_bridge_check_health_success():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        ok = await bridge.check_health()
        assert ok is True
        assert bridge.connected is True
        assert bridge.last_error is None
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_check_health_failure():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    
    with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("Connection refused")):
        ok = await bridge.check_health()
        assert ok is False
        assert bridge.connected is False
        assert "Health check failed" in (bridge.last_error or "")
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_ensure_active_batch_existing():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": "batch-1234", "code": "BATCH-EXISTING"}]

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        batch_id = await bridge.ensure_active_batch()
        assert batch_id == "batch-1234"
        assert bridge.active_batch_code == "BATCH-EXISTING"
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_ensure_active_batch_creates_new_if_none():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    
    query_resp = MagicMock()
    query_resp.status_code = 200
    query_resp.json.return_value = []

    create_resp = MagicMock()
    create_resp.status_code = 201
    create_resp.json.return_value = {"id": "batch-new-999"}

    start_resp = MagicMock()
    start_resp.status_code = 200
    start_resp.json.return_value = {"status": "InProgress"}

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=query_resp):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [create_resp, start_resp]
            batch_id = await bridge.ensure_active_batch()
            assert batch_id == "batch-new-999"
            assert bridge.active_batch_id == "batch-new-999"
            assert "BATCH-TWIN-" in (bridge.active_batch_code or "")
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_send_frame_payload_structure():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    bridge.active_batch_id = "batch-active-1"
    bridge.active_batch_code = "BATCH-TEST"
    bridge.connected = True

    frame = PlantTelemetry(
        timestamp=datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc),
        tank_level_m=5.43,
        tank_volume_m3=38.4,
        feed_flow_m3_h=20.0,
        discharge_flow_m3_h=19.8,
        reactor_temp_c=88.5,
        reactor_pressure_bar=2.34,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        status=PlantStatus.NORMAL
    )

    ingest_resp = MagicMock()
    ingest_resp.status_code = 200
    ingest_resp.json.return_value = {"measurementsIngested": 6, "alertsTriggered": 0}

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        # Check that existing batch check returns InProgress
        batch_get_resp = MagicMock()
        batch_get_resp.status_code = 200
        batch_get_resp.json.return_value = {"status": "InProgress", "code": "BATCH-TEST"}
        mock_get.return_value = batch_get_resp

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=ingest_resp) as mock_post:
            res = await bridge.send_frame(frame)
            assert res == {"measurementsIngested": 6, "alertsTriggered": 0}
            assert bridge.frames_sent == 1
            assert bridge.measurements_ingested == 6
            assert bridge.alerts_triggered == 0

            # Verify payload passed to .NET ingest API
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert payload["batchId"] == "batch-active-1"
            assert len(payload["measurements"]) == 6
            tags = [m["sensorTag"] for m in payload["measurements"]]
            assert "TT-101" in tags
            assert "PT-101" in tags
            assert "LT-101" in tags
            assert "FT-101" in tags
            assert "FT-102" in tags
            assert "TCV-101" in tags

            # Verify units
            units = {m["sensorTag"]: m["unit"] for m in payload["measurements"]}
            assert units["TT-101"] == "Celsius"
            assert units["PT-101"] == "Bar"
            assert units["LT-101"] == "Meters"
            assert units["FT-101"] == "M3PerHour"
            assert units["TCV-101"] == "Percent"
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_abort_batch_on_interlock():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    bridge.active_batch_id = "batch-trip-1"

    abort_resp = MagicMock()
    abort_resp.status_code = 200
    abort_resp.json.return_value = {"status": "Aborted"}

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=abort_resp) as mock_post:
        res = await bridge.abort_batch("Safety Instrumented System trip")
        assert res == {"status": "Aborted"}
        assert bridge.active_batch_id is None
        mock_post.assert_called_once()
        assert "abort" in mock_post.call_args.args[0]
    await bridge.close()


def test_api_bridge_endpoints():
    client = TestClient(app)
    
    # Status endpoint
    resp = client.get("/api/bridge/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert "connected" in data

    # Toggle endpoint
    toggle_resp = client.post("/api/bridge/toggle")
    assert toggle_resp.status_code == 200
    new_state = toggle_resp.json()["enabled"]
    
    # Toggle back
    client.post("/api/bridge/toggle")


@pytest.mark.asyncio
async def test_bridge_send_frame_channel_mode():
    bridge = IndustrialTelemetryBridge(
        engine_url="http://mock-net-engine:5100",
        ingestion_mode="channel"
    )
    bridge.active_batch_id = "batch-chan-test"
    bridge.connected = True

    frame = PlantTelemetry(
        timestamp=datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc),
        tank_level_m=5.0,
        tank_volume_m3=35.0,
        feed_flow_m3_h=20.0,
        discharge_flow_m3_h=20.0,
        reactor_temp_c=90.0,
        reactor_pressure_bar=2.5,
        cooling_valve_pct=50.0,
        feed_valve_open=True,
        status=PlantStatus.NORMAL
    )

    channel_resp = MagicMock()
    channel_resp.status_code = 202
    channel_resp.json.return_value = {"acceptedCount": 6, "currentQueueDepth": 12}

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"status": "InProgress", "code": "BATCH-CHAN-TEST"})
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=channel_resp) as mock_post:
            res = await bridge.send_frame(frame)
            assert res == {"acceptedCount": 6, "currentQueueDepth": 12}
            assert bridge.frames_sent == 1
            assert bridge.measurements_ingested == 6
            assert "/api/telemetry/ingest/channel" in mock_post.call_args.args[0]
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_send_frame_with_ekf_and_nmpc_tags():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    bridge.active_batch_id = "batch-ekf-1"
    bridge.active_batch_code = "BATCH-EKF-TEST"
    bridge.connected = True

    frame = PlantTelemetry(
        timestamp=datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc),
        tank_level_m=5.0,
        tank_volume_m3=35.0,
        feed_flow_m3_h=20.0,
        discharge_flow_m3_h=20.0,
        reactor_temp_c=88.5,
        reactor_pressure_bar=2.34,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        status=PlantStatus.NORMAL,
        estimated_temp_c=88.42,
        estimated_ca_mol_l=1.2345,
        estimated_conversion_pct=75.31,
        nmpc_optimal_cooling_pct=52.4,
        nmpc_target_temp_c=88.0,
        ekf_nis=1.2
    )

    ingest_resp = MagicMock(status_code=200, json=lambda: {"measurementsIngested": 11, "alertsTriggered": 0})
    batch_get_resp = MagicMock(status_code=200, json=lambda: {"status": "InProgress", "code": "BATCH-EKF-TEST"})

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=batch_get_resp):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=ingest_resp) as mock_post:
            res = await bridge.send_frame(frame)
            assert res == {"measurementsIngested": 11, "alertsTriggered": 0}
            assert bridge.frames_sent == 1
            assert bridge.measurements_ingested == 11

            payload = mock_post.call_args.kwargs["json"]
            assert payload["batchId"] == "batch-ekf-1"
            assert len(payload["measurements"]) == 11

            measurements_by_tag = {m["sensorTag"]: m for m in payload["measurements"]}
            
            # Check EKF state estimation tags
            assert "TIT-101-EST" in measurements_by_tag
            assert measurements_by_tag["TIT-101-EST"]["value"] == 88.42
            assert measurements_by_tag["TIT-101-EST"]["unit"] == "Celsius"
            assert measurements_by_tag["TIT-101-EST"]["quality"] == "Good"

            assert "AT-101-EST" in measurements_by_tag
            assert measurements_by_tag["AT-101-EST"]["value"] == 1.2345
            assert measurements_by_tag["AT-101-EST"]["unit"] == "MolPerLiter"
            assert measurements_by_tag["AT-101-EST"]["quality"] == "Good"

            assert "X-101-EST" in measurements_by_tag
            assert measurements_by_tag["X-101-EST"]["value"] == 75.31
            assert measurements_by_tag["X-101-EST"]["unit"] == "Percent"

            # Check NMPC tags
            assert "TCV-101-NMPC" in measurements_by_tag
            assert measurements_by_tag["TCV-101-NMPC"]["value"] == 52.4
            assert measurements_by_tag["TCV-101-NMPC"]["unit"] == "Percent"

            assert "TIC-101-SP" in measurements_by_tag
            assert measurements_by_tag["TIC-101-SP"]["value"] == 88.0
            assert measurements_by_tag["TIC-101-SP"]["unit"] == "Celsius"
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_send_frame_ekf_nis_quality_degradation():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100")
    bridge.active_batch_id = "batch-nis-1"
    bridge.connected = True

    base_kwargs = dict(
        timestamp=datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc),
        tank_level_m=5.0,
        tank_volume_m3=35.0,
        feed_flow_m3_h=20.0,
        discharge_flow_m3_h=20.0,
        reactor_temp_c=95.0,
        reactor_pressure_bar=2.8,
        cooling_valve_pct=60.0,
        feed_valve_open=True,
        estimated_temp_c=89.0,
        status=PlantStatus.NORMAL
    )

    batch_get_resp = MagicMock(status_code=200, json=lambda: {"status": "InProgress", "code": "BATCH-NIS-TEST"})
    ingest_resp = MagicMock(status_code=200, json=lambda: {"measurementsIngested": 7, "alertsTriggered": 0})

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=batch_get_resp):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=ingest_resp) as mock_post:
            # 1. Moderate drift: NIS = 5.2 (between 3.84 and 6.63) -> Uncertain
            frame_uncertain = PlantTelemetry(**base_kwargs, ekf_nis=5.2)
            await bridge.send_frame(frame_uncertain)
            payload = mock_post.call_args.kwargs["json"]
            measurements_by_tag = {m["sensorTag"]: m for m in payload["measurements"]}
            assert measurements_by_tag["TT-101"]["quality"] == "Uncertain"
            assert measurements_by_tag["TIT-101-EST"]["quality"] == "Uncertain"

            # 2. Severe sensor failure/drift: NIS = 12.5 (> 6.63) -> Bad
            frame_bad = PlantTelemetry(**base_kwargs, ekf_nis=12.5)
            await bridge.send_frame(frame_bad)
            payload = mock_post.call_args.kwargs["json"]
            measurements_by_tag = {m["sensorTag"]: m for m in payload["measurements"]}
            assert measurements_by_tag["TT-101"]["quality"] == "Bad"
            assert measurements_by_tag["TIT-101-EST"]["quality"] == "Bad"
    await bridge.close()


@pytest.mark.asyncio
async def test_bridge_send_frame_wire_protocol_mode():
    bridge = IndustrialTelemetryBridge(
        engine_url="http://mock-net-engine:5100",
        ingestion_mode="wire"
    )
    bridge.active_batch_id = "batch-wire-abc"
    bridge.connected = True

    frame = PlantTelemetry(
        timestamp=datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc),
        tank_level_m=6.0,
        tank_volume_m3=42.0,
        feed_flow_m3_h=22.0,
        discharge_flow_m3_h=22.0,
        reactor_temp_c=87.5,
        reactor_pressure_bar=2.1,
        cooling_valve_pct=40.0,
        feed_valve_open=True,
        estimated_temp_c=87.4,
        estimated_ca_mol_l=1.15,
        ekf_nis=1.1,
        status=PlantStatus.NORMAL
    )

    wire_resp = MagicMock(
        status_code=202,
        json=lambda: {
            "totalLinesParsed": 8,
            "validMeasurements": 8,
            "skippedLines": 0,
            "elapsedMilliseconds": 0.45,
            "recordsPerSecond": 17777.0,
            "ingestMode": "Channel",
            "alertsTriggered": 0,
            "channelQueueDepth": 8
        }
    )
    batch_get_resp = MagicMock(status_code=200, json=lambda: {"status": "InProgress", "code": "BATCH-WIRE-TEST"})

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=batch_get_resp):
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=wire_resp) as mock_post:
            res = await bridge.send_frame(frame)
            assert res["validMeasurements"] == 8
            assert bridge.frames_sent == 1
            assert bridge.measurements_ingested == 8

            call_args = mock_post.call_args
            assert "/api/telemetry/batch/batch-wire-abc/ingest/wire?mode=Channel" in call_args.args[0]
            assert call_args.kwargs["headers"]["Content-Type"] == "text/plain"
            content = call_args.kwargs["content"]
            lines = content.split("\n")
            assert len(lines) == 8
            # Verify pipe format: Timestamp|SensorTag|Value|Unit|Quality
            assert "TT-101|87.5|Celsius|Good" in lines[0]
            assert "AT-101-EST|1.15|MolPerLiter|Good" in content
    await bridge.close()
