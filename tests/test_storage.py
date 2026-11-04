import os
import pytest
from datetime import datetime, timezone
from process_twin.db.storage import TelemetryDatabase
from process_twin.core.models import PlantTelemetry, AlarmEvent, PlantStatus, AlarmSeverity


@pytest.fixture
async def temp_db(tmp_path):
    db_file = tmp_path / "test_twin.db"
    db = TelemetryDatabase(str(db_file))
    await db.init_db()
    return db


async def test_insert_and_retrieve_telemetry(temp_db):
    frame = PlantTelemetry(
        timestamp=datetime.now(timezone.utc),
        tank_level_m=6.5,
        tank_volume_m3=32.5,
        feed_flow_m3_h=25.0,
        discharge_flow_m3_h=25.0,
        reactor_temp_c=88.5,
        reactor_pressure_bar=1.2,
        cooling_valve_pct=50.0,
        feed_valve_open=True,
        interlock_tripped=False,
        status=PlantStatus.NORMAL,
    )

    await temp_db.insert_telemetry(frame)
    rows = await temp_db.get_recent_telemetry(limit=10)

    assert len(rows) == 1
    assert rows[0]["tank_level_m"] == 6.5
    assert rows[0]["reactor_temp_c"] == 88.5
    assert rows[0]["feed_valve_open"] == 1
    assert rows[0]["status"] == "NORMAL"


async def test_insert_and_retrieve_alarms(temp_db):
    alarm = AlarmEvent(
        id="evt-404",
        timestamp=datetime.now(timezone.utc),
        severity=AlarmSeverity.CRITICAL,
        tag="LT-101",
        message="Critical high level interlock tripped",
        acknowledged=False,
    )

    await temp_db.insert_alarm(alarm)
    alarms = await temp_db.get_recent_alarms(limit=10)

    assert len(alarms) == 1
    assert alarms[0]["id"] == "evt-404"
    assert alarms[0]["severity"] == "CRITICAL"
    assert alarms[0]["tag"] == "LT-101"
    assert alarms[0]["acknowledged"] == 0
