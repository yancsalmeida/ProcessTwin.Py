from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from process_twin.core.models import (
    PlantTelemetry,
    AlarmEvent,
    ActuatorCommand,
    PlantStatus,
    AlarmSeverity,
)


def test_plant_telemetry_valid():
    now = datetime.now(timezone.utc)
    telemetry = PlantTelemetry(
        timestamp=now,
        tank_level_m=5.25,
        tank_volume_m3=26.25,
        feed_flow_m3_h=24.0,
        discharge_flow_m3_h=24.0,
        reactor_temp_c=85.2,
        reactor_pressure_bar=1.12,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        interlock_tripped=False,
        status=PlantStatus.NORMAL,
    )
    assert telemetry.tank_level_m == 5.25
    assert telemetry.tank_volume_m3 == 26.25
    assert telemetry.status == PlantStatus.NORMAL
    assert telemetry.feed_valve_open is True


def test_plant_telemetry_invalid_level():
    with pytest.raises(ValidationError):
        PlantTelemetry(
            tank_level_m=-1.0,  # ge=0.0 violation
            tank_volume_m3=0.0,
            feed_flow_m3_h=24.0,
            discharge_flow_m3_h=24.0,
            reactor_temp_c=85.0,
            reactor_pressure_bar=1.0,
            cooling_valve_pct=50.0,
            feed_valve_open=True,
        )

    with pytest.raises(ValidationError):
        PlantTelemetry(
            tank_level_m=15.0,  # le=12.0 violation
            tank_volume_m3=75.0,
            feed_flow_m3_h=24.0,
            discharge_flow_m3_h=24.0,
            reactor_temp_c=85.0,
            reactor_pressure_bar=1.0,
            cooling_valve_pct=50.0,
            feed_valve_open=True,
        )


def test_actuator_command_validation():
    cmd = ActuatorCommand(cooling_valve_pct=75.5, feed_valve_open=False)
    assert cmd.cooling_valve_pct == 75.5
    assert cmd.feed_valve_open is False
    assert cmd.target_feed_flow_m3_h is None

    # Out of range cooling valve (> 100%)
    with pytest.raises(ValidationError):
        ActuatorCommand(cooling_valve_pct=120.0)

    # Out of range flow (> 60 m³/h)
    with pytest.raises(ValidationError):
        ActuatorCommand(target_feed_flow_m3_h=99.0)


def test_alarm_event_creation():
    alarm = AlarmEvent(
        id="alarm-101",
        severity=AlarmSeverity.CRITICAL,
        tag="LT-101",
        message="Critical high level",
    )
    assert alarm.id == "alarm-101"
    assert alarm.severity == AlarmSeverity.CRITICAL
    assert alarm.tag == "LT-101"
    assert alarm.acknowledged is False
    assert alarm.resolved is False


def test_nmpc_models_validation():
    from process_twin.core.models import NmpcToggleCommand, NmpcSetpointsCommand

    toggle_cmd = NmpcToggleCommand(enabled=True, temp_setpoint_c=86.5)
    assert toggle_cmd.enabled is True
    assert toggle_cmd.temp_setpoint_c == 86.5
    assert toggle_cmd.ca_setpoint is None

    sp_cmd = NmpcSetpointsCommand(temp_setpoint_c=90.0, ca_setpoint=0.45)
    assert sp_cmd.temp_setpoint_c == 90.0
    assert sp_cmd.ca_setpoint == 0.45

    # Out of range temperature (< 20 or > 150)
    with pytest.raises(ValidationError):
        NmpcSetpointsCommand(temp_setpoint_c=180.0)

    # ActuatorCommand with NMPC fields
    act_cmd = ActuatorCommand(nmpc_enabled=True, nmpc_target_temp_c=83.0)
    assert act_cmd.nmpc_enabled is True
    assert act_cmd.nmpc_target_temp_c == 83.0
