from datetime import datetime, timezone
from process_twin.core.models import PlantTelemetry, PlantStatus, AlarmSeverity
from process_twin.core.alarms import evaluate_safety_alarms, AlarmThresholds


def create_baseline_telemetry(**kwargs) -> PlantTelemetry:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        tank_level_m=5.0,
        tank_volume_m3=25.0,
        feed_flow_m3_h=24.0,
        discharge_flow_m3_h=24.0,
        reactor_temp_c=85.0,
        reactor_pressure_bar=1.1,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        interlock_tripped=False,
        status=PlantStatus.NORMAL,
    )
    defaults.update(kwargs)
    return PlantTelemetry(**defaults)


def test_normal_operating_conditions():
    telemetry = create_baseline_telemetry()
    status, alarms, tripped = evaluate_safety_alarms(telemetry)

    assert status == PlantStatus.NORMAL
    assert len(alarms) == 0
    assert tripped is False


def test_level_high_warning():
    telemetry = create_baseline_telemetry(tank_level_m=8.5)
    status, alarms, tripped = evaluate_safety_alarms(telemetry)

    assert status == PlantStatus.HIGH_ALARM
    assert len(alarms) == 1
    assert alarms[0].tag == "LT-101"
    assert alarms[0].severity == AlarmSeverity.WARNING
    assert tripped is False


def test_level_critical_interlock_trip():
    telemetry = create_baseline_telemetry(tank_level_m=9.8)
    status, alarms, tripped = evaluate_safety_alarms(telemetry)

    assert status == PlantStatus.CRITICAL_SHUTDOWN
    assert len(alarms) == 1
    assert alarms[0].tag == "LT-101"
    assert alarms[0].severity == AlarmSeverity.CRITICAL
    assert tripped is True


def test_temperature_warning_and_runaway():
    # Warning
    tel_warn = create_baseline_telemetry(reactor_temp_c=120.0)
    status_w, alarms_w, tripped_w = evaluate_safety_alarms(tel_warn)
    assert status_w == PlantStatus.HIGH_ALARM
    assert len(alarms_w) == 1
    assert alarms_w[0].tag == "TT-102"
    assert alarms_w[0].severity == AlarmSeverity.WARNING
    assert tripped_w is False

    # Critical Runaway
    tel_crit = create_baseline_telemetry(reactor_temp_c=142.0)
    status_c, alarms_c, tripped_c = evaluate_safety_alarms(tel_crit)
    assert status_c == PlantStatus.CRITICAL_SHUTDOWN
    assert len(alarms_c) == 1
    assert alarms_c[0].tag == "TT-102"
    assert alarms_c[0].severity == AlarmSeverity.CRITICAL
    assert tripped_c is True


def test_pressure_overpressure_alarms():
    # Warning
    tel_warn = create_baseline_telemetry(reactor_pressure_bar=4.5)
    status_w, alarms_w, tripped_w = evaluate_safety_alarms(tel_warn)
    assert status_w == PlantStatus.HIGH_ALARM
    assert len(alarms_w) == 1
    assert alarms_w[0].tag == "PT-103"
    assert alarms_w[0].severity == AlarmSeverity.WARNING

    # Critical Overpressure
    tel_crit = create_baseline_telemetry(reactor_pressure_bar=5.8)
    status_c, alarms_c, tripped_c = evaluate_safety_alarms(tel_crit)
    assert status_c == PlantStatus.CRITICAL_SHUTDOWN
    assert len(alarms_c) == 1
    assert alarms_c[0].tag == "PT-103"
    assert alarms_c[0].severity == AlarmSeverity.CRITICAL
