import pytest
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.models import ActuatorCommand, PlantStatus


def test_simulator_initial_state():
    twin = PlantDigitalTwin()
    assert twin.tank_level_m == 5.0
    assert twin.tank_area_m2 == 5.0
    assert twin.feed_valve_open is True
    assert twin.interlock_tripped is False
    assert twin.cooling_valve_pct == 45.0
    assert twin.status == PlantStatus.NORMAL


def test_mass_balance_accumulation():
    twin = PlantDigitalTwin()
    twin.time_scale = 1.0
    # Suppress noise for deterministic verification
    twin.level_noise_std = 0.0

    # Increase feed flow above discharge
    # Feed: 44 m3/h, Discharge: 24 m3/h -> Net: +20 m3/h
    # In 360 seconds (0.1 hours), accumulation = 2.0 m3
    # Height increase = 2.0 / 5.0 = 0.4 m -> new level should be ~5.4 m
    twin.apply_actuator_command(ActuatorCommand(target_feed_flow_m3_h=44.0))

    for _ in range(360):
        frame = twin.step(dt_seconds=1.0)

    assert frame.tank_level_m == pytest.approx(5.4, abs=0.05)
    assert frame.feed_flow_m3_h == 44.0
    assert frame.discharge_flow_m3_h == 24.0


def test_mass_balance_depletion():
    twin = PlantDigitalTwin()
    twin.time_scale = 1.0
    twin.level_noise_std = 0.0

    # Feed: 0 m3/h, Discharge: 30 m3/h -> Net: -30 m3/h
    # In 180 seconds (0.05 hours), depletion = 1.5 m3
    # Height drop = 1.5 / 5.0 = 0.3 m -> new level should be ~4.7 m
    twin.apply_actuator_command(
        ActuatorCommand(target_feed_flow_m3_h=0.0, target_discharge_flow_m3_h=30.0)
    )

    for _ in range(180):
        frame = twin.step(dt_seconds=1.0)

    assert frame.tank_level_m == pytest.approx(4.7, abs=0.05)


def test_automated_interlock_trip_on_high_level():
    twin = PlantDigitalTwin()
    twin.level_noise_std = 0.0
    # Set level directly above trip threshold (9.5 m)
    twin.tank_level_m = 9.55

    # Step evaluates safety interlock and trips automated emergency shutoff
    frame = twin.step(dt_seconds=0.2)

    assert frame.interlock_tripped is True
    assert twin.feed_valve_open is False
    assert frame.status == PlantStatus.CRITICAL_SHUTDOWN
    assert any(a.tag == "LT-101" for a in twin.active_alarms)


def test_automated_interlock_trip_on_thermal_runaway():
    twin = PlantDigitalTwin()
    twin.temp_noise_std = 0.0
    # Set reactor temperature above critical runaway threshold (135.0 °C)
    twin.reactor_temp_c = 136.5

    frame = twin.step(dt_seconds=0.2)

    assert twin.interlock_tripped is True
    assert twin.cooling_valve_pct == 100.0  # Emergency cooling flood
    assert twin.feed_valve_open is False
    assert frame.status == PlantStatus.CRITICAL_SHUTDOWN


def test_alarm_acknowledgment():
    twin = PlantDigitalTwin()
    twin.tank_level_m = 8.5  # High warning
    twin.step(dt_seconds=0.1)

    assert len(twin.active_alarms) > 0
    alarm_id = twin.active_alarms[0].id
    assert twin.active_alarms[0].acknowledged is False

    acknowledged = twin.acknowledge_alarm(alarm_id)
    assert acknowledged is True
    assert twin.active_alarms[0].acknowledged is True

    # Unknown alarm id returns False
    assert twin.acknowledge_alarm("non-existent-id") is False


def test_simulator_reset():
    twin = PlantDigitalTwin()
    twin.tank_level_m = 9.8
    twin.reactor_temp_c = 240.0
    twin.cooling_valve_pct = 100.0
    twin.feed_valve_open = False
    twin.interlock_tripped = True
    twin.status = PlantStatus.CRITICAL_SHUTDOWN

    twin.reset()

    assert twin.tank_level_m == 5.0
    assert twin.reactor_temp_c == 85.0
    assert twin.cooling_valve_pct == 45.0
    assert twin.feed_valve_open is True
    assert twin.interlock_tripped is False
    assert twin.status == PlantStatus.NORMAL
    assert len(twin.active_alarms) == 0


def test_thermal_runaway_can_cool_down():
    """Verify that even at max temperature (250°C), 100% cooling duty overpowers heat generation."""
    twin = PlantDigitalTwin()
    twin.temp_noise_std = 0.0
    twin.reactor_temp_c = 250.0
    twin.cooling_valve_pct = 100.0
    twin.feed_valve_open = False  # typical after trip

    initial_temp = twin.reactor_temp_c
    frame = twin.step(dt_seconds=1.0)

    # Temperature must decrease, proving no unphysical kinetic trapping
    assert frame.reactor_temp_c < initial_temp
