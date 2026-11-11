import pytest
from httpx import AsyncClient, ASGITransport

from process_twin.core.nmpc import CstrNonlinearMpc, NmpcConfig, NmpcResult
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.models import ActuatorCommand, PlantStatus
from process_twin.api.app import app, twin


# =====================================================================
# Unit Tests: CstrNonlinearMpc
# =====================================================================

def test_nmpc_config_defaults():
    config = NmpcConfig()
    assert config.prediction_horizon == 15
    assert config.control_horizon == 4
    assert config.temp_setpoint_c == 85.0
    assert config.u_min_pct == 0.0
    assert config.u_max_pct == 100.0
    assert config.max_delta_u_pct == 15.0


def test_nmpc_projection_slew_rate_and_bounds():
    controller = CstrNonlinearMpc()
    u_prev = 50.0

    # Test move exceeding positive slew limit (+25% requested, max allowed +15% => 65%)
    moves = [75.0, 95.0]
    proj = controller._project_moves(moves, u_prev)
    assert proj[0] == pytest.approx(65.0, abs=1e-5)
    assert proj[1] == pytest.approx(80.0, abs=1e-5)  # 65 + 15

    # Test move exceeding negative slew limit (-30% requested, max allowed -15% => 35%)
    moves_down = [20.0]
    proj_down = controller._project_moves(moves_down, u_prev)
    assert proj_down[0] == pytest.approx(35.0, abs=1e-5)

    # Test hard box constraints
    moves_oob = [120.0, -10.0]
    proj_oob = controller._project_moves(moves_oob, 95.0)
    assert proj_oob[0] <= 100.0


def test_nmpc_cooling_response_to_high_temperature():
    """When temperature is above setpoint (92 °C vs 85 °C), optimizer must increase cooling valve opening."""
    controller = CstrNonlinearMpc()
    controller.reset(initial_u=45.0)

    res = controller.compute_optimal_move(
        current_ca=0.5,
        current_temp_c=92.0,
        flow_m3_h=24.0,
        feed_active=True,
        u_prev=45.0,
    )

    assert isinstance(res, NmpcResult)
    # Valve opening must increase to reject heat
    assert res.optimal_cooling_valve_pct > 45.0
    assert res.temp_tracking_error == pytest.approx(7.0, abs=0.1)
    assert len(res.predicted_temp_trajectory) == controller.config.prediction_horizon
    assert len(res.planned_control_moves) == controller.config.control_horizon
    assert res.solve_time_ms >= 0.0


def test_nmpc_heating_throttling_to_low_temperature():
    """When temperature is below setpoint (78 °C vs 85 °C), optimizer must throttle cooling valve down."""
    controller = CstrNonlinearMpc()
    controller.reset(initial_u=45.0)

    res = controller.compute_optimal_move(
        current_ca=0.5,
        current_temp_c=78.0,
        flow_m3_h=24.0,
        feed_active=True,
        u_prev=45.0,
    )

    assert res.optimal_cooling_valve_pct < 45.0
    assert res.temp_tracking_error == pytest.approx(-7.0, abs=0.1)


def test_nmpc_thermal_runaway_slack_penalty():
    """Approaching runaway safety limits (97 °C > 95 °C max safe) triggers massive slack penalty."""
    cfg = NmpcConfig(temp_setpoint_c=85.0, max_safe_temp_c=95.0)
    controller = CstrNonlinearMpc(config=cfg)

    # At runaway condition
    res_runaway = controller.compute_optimal_move(
        current_ca=0.6,
        current_temp_c=97.0,
        flow_m3_h=24.0,
        feed_active=True,
        u_prev=50.0,
    )
    # Aggressively open cooling valve to max allowable rate
    assert res_runaway.optimal_cooling_valve_pct >= 60.0


def test_nmpc_setpoint_update():
    controller = CstrNonlinearMpc()
    controller.update_setpoints(temp_setpoint_c=90.0, ca_setpoint=0.4)
    assert controller.config.temp_setpoint_c == 90.0
    assert controller.config.ca_setpoint == 0.4


# =====================================================================
# Simulator Integration Tests: PlantDigitalTwin + NMPC
# =====================================================================

def test_twin_nmpc_toggle_and_execution():
    sim = PlantDigitalTwin()
    assert not sim.nmpc_enabled

    # Enable NMPC with setpoint 82.0 °C
    sim.enable_nmpc(True, temp_setpoint_c=82.0)
    assert sim.nmpc_enabled
    assert sim.nmpc.config.temp_setpoint_c == 82.0

    # Advance simulation by several steps
    for _ in range(5):
        frame = sim.step(dt_seconds=0.2)
        assert frame.nmpc_enabled is True
        assert frame.nmpc_target_temp_c == 82.0
        assert frame.nmpc_optimal_cooling_pct is not None
        assert frame.nmpc_solve_time_ms is not None
        assert frame.nmpc_cost is not None

    diag = sim.get_nmpc_diagnostics()
    assert diag["enabled"] is True
    assert diag["config"]["temp_setpoint_c"] == 82.0
    assert diag["latest_result"] is not None


def test_twin_actuator_override_disables_nmpc():
    sim = PlantDigitalTwin()
    sim.enable_nmpc(True)
    assert sim.nmpc_enabled

    # Operator manually forces cooling valve to 70% without passing nmpc_enabled
    cmd = ActuatorCommand(cooling_valve_pct=70.0)
    sim.apply_actuator_command(cmd)

    # NMPC should yield to manual operator control
    assert sim.nmpc_enabled is False
    assert sim.cooling_valve_pct == 70.0


def test_twin_interlock_trip_overrides_nmpc():
    sim = PlantDigitalTwin()
    sim.enable_nmpc(True)
    assert sim.nmpc_enabled

    # Force reactor core to extreme runaway temperature triggering critical SIS trip (threshold 135.0 °C)
    sim.reactor_temp_c = 140.0
    frame = sim.step(dt_seconds=0.2)

    assert frame.interlock_tripped is True
    assert frame.nmpc_enabled is False
    assert sim.nmpc_enabled is False
    assert sim.cooling_valve_pct == 100.0
    assert sim.feed_valve_open is False


# =====================================================================
# API Integration Tests: FastAPI Endpoints
# =====================================================================

@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_api_nmpc_diagnostics(async_client):
    response = await async_client.get("/api/nmpc/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "config" in data
    assert data["config"]["prediction_horizon"] == 15


async def test_api_nmpc_toggle_and_setpoint(async_client):
    # Toggle NMPC on via POST
    toggle_resp = await async_client.post("/api/nmpc/toggle", json={"enabled": True, "temp_setpoint_c": 83.5})
    assert toggle_resp.status_code == 200
    data = toggle_resp.json()
    assert data["status"] == "success"
    assert data["nmpc_enabled"] is True
    assert data["temp_setpoint_c"] == 83.5

    # Check setpoint endpoint
    sp_resp = await async_client.post("/api/nmpc/setpoint", json={"temp_setpoint_c": 87.0, "ca_setpoint": 0.45})
    assert sp_resp.status_code == 200
    sp_data = sp_resp.json()
    assert sp_data["status"] == "success"
    assert sp_data["temp_setpoint_c"] == 87.0
    assert sp_data["ca_setpoint"] == 0.45

    # Toggle off
    toggle_off = await async_client.post("/api/nmpc/toggle", json={"enabled": False})
    assert toggle_off.status_code == 200
    assert toggle_off.json()["nmpc_enabled"] is False

    # Toggle without body (inverts status)
    toggle_flip = await async_client.post("/api/nmpc/toggle")
    assert toggle_flip.status_code == 200
    assert toggle_flip.json()["nmpc_enabled"] is True
