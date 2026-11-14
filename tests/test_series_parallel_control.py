"""
Comprehensive Automated Tests for Option 1:
- 6-State Series-Parallel CSTR EKF (state observability, convergence from wrong IC, NIS fault detection)
- Economic / Selectivity-Maximizing NMPC + Thermal Runaway Rejection
- Pure-Python Fallback Determinism (CI compatibility without C11 shared library)
- PlantDigitalTwin Integration, SP-TT-201 Safety Alarm & IndustrialTelemetryBridge Tags
- Bi-directional OPC UA (SeriesParallelCstr nodes) & Modbus TCP (HR 30..41, Coils 6..7) Sync
- FastAPI REST Endpoints & SCADA Dashboard Controls
"""

import random
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from asyncua import Client
from fastapi.testclient import TestClient
from pymodbus.client import AsyncModbusTcpClient

from process_twin.api.app import app, twin
from process_twin.core.alarms import evaluate_safety_alarms
from process_twin.core.models import (
    PlantStatus,
    PlantTelemetry,
    SeriesParallelControlConfigCommand,
    SeriesParallelControlTelemetry,
)
from process_twin.core.native_reactor import NativeReactorCore
from process_twin.core.series_parallel_control import (
    SeriesParallelExtendedKalmanFilter,
    SeriesParallelNonlinearMpc,
    SpNmpcConfig,
    build_series_parallel_params,
    simulate_series_parallel_closed_loop_comparison,
    step_series_parallel_substepped,
)
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.telemetry_bridge import IndustrialTelemetryBridge
from process_twin.industrial.modbus_server import ProcessTwinModbusServer
from process_twin.industrial.opcua_server import ProcessTwinOpcUaServer


def test_ekf6_converges_from_wrong_initial_concentrations():
    core = NativeReactorCore()
    ekf = SeriesParallelExtendedKalmanFilter(native_reactor=core)
    ekf.reset(
        initial_ca=0.88,
        initial_cb=0.04,
        initial_cc=0.05,
        initial_cd=0.03,
        initial_temp_c=60.0,
        initial_jacket_temp_c=35.0,
    )

    true_state = (0.45, 0.30, 0.21, 0.04, 62.0 + 273.15, 36.0 + 273.15)
    params = build_series_parallel_params(core, cooling_valve_pct=58.0, feed_temp_c=51.85)
    rng = random.Random(101)

    est = None
    for _ in range(55):
        true_state = step_series_parallel_substepped(core, params, true_state, dt_min=0.05)
        t_c = true_state[4] - 273.15
        tj_c = true_state[5] - 273.15
        est = ekf.step(
            dt_min=0.05,
            params=params,
            measured_temp_c=t_c + rng.gauss(0.0, 0.12),
            measured_jacket_temp_c=tj_c + rng.gauss(0.0, 0.08),
        )

    assert est is not None
    assert abs(est.ca_mol_per_l - true_state[0]) < 0.015
    assert abs(est.cb_mol_per_l - true_state[1]) < 0.015
    assert abs(est.temp_c - (true_state[4] - 273.15)) < 0.6
    assert est.sensor_healthy is True
    assert est.nis < ekf.NIS_99_CHI2_2DOF


def test_ekf6_nis_detects_sensor_fault():
    core = NativeReactorCore()
    ekf = SeriesParallelExtendedKalmanFilter(native_reactor=core)
    params = build_series_parallel_params(core, cooling_valve_pct=55.0)
    ekf.reset(initial_ca=0.48, initial_cb=0.28, initial_cc=0.21, initial_cd=0.03, initial_temp_c=62.0, initial_jacket_temp_c=36.0)

    for _ in range(15):
        est = ekf.step(dt_min=0.05, params=params, measured_temp_c=62.0, measured_jacket_temp_c=36.0)

    assert est.sensor_healthy is True

    # Sudden +18C sensor bias fault on TT-201
    fault_est = ekf.step(dt_min=0.05, params=params, measured_temp_c=80.0, measured_jacket_temp_c=36.0)
    assert fault_est.nis > ekf.NIS_99_CHI2_2DOF
    assert fault_est.sensor_healthy is False


def test_closed_loop_economic_nmpc_prevents_runaway_and_maximizes_cb():
    res = simulate_series_parallel_closed_loop_comparison(
        native_core=NativeReactorCore(),
        feed_temp_nominal_c=51.85,
        disturbance_feed_temp_c=76.85,
        disturbance_step_index=12,
        initial_cooling_valve_pct=15.0,
        dt_min=0.05,
        steps=80,
        noise_seed=42,
    )

    assert res["open_loop_has_runaway"] is True
    assert res["open_loop_peak_temp_c"] > res["max_safe_temp_c"]
    assert res["closed_loop_has_runaway"] is False
    assert res["closed_loop_peak_temp_c"] < res["max_safe_temp_c"] - 10.0
    assert res["closed_loop_final_cb_mol_per_l"] > 5.0 * res["open_loop_final_cb_mol_per_l"]
    assert res["closed_loop_final_selectivity_pct"] > 5.0 * res["open_loop_final_selectivity_pct"]
    assert res["final_ekf_cb_error"] < 0.01


def test_closed_loop_works_in_pure_python_fallback():
    core = NativeReactorCore()
    core._lib = None
    assert not core.is_available

    res = simulate_series_parallel_closed_loop_comparison(native_core=core, steps=45)
    assert res["engine_mode"] == "Python RK4 Fallback"
    assert res["closed_loop_has_runaway"] is False
    assert res["closed_loop_peak_temp_c"] < res["open_loop_peak_temp_c"]
    assert res["closed_loop_final_cb_mol_per_l"] > res["open_loop_final_cb_mol_per_l"]


def test_nmpc_target_cb_setpoint_mode():
    core = NativeReactorCore()
    nmpc = SeriesParallelNonlinearMpc(config=SpNmpcConfig(target_cb_mol_per_l=0.25), native_reactor=core)
    params = build_series_parallel_params(core, cooling_valve_pct=55.0)
    out = nmpc.compute_optimal_move(
        current_state_c=(0.48, 0.28, 0.21, 0.03, 62.0, 36.0),
        base_params=params,
        u_prev=55.0,
    )
    assert 0.0 <= out.optimal_cooling_valve_pct <= 100.0
    assert len(out.predicted_cb_trajectory) == nmpc.config.prediction_horizon
    assert out.solve_time_ms >= 0.0


def test_twin_step_and_alarm_integration():
    local_twin = PlantDigitalTwin()
    diag = local_twin.update_series_parallel_control(
        SeriesParallelControlConfigCommand(nmpc_enabled=True, cooling_valve_pct=30.0)
    )
    assert diag["nmpc"]["enabled"] is True

    frame = local_twin.step(dt_seconds=0.2)
    assert frame.sp_control is not None
    assert frame.sp_control.nmpc_enabled is True
    assert frame.sp_control.est_cb_mol_per_l > 0.0

    # Verify SP-TT-201 alarm evaluates critical runaway cleanly
    runaway_frame = PlantTelemetry(
        tank_level_m=5.0,
        tank_volume_m3=25.0,
        feed_flow_m3_h=24.0,
        discharge_flow_m3_h=24.0,
        reactor_temp_c=85.0,
        reactor_pressure_bar=1.013,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        sp_control=SeriesParallelControlTelemetry(
            ca_mol_per_l=0.02,
            cb_mol_per_l=0.03,
            cc_mol_per_l=0.94,
            cd_mol_per_l=0.01,
            reactor_temp_c=135.2,
            jacket_temp_c=60.0,
            conversion_a_pct=98.0,
            yield_b_pct=3.0,
            selectivity_b_pct=3.1,
            est_ca_mol_per_l=0.02,
            est_cb_mol_per_l=0.03,
            est_cc_mol_per_l=0.94,
            est_cd_mol_per_l=0.01,
            est_temp_c=135.1,
            est_jacket_temp_c=60.0,
            ekf_nis=1.2,
            cooling_valve_pct=15.0,
            feed_temp_c=76.85,
            jacket_inlet_temp_c=21.85,
            max_safe_temp_c=126.85,
            is_thermal_runaway=True,
        ),
    )
    status, alarms, tripped = evaluate_safety_alarms(runaway_frame)
    assert status == PlantStatus.CRITICAL_SHUTDOWN
    assert tripped is True
    assert any(a.tag == "SP-TT-201" for a in alarms)


@pytest.mark.asyncio
async def test_telemetry_bridge_forwards_series_parallel_tags():
    bridge = IndustrialTelemetryBridge(engine_url="http://mock-net-engine:5100", ingestion_mode="transactional")
    bridge.active_batch_id = "batch-sp-001"
    bridge.active_batch_code = "BATCH-SP"

    local_twin = PlantDigitalTwin()
    frame = local_twin.step(dt_seconds=0.2)
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        return MagicMock(
            status_code=200,
            json=lambda: {"measurementsIngested": len(json["measurements"]), "alertsTriggered": 0},
        )

    try:
        with patch.object(bridge, "ensure_active_batch", new_callable=AsyncMock, return_value="batch-sp-001"):
            with patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
                res = await bridge.send_frame(frame)
    finally:
        await bridge.close()

    assert res is not None
    tags = {m["sensorTag"]: m for m in captured["json"]["measurements"]}
    assert "SP-TT-201" in tags
    assert "SP-TT-202" in tags
    assert "SP-AT-201-CB" in tags and tags["SP-AT-201-CB"]["unit"] == "MolPerLiter"
    assert "SP-SB-201" in tags and tags["SP-SB-201"]["unit"] == "Percent"
    assert "SP-TCV-201-NMPC" in tags


@pytest.mark.asyncio
async def test_opcua_series_parallel_sync_and_client_writes():
    local_twin = PlantDigitalTwin()
    endpoint = "opc.tcp://127.0.0.1:48409/processtwin/sp-test/"
    server = ProcessTwinOpcUaServer(endpoint=endpoint, server_name="SP Test OPC UA", twin_ref=local_twin)
    await server.start()
    try:
        frame = local_twin.step(dt_seconds=0.2)
        await server.sync_telemetry(frame)

        async with Client(endpoint) as client:
            ns_idx = await client.get_namespace_index(server.namespace_uri)
            cb_node = await client.nodes.objects.get_child(
                [f"{ns_idx}:ProcessTwin", f"{ns_idx}:SeriesParallelCstr", f"{ns_idx}:SP_CB_MolPerL"]
            )
            val_cb = await cb_node.read_value()
            assert val_cb == pytest.approx(frame.sp_control.cb_mol_per_l, abs=1e-4)

            en_node = await client.nodes.objects.get_child(
                [f"{ns_idx}:ProcessTwin", f"{ns_idx}:SeriesParallelCstr", f"{ns_idx}:SP_NmpcEnabled"]
            )
            feed_node = await client.nodes.objects.get_child(
                [f"{ns_idx}:ProcessTwin", f"{ns_idx}:SeriesParallelCstr", f"{ns_idx}:SP_FeedTemp_C"]
            )
            await en_node.write_value(True)
            await feed_node.write_value(76.85)

        await server.check_client_writes()
        assert local_twin.sp_nmpc_enabled is True
        assert local_twin.sp_feed_temp_c == pytest.approx(76.85, abs=0.01)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_modbus_series_parallel_registers_and_coils():
    local_twin = PlantDigitalTwin()
    port = 50229
    server = ProcessTwinModbusServer(host="127.0.0.1", port=port, twin_ref=local_twin)
    await server.start()
    client = AsyncModbusTcpClient("127.0.0.1", port=port)
    await client.connect()
    try:
        assert client.connected
        frame = local_twin.step(dt_seconds=0.2)
        server.sync_telemetry(frame)

        regs = await client.read_holding_registers(30, count=12)
        assert not regs.isError()
        assert regs.registers[1] == int(round(frame.sp_control.cb_mol_per_l * 1000))
        assert regs.registers[4] == int(round(frame.sp_control.reactor_temp_c * 10))
        assert regs.registers[6] == int(round(frame.sp_control.selectivity_b_pct * 10))

        # Enable Economic NMPC via Coil 6 and update cooling valve / feed temp via HR 39, 40
        await client.write_coil(6, True)
        await client.write_register(39, 825)  # 82.5%
        await client.write_register(40, 650)  # 65.0 C

        assert local_twin.sp_nmpc_enabled is True
        assert local_twin.sp_cooling_valve_pct == pytest.approx(82.5, abs=0.1)
        assert local_twin.sp_feed_temp_c == pytest.approx(65.0, abs=0.1)
    finally:
        client.close()
        await server.stop()


def test_fastapi_series_parallel_control_endpoints():
    client = TestClient(app)
    try:
        r_status = client.get("/api/series-parallel/control/status")
        assert r_status.status_code == 200
        assert "ekf" in r_status.json()
        assert "nmpc" in r_status.json()

        r_cfg = client.post(
            "/api/series-parallel/control/config",
            json={"nmpc_enabled": True, "feed_temp_c": 70.0},
        )
        assert r_cfg.status_code == 200
        assert r_cfg.json()["nmpc"]["enabled"] is True
        assert r_cfg.json()["telemetry"]["feed_temp_c"] == pytest.approx(70.0)

        r_cl = client.post("/api/series-parallel/closed-loop", json={"steps": 50})
        assert r_cl.status_code == 200
        body = r_cl.json()
        assert len(body["closed_loop_trajectory"]) == 50
        assert body["closed_loop_has_runaway"] is False
        assert body["closed_loop_peak_temp_c"] < body["open_loop_peak_temp_c"]

        r_reset = client.post("/api/series-parallel/control/reset")
        assert r_reset.status_code == 200
        assert r_reset.json()["nmpc"]["enabled"] is False

        r_ui = client.get("/")
        assert r_ui.status_code == 200
        assert "btnToggleSpNmpc" in r_ui.text
        assert "triggerSeriesParallelClosedLoop" in r_ui.text
    finally:
        twin.reset_series_parallel_control()
