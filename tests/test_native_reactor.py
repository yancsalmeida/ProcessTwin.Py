"""
Automated Unit Tests for Native C-ABI FFI Bridge (ReactorCore.C -> ProcessTwin.Py).
Validates:
1. Native library loading and version probe.
2. Double-precision IEEE-754 numerical parity between C11 RK4 and Python reference.
3. Batch trajectory solver parity over multi-minute non-linear CSTR dynamics.
4. Detailed telemetry metrics (conversion %, reaction rate, thermal runaway interlocks).
5. Graceful fallback behavior when compiled native library is absent.
"""

import pytest
import math
from process_twin.core.native_reactor import (
    NativeReactorCore,
    NativeCstrParams,
    NativeCstrState,
    TrajectoryResult,
    NativeCstrSeriesParallelParams,
    NativeCstrSeriesParallelState,
    SeriesParallelTrajectoryResult,
    SeriesParallelBatchTrajectoryResult,
)


@pytest.fixture
def native_engine() -> NativeReactorCore:
    return NativeReactorCore()


def test_native_library_loaded_and_version(native_engine: NativeReactorCore):
    assert native_engine.is_available is True
    version = native_engine.get_version()
    assert "ReactorCore.C" in version
    assert "1.1.0-native-ffi" in version


def test_default_parameters_parity(native_engine: NativeReactorCore):
    params = native_engine.get_default_params()
    assert params.volumetric_flow_l_per_min == 100.0
    assert params.volume_liters == 100.0
    assert params.ca_feed_mol_per_l == 1.0
    assert params.t_feed_kelvin == 350.0
    assert params.t_coolant_kelvin == 300.0


def test_reaction_rate_parity(native_engine: NativeReactorCore):
    params = native_engine.get_default_params()
    rate_native = native_engine.reaction_rate(params, ca=1.0, temp_k=350.0)
    rate_py = NativeReactorCore._py_reaction_rate(params, ca=1.0, temp_k=350.0)
    assert abs(rate_native - rate_py) < 1e-14


def test_single_step_rk4_analytical_parity(native_engine: NativeReactorCore):
    params = native_engine.get_default_params()
    ca_init = 1.0
    temp_init = 350.0
    dt = 0.005  # 0.3 seconds

    # Step in Native C
    ca_native, temp_native = native_engine.step_rk4(params, ca_init, temp_init, dt)

    # Step in Python Reference
    ca_py, temp_py = NativeReactorCore._py_step_rk4(params, ca_init, temp_init, dt)

    # Parity check: difference must be near floating-point epsilon (< 1e-12)
    assert abs(ca_native - ca_py) < 1e-12, f"Ca mismatch: {ca_native} vs {ca_py}"
    assert abs(temp_native - temp_py) < 1e-12, f"Temp mismatch: {temp_native} vs {temp_py}"

    # Physical sanity
    assert ca_native < ca_init, "Reactant concentration must decrease"
    assert temp_native > temp_init, "Exothermic reaction must release heat"


def test_multi_step_trajectory_parity(native_engine: NativeReactorCore):
    params = native_engine.get_default_params()
    ca0 = 1.0
    temp0 = 350.0
    dt = 0.005
    steps = 200  # 1.0 minute simulated time

    # Run C batch solver
    res_native = native_engine.simulate_trajectory(params, ca0, temp0, dt, steps)
    # Run Python reference solver
    res_py = NativeReactorCore._py_simulate_trajectory(params, ca0, temp0, dt, steps)

    assert len(res_native.times_min) == steps + 1
    assert len(res_native.ca_mol_per_l) == steps + 1
    assert len(res_native.temp_kelvin) == steps + 1

    # Check analytical parity across every single step
    max_ca_err = max(abs(cn - cp) for cn, cp in zip(res_native.ca_mol_per_l, res_py.ca_mol_per_l))
    max_temp_err = max(abs(tn - tp) for tn, tp in zip(res_native.temp_kelvin, res_py.temp_kelvin))

    assert max_ca_err < 1e-10, f"Max Ca error exceeded: {max_ca_err}"
    assert max_temp_err < 1e-10, f"Max Temp error exceeded: {max_temp_err}"


def test_step_detailed_and_runaway(native_engine: NativeReactorCore):
    params = native_engine.get_default_params()
    ca = 1.0
    temp = 350.0
    dt = 0.005

    state = native_engine.step_detailed(params, ca, temp, dt, runaway_threshold_k=395.0)
    assert state.ca_mol_per_l < 1.0
    assert state.temp_kelvin > 350.0
    assert state.conversion_pct > 0.0
    assert state.reaction_rate_mol_per_l_min > 0.0
    assert state.is_runaway is False

    # Simulate near runaway conditions (very hot feed / low cooling)
    hot_params = NativeCstrParams(
        ua_j_per_min_k=0.0,  # Cooling failure
        t_feed_kelvin=390.0,
    )
    runaway_state = native_engine.step_detailed(
        hot_params, ca=0.5, temp_k=396.0, dt=dt, runaway_threshold_k=395.0
    )
    assert runaway_state.is_runaway is True


def test_graceful_fallback_when_binary_missing():
    # Force loading a nonexistent path
    fallback_engine = NativeReactorCore(lib_path="/nonexistent/libreactorcore.so")
    assert fallback_engine.is_available is False
    assert "Fallback" in fallback_engine.get_version()

    params = fallback_engine.get_default_params()
    ca, temp = fallback_engine.step_rk4(params, 1.0, 350.0, 0.005)
    assert ca < 1.0
    assert temp > 350.0

    traj = fallback_engine.simulate_trajectory(params, 1.0, 350.0, 0.005, 10)
    assert len(traj.times_min) == 11


def test_simd_hardware_detection(native_engine: NativeReactorCore):
    has_simd = native_engine.has_simd()
    assert isinstance(has_simd, bool)


def test_batch_rk4_parity(native_engine: NativeReactorCore):
    params_list = [native_engine.get_default_params() for _ in range(4)]
    ca_list = [1.0, 0.8, 0.6, 0.4]
    temp_list = [350.0, 355.0, 360.0, 365.0]
    dt = 0.005

    ca_out_native, temp_out_native = native_engine.step_batch_rk4(params_list, ca_list, temp_list, dt)
    ca_out_py, temp_out_py = NativeReactorCore._py_step_batch_rk4(params_list, ca_list, temp_list, dt)

    assert len(ca_out_native) == 4
    assert len(temp_out_native) == 4
    for cn, cp in zip(ca_out_native, ca_out_py):
        assert abs(cn - cp) < 1e-12
    for tn, tp in zip(temp_out_native, temp_out_py):
        assert abs(tn - tp) < 1e-12


def test_batch_detailed_and_runaway(native_engine: NativeReactorCore):
    params_list = [native_engine.get_default_params() for _ in range(4)]
    # Make 4th reactor runaway
    params_list[3] = NativeCstrParams(ua_j_per_min_k=0.0, t_feed_kelvin=390.0)
    ca_list = [1.0, 0.8, 0.6, 0.5]
    temp_list = [350.0, 355.0, 360.0, 396.0]
    dt = 0.005

    states = native_engine.step_batch_detailed(params_list, ca_list, temp_list, dt, runaway_threshold_k=395.0)
    assert len(states) == 4
    assert states[0].is_runaway is False
    assert states[3].is_runaway is True
    assert states[3].temp_kelvin >= 395.0


def test_simulate_batch_trajectory_parity(native_engine: NativeReactorCore):
    params_list = [native_engine.get_default_params() for _ in range(4)]
    ca_list = [1.0, 0.8, 0.6, 0.4]
    temp_list = [350.0, 355.0, 360.0, 365.0]
    dt = 0.005
    steps = 50

    res_native = native_engine.simulate_batch_trajectory(params_list, ca_list, temp_list, dt, steps)
    res_py = NativeReactorCore._py_simulate_batch_trajectory(params_list, ca_list, temp_list, dt, steps)

    assert len(res_native.times_min) == steps + 1
    assert len(res_native.ca_matrix) == steps + 1
    assert len(res_native.temp_matrix) == steps + 1
    assert len(res_native.ca_matrix[0]) == 4
    assert len(res_native.temp_matrix[0]) == 4

    for s in range(steps + 1):
        for r in range(4):
            assert abs(res_native.ca_matrix[s][r] - res_py.ca_matrix[s][r]) < 1e-10
            assert abs(res_native.temp_matrix[s][r] - res_py.temp_matrix[s][r]) < 1e-10


def test_series_parallel_default_parameters_and_simd_level(native_engine: NativeReactorCore):
    simd_level = native_engine.get_simd_level()
    assert simd_level in (0, 256, 512)

    params = native_engine.get_series_parallel_default_params()
    assert params.volume_liters == 100.0
    assert params.volumetric_flow_l_per_min == 100.0
    assert params.k0_1_per_min == 7.08e10
    assert params.k0_2_per_min == 3.50e9
    assert params.k0_3_per_min == 1.20e9
    assert params.ca_feed_mol_per_l == 1.0
    assert params.cb_feed_mol_per_l == 0.0
    assert params.t_feed_kelvin == 350.0
    assert params.jacket_volume_liters == 25.0
    assert params.jacket_flow_l_per_min == 50.0
    assert params.jacket_t_in_kelvin == 295.0


def test_series_parallel_reaction_rates_parity(native_engine: NativeReactorCore):
    params = native_engine.get_series_parallel_default_params()
    r1_n, r2_n, r3_n = native_engine.series_parallel_rates(params, ca=1.0, cb=0.2, temp_k=350.0)
    r1_p, r2_p, r3_p = NativeReactorCore._py_series_parallel_rates(params, ca=1.0, cb=0.2, temp_k=350.0)

    assert abs(r1_n - r1_p) < 1e-14
    assert abs(r2_n - r2_p) < 1e-14
    assert abs(r3_n - r3_p) < 1e-14
    assert r1_n > 0.0
    assert r2_n > 0.0
    assert r3_n > 0.0


def test_series_parallel_single_step_rk4_parity(native_engine: NativeReactorCore):
    params = native_engine.get_series_parallel_default_params()
    ca, cb, cc, cd, temp, temp_j = 1.0, 0.0, 0.0, 0.0, 350.0, 295.0
    dt = 0.005

    # Step in C
    n_ca, n_cb, n_cc, n_cd, n_temp, n_temp_j = native_engine.step_series_parallel_rk4(
        params, ca, cb, cc, cd, temp, temp_j, dt
    )

    # Step in Python reference
    p_ca, p_cb, p_cc, p_cd, p_temp, p_temp_j = NativeReactorCore._py_series_parallel_step_rk4(
        params, ca, cb, cc, cd, temp, temp_j, dt
    )

    assert abs(n_ca - p_ca) < 1e-12
    assert abs(n_cb - p_cb) < 1e-12
    assert abs(n_cc - p_cc) < 1e-12
    assert abs(n_cd - p_cd) < 1e-12
    assert abs(n_temp - p_temp) < 1e-12
    assert abs(n_temp_j - p_temp_j) < 1e-12

    # Physical sanity
    assert n_ca < 1.0, "Reactant A must be consumed"
    assert n_cb > 0.0, "Product B must be formed"
    assert n_temp > 350.0, "Exothermic reaction must increase reactor temperature"
    assert n_temp_j > 295.0, "Cooling jacket must absorb heat from reactor"


def test_series_parallel_step_detailed_and_runaway(native_engine: NativeReactorCore):
    params = native_engine.get_series_parallel_default_params()
    ca, cb, cc, cd, temp, temp_j = 1.0, 0.0, 0.0, 0.0, 350.0, 295.0
    dt = 0.005

    st = native_engine.step_series_parallel_detailed(
        params, ca, cb, cc, cd, temp, temp_j, dt, runaway_threshold_k=400.0
    )

    assert st.conversion_a_pct > 0.0
    assert st.yield_b_pct > 0.0
    assert st.selectivity_b_pct > 0.0
    assert st.heat_generated_watts > 0.0
    assert st.heat_removed_watts > 0.0
    assert st.is_thermal_runaway is False

    # Simulate thermal runaway: low coolant flow, zero jacket cooling, hot feed
    runaway_params = NativeCstrSeriesParallelParams(
        ua_j_per_min_k=0.0,
        jacket_flow_l_per_min=0.0,
        t_feed_kelvin=395.0,
    )
    st_runaway = native_engine.step_series_parallel_detailed(
        runaway_params, ca=0.8, cb=0.1, cc=0.05, cd=0.05, temp_k=402.0, temp_j_k=300.0, dt=dt, runaway_threshold_k=400.0
    )
    assert st_runaway.is_thermal_runaway is True
    assert st_runaway.temp_kelvin >= 400.0


def test_series_parallel_batch_rk4_parity(native_engine: NativeReactorCore):
    params = native_engine.get_series_parallel_default_params()
    n = 4
    ca_list = [1.0, 0.8, 0.5, 0.2]
    cb_list = [0.0, 0.1, 0.2, 0.3]
    cc_list = [0.0, 0.02, 0.05, 0.1]
    cd_list = [0.0, 0.01, 0.03, 0.05]
    temp_list = [350.0, 355.0, 360.0, 365.0]
    temp_j_list = [295.0, 298.0, 300.0, 305.0]
    dt = 0.005

    n_out = native_engine.step_series_parallel_batch_rk4(
        params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt
    )
    p_out = NativeReactorCore._py_series_parallel_step_batch_rk4(
        params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt
    )

    for i in range(6):
        assert len(n_out[i]) == n
        for val_n, val_p in zip(n_out[i], p_out[i]):
            assert abs(val_n - val_p) < 1e-12


def test_series_parallel_batch_trajectory_parity(native_engine: NativeReactorCore):
    params = native_engine.get_series_parallel_default_params()
    n = 4
    ca_list = [1.0, 0.8, 0.6, 0.4]
    cb_list = [0.0, 0.05, 0.1, 0.15]
    cc_list = [0.0, 0.01, 0.02, 0.03]
    cd_list = [0.0, 0.01, 0.02, 0.03]
    temp_list = [350.0, 352.0, 355.0, 360.0]
    temp_j_list = [295.0, 296.0, 298.0, 300.0]
    dt = 0.005
    steps = 40

    res_n = native_engine.simulate_series_parallel_batch_trajectory(
        params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt, steps
    )
    res_p = NativeReactorCore._py_series_parallel_simulate_batch_trajectory(
        params, ca_list, cb_list, cc_list, cd_list, temp_list, temp_j_list, dt, steps
    )

    assert len(res_n.times_min) == steps + 1
    for s in range(steps + 1):
        for r in range(n):
            assert abs(res_n.ca_matrix[s][r] - res_p.ca_matrix[s][r]) < 1e-10
            assert abs(res_n.cb_matrix[s][r] - res_p.cb_matrix[s][r]) < 1e-10
            assert abs(res_n.cc_matrix[s][r] - res_p.cc_matrix[s][r]) < 1e-10
            assert abs(res_n.cd_matrix[s][r] - res_p.cd_matrix[s][r]) < 1e-10
            assert abs(res_n.temp_matrix[s][r] - res_p.temp_matrix[s][r]) < 1e-10
            assert abs(res_n.jacket_temp_matrix[s][r] - res_p.jacket_temp_matrix[s][r]) < 1e-10


