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
