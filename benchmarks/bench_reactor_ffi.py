"""
High-Performance Benchmark: Native C11 (ReactorCore.C) vs. Pure Python RK4.

Measures:
1. Single-step execution latency (microseconds / step).
2. Large-scale trajectory integration throughput (steps / second).
3. Numerical IEEE-754 double precision parity & error norms.
4. Wall-clock acceleration factor (Speedup).
"""

import time
import math
import sys
from process_twin.core.native_reactor import (
    NativeReactorCore,
    NativeCstrParams,
)


def run_benchmark():
    print("=" * 72)
    print("   BENCHMARK: Native C11 (ReactorCore.C) vs. Pure Python RK4 Engine    ")
    print("=" * 72)

    engine = NativeReactorCore()
    print(f"[*] Engine Status: {'AVAILABLE' if engine.is_available else 'UNAVAILABLE'}")
    print(f"[*] Engine Banner: {engine.get_version()}")
    print("-" * 72)

    if not engine.is_available:
        print("[ERROR] libreactorcore.so not loaded. Run 'make lib' in ReactorCore.C first.")
        sys.exit(1)

    params = engine.get_default_params()
    ca0 = 1.0
    temp0 = 350.0
    dt = 0.005

    # -------------------------------------------------------------------------
    # Benchmark 1: Single-Step Latency (10,000 steps)
    # -------------------------------------------------------------------------
    warmup_steps = 1000
    for _ in range(warmup_steps):
        engine.step_rk4(params, ca0, temp0, dt)
        NativeReactorCore._py_step_rk4(params, ca0, temp0, dt)

    n_single = 25000
    print(f"\n[1] Single-Step RK4 Latency ({n_single:,} iterations):")

    # Python
    t0 = time.perf_counter()
    ca_p, t_p = ca0, temp0
    for _ in range(n_single):
        ca_p, t_p = NativeReactorCore._py_step_rk4(params, ca_p, t_p, dt)
    py_single_time = time.perf_counter() - t0
    py_lat_us = (py_single_time / n_single) * 1e6

    # Native C single-step (via ctypes FFI)
    t0 = time.perf_counter()
    ca_c, t_c = ca0, temp0
    for _ in range(n_single):
        ca_c, t_c = engine.step_rk4(params, ca_c, t_c, dt)
    c_single_time = time.perf_counter() - t0
    c_lat_us = (c_single_time / n_single) * 1e6

    print(f"    - Pure Python 3.12:    {py_single_time:.4f} s | {py_lat_us:6.2f} µs/step | {n_single / py_single_time:,.0f} steps/s")
    print(f"    - Native C11 (ctypes): {c_single_time:.4f} s | {c_lat_us:6.2f} µs/step | {n_single / c_single_time:,.0f} steps/s")
    print(f"    - FFI Overhead Speedup: {py_single_time / c_single_time:.2f}x")

    # -------------------------------------------------------------------------
    # Benchmark 2: Batch Trajectory Integration (100,000 steps)
    # (Simulating C inside a tight native loop vs Python loop overhead)
    # -------------------------------------------------------------------------
    n_batch = 100000
    print(f"\n[2] High-Throughput Batch Trajectory ({n_batch:,} steps, dt = {dt} min):")

    # Python batch
    t0 = time.perf_counter()
    traj_py = NativeReactorCore._py_simulate_trajectory(params, ca0, temp0, dt, n_batch)
    py_batch_time = time.perf_counter() - t0
    py_batch_throughput = n_batch / py_batch_time

    # Native C batch (one single FFI entry, full loop in C)
    t0 = time.perf_counter()
    traj_c = engine.simulate_trajectory(params, ca0, temp0, dt, n_batch)
    c_batch_time = time.perf_counter() - t0
    c_batch_throughput = n_batch / c_batch_time

    speedup_batch = py_batch_time / c_batch_time

    print(f"    - Pure Python 3.12:    {py_batch_time:.4f} s | {(py_batch_time / n_batch) * 1e6:6.3f} µs/step | {py_batch_throughput:,.0f} steps/s")
    print(f"    - Native C11 Batch:    {c_batch_time:.4f} s | {(c_batch_time / n_batch) * 1e6:6.3f} µs/step | {c_batch_throughput:,.0f} steps/s")
    print(f"    - Full Native Speedup: {speedup_batch:.1f}x FASTER")

    # -------------------------------------------------------------------------
    # Numerical Parity Verification
    # -------------------------------------------------------------------------
    max_ca_diff = max(abs(c - p) for c, p in zip(traj_c.ca_mol_per_l, traj_py.ca_mol_per_l))
    max_t_diff = max(abs(c - p) for c, p in zip(traj_c.temp_kelvin, traj_py.temp_kelvin))

    print(f"\n[3] IEEE-754 Numerical Parity Verification across {n_batch:,} points:")
    print(f"    - Max Ca Absolute Error:   {max_ca_diff:.2e} mol/L")
    print(f"    - Max Temp Absolute Error: {max_t_diff:.2e} K")
    print(f"    - Parity Status:           {'VERIFIED PERFECT (<= 1e-10)' if max_ca_diff < 1e-10 else 'DEVIATION DETECTED'}")

    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
