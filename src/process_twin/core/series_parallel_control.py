"""
Advanced Estimation & Economic Optimal Control for the 6-State Series-Parallel CSTR:
    A -> B -> C,  A -> D  (with dynamic jacket temperature Tj)

State Vector x = [Ca, Cb, Cc, Cd, T_K, Tj_K]^T:
  1. SeriesParallelExtendedKalmanFilter:
     - 6-state Continuous-Discrete EKF backed by NativeReactorCore (C11/RK4 + fallback)
     - Discrete Jacobian Phi computed via central finite differences over exact RK4 map
     - Multi-sensor measurement update y = [T_K, Tj_K]^T (H in R^(2x6))
     - Joseph stabilized covariance update + 2-DOF Chi-square NIS sensor health check

  2. SeriesParallelNonlinearMpc:
     - Economic / Selectivity-Maximizing Receding-Horizon NMPC
     - Maximizes desired intermediate product Cb & selectivity while penalizing C/D degradation
     - Enforces thermal runaway safety constraint (T <= max_safe_temp_c) via soft-margin + barrier
     - Manipulates cooling jacket valve opening u in [min_valve_pct, max_valve_pct] with slew limits
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from process_twin.core.native_reactor import (
    NativeCstrSeriesParallelParams,
    NativeCstrSeriesParallelState,
    NativeReactorCore,
)


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiplies two dense matrices A (m x k) and B (k x n)."""
    rows = len(a)
    inner = len(b)
    cols = len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        row_out = out[i]
        for k in range(inner):
            aik = row_a[k]
            if aik == 0.0:
                continue
            row_b = b[k]
            for j in range(cols):
                row_out[j] += aik * row_b[j]
    return out


def _mat_mul_transpose(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Computes A @ B^T for dense matrices A (m x k) and B (n x k)."""
    rows = len(a)
    cols = len(b)
    inner = len(a[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        for j in range(cols):
            row_b = b[j]
            s = 0.0
            for k in range(inner):
                s += row_a[k] * row_b[k]
            out[i][j] = s
    return out


def _inv_2x2(s: List[List[float]]) -> List[List[float]]:
    """Analytical stabilized inverse of a 2x2 symmetric matrix S."""
    s00, s01 = s[0][0], s[0][1]
    s10, s11 = s[1][0], s[1][1]
    det = s00 * s11 - s01 * s10
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0.0 else -1e-12
    inv_det = 1.0 / det
    return [
        [s11 * inv_det, -s01 * inv_det],
        [-s10 * inv_det, s00 * inv_det],
    ]


def build_series_parallel_params(
    native_core: NativeReactorCore,
    cooling_valve_pct: float = 55.0,
    feed_temp_c: float = 51.85,
    jacket_inlet_temp_c: float = 21.85,
    ca_feed_mol_per_l: float = 1.0,
    volumetric_flow_l_per_min: float = 100.0,
    reactor_volume_liters: float = 100.0,
    ua_max_j_per_min_k: float = 600000.0,
    cooling_ua_override_w_per_k: Optional[float] = None,
) -> NativeCstrSeriesParallelParams:
    """
    Constructs NativeCstrSeriesParallelParams mapping control valve opening (0-100%)
    and boundary temperatures to jacket flow and overall heat-transfer UA.
    """
    params = native_core.get_series_parallel_default_params()
    valve_frac = max(0.0, min(100.0, float(cooling_valve_pct))) / 100.0

    params.volumetric_flow_l_per_min = max(1.0, float(volumetric_flow_l_per_min))
    params.volume_liters = max(1.0, float(reactor_volume_liters))
    params.ca_feed_mol_per_l = max(0.0, float(ca_feed_mol_per_l))
    params.t_feed_kelvin = float(feed_temp_c) + 273.15
    params.jacket_t_in_kelvin = float(jacket_inlet_temp_c) + 273.15

    if cooling_ua_override_w_per_k is not None:
        params.ua_j_per_min_k = max(0.0, float(cooling_ua_override_w_per_k) * 60.0)
    else:
        params.ua_j_per_min_k = (0.04 + 0.96 * valve_frac) * max(1000.0, float(ua_max_j_per_min_k))

    params.jacket_flow_l_per_min = 15.0 + 225.0 * valve_frac
    return params


def step_series_parallel_substepped(
    native_core: NativeReactorCore,
    params: NativeCstrSeriesParallelParams,
    state: Tuple[float, float, float, float, float, float],
    dt_min: float,
    max_substep_min: float = 0.02,
) -> Tuple[float, float, float, float, float, float]:
    """Integrates the 6-state ODE via NativeReactorCore RK4 with automatic substepping."""
    if dt_min <= 0.0:
        return state
    n_sub = max(1, int(math.ceil(dt_min / max_substep_min)))
    h = dt_min / n_sub
    ca, cb, cc, cd, temp_k, jacket_k = state
    for _ in range(n_sub):
        ca, cb, cc, cd, temp_k, jacket_k = native_core.step_series_parallel_rk4(
            params, ca, cb, cc, cd, temp_k, jacket_k, dt_min=h
        )
    return (ca, cb, cc, cd, temp_k, jacket_k)


def compute_series_parallel_kpis(
    ca: float,
    cb: float,
    ca_feed: float,
) -> Tuple[float, float, float]:
    """Returns (conversion_a_pct, yield_b_pct, selectivity_b_pct) bounded safely in [0, 100]."""
    if ca_feed <= 1e-9:
        return (0.0, 0.0, 0.0)
    reacted = max(0.0, ca_feed - ca)
    conversion_a = max(0.0, min(100.0, (reacted / ca_feed) * 100.0))
    yield_b = max(0.0, min(100.0, (max(0.0, cb) / ca_feed) * 100.0))
    selectivity_b = max(0.0, min(100.0, (max(0.0, cb) / reacted) * 100.0)) if reacted > 1e-6 else 0.0
    return (conversion_a, yield_b, selectivity_b)


@dataclass
class SpEkfEstimate:
    """State estimate & diagnostics from the 6-state Series-Parallel EKF."""

    ca_mol_per_l: float
    cb_mol_per_l: float
    cc_mol_per_l: float
    cd_mol_per_l: float
    temp_c: float
    jacket_temp_c: float
    conversion_a_pct: float
    yield_b_pct: float
    selectivity_b_pct: float
    temp_residual_c: float
    jacket_residual_c: float
    nis: float
    sensor_healthy: bool
    cov_trace: float


class SeriesParallelExtendedKalmanFilter:
    """
    6-State Continuous-Discrete Extended Kalman Filter (EKF) for the Series-Parallel CSTR.
    Estimates unmeasured species [Ca, Cb, Cc, Cd] and filtered [T, Tj] solely from
    noisy dual-RTD temperature measurements (y = [T_K, Tj_K]^T).
    """

    STATE_DIM = 6
    MEAS_DIM = 2
    NIS_99_CHI2_2DOF = 9.21  # Chi-square(df=2, alpha=0.01)

    def __init__(self, native_reactor: Optional[NativeReactorCore] = None) -> None:
        self.native_reactor = native_reactor or NativeReactorCore()
        self.q_diag = [0.005, 0.005, 0.004, 0.003, 0.12, 0.08]
        self.r_temp_k2 = 0.04      # sigma_T = 0.20 K
        self.r_jacket_k2 = 0.0225  # sigma_Tj = 0.15 K
        self.state: List[float] = [0.50, 0.25, 0.20, 0.05, 333.15, 310.15]
        self.P: List[List[float]] = [[0.0] * 6 for _ in range(6)]
        self.last_temp_residual_k = 0.0
        self.last_jacket_residual_k = 0.0
        self.last_nis = 0.0
        self.reset()

    def reset(
        self,
        initial_ca: float = 0.50,
        initial_cb: float = 0.25,
        initial_cc: float = 0.20,
        initial_cd: float = 0.05,
        initial_temp_c: float = 60.0,
        initial_jacket_temp_c: float = 36.0,
    ) -> None:
        """Resets 6-state mean and initial error covariance matrix P_0."""
        self.state = [
            float(initial_ca),
            float(initial_cb),
            float(initial_cc),
            float(initial_cd),
            float(initial_temp_c) + 273.15,
            float(initial_jacket_temp_c) + 273.15,
        ]
        init_vars = [0.15, 0.12, 0.10, 0.05, 4.0, 2.0]
        self.P = [[init_vars[i] if i == j else 0.0 for j in range(6)] for i in range(6)]
        self.last_temp_residual_k = 0.0
        self.last_jacket_residual_k = 0.0
        self.last_nis = 0.0

    def _clamp_state(self, ca_feed: float) -> None:
        max_c = max(2.0, ca_feed * 2.0)
        for i in range(4):
            self.state[i] = max(0.0, min(max_c, self.state[i]))
        self.state[4] = max(250.0, min(800.0, self.state[4]))
        self.state[5] = max(240.0, min(650.0, self.state[5]))

    def _discrete_jacobian(
        self,
        params: NativeCstrSeriesParallelParams,
        x0: Tuple[float, float, float, float, float, float],
        dt_min: float,
    ) -> List[List[float]]:
        """Computes exact discrete-time Jacobian Phi = d f_RK4(x, u) / d x in R^(6x6)."""
        eps = [1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.05]
        phi = [[0.0] * 6 for _ in range(6)]
        for col in range(6):
            delta = eps[col]
            x_plus = list(x0)
            x_minus = list(x0)
            x_plus[col] += delta
            x_minus[col] = max(0.0 if col < 4 else 220.0, x_minus[col] - delta)
            actual_span = x_plus[col] - x_minus[col]
            if actual_span < 1e-12:
                phi[col][col] = 1.0
                continue
            f_plus = step_series_parallel_substepped(
                self.native_reactor, params, tuple(x_plus), dt_min  # type: ignore[arg-type]
            )
            f_minus = step_series_parallel_substepped(
                self.native_reactor, params, tuple(x_minus), dt_min  # type: ignore[arg-type]
            )
            inv_span = 1.0 / actual_span
            for row in range(6):
                phi[row][col] = (f_plus[row] - f_minus[row]) * inv_span
        return phi

    def predict(self, dt_min: float, params: NativeCstrSeriesParallelParams) -> None:
        """EKF Time-Update (Predict) step using substepped RK4 and discrete Jacobian Phi."""
        if dt_min <= 0.0:
            return
        x0 = tuple(self.state)
        phi = self._discrete_jacobian(params, x0, dt_min)  # type: ignore[arg-type]
        x_pred = step_series_parallel_substepped(
            self.native_reactor, params, x0, dt_min  # type: ignore[arg-type]
        )
        self.state = list(x_pred)
        self._clamp_state(params.ca_feed_mol_per_l)

        # P_prior = Phi * P * Phi^T + Q * dt
        p_prior = _mat_mul_transpose(_mat_mul(phi, self.P), phi)
        for i in range(6):
            p_prior[i][i] = max(1e-8, p_prior[i][i] + self.q_diag[i] * dt_min)
            for j in range(i + 1, 6):
                sym = 0.5 * (p_prior[i][j] + p_prior[j][i])
                p_prior[i][j] = sym
                p_prior[j][i] = sym
        self.P = p_prior

    def update(
        self,
        measured_temp_c: float,
        measured_jacket_temp_c: float,
        ca_feed: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        EKF Measurement-Update using dual temperature transmitters y = [T_K, Tj_K]^T.
        H selects states 4 and 5: H = [[0,0,0,0,1,0], [0,0,0,0,0,1]].
        Returns (temp_residual_k, jacket_residual_k, nis).
        """
        y_t = float(measured_temp_c) + 273.15
        y_j = float(measured_jacket_temp_c) + 273.15
        nu0 = y_t - self.state[4]
        nu1 = y_j - self.state[5]
        self.last_temp_residual_k = nu0
        self.last_jacket_residual_k = nu1

        # S = H * P * H^T + R (2x2 block of P rows/cols 4..5 + R)
        S = [
            [self.P[4][4] + self.r_temp_k2, self.P[4][5]],
            [self.P[5][4], self.P[5][5] + self.r_jacket_k2],
        ]
        S_inv = _inv_2x2(S)

        nis = (
            nu0 * (S_inv[0][0] * nu0 + S_inv[0][1] * nu1)
            + nu1 * (S_inv[1][0] * nu0 + S_inv[1][1] * nu1)
        )
        nis = max(0.0, nis)
        self.last_nis = nis

        # Kalman Gain K = P * H^T * S^-1 in R^(6x2)
        K = [[0.0, 0.0] for _ in range(6)]
        for i in range(6):
            p_i4 = self.P[i][4]
            p_i5 = self.P[i][5]
            K[i][0] = p_i4 * S_inv[0][0] + p_i5 * S_inv[1][0]
            K[i][1] = p_i4 * S_inv[0][1] + p_i5 * S_inv[1][1]

        # State update: x = x + K * nu
        for i in range(6):
            self.state[i] += K[i][0] * nu0 + K[i][1] * nu1
        self._clamp_state(ca_feed)

        # Joseph stabilized covariance update:
        # P_post = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        A = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        for i in range(6):
            A[i][4] -= K[i][0]
            A[i][5] -= K[i][1]

        p_post = _mat_mul_transpose(_mat_mul(A, self.P), A)
        for i in range(6):
            ki0, ki1 = K[i][0], K[i][1]
            for j in range(6):
                p_post[i][j] += ki0 * self.r_temp_k2 * K[j][0] + ki1 * self.r_jacket_k2 * K[j][1]

        for i in range(6):
            p_post[i][i] = max(1e-8, p_post[i][i])
            for j in range(i + 1, 6):
                sym = 0.5 * (p_post[i][j] + p_post[j][i])
                p_post[i][j] = sym
                p_post[j][i] = sym
        self.P = p_post

        return (nu0, nu1, nis)

    def step(
        self,
        dt_min: float,
        params: NativeCstrSeriesParallelParams,
        measured_temp_c: float,
        measured_jacket_temp_c: float,
    ) -> SpEkfEstimate:
        """Executes full 6-state Predict + Update cycle and returns SpEkfEstimate."""
        self.predict(dt_min, params)
        nu0, nu1, nis = self.update(
            measured_temp_c=measured_temp_c,
            measured_jacket_temp_c=measured_jacket_temp_c,
            ca_feed=params.ca_feed_mol_per_l,
        )
        ca, cb, cc, cd, t_k, tj_k = self.state
        conv_a, yield_b, sel_b = compute_series_parallel_kpis(
            ca=ca, cb=cb, ca_feed=params.ca_feed_mol_per_l
        )
        cov_trace = sum(self.P[i][i] for i in range(6))

        return SpEkfEstimate(
            ca_mol_per_l=round(ca, 4),
            cb_mol_per_l=round(cb, 4),
            cc_mol_per_l=round(cc, 4),
            cd_mol_per_l=round(cd, 4),
            temp_c=round(t_k - 273.15, 2),
            jacket_temp_c=round(tj_k - 273.15, 2),
            conversion_a_pct=round(conv_a, 2),
            yield_b_pct=round(yield_b, 2),
            selectivity_b_pct=round(sel_b, 2),
            temp_residual_c=round(nu0, 4),
            jacket_residual_c=round(nu1, 4),
            nis=round(nis, 3),
            sensor_healthy=(nis < self.NIS_99_CHI2_2DOF),
            cov_trace=round(cov_trace, 6),
        )


@dataclass
class SpNmpcConfig:
    """Tuning and safety constraint parameters for the Economic Series-Parallel NMPC."""

    prediction_horizon: int = 12
    control_horizon: int = 4
    dt_step_min: float = 0.08             # 0.96 min total lookahead (~1 residence time tau)
    max_safe_temp_c: float = 126.85       # 400.0 K runaway trip threshold
    temp_safety_margin_c: float = 14.0    # Preemptive soft barrier starts at T_safe - 14C
    target_cb_mol_per_l: Optional[float] = None
    weight_cb_economic: float = 42.0      # Economic incentive maximizing B (mol/L)
    weight_cb_tracking: float = 12.0      # Optional setpoint tracking term when target_cb set
    weight_byproduct: float = 4.0         # Penalizes C & D degradation byproducts
    weight_utility_effort: float = 0.35   # Small utility pumping cost (u / 100)^2
    weight_delta_u: float = 0.002         # Smoothness penalty on delta_u
    weight_safety_soft: float = 1.5       # Preemptive soft temperature barrier weight
    weight_safety_hard: float = 45.0      # Steep penalty above max_safe_temp_c
    min_valve_pct: float = 0.0
    max_valve_pct: float = 100.0
    max_delta_u_pct: float = 25.0         # Max slew per NMPC step
    max_iterations: int = 6
    initial_step_size: float = 2.0
    tolerance: float = 1e-3


@dataclass
class SpNmpcResult:
    """Output of a single receding-horizon Economic NMPC optimization step."""

    optimal_cooling_valve_pct: float
    predicted_cb_trajectory: List[float]
    predicted_temp_c_trajectory: List[float]
    predicted_selectivity_trajectory: List[float]
    planned_control_moves: List[float]
    cost: float
    iterations: int
    solve_time_ms: float
    converged: bool


class SeriesParallelNonlinearMpc:
    """
    Economic Receding-Horizon NMPC for the 6-State Series-Parallel CSTR.
    Optimizes cooling valve opening u in [0, 100%] to maximize intermediate
    product Cb & selectivity while rejecting disturbances and respecting
    hard/soft thermal runaway constraints (T <= max_safe_temp_c).
    """

    def __init__(
        self,
        config: Optional[SpNmpcConfig] = None,
        native_reactor: Optional[NativeReactorCore] = None,
    ) -> None:
        self.config = config or SpNmpcConfig()
        self.native_reactor = native_reactor or NativeReactorCore()
        self._warm_moves: List[float] = [58.0] * self.config.control_horizon
        self.last_applied_u: float = 58.0
        self.last_result: Optional[SpNmpcResult] = None

    def reset(self, initial_u: float = 58.0) -> None:
        self.last_applied_u = initial_u
        self._warm_moves = [initial_u] * self.config.control_horizon
        self.last_result = None

    def update_config(
        self,
        max_safe_temp_c: Optional[float] = None,
        target_cb_mol_per_l: Optional[float] = None,
        clear_target_cb: bool = False,
    ) -> None:
        if max_safe_temp_c is not None:
            self.config.max_safe_temp_c = max(70.0, min(250.0, float(max_safe_temp_c)))
        if clear_target_cb:
            self.config.target_cb_mol_per_l = None
        elif target_cb_mol_per_l is not None:
            self.config.target_cb_mol_per_l = max(0.0, min(5.0, float(target_cb_mol_per_l)))

    def _project_moves(self, moves: List[float], u_prev: float) -> List[float]:
        cfg = self.config
        projected: List[float] = []
        prev = u_prev
        for u in moves:
            clamped = max(cfg.min_valve_pct, min(cfg.max_valve_pct, u))
            slew = max(prev - cfg.max_delta_u_pct, min(prev + cfg.max_delta_u_pct, clamped))
            projected.append(slew)
            prev = slew
        return projected

    def _simulate_horizon(
        self,
        initial_state_k: Tuple[float, float, float, float, float, float],
        u_moves: List[float],
        base_params: NativeCstrSeriesParallelParams,
        ua_max_j_per_min_k: float,
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        cfg = self.config
        state = initial_state_k
        feed_c = base_params.t_feed_kelvin - 273.15
        jacket_in_c = base_params.jacket_t_in_kelvin - 273.15

        cb_traj: List[float] = []
        byprod_traj: List[float] = []
        t_c_traj: List[float] = []
        sel_traj: List[float] = []
        u_traj: List[float] = []

        for k in range(cfg.prediction_horizon):
            u_k = u_moves[k] if k < len(u_moves) else u_moves[-1]
            u_traj.append(u_k)
            params_k = build_series_parallel_params(
                native_core=self.native_reactor,
                cooling_valve_pct=u_k,
                feed_temp_c=feed_c,
                jacket_inlet_temp_c=jacket_in_c,
                ca_feed_mol_per_l=base_params.ca_feed_mol_per_l,
                volumetric_flow_l_per_min=base_params.volumetric_flow_l_per_min,
                reactor_volume_liters=base_params.volume_liters,
                ua_max_j_per_min_k=ua_max_j_per_min_k,
            )
            state = step_series_parallel_substepped(
                self.native_reactor, params_k, state, cfg.dt_step_min, max_substep_min=0.04
            )
            ca, cb, cc, cd, t_k, _ = state
            _, _, sel_b = compute_series_parallel_kpis(ca, cb, base_params.ca_feed_mol_per_l)
            cb_traj.append(cb)
            byprod_traj.append(cc + 1.5 * cd)
            t_c_traj.append(t_k - 273.15)
            sel_traj.append(sel_b)

        return (cb_traj, byprod_traj, t_c_traj, sel_traj, u_traj)

    def _cost_function(
        self,
        cb_traj: List[float],
        byprod_traj: List[float],
        t_c_traj: List[float],
        u_traj: List[float],
        u_prev: float,
    ) -> float:
        cfg = self.config
        soft_limit_c = cfg.max_safe_temp_c - cfg.temp_safety_margin_c
        prev_u = u_prev
        cost = 0.0

        for k in range(len(cb_traj)):
            terminal_w = 1.6 if k == len(cb_traj) - 1 else 1.0
            cb_k = cb_traj[k]
            t_k = t_c_traj[k]
            u_k = u_traj[k]

            # Economic term: maximize Cb & penalize over-reaction C and byproduct D
            cost += terminal_w * (-cfg.weight_cb_economic * cb_k + cfg.weight_byproduct * byprod_traj[k])

            if cfg.target_cb_mol_per_l is not None:
                cost += terminal_w * cfg.weight_cb_tracking * (cb_k - cfg.target_cb_mol_per_l) ** 2

            # Safety temperature barrier
            if t_k > soft_limit_c:
                cost += cfg.weight_safety_soft * (t_k - soft_limit_c) ** 2
            if t_k > cfg.max_safe_temp_c:
                cost += cfg.weight_safety_hard * (t_k - cfg.max_safe_temp_c) ** 2

            # Utility cost + move suppression
            du = u_k - prev_u
            cost += cfg.weight_utility_effort * ((u_k / 100.0) ** 2) + cfg.weight_delta_u * (du * du)
            prev_u = u_k

        return cost

    def compute_optimal_move(
        self,
        current_state_c: Tuple[float, float, float, float, float, float],
        base_params: NativeCstrSeriesParallelParams,
        u_prev: Optional[float] = None,
        ua_max_j_per_min_k: float = 600000.0,
    ) -> SpNmpcResult:
        """
        Solves the Economic NMPC optimal control problem for current_state_c =
        (ca, cb, cc, cd, temp_c, jacket_temp_c). Returns SpNmpcResult with u0*.
        """
        t_start = time.perf_counter()
        cfg = self.config
        prev_u = self.last_applied_u if u_prev is None else float(u_prev)

        x0_k: Tuple[float, float, float, float, float, float] = (
            max(0.0, current_state_c[0]),
            max(0.0, current_state_c[1]),
            max(0.0, current_state_c[2]),
            max(0.0, current_state_c[3]),
            current_state_c[4] + 273.15,
            current_state_c[5] + 273.15,
        )

        # Multi-start candidate screening avoids flat thermal saturation local minima
        shifted = self._warm_moves[1:] + [self._warm_moves[-1]]
        candidates = [
            shifted,
            [prev_u] * cfg.control_horizon,
            [max(cfg.min_valve_pct, prev_u - cfg.max_delta_u_pct * 0.7)] * cfg.control_horizon,
            [min(cfg.max_valve_pct, prev_u + cfg.max_delta_u_pct * 0.7)] * cfg.control_horizon,
            [58.0] * cfg.control_horizon,
            [95.0] * cfg.control_horizon,
        ]

        u_current: List[float] = []
        cost_current = float("inf")
        cb_traj: List[float] = []
        by_traj: List[float] = []
        t_traj: List[float] = []
        sel_traj: List[float] = []

        for cand in candidates:
            proj = self._project_moves(cand, prev_u)
            cb_c, by_c, t_c, sel_c, u_c = self._simulate_horizon(x0_k, proj, base_params, ua_max_j_per_min_k)
            c_val = self._cost_function(cb_c, by_c, t_c, u_c, prev_u)
            if c_val < cost_current:
                cost_current = c_val
                u_current = proj
                cb_traj, by_traj, t_traj, sel_traj = cb_c, by_c, t_c, sel_c

        nc = cfg.control_horizon
        step_size = cfg.initial_step_size
        converged = False
        iteration_count = 0
        eps = 0.5

        for _ in range(cfg.max_iterations):
            iteration_count += 1
            grad: List[float] = []
            for j in range(nc):
                up = list(u_current)
                up[j] = min(cfg.max_valve_pct, up[j] + eps)
                um = list(u_current)
                um[j] = max(cfg.min_valve_pct, um[j] - eps)
                span = up[j] - um[j]
                if span < 1e-6:
                    grad.append(0.0)
                    continue
                cb_p, by_p, t_p, _, u_p = self._simulate_horizon(x0_k, up, base_params, ua_max_j_per_min_k)
                cost_p = self._cost_function(cb_p, by_p, t_p, u_p, prev_u)

                cb_m, by_m, t_m, _, u_m = self._simulate_horizon(x0_k, um, base_params, ua_max_j_per_min_k)
                cost_m = self._cost_function(cb_m, by_m, t_m, u_m, prev_u)

                grad.append((cost_p - cost_m) / span)

            if max(abs(g) for g in grad) < cfg.tolerance:
                converged = True
                break

            alpha = step_size
            improved = False
            for _ in range(5):
                cand_u = self._project_moves([u_current[j] - alpha * grad[j] for j in range(nc)], prev_u)
                cb_c, by_c, t_c, sel_c, u_full = self._simulate_horizon(
                    x0_k, cand_u, base_params, ua_max_j_per_min_k
                )
                c_cand = self._cost_function(cb_c, by_c, t_c, u_full, prev_u)
                if c_cand < cost_current:
                    u_current = cand_u
                    cost_current = c_cand
                    cb_traj, by_traj, t_traj, sel_traj = cb_c, by_c, t_c, sel_c
                    improved = True
                    break
                alpha *= 0.5

            if improved:
                step_size = min(8.0, alpha * 1.15)
            else:
                step_size *= 0.4
                if step_size < 1e-3:
                    break

        self._warm_moves = list(u_current)
        optimal_u0 = round(u_current[0], 2)
        self.last_applied_u = optimal_u0
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        result = SpNmpcResult(
            optimal_cooling_valve_pct=optimal_u0,
            predicted_cb_trajectory=[round(v, 4) for v in cb_traj],
            predicted_temp_c_trajectory=[round(v, 2) for v in t_traj],
            predicted_selectivity_trajectory=[round(v, 2) for v in sel_traj],
            planned_control_moves=[round(v, 2) for v in u_current],
            cost=round(cost_current, 3),
            iterations=iteration_count,
            solve_time_ms=round(elapsed_ms, 2),
            converged=converged,
        )
        self.last_result = result
        return result


def simulate_series_parallel_closed_loop_comparison(
    native_core: Optional[NativeReactorCore] = None,
    feed_temp_nominal_c: float = 51.85,
    disturbance_feed_temp_c: float = 76.85,
    disturbance_step_index: int = 12,
    jacket_inlet_temp_c: float = 21.85,
    max_safe_temp_c: float = 126.85,
    initial_cooling_valve_pct: float = 15.0,
    dt_min: float = 0.05,
    steps: int = 80,
    noise_seed: int = 42,
) -> dict:
    """
    Runs a deterministic side-by-side closed-loop comparison from identical initial conditions
    and hot-feed disturbance (51.85 C -> 76.85 C at step `disturbance_step_index`):
      1. Open-loop manual plant (fixed cooling valve at initial_cooling_valve_pct)
      2. Closed-loop 6-state EKF + Economic NMPC (estimates Ca..Cd from noisy T, Tj
         and modulates cooling valve to maximize Cb while preventing thermal runaway).
    """
    import random

    core = native_core or NativeReactorCore()
    rng = random.Random(noise_seed)

    # Identical realistic initial operating point (off-design start)
    ol_state = (0.50, 0.25, 0.20, 0.05, 58.0 + 273.15, 35.0 + 273.15)
    cl_state = ol_state

    ekf = SeriesParallelExtendedKalmanFilter(native_reactor=core)
    # Intentionally perturb initial concentration guess in EKF to prove convergence from wrong IC!
    ekf.reset(initial_ca=0.80, initial_cb=0.08, initial_cc=0.08, initial_cd=0.04, initial_temp_c=58.0, initial_jacket_temp_c=35.0)

    nmpc = SeriesParallelNonlinearMpc(
        config=SpNmpcConfig(max_safe_temp_c=max_safe_temp_c),
        native_reactor=core,
    )
    nmpc.reset(initial_u=initial_cooling_valve_pct)

    open_loop_traj = []
    closed_loop_traj = []
    u_cl = float(initial_cooling_valve_pct)

    ol_peak_temp = -999.0
    cl_peak_temp = -999.0
    ol_has_runaway = False
    cl_has_runaway = False
    safe_k = max_safe_temp_c + 273.15

    for idx in range(1, steps + 1):
        t_min = round(idx * dt_min, 4)
        feed_c = disturbance_feed_temp_c if idx >= disturbance_step_index else feed_temp_nominal_c

        # 1. Open-loop step
        ol_params = build_series_parallel_params(
            core,
            cooling_valve_pct=initial_cooling_valve_pct,
            feed_temp_c=feed_c,
            jacket_inlet_temp_c=jacket_inlet_temp_c,
        )
        ol_state = step_series_parallel_substepped(core, ol_params, ol_state, dt_min)
        ol_ca, ol_cb, ol_cc, ol_cd, ol_tk, ol_tjk = ol_state
        ol_tc = ol_tk - 273.15
        ol_tjc = ol_tjk - 273.15
        ol_conv, ol_yb, ol_selb = compute_series_parallel_kpis(ol_ca, ol_cb, ol_params.ca_feed_mol_per_l)
        ol_peak_temp = max(ol_peak_temp, ol_tc)
        ol_run = ol_tk > safe_k
        ol_has_runaway = ol_has_runaway or ol_run
        open_loop_traj.append({
            "step": idx,
            "time_min": t_min,
            "feed_temp_c": round(feed_c, 2),
            "cooling_valve_pct": round(initial_cooling_valve_pct, 2),
            "true_ca_mol_per_l": round(ol_ca, 4),
            "true_cb_mol_per_l": round(ol_cb, 4),
            "true_cc_mol_per_l": round(ol_cc, 4),
            "true_cd_mol_per_l": round(ol_cd, 4),
            "true_temp_c": round(ol_tc, 2),
            "true_jacket_temp_c": round(ol_tjc, 2),
            "conversion_a_pct": round(ol_conv, 2),
            "selectivity_b_pct": round(ol_selb, 2),
            "is_thermal_runaway": ol_run,
        })

        # 2. Closed-loop Plant step -> Noisy Sensors -> 6-state EKF -> Economic NMPC
        cl_params = build_series_parallel_params(
            core,
            cooling_valve_pct=u_cl,
            feed_temp_c=feed_c,
            jacket_inlet_temp_c=jacket_inlet_temp_c,
        )
        cl_state = step_series_parallel_substepped(core, cl_params, cl_state, dt_min)
        cl_ca, cl_cb, cl_cc, cl_cd, cl_tk, cl_tjk = cl_state
        cl_tc = cl_tk - 273.15
        cl_tjc = cl_tjk - 273.15

        meas_t = cl_tc + rng.gauss(0.0, 0.20)
        meas_tj = cl_tjc + rng.gauss(0.0, 0.15)
        est = ekf.step(dt_min=dt_min, params=cl_params, measured_temp_c=meas_t, measured_jacket_temp_c=meas_tj)

        # Economic NMPC acts exclusively from EKF estimated 6-state vector!
        nmpc_res = nmpc.compute_optimal_move(
            current_state_c=(
                est.ca_mol_per_l,
                est.cb_mol_per_l,
                est.cc_mol_per_l,
                est.cd_mol_per_l,
                est.temp_c,
                est.jacket_temp_c,
            ),
            base_params=cl_params,
            u_prev=u_cl,
        )
        u_cl = nmpc_res.optimal_cooling_valve_pct

        cl_conv, cl_yb, cl_selb = compute_series_parallel_kpis(cl_ca, cl_cb, cl_params.ca_feed_mol_per_l)
        cl_peak_temp = max(cl_peak_temp, cl_tc)
        cl_run = cl_tk > safe_k
        cl_has_runaway = cl_has_runaway or cl_run
        closed_loop_traj.append({
            "step": idx,
            "time_min": t_min,
            "feed_temp_c": round(feed_c, 2),
            "cooling_valve_pct": round(u_cl, 2),
            "true_ca_mol_per_l": round(cl_ca, 4),
            "true_cb_mol_per_l": round(cl_cb, 4),
            "true_cc_mol_per_l": round(cl_cc, 4),
            "true_cd_mol_per_l": round(cl_cd, 4),
            "true_temp_c": round(cl_tc, 2),
            "true_jacket_temp_c": round(cl_tjc, 2),
            "est_ca_mol_per_l": est.ca_mol_per_l,
            "est_cb_mol_per_l": est.cb_mol_per_l,
            "est_cc_mol_per_l": est.cc_mol_per_l,
            "est_cd_mol_per_l": est.cd_mol_per_l,
            "est_temp_c": est.temp_c,
            "est_jacket_temp_c": est.jacket_temp_c,
            "ekf_nis": est.nis,
            "sensor_healthy": est.sensor_healthy,
            "conversion_a_pct": round(cl_conv, 2),
            "selectivity_b_pct": round(cl_selb, 2),
            "nmpc_solve_time_ms": nmpc_res.solve_time_ms,
            "is_thermal_runaway": cl_run,
        })

    return {
        "engine_mode": f"Native C11 ({core.get_version()})" if core.is_available else "Python RK4 Fallback",
        "max_safe_temp_c": round(max_safe_temp_c, 2),
        "disturbance_step": disturbance_step_index,
        "open_loop_peak_temp_c": round(ol_peak_temp, 2),
        "open_loop_has_runaway": ol_has_runaway,
        "open_loop_final_cb_mol_per_l": open_loop_traj[-1]["true_cb_mol_per_l"] if open_loop_traj else 0.0,
        "open_loop_final_selectivity_pct": open_loop_traj[-1]["selectivity_b_pct"] if open_loop_traj else 0.0,
        "closed_loop_peak_temp_c": round(cl_peak_temp, 2),
        "closed_loop_has_runaway": cl_has_runaway,
        "closed_loop_final_cb_mol_per_l": closed_loop_traj[-1]["true_cb_mol_per_l"] if closed_loop_traj else 0.0,
        "closed_loop_final_selectivity_pct": closed_loop_traj[-1]["selectivity_b_pct"] if closed_loop_traj else 0.0,
        "final_ekf_cb_error": round(
            abs(closed_loop_traj[-1]["true_cb_mol_per_l"] - closed_loop_traj[-1]["est_cb_mol_per_l"]), 4
        ) if closed_loop_traj else 0.0,
        "open_loop_trajectory": open_loop_traj,
        "closed_loop_trajectory": closed_loop_traj,
    }
