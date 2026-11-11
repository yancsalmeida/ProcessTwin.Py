"""
Non-Linear Model Predictive Control (NMPC) for Exothermic Continuous Stirred-Tank Reactor (CSTR).

Coupled with the real-time state estimates [Ca, T] from the Extended Kalman Filter (EKF),
this controller computes optimal cooling jacket valve trajectories over a receding finite prediction
horizon, satisfying physical actuator constraints (0-100%), valve slew rates, and thermal runaway safety limits.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class NmpcConfig:
    """Tuning parameters, horizons, and physical constraints for CSTR NMPC."""
    prediction_horizon: int = 15        # Np: Prediction horizon steps
    control_horizon: int = 4           # Nc: Control horizon moves
    dt_seconds: float = 1.0            # Internal step duration (seconds)

    # Setpoints
    temp_setpoint_c: float = 85.0      # Target reactor core temperature (°C)
    ca_setpoint: Optional[float] = None # Optional target reactant concentration (mol/L)

    # Cost function weighting matrices
    weight_temp: float = 12.0          # Q_T: Tracking penalty for temperature deviation
    weight_ca: float = 4.0             # Q_Ca: Tracking penalty for concentration deviation
    weight_u_rate: float = 0.8         # R_delta_u: Actuator slew rate penalty
    weight_u_deviation: float = 0.02   # R_u: Input deviation penalty from nominal
    weight_safety_slack: float = 500.0 # S_slack: Penalty for approaching runaway limit

    # Physical actuator constraints
    u_min_pct: float = 0.0             # Fully closed cooling valve (%)
    u_max_pct: float = 100.0           # Fully open cooling valve (%)
    max_delta_u_pct: float = 15.0      # Maximum allowable valve move per second (% / dt)
    nominal_u_pct: float = 45.0        # Nominal baseline valve opening (%)
    max_safe_temp_c: float = 95.0      # Maximum allowable reactor temperature (°C)

    # Optimizer settings
    max_iterations: int = 25           # Maximum projected gradient iterations per cycle
    tolerance: float = 1e-4            # Gradient infinity norm convergence tolerance
    initial_step_size: float = 2.0     # Initial gradient descent step size


@dataclass
class NmpcResult:
    """Diagnostic output of a single NMPC optimization cycle."""
    optimal_cooling_valve_pct: float
    predicted_temp_trajectory: List[float] = field(default_factory=list)
    predicted_ca_trajectory: List[float] = field(default_factory=list)
    planned_control_moves: List[float] = field(default_factory=list)
    cost: float = 0.0
    iterations: int = 0
    solve_time_ms: float = 0.0
    converged: bool = False
    temp_tracking_error: float = 0.0


class CstrNonlinearMpc:
    """
    Nonlinear Model Predictive Controller (NMPC) for CSTR Temperature & Concentration Control.

    Uses receding horizon optimization with warm-started Projected Gradient Descent,
    subject to actuator box constraints and slew-rate limits.
    """

    def __init__(
        self,
        config: Optional[NmpcConfig] = None,
        ca_feed: float = 1.0,
        temp_feed_c: float = 25.0,
        coolant_temp_c: float = 20.0,
        reactor_volume_m3: float = 12.0,
        nominal_flow_m3_h: float = 24.0,
    ) -> None:
        self.config = config or NmpcConfig()
        self.ca_feed = float(ca_feed)
        self.temp_feed_c = float(temp_feed_c)
        self.coolant_temp_c = float(coolant_temp_c)
        self.reactor_volume_m3 = float(reactor_volume_m3)
        self.nominal_flow_m3_h = float(nominal_flow_m3_h)

        # Warm-start control move sequence [u_0, u_1, ..., u_{Nc-1}]
        self._warm_moves = [self.config.nominal_u_pct] * self.config.control_horizon
        self.last_applied_u = self.config.nominal_u_pct
        self.last_result: Optional[NmpcResult] = None

    def reset(self, initial_u: Optional[float] = None) -> None:
        """Resets the controller internal warm-start memory."""
        u0 = initial_u if initial_u is not None else self.config.nominal_u_pct
        self.last_applied_u = max(self.config.u_min_pct, min(self.config.u_max_pct, u0))
        self._warm_moves = [self.last_applied_u] * self.config.control_horizon
        self.last_result = None

    def update_setpoints(
        self,
        temp_setpoint_c: Optional[float] = None,
        ca_setpoint: Optional[float] = None,
    ) -> None:
        """Dynamically updates target setpoints from supervisory control."""
        if temp_setpoint_c is not None:
            self.config.temp_setpoint_c = max(20.0, min(150.0, float(temp_setpoint_c)))
        if ca_setpoint is not None:
            self.config.ca_setpoint = max(0.01, min(self.ca_feed, float(ca_setpoint))) if ca_setpoint > 0 else None

    @staticmethod
    def _arrhenius_k(temp_c: float) -> float:
        """Arrhenius rate constant calibrated around 85 °C (358.15 K)."""
        temp_k = temp_c + 273.15
        if temp_k <= 50.0:
            return 0.0
        arg = -4200.0 * (1.0 / temp_k - 1.0 / 358.15)
        arg = max(-50.0, min(50.0, arg))
        return math.exp(arg)

    def _derivatives(
        self,
        ca: float,
        temp_c: float,
        u_cooling_pct: float,
        flow_m3_h: float,
        feed_active: bool,
    ) -> Tuple[float, float]:
        """Evaluates non-linear CSTR ODEs: dCa/dt and dT/dt."""
        flow_m3_s = (flow_m3_h if feed_active else 0.0) / 3600.0
        space_inv = flow_m3_s / self.reactor_volume_m3

        k = self._arrhenius_k(temp_c)
        r_ca = space_inv * self.ca_feed * (ca / self.ca_feed) * k
        dca_dt = space_inv * (self.ca_feed - ca) - r_ca

        feed_avail = (flow_m3_h / self.nominal_flow_m3_h) if feed_active else 0.15
        ca_ratio = ca / (self.ca_feed * 0.5)
        conv_factor = (2.0 * k) / (1.0 + k)
        heat_gen = 0.26733 * conv_factor * ca_ratio * max(0.1, min(1.5, feed_avail))

        cooling_duty = (u_cooling_pct / 100.0) * 0.008 * (temp_c - self.coolant_temp_c)
        feed_cooling = space_inv * (self.temp_feed_c - temp_c)
        dt_dt = heat_gen + feed_cooling - cooling_duty

        return (dca_dt, dt_dt)

    def _rk4_step(
        self,
        ca: float,
        temp_c: float,
        u_cooling_pct: float,
        dt: float,
        flow_m3_h: float,
        feed_active: bool,
    ) -> Tuple[float, float]:
        """Advances state [Ca, T] by dt seconds using 4th-Order Runge-Kutta."""
        h2 = 0.5 * dt
        k1_ca, k1_t = self._derivatives(ca, temp_c, u_cooling_pct, flow_m3_h, feed_active)
        k2_ca, k2_t = self._derivatives(ca + h2 * k1_ca, temp_c + h2 * k1_t, u_cooling_pct, flow_m3_h, feed_active)
        k3_ca, k3_t = self._derivatives(ca + h2 * k2_ca, temp_c + h2 * k2_t, u_cooling_pct, flow_m3_h, feed_active)
        k4_ca, k4_t = self._derivatives(ca + dt * k3_ca, temp_c + dt * k3_t, u_cooling_pct, flow_m3_h, feed_active)

        h6 = dt / 6.0
        next_ca = ca + h6 * (k1_ca + 2.0 * k2_ca + 2.0 * k3_ca + k4_ca)
        next_temp = temp_c + h6 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)

        return (
            max(0.0, min(self.ca_feed * 1.5, next_ca)),
            max(15.0, min(250.0, next_temp)),
        )

    def _simulate_trajectory(
        self,
        initial_ca: float,
        initial_temp_c: float,
        control_moves: List[float],
        flow_m3_h: float,
        feed_active: bool,
    ) -> Tuple[List[float], List[float]]:
        """Simulates state forward over the entire prediction horizon Np."""
        np_horizon = self.config.prediction_horizon
        nc_horizon = self.config.control_horizon
        dt = self.config.dt_seconds

        ca_traj: List[float] = []
        temp_traj: List[float] = []

        cur_ca = initial_ca
        cur_t = initial_temp_c

        for i in range(np_horizon):
            u_i = control_moves[min(i, nc_horizon - 1)]
            cur_ca, cur_t = self._rk4_step(cur_ca, cur_t, u_i, dt, flow_m3_h, feed_active)
            ca_traj.append(cur_ca)
            temp_traj.append(cur_t)

        return ca_traj, temp_traj

    def _cost_function(
        self,
        ca_traj: List[float],
        temp_traj: List[float],
        control_moves: List[float],
        u_prev: float,
    ) -> float:
        """
        Evaluates cumulative quadratic cost function:
        J = sum_{i=1}^{Np} [ w_T*(T_i - T_sp)^2 + w_Ca*(Ca_i - Ca_sp)^2 + w_slack*max(0, T_i - T_max)^2 ]
          + sum_{j=0}^{Nc-1} [ w_rate*(u_j - u_{j-1})^2 + w_dev*(u_j - u_nom)^2 ]
        """
        cfg = self.config
        total_cost = 0.0

        # State tracking cost over prediction horizon
        for ca_i, t_i in zip(ca_traj, temp_traj):
            temp_err = t_i - cfg.temp_setpoint_c
            total_cost += cfg.weight_temp * (temp_err * temp_err)

            if cfg.ca_setpoint is not None:
                ca_err = ca_i - cfg.ca_setpoint
                total_cost += cfg.weight_ca * (ca_err * ca_err)

            if t_i > cfg.max_safe_temp_c:
                slack = t_i - cfg.max_safe_temp_c
                total_cost += cfg.weight_safety_slack * (slack * slack)

        # Control effort and slew rate penalties over control horizon
        prev_move = u_prev
        for move in control_moves:
            delta_u = move - prev_move
            total_cost += cfg.weight_u_rate * (delta_u * delta_u)

            u_dev = move - cfg.nominal_u_pct
            total_cost += cfg.weight_u_deviation * (u_dev * u_dev)

            prev_move = move

        return total_cost

    def _project_moves(self, control_moves: List[float], u_prev: float) -> List[float]:
        """
        Projects control moves onto admissible convex set satisfying:
        1. Actuator box constraints: u_min <= u_i <= u_max
        2. Actuator slew rate limits: |u_i - u_{i-1}| <= max_delta_u
        """
        cfg = self.config
        projected: List[float] = []
        prev = u_prev

        for move in control_moves:
            # 1. Box constraint
            clamped = max(cfg.u_min_pct, min(cfg.u_max_pct, move))
            # 2. Rate of change constraint
            rate_limited = max(prev - cfg.max_delta_u_pct, min(prev + cfg.max_delta_u_pct, clamped))
            projected.append(rate_limited)
            prev = rate_limited

        return projected

    def compute_optimal_move(
        self,
        current_ca: float,
        current_temp_c: float,
        flow_m3_h: float = 24.0,
        feed_active: bool = True,
        u_prev: Optional[float] = None,
    ) -> NmpcResult:
        """
        Executes NMPC optimization cycle:
        Solves for control sequence U* minimizing cost J, returning optimal move u_0*.
        """
        t_start = time.perf_counter()
        cfg = self.config
        prev_valve = self.last_applied_u if u_prev is None else u_prev

        # 1. Warm start: shift previous optimal moves by 1 step
        warm = self._warm_moves[1:] + [self._warm_moves[-1]]
        u_current = self._project_moves(warm, prev_valve)

        nc = cfg.control_horizon
        step_size = cfg.initial_step_size
        converged = False
        iteration_count = 0

        # Baseline evaluation
        ca_traj, t_traj = self._simulate_trajectory(current_ca, current_temp_c, u_current, flow_m3_h, feed_active)
        cost_current = self._cost_function(ca_traj, t_traj, u_current, prev_valve)

        # 2. Projected Gradient Descent with Armijo Line Search
        eps = 0.25  # Finite difference perturbation (%)

        for it in range(cfg.max_iterations):
            iteration_count += 1

            # Compute numerical gradient vector dJ/du
            gradient: List[float] = []
            for j in range(nc):
                u_perturbed = list(u_current)

                u_perturbed[j] = u_current[j] + eps
                ca_p, t_p = self._simulate_trajectory(current_ca, current_temp_c, u_perturbed, flow_m3_h, feed_active)
                cost_plus = self._cost_function(ca_p, t_p, u_perturbed, prev_valve)

                u_perturbed[j] = u_current[j] - eps
                ca_m, t_m = self._simulate_trajectory(current_ca, current_temp_c, u_perturbed, flow_m3_h, feed_active)
                cost_minus = self._cost_function(ca_m, t_m, u_perturbed, prev_valve)

                gradient.append((cost_plus - cost_minus) / (2.0 * eps))

            grad_norm = max(abs(g) for g in gradient)
            if grad_norm < cfg.tolerance:
                converged = True
                break

            # Backtracking Armijo line search along projected gradient direction
            alpha = step_size
            improved = False
            best_candidate = u_current
            best_candidate_cost = cost_current
            best_candidate_ca = ca_traj
            best_candidate_t = t_traj

            for _ in range(6):
                candidate_moves = [u_current[j] - alpha * gradient[j] for j in range(nc)]
                candidate_moves = self._project_moves(candidate_moves, prev_valve)

                ca_c, t_c = self._simulate_trajectory(current_ca, current_temp_c, candidate_moves, flow_m3_h, feed_active)
                cost_cand = self._cost_function(ca_c, t_c, candidate_moves, prev_valve)

                if cost_cand < cost_current:
                    improved = True
                    best_candidate = candidate_moves
                    best_candidate_cost = cost_cand
                    best_candidate_ca = ca_c
                    best_candidate_t = t_c
                    break
                alpha *= 0.5

            if improved:
                u_current = best_candidate
                cost_current = best_candidate_cost
                ca_traj = best_candidate_ca
                t_traj = best_candidate_t
                step_size = min(5.0, alpha * 1.1)  # Adaptive step expansion
            else:
                step_size *= 0.5
                if step_size < 1e-4:
                    break

        # 3. Store warm moves and apply first optimal move (Receding Horizon Principle)
        self._warm_moves = list(u_current)
        optimal_u0 = round(u_current[0], 2)
        self.last_applied_u = optimal_u0

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        result = NmpcResult(
            optimal_cooling_valve_pct=optimal_u0,
            predicted_temp_trajectory=[round(t, 2) for t in t_traj],
            predicted_ca_trajectory=[round(ca, 4) for ca in ca_traj],
            planned_control_moves=[round(u, 2) for u in u_current],
            cost=round(cost_current, 3),
            iterations=iteration_count,
            solve_time_ms=round(t_elapsed_ms, 2),
            converged=converged,
            temp_tracking_error=round(current_temp_c - cfg.temp_setpoint_c, 2),
        )

        self.last_result = result
        return result
