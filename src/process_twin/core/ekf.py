"""
Extended Kalman Filter (EKF) for Real-Time State & Concentration Estimation in Exothermic CSTR.

In industrial continuous chemical reactors, reactor temperature (T) is readily measured via
high-frequency RTDs/thermocouples (e.g., TT-102), but internal reactant concentration (Ca)
and reaction rates cannot be measured online without significant analytical delays or high CAPEX.

This module provides a non-linear Extended Kalman Filter that:
1. Reconstructs unmeasured internal reactant concentration Ca(t) from noisy temperature measurements.
2. Estimates unmeasured reaction rate r(t) and conversion percentage X(t).
3. Evaluates Normalized Innovation Squared (NIS) for automated sensor anomaly and drift detection.
4. Uses Joseph stabilized covariance updates ensuring positive semi-definiteness and numerical stability.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class EkfEstimate:
    """Instantaneous filtered state estimate and diagnostic metrics."""
    ca_mol_per_l: float
    temp_celsius: float
    temp_kelvin: float
    conversion_pct: float
    reaction_rate: float
    temp_residual: float
    nis: float
    p_ca: float
    p_temp: float
    p_ca_temp: float
    kalman_gain_ca: float
    kalman_gain_temp: float


class CstrExtendedKalmanFilter:
    """
    Continuous-Discrete Extended Kalman Filter (EKF) for exothermic CSTR dynamics.

    State vector:
        x = [Ca, T]^T
        Ca: Reactant concentration in kmol/m³ (mol/L)
        T: Reactor temperature in °C

    Non-linear dynamic model:
        dCa/dt = (Fin/V)*(Ca_feed - Ca) - r(Ca, T)
        dT/dt  = delta_T_ad * r(Ca, T) + (Fin/V)*(T_feed - T) - (UA(u_c)/(V*rho*Cp))*(T - T_coolant)

    Measurement model:
        y_k = T_measured + v_k, where v_k ~ N(0, R)
    """

    def __init__(
        self,
        initial_ca: float = 0.5,
        initial_temp_c: float = 85.0,
        ca_feed: float = 1.0,
        temp_feed_c: float = 25.0,
        coolant_temp_c: float = 20.0,
        reactor_volume_m3: float = 12.0,
        nominal_flow_m3_h: float = 24.0,
        q_ca: float = 1e-4,
        q_temp: float = 1e-2,
        r_temp: float = 0.15 ** 2,
        initial_p_ca: float = 0.04,
        initial_p_temp: float = 1.0,
    ) -> None:
        # Physical parameters
        self.ca_feed = ca_feed
        self.temp_feed_c = temp_feed_c
        self.coolant_temp_c = coolant_temp_c
        self.reactor_volume_m3 = reactor_volume_m3
        self.nominal_flow_m3_h = nominal_flow_m3_h

        # State estimate: [Ca, T]
        self.ca = float(initial_ca)
        self.temp_c = float(initial_temp_c)

        # Error covariance matrix P = [[P11, P12], [P21, P22]]
        self.p11 = float(initial_p_ca)
        self.p12 = 0.0
        self.p21 = 0.0
        self.p22 = float(initial_p_temp)

        # Process noise covariance Q (continuous-time spectral density)
        self.q_ca = float(q_ca)
        self.q_temp = float(q_temp)

        # Measurement noise variance R
        self.r_temp = float(r_temp)

        # Latest step diagnostics
        self.last_residual = 0.0
        self.last_nis = 0.0
        self.last_k_ca = 0.0
        self.last_k_temp = 0.0

    def reset(
        self,
        ca: float = 0.5,
        temp_c: float = 85.0,
        p_ca: float = 0.04,
        p_temp: float = 1.0,
    ) -> None:
        """Resets state estimate and covariance matrix."""
        self.ca = float(ca)
        self.temp_c = float(temp_c)
        self.p11 = float(p_ca)
        self.p12 = 0.0
        self.p21 = 0.0
        self.p22 = float(p_temp)
        self.last_residual = 0.0
        self.last_nis = 0.0
        self.last_k_ca = 0.0
        self.last_k_temp = 0.0

    @property
    def conversion_pct(self) -> float:
        """Instantaneous estimated reactant conversion percentage."""
        return max(0.0, min(100.0, (1.0 - (self.ca / self.ca_feed)) * 100.0))

    @staticmethod
    def _arrhenius_k(temp_c: float) -> float:
        """Arrhenius dimensionless rate constant scaled around 85 °C (358.15 K)."""
        temp_k = temp_c + 273.15
        if temp_k <= 50.0:
            return 0.0
        arg = -4200.0 * (1.0 / temp_k - 1.0 / 358.15)
        # Numerical clamping to avoid floating-point overflow during runaway
        arg = max(-50.0, min(50.0, arg))
        return math.exp(arg)

    def reaction_rate(self, ca: float, temp_c: float, feed_avail: float = 1.0) -> float:
        """
        Calculates instantaneous reaction rate:
        r = 0.26733 * ((2 * k) / (1 + k)) * (Ca / Ca_feed) * feed_avail
        """
        if ca <= 0.0:
            return 0.0
        k = self._arrhenius_k(temp_c)
        conv_factor = (2.0 * k) / (1.0 + k)
        ca_ratio = max(0.0, min(1.5, ca / self.ca_feed))
        return 0.26733 * conv_factor * ca_ratio * max(0.1, min(1.5, feed_avail))

    def _derivatives(
        self,
        ca: float,
        temp_c: float,
        flow_m3_h: float,
        cooling_valve_pct: float,
        feed_active: bool,
    ) -> Tuple[float, float]:
        """Evaluates non-linear continuous ODEs [dCa/dt, dT/dt]."""
        feed_avail = (flow_m3_h / self.nominal_flow_m3_h) if feed_active else 0.15
        r = self.reaction_rate(ca, temp_c, feed_avail)

        flow_m3_s = (flow_m3_h if feed_active else 0.0) / 3600.0
        space_inv = flow_m3_s / self.reactor_volume_m3  # s^-1

        # Species balance: dCa/dt = (Fin/V)*(Ca_feed - Ca) - r_ca
        # At nominal steady-state Ca=0.5, Fin=24 m3/h, V=12 m3, k=1.0 -> dCa/dt = 0
        r_ca = space_inv * self.ca_feed * (ca / self.ca_feed) * self._arrhenius_k(temp_c)
        dca_dt = space_inv * (self.ca_feed - ca) - r_ca

        # Energy balance: reaction heat generation coupled to Ca concentration
        ca_ratio = ca / (self.ca_feed * 0.5)
        k = self._arrhenius_k(temp_c)
        conv_factor = (2.0 * k) / (1.0 + k)
        heat_gen = 0.26733 * conv_factor * ca_ratio * max(0.1, min(1.5, feed_avail))
        cooling_duty = (cooling_valve_pct / 100.0) * 0.008 * (temp_c - self.coolant_temp_c)
        feed_cooling = space_inv * (self.temp_feed_c - temp_c)
        dt_dt = heat_gen + feed_cooling - cooling_duty

        return (dca_dt, dt_dt)

    def _jacobian_f(
        self,
        ca: float,
        temp_c: float,
        flow_m3_h: float,
        cooling_valve_pct: float,
        feed_active: bool,
    ) -> Tuple[float, float, float, float]:
        """
        Computes analytical continuous Jacobian matrix F = [[f11, f12], [f21, f22]]
        where f1 = dCa/dt and f2 = dT/dt.
        """
        flow_m3_s = (flow_m3_h if feed_active else 0.0) / 3600.0
        space_inv = flow_m3_s / self.reactor_volume_m3

        k = self._arrhenius_k(temp_c)
        temp_k = temp_c + 273.15
        dk_dt = k * (4200.0 / (temp_k * temp_k))

        # df1/dCa = -space_inv * (1 + k)
        f11 = -space_inv * (1.0 + k)
        # df1/dT = -space_inv * Ca * dk_dt
        f12 = -space_inv * ca * dk_dt

        # df2/dCa = 0.26733 * (2k/(1+k)) * (1 / (0.5 * Ca_feed)) * feed_avail
        feed_avail = (flow_m3_h / self.nominal_flow_m3_h) if feed_active else 0.15
        ca_scale = 1.0 / max(0.01, self.ca_feed * 0.5)
        conv_factor = (2.0 * k) / (1.0 + k)
        f21 = 0.26733 * conv_factor * ca_scale * max(0.1, min(1.5, feed_avail))

        # df2/dT = d(heat_gen)/dT + d(feed_cooling)/dT - d(cooling_duty)/dT
        # d/dT (2k/(1+k)) = 2 / (1+k)^2 * dk_dt
        d_conv_dt = (2.0 / ((1.0 + k) * (1.0 + k))) * dk_dt
        d_heat_dt = 0.26733 * d_conv_dt * (ca * ca_scale) * max(0.1, min(1.5, feed_avail))
        d_feed_dt = -space_inv
        d_cool_dt = (cooling_valve_pct / 100.0) * 0.008
        f22 = d_heat_dt + d_feed_dt - d_cool_dt

        return (f11, f12, f21, f22)

    def predict(
        self,
        dt_seconds: float,
        flow_m3_h: float = 24.0,
        cooling_valve_pct: float = 45.0,
        feed_active: bool = True,
    ) -> None:
        """
        EKF Time Propagation / Prediction step:
        1. Non-linear state integration via 4th-Order Runge-Kutta (RK4).
        2. Covariance propagation: P = Phi * P * Phi^T + Q * dt.
        """
        if dt_seconds <= 0.0:
            return

        # 1. State RK4 integration
        h = dt_seconds
        half_h = 0.5 * h

        k1_ca, k1_t = self._derivatives(self.ca, self.temp_c, flow_m3_h, cooling_valve_pct, feed_active)
        k2_ca, k2_t = self._derivatives(
            self.ca + half_h * k1_ca,
            self.temp_c + half_h * k1_t,
            flow_m3_h,
            cooling_valve_pct,
            feed_active,
        )
        k3_ca, k3_t = self._derivatives(
            self.ca + half_h * k2_ca,
            self.temp_c + half_h * k2_t,
            flow_m3_h,
            cooling_valve_pct,
            feed_active,
        )
        k4_ca, k4_t = self._derivatives(
            self.ca + h * k3_ca,
            self.temp_c + h * k3_t,
            flow_m3_h,
            cooling_valve_pct,
            feed_active,
        )

        h6 = h / 6.0
        self.ca += h6 * (k1_ca + 2.0 * k2_ca + 2.0 * k3_ca + k4_ca)
        self.temp_c += h6 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t)

        # Enforce physical state bounds
        self.ca = max(0.0, min(self.ca_feed * 1.5, self.ca))
        self.temp_c = max(15.0, min(250.0, self.temp_c))

        # 2. Covariance propagation via State Transition Matrix Phi = I + F*dt + 0.5*(F*dt)^2
        f11, f12, f21, f22 = self._jacobian_f(self.ca, self.temp_c, flow_m3_h, cooling_valve_pct, feed_active)

        phi11 = 1.0 + f11 * h
        phi12 = f12 * h
        phi21 = f21 * h
        phi22 = 1.0 + f22 * h

        # M = Phi * P
        m11 = phi11 * self.p11 + phi12 * self.p21
        m12 = phi11 * self.p12 + phi12 * self.p22
        m21 = phi21 * self.p11 + phi22 * self.p21
        m22 = phi21 * self.p12 + phi22 * self.p22

        # P_prior = M * Phi^T + Q * dt
        self.p11 = m11 * phi11 + m12 * phi12 + self.q_ca * h
        self.p12 = m11 * phi21 + m12 * phi22
        self.p21 = m21 * phi11 + m22 * phi12
        self.p22 = m21 * phi21 + m22 * phi22 + self.q_temp * h

        # Symmetrize
        p_cross = 0.5 * (self.p12 + self.p21)
        self.p12 = p_cross
        self.p21 = p_cross

    def update_temperature(self, measured_temp_c: float) -> Tuple[float, float]:
        """
        EKF Measurement Update step using online temperature transmitter y = T:
        H = [0, 1]
        Returns (residual, NIS).
        """
        # Innovation (residual)
        residual = measured_temp_c - self.temp_c
        self.last_residual = residual

        # Innovation covariance S = H * P * H^T + R = P22 + R
        s = self.p22 + self.r_temp
        if s <= 1e-12:
            s = 1e-12

        # Normalized Innovation Squared (NIS) ~ chi2(1)
        nis = (residual * residual) / s
        self.last_nis = nis

        # Kalman Gain: K = P * H^T / S = [P12 / S, P22 / S]^T
        k_ca = self.p12 / s
        k_temp = self.p22 / s
        self.last_k_ca = k_ca
        self.last_k_temp = k_temp

        # State update: x = x_prior + K * residual
        self.ca += k_ca * residual
        self.temp_c += k_temp * residual

        # Physical clamping
        self.ca = max(0.0, min(self.ca_feed * 1.5, self.ca))
        self.temp_c = max(15.0, min(250.0, self.temp_c))

        # Covariance update via Joseph stabilized form:
        # P_post = (I - K*H) * P_prior * (I - K*H)^T + K * R * K^T
        # I - K*H = [[1, -k_ca], [0, 1 - k_temp]]
        a11 = 1.0
        a12 = -k_ca
        a21 = 0.0
        a22 = 1.0 - k_temp

        # W = (I - K*H) * P
        w11 = a11 * self.p11 + a12 * self.p21
        w12 = a11 * self.p12 + a12 * self.p22
        w21 = a21 * self.p11 + a22 * self.p21
        w22 = a21 * self.p12 + a22 * self.p22

        # P_post = W * (I - K*H)^T + K * R * K^T
        self.p11 = w11 * a11 + w12 * a12 + k_ca * self.r_temp * k_ca
        self.p12 = w11 * a21 + w12 * a22 + k_ca * self.r_temp * k_temp
        self.p21 = w21 * a11 + w22 * a12 + k_temp * self.r_temp * k_ca
        self.p22 = w21 * a21 + w22 * a22 + k_temp * self.r_temp * k_temp

        # Guarantee symmetry and positive lower variance bound
        self.p11 = max(1e-8, self.p11)
        self.p22 = max(1e-8, self.p22)
        p_sym = 0.5 * (self.p12 + self.p21)
        self.p12 = p_sym
        self.p21 = p_sym

        return (residual, nis)

    def step(
        self,
        dt_seconds: float,
        measured_temp_c: float,
        flow_m3_h: float = 24.0,
        cooling_valve_pct: float = 45.0,
        feed_active: bool = True,
    ) -> EkfEstimate:
        """
        Executes a complete EKF cycle (Predict + Update) and returns the state estimate.
        """
        self.predict(dt_seconds, flow_m3_h, cooling_valve_pct, feed_active)
        res, nis = self.update_temperature(measured_temp_c)

        feed_avail = (flow_m3_h / self.nominal_flow_m3_h) if feed_active else 0.15
        r = self.reaction_rate(self.ca, self.temp_c, feed_avail)
        conv_pct = max(0.0, min(100.0, (1.0 - (self.ca / self.ca_feed)) * 100.0))

        return EkfEstimate(
            ca_mol_per_l=round(self.ca, 4),
            temp_celsius=round(self.temp_c, 2),
            temp_kelvin=round(self.temp_c + 273.15, 2),
            conversion_pct=round(conv_pct, 2),
            reaction_rate=round(r, 5),
            temp_residual=round(res, 4),
            nis=round(nis, 3),
            p_ca=round(self.p11, 6),
            p_temp=round(self.p22, 6),
            p_ca_temp=round(self.p12, 6),
            kalman_gain_ca=round(self.last_k_ca, 5),
            kalman_gain_temp=round(self.last_k_temp, 5),
        )
