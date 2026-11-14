import math
import random
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional
from process_twin.core.models import (
    PlantTelemetry,
    AlarmEvent,
    ActuatorCommand,
    PlantStatus,
    TransportProperties,
    CstrBatteryTelemetry,
    SeriesParallelControlTelemetry,
    SeriesParallelControlConfigCommand,
)
from process_twin.core.alarms import evaluate_safety_alarms
from process_twin.core.ekf import CstrExtendedKalmanFilter, EkfEstimate
from process_twin.core.native_reactor import NativeReactorCore
from process_twin.core.nmpc import CstrNonlinearMpc, NmpcConfig, NmpcResult
from process_twin.core.series_parallel_control import (
    SeriesParallelExtendedKalmanFilter,
    SeriesParallelNonlinearMpc,
    SpEkfEstimate,
    SpNmpcConfig,
    SpNmpcResult,
    build_series_parallel_params,
    compute_series_parallel_kpis,
    step_series_parallel_substepped,
)


class PlantDigitalTwin:
    """
    Asynchronous Digital Twin simulating a coupled Buffer Surge Tank and Exothermic CSTR.
    Features:
    - Mass balance for surge tank: dh/dt = (Fin - Fout) / Area
    - Dynamic energy balance for CSTR temperature & vapor pressure
    - Stochastic Gaussian noise representing 4-20mA industrial transmitter drift
    - Real-time Safety Instrumented System (SIS) automated interlock actuation
    """

    def __init__(self) -> None:
        # Buffer Tank physical dimensions
        self.tank_area_m2 = 5.0       # 5 m² cross-sectional area (50 m³ at 10m height)
        self.tank_max_height_m = 10.0

        # State variables
        self.tank_level_m = 5.0       # Initial 50% fill (5.0 m)
        self.feed_valve_open = True
        self.feed_flow_m3_h = 24.0    # 24 m³/h nominal inlet
        self.discharge_flow_m3_h = 24.0 # 24 m³/h nominal discharge to reactor

        # Reactor state variables
        self.reactor_temp_c = 85.0    # 85 °C steady operating point
        self.cooling_valve_pct = 45.0 # 45% valve opening
        self.reactor_volume_m3 = 12.0
        self.coolant_temp_c = 20.0

        # Simulation speed factor (default 5.0x for responsive interactive SCADA feedback)
        self.time_scale = 5.0

        # 6-State Series-Parallel CSTR (A -> B -> C, A -> D) + EKF + Economic NMPC
        self.sp_core = NativeReactorCore()
        self.sp_ekf = SeriesParallelExtendedKalmanFilter(native_reactor=self.sp_core)
        self.sp_nmpc = SeriesParallelNonlinearMpc(config=SpNmpcConfig(), native_reactor=self.sp_core)
        self.sp_nmpc_enabled: bool = False
        self.sp_cooling_valve_pct: float = 55.0
        self.sp_feed_temp_c: float = 51.85
        self.sp_jacket_inlet_temp_c: float = 21.85
        self.sp_ca_feed: float = 1.0
        self.sp_state = (0.48, 0.28, 0.21, 0.03, 62.0 + 273.15, 36.0 + 273.15)
        self.latest_sp_ekf: Optional[SpEkfEstimate] = None
        self.latest_sp_nmpc: Optional[SpNmpcResult] = None
        self.latest_sp_telemetry: Optional[SeriesParallelControlTelemetry] = None
        self.latest_sp_closed_loop: Optional[dict] = None

        # Alarms & events
        self.active_alarms: List[AlarmEvent] = []
        self.interlock_tripped = False
        self.status = PlantStatus.NORMAL

        # Sensor noise settings (standard deviation)
        self.level_noise_std = 0.02   # ± 2 cm noise
        self.temp_noise_std = 0.15    # ± 0.15 °C noise
        self.press_noise_std = 0.015  # ± 0.015 bar noise

        # Real-fluid thermodynamics state from ChemFlow.NET
        self.latest_beta: Optional[float] = None
        self.latest_phase: Optional[str] = "SubcooledLiquid"
        self.latest_thermo_model: Optional[str] = "Peng-Robinson EOS (1976)"
        self.latest_molar_enthalpy: Optional[float] = None
        self.latest_transport: Optional[TransportProperties] = None
        self.latest_battery: Optional[CstrBatteryTelemetry] = None
        self.latest_series_parallel_plant: Optional[dict] = None
        self.latest_series_parallel_profile: Optional[dict] = None

        # Extended Kalman Filter (EKF) for non-linear state & concentration estimation
        self.ekf = CstrExtendedKalmanFilter(
            initial_ca=0.5,
            initial_temp_c=self.reactor_temp_c,
            r_temp=self.temp_noise_std ** 2,
        )
        self.latest_ekf: Optional[EkfEstimate] = None

        # Non-Linear Model Predictive Controller (NMPC) for CSTR
        self.nmpc = CstrNonlinearMpc(
            reactor_volume_m3=self.reactor_volume_m3,
            coolant_temp_c=self.coolant_temp_c,
            nominal_flow_m3_h=24.0,
        )
        self.nmpc_enabled = False
        self.latest_nmpc: Optional[NmpcResult] = None

    @property
    def reactor_pressure_bar(self) -> float:
        """Estimated base reactor pressure in bar."""
        return round(1.013 + (self.reactor_temp_c - 85.0) * 0.05, 3)

    def update_thermodynamics(self, flash_data: Optional[dict]) -> None:
        """Updates real-fluid thermodynamic state resolved from ChemFlow.NET."""
        if not flash_data:
            return
        self.latest_beta = flash_data.get("vaporFractionBeta")
        self.latest_phase = flash_data.get("phase")
        self.latest_thermo_model = flash_data.get("thermodynamicModel")
        self.latest_molar_enthalpy = flash_data.get("molarEnthalpyJPerMol")

    def update_transport_properties(self, tp: Optional[TransportProperties]) -> None:
        """Updates rigorous fluid transport properties resolved from ChemFlow.NET."""
        self.latest_transport = tp

    def update_cstr_battery(self, battery: Optional[CstrBatteryTelemetry]) -> None:
        """Updates parallel CSTR battery simulation state resolved from ChemFlow.NET."""
        self.latest_battery = battery

    def update_series_parallel_plant(self, plant_summary: Optional[dict]) -> None:
        """Stores the latest closed-loop series-parallel recycle plant summary."""
        self.latest_series_parallel_plant = plant_summary

    def update_series_parallel_profile(self, profile: Optional[dict]) -> None:
        """Stores the latest series-parallel nominal vs runaway thermal trajectory comparison."""
        self.latest_series_parallel_profile = profile

    def reset(self) -> None:
        """Resets the plant simulator back to its initial normal steady-state conditions."""
        self.tank_level_m = 5.0
        self.feed_valve_open = True
        self.feed_flow_m3_h = 24.0
        self.discharge_flow_m3_h = 24.0
        self.reactor_temp_c = 85.0
        self.cooling_valve_pct = 45.0
        self.interlock_tripped = False
        self.status = PlantStatus.NORMAL
        self.active_alarms.clear()
        self.ekf.reset(ca=0.5, temp_c=85.0)
        self.latest_ekf = None
        self.nmpc_enabled = False
        self.nmpc.reset(initial_u=45.0)
        self.latest_nmpc = None

    def apply_actuator_command(self, cmd: ActuatorCommand) -> None:
        """Applies manual operator overrides from SCADA/HMI interface."""
        if cmd.time_scale is not None:
            self.time_scale = max(1.0, min(20.0, cmd.time_scale))

        if cmd.nmpc_enabled is not None:
            self.nmpc_enabled = cmd.nmpc_enabled
            if self.nmpc_enabled:
                self.nmpc.reset(initial_u=self.cooling_valve_pct)

        if cmd.nmpc_target_temp_c is not None or cmd.nmpc_target_ca is not None:
            self.nmpc.update_setpoints(
                temp_setpoint_c=cmd.nmpc_target_temp_c,
                ca_setpoint=cmd.nmpc_target_ca,
            )

        if cmd.cooling_valve_pct is not None:
            self.cooling_valve_pct = max(0.0, min(100.0, cmd.cooling_valve_pct))
            # If manual cooling valve setpoint is explicitly given without specifying nmpc_enabled, disable NMPC
            if cmd.nmpc_enabled is None:
                self.nmpc_enabled = False

        if cmd.feed_valve_open is not None:
            self.feed_valve_open = cmd.feed_valve_open
            # If interlock was tripped, operator can reset it when conditions are not critical
            if cmd.feed_valve_open and self.status != PlantStatus.CRITICAL_SHUTDOWN:
                self.interlock_tripped = False

        if cmd.target_feed_flow_m3_h is not None:
            self.feed_flow_m3_h = max(0.0, min(60.0, cmd.target_feed_flow_m3_h))

        if cmd.target_discharge_flow_m3_h is not None:
            self.discharge_flow_m3_h = max(0.0, min(60.0, cmd.target_discharge_flow_m3_h))

    def step(self, dt_seconds: float = 0.2) -> PlantTelemetry:
        """
        Advances the physical plant simulation by dt_seconds using Euler integration.
        Returns the instantaneous noisy telemetry frame.
        """
        dt_eff = dt_seconds * self.time_scale
        dt_hours = dt_eff / 3600.0

        # Effective inlet flow depends on emergency feed valve state
        actual_feed_flow = self.feed_flow_m3_h if self.feed_valve_open else 0.0

        # Tank mass balance: dV/dt = Fin - Fout  =>  dh = (dV) / Area
        net_flow_m3 = (actual_feed_flow - self.discharge_flow_m3_h) * dt_hours
        self.tank_level_m += net_flow_m3 / self.tank_area_m2
        self.tank_level_m = max(0.0, min(self.tank_max_height_m + 1.0, self.tank_level_m))
        tank_volume = self.tank_level_m * self.tank_area_m2

        # CSTR energy balance:
        # Non-linear Arrhenius kinetics with conversion asymptotic bounds (Michaelis-Menten / CSTR X = Da / (1 + Da))
        # This prevents unphysical infinite reaction rate traps during thermal runaway excursions.
        # Calibrated around nominal operating point T = 85 °C (358.15 K)
        arrhenius_k = math.exp(-4200.0 * (1.0 / (self.reactor_temp_c + 273.15) - 1.0 / 358.15))
        conversion_factor = (2.0 * arrhenius_k) / (1.0 + arrhenius_k)
        # Reactant feed entering the CSTR is supplied by buffer tank discharge
        reactor_feed_active = self.feed_valve_open and self.tank_level_m > 0.05
        feed_avail = (self.discharge_flow_m3_h / 24.0) if reactor_feed_active else 0.15
        heat_generation = 0.26733 * conversion_factor * max(0.1, min(1.5, feed_avail))

        cooling_duty = (self.cooling_valve_pct / 100.0) * 0.008 * (self.reactor_temp_c - self.coolant_temp_c)
        feed_cooling = (self.discharge_flow_m3_h / (self.reactor_volume_m3 * 3600.0)) * (25.0 - self.reactor_temp_c)

        dT_dt = heat_generation + feed_cooling - cooling_duty
        self.reactor_temp_c += dT_dt * dt_eff
        self.reactor_temp_c = max(15.0, min(250.0, self.reactor_temp_c))

        # Vapor space pressure Antoine approximation (blanket + solvent vapor in bar)
        p_vap = math.exp(11.68 - 3816.44 / (self.reactor_temp_c + 227.02))
        reactor_pressure = 1.013 + p_vap

        # Construct measured telemetry with transmitter noise
        noisy_level = max(0.0, self.tank_level_m + random.gauss(0, self.level_noise_std))
        noisy_temp = max(15.0, self.reactor_temp_c + random.gauss(0, self.temp_noise_std))
        noisy_pressure = max(0.5, reactor_pressure + random.gauss(0, self.press_noise_std))

        # Run Extended Kalman Filter (EKF) step for unmeasured state & concentration estimation
        self.latest_ekf = self.ekf.step(
            dt_seconds=dt_eff,
            measured_temp_c=noisy_temp,
            flow_m3_h=self.discharge_flow_m3_h,
            cooling_valve_pct=self.cooling_valve_pct,
            feed_active=reactor_feed_active,
        )

        # NMPC Closed-Loop Execution (modulate cooling jacket valve using real-time EKF state estimates)
        if self.nmpc_enabled and not self.interlock_tripped and dt_seconds > 0.0:
            est_ca = self.latest_ekf.ca_mol_per_l if self.latest_ekf else 0.5
            est_temp = self.latest_ekf.temp_celsius if self.latest_ekf else noisy_temp
            self.latest_nmpc = self.nmpc.compute_optimal_move(
                current_ca=est_ca,
                current_temp_c=est_temp,
                flow_m3_h=self.discharge_flow_m3_h,
                feed_active=reactor_feed_active,
                u_prev=self.cooling_valve_pct,
            )
            self.cooling_valve_pct = self.latest_nmpc.optimal_cooling_valve_pct

        frame = PlantTelemetry(
            timestamp=datetime.now(timezone.utc),
            tank_level_m=round(noisy_level, 3),
            tank_volume_m3=round(noisy_level * self.tank_area_m2, 2),
            feed_flow_m3_h=round(actual_feed_flow, 2),
            discharge_flow_m3_h=round(self.discharge_flow_m3_h, 2),
            reactor_temp_c=round(noisy_temp, 2),
            reactor_pressure_bar=round(noisy_pressure, 3),
            cooling_valve_pct=round(self.cooling_valve_pct, 1),
            feed_valve_open=self.feed_valve_open,
            interlock_tripped=self.interlock_tripped,
            status=self.status,
            time_scale=self.time_scale,
            vapor_fraction_beta=self.latest_beta,
            thermo_phase=self.latest_phase,
            thermo_model=self.latest_thermo_model,
            molar_enthalpy_j_mol=self.latest_molar_enthalpy,
            estimated_ca_mol_l=self.latest_ekf.ca_mol_per_l if self.latest_ekf else None,
            estimated_temp_c=self.latest_ekf.temp_celsius if self.latest_ekf else None,
            estimated_conversion_pct=self.latest_ekf.conversion_pct if self.latest_ekf else None,
            estimated_reaction_rate=self.latest_ekf.reaction_rate if self.latest_ekf else None,
            ekf_temp_residual_c=self.latest_ekf.temp_residual if self.latest_ekf else None,
            ekf_nis=self.latest_ekf.nis if self.latest_ekf else None,
            nmpc_enabled=self.nmpc_enabled,
            nmpc_target_temp_c=self.nmpc.config.temp_setpoint_c if self.nmpc_enabled else None,
            nmpc_optimal_cooling_pct=self.latest_nmpc.optimal_cooling_valve_pct if (self.nmpc_enabled and self.latest_nmpc) else None,
            nmpc_solve_time_ms=self.latest_nmpc.solve_time_ms if (self.nmpc_enabled and self.latest_nmpc) else None,
            nmpc_cost=self.latest_nmpc.cost if (self.nmpc_enabled and self.latest_nmpc) else None,
            nmpc_predicted_temp_trajectory=self.latest_nmpc.predicted_temp_trajectory if (self.nmpc_enabled and self.latest_nmpc) else None,
            transport_properties=self.latest_transport,
            cstr_battery=self.latest_battery,
        )

        # Safety Interlock & Alarm Evaluation (SIS)
        status, new_alarms, should_trip = evaluate_safety_alarms(frame)
        self.status = status

        if should_trip and not self.interlock_tripped:
            # Automated safety response: shut inlet feed valve, disable NMPC, and flood cooling jacket!
            self.interlock_tripped = True
            self.nmpc_enabled = False
            self.sp_nmpc_enabled = False
            self.sp_cooling_valve_pct = 100.0
            self.feed_valve_open = False
            self.cooling_valve_pct = 100.0
            frame.nmpc_enabled = False
            frame.interlock_tripped = True
            frame.feed_valve_open = False
            frame.cooling_valve_pct = 100.0

        frame.status = self.status
        frame.sp_control = self._step_series_parallel(dt_seconds)

        # Deduplicate and register active alarms
        for alarm in new_alarms:
            if not any(a.tag == alarm.tag and not a.resolved for a in self.active_alarms):
                self.active_alarms.append(alarm)

        return frame

    def acknowledge_alarm(self, alarm_id: str) -> bool:
        """Acknowledges an active alarm event by ID."""
        for alarm in self.active_alarms:
            if alarm.id == alarm_id:
                alarm.acknowledged = True
                return True
        return False

    def enable_nmpc(
        self,
        enabled: bool,
        temp_setpoint_c: Optional[float] = None,
        ca_setpoint: Optional[float] = None,
    ) -> bool:
        """Enables or disables NMPC closed-loop controller."""
        self.nmpc_enabled = enabled
        if enabled:
            self.nmpc.update_setpoints(temp_setpoint_c=temp_setpoint_c, ca_setpoint=ca_setpoint)
            self.nmpc.reset(initial_u=self.cooling_valve_pct)
        return self.nmpc_enabled

    def update_nmpc_setpoints(
        self,
        temp_setpoint_c: Optional[float] = None,
        ca_setpoint: Optional[float] = None,
    ) -> None:
        """Updates NMPC tracking setpoints."""
        self.nmpc.update_setpoints(temp_setpoint_c=temp_setpoint_c, ca_setpoint=ca_setpoint)

    def get_nmpc_diagnostics(self) -> dict:
        """Returns comprehensive diagnostic telemetry from NMPC optimizer."""
        return {
            "enabled": self.nmpc_enabled,
            "config": {
                "prediction_horizon": self.nmpc.config.prediction_horizon,
                "control_horizon": self.nmpc.config.control_horizon,
                "temp_setpoint_c": self.nmpc.config.temp_setpoint_c,
                "ca_setpoint": self.nmpc.config.ca_setpoint,
                "max_safe_temp_c": self.nmpc.config.max_safe_temp_c,
                "max_delta_u_pct": self.nmpc.config.max_delta_u_pct,
                "nominal_u_pct": self.nmpc.config.nominal_u_pct,
            },
            "latest_result": {
                "optimal_cooling_valve_pct": self.latest_nmpc.optimal_cooling_valve_pct,
                "predicted_temp_trajectory": self.latest_nmpc.predicted_temp_trajectory,
                "predicted_ca_trajectory": self.latest_nmpc.predicted_ca_trajectory,
                "planned_control_moves": self.latest_nmpc.planned_control_moves,
                "cost": self.latest_nmpc.cost,
                "iterations": self.latest_nmpc.iterations,
                "solve_time_ms": self.latest_nmpc.solve_time_ms,
                "converged": self.latest_nmpc.converged,
                "temp_tracking_error": self.latest_nmpc.temp_tracking_error,
            } if self.latest_nmpc else None,
        }

    def _step_series_parallel(self, dt_seconds: float) -> SeriesParallelControlTelemetry:
        """Advances the 6-state Series-Parallel CSTR plant, 6-state EKF, and optional Economic NMPC."""
        dt_min = (max(0.0, float(dt_seconds)) * self.time_scale) / 60.0
        params = build_series_parallel_params(
            native_core=self.sp_core,
            cooling_valve_pct=self.sp_cooling_valve_pct,
            feed_temp_c=self.sp_feed_temp_c,
            jacket_inlet_temp_c=self.sp_jacket_inlet_temp_c,
            ca_feed_mol_per_l=self.sp_ca_feed,
        )

        if dt_min > 0.0:
            self.sp_state = step_series_parallel_substepped(self.sp_core, params, self.sp_state, dt_min)
            ca, cb, cc, cd, t_k, tj_k = self.sp_state
            meas_temp_c = (t_k - 273.15) + random.gauss(0.0, 0.15)
            meas_jacket_c = (tj_k - 273.15) + random.gauss(0.0, 0.10)
            self.latest_sp_ekf = self.sp_ekf.step(
                dt_min=dt_min,
                params=params,
                measured_temp_c=meas_temp_c,
                measured_jacket_temp_c=meas_jacket_c,
            )

            if self.sp_nmpc_enabled and not self.interlock_tripped:
                est = self.latest_sp_ekf
                self.latest_sp_nmpc = self.sp_nmpc.compute_optimal_move(
                    current_state_c=(
                        est.ca_mol_per_l,
                        est.cb_mol_per_l,
                        est.cc_mol_per_l,
                        est.cd_mol_per_l,
                        est.temp_c,
                        est.jacket_temp_c,
                    ),
                    base_params=params,
                    u_prev=self.sp_cooling_valve_pct,
                )
                self.sp_cooling_valve_pct = self.latest_sp_nmpc.optimal_cooling_valve_pct
        elif self.latest_sp_ekf is None:
            ca, cb, cc, cd, t_k, tj_k = self.sp_state
            self.latest_sp_ekf = self.sp_ekf.step(
                dt_min=0.0,
                params=params,
                measured_temp_c=t_k - 273.15,
                measured_jacket_temp_c=tj_k - 273.15,
            )

        ca, cb, cc, cd, t_k, tj_k = self.sp_state
        t_c = t_k - 273.15
        tj_c = tj_k - 273.15
        conv_a, yield_b, sel_b = compute_series_parallel_kpis(ca, cb, params.ca_feed_mol_per_l)
        est = self.latest_sp_ekf

        telemetry = SeriesParallelControlTelemetry(
            ca_mol_per_l=round(ca, 4),
            cb_mol_per_l=round(cb, 4),
            cc_mol_per_l=round(cc, 4),
            cd_mol_per_l=round(cd, 4),
            reactor_temp_c=round(t_c, 2),
            jacket_temp_c=round(tj_c, 2),
            conversion_a_pct=round(conv_a, 2),
            yield_b_pct=round(yield_b, 2),
            selectivity_b_pct=round(sel_b, 2),
            est_ca_mol_per_l=est.ca_mol_per_l if est else round(ca, 4),
            est_cb_mol_per_l=est.cb_mol_per_l if est else round(cb, 4),
            est_cc_mol_per_l=est.cc_mol_per_l if est else round(cc, 4),
            est_cd_mol_per_l=est.cd_mol_per_l if est else round(cd, 4),
            est_temp_c=est.temp_c if est else round(t_c, 2),
            est_jacket_temp_c=est.jacket_temp_c if est else round(tj_c, 2),
            ekf_nis=est.nis if est else 0.0,
            sensor_healthy=est.sensor_healthy if est else True,
            nmpc_enabled=self.sp_nmpc_enabled,
            cooling_valve_pct=round(self.sp_cooling_valve_pct, 2),
            feed_temp_c=round(self.sp_feed_temp_c, 2),
            jacket_inlet_temp_c=round(self.sp_jacket_inlet_temp_c, 2),
            max_safe_temp_c=round(self.sp_nmpc.config.max_safe_temp_c, 2),
            target_cb_mol_per_l=self.sp_nmpc.config.target_cb_mol_per_l,
            nmpc_solve_time_ms=self.latest_sp_nmpc.solve_time_ms if self.latest_sp_nmpc else None,
            nmpc_cost=self.latest_sp_nmpc.cost if self.latest_sp_nmpc else None,
            is_thermal_runaway=(t_c > self.sp_nmpc.config.max_safe_temp_c),
        )
        self.latest_sp_telemetry = telemetry
        return telemetry

    def reset_series_parallel_control(self) -> None:
        """Resets 6-state Series-Parallel CSTR, EKF, and Economic NMPC to nominal steady state."""
        self.sp_nmpc_enabled = False
        self.sp_cooling_valve_pct = 55.0
        self.sp_feed_temp_c = 51.85
        self.sp_jacket_inlet_temp_c = 21.85
        self.sp_state = (0.48, 0.28, 0.21, 0.03, 62.0 + 273.15, 36.0 + 273.15)
        self.sp_ekf.reset()
        self.sp_nmpc.reset(initial_u=self.sp_cooling_valve_pct)
        self.latest_sp_ekf = None
        self.latest_sp_nmpc = None
        self._step_series_parallel(0.0)

    def update_series_parallel_control(self, cmd: SeriesParallelControlConfigCommand) -> dict:
        """Applies operator / OPC UA / Modbus configuration commands to 6-state Series-Parallel control."""
        if cmd.cooling_valve_pct is not None:
            self.sp_cooling_valve_pct = max(0.0, min(100.0, float(cmd.cooling_valve_pct)))
        if cmd.feed_temp_c is not None:
            self.sp_feed_temp_c = max(10.0, min(180.0, float(cmd.feed_temp_c)))
        if cmd.jacket_inlet_temp_c is not None:
            self.sp_jacket_inlet_temp_c = max(-20.0, min(120.0, float(cmd.jacket_inlet_temp_c)))

        self.sp_nmpc.update_config(
            max_safe_temp_c=cmd.max_safe_temp_c,
            target_cb_mol_per_l=cmd.target_cb_mol_per_l,
            clear_target_cb=cmd.clear_target_cb,
        )

        if cmd.nmpc_enabled is not None:
            self.sp_nmpc_enabled = bool(cmd.nmpc_enabled)
            if self.sp_nmpc_enabled:
                self.sp_nmpc.reset(initial_u=self.sp_cooling_valve_pct)

        snap = self._step_series_parallel(0.0)
        return self.get_series_parallel_control_diagnostics(snapshot=snap)

    def get_series_parallel_control_diagnostics(
        self,
        snapshot: Optional[SeriesParallelControlTelemetry] = None,
    ) -> dict:
        """Returns comprehensive diagnostics for the 6-state Series-Parallel EKF and Economic NMPC."""
        snap = snapshot or self.latest_sp_telemetry or self._step_series_parallel(0.0)
        return {
            "telemetry": snap.model_dump(),
            "ekf": {
                "cov_trace": self.latest_sp_ekf.cov_trace if self.latest_sp_ekf else None,
                "temp_residual_c": self.latest_sp_ekf.temp_residual_c if self.latest_sp_ekf else None,
                "jacket_residual_c": self.latest_sp_ekf.jacket_residual_c if self.latest_sp_ekf else None,
                "nis": self.latest_sp_ekf.nis if self.latest_sp_ekf else None,
                "sensor_healthy": self.latest_sp_ekf.sensor_healthy if self.latest_sp_ekf else True,
            },
            "nmpc": {
                "enabled": self.sp_nmpc_enabled,
                "config": {
                    "prediction_horizon": self.sp_nmpc.config.prediction_horizon,
                    "control_horizon": self.sp_nmpc.config.control_horizon,
                    "dt_step_min": self.sp_nmpc.config.dt_step_min,
                    "max_safe_temp_c": self.sp_nmpc.config.max_safe_temp_c,
                    "target_cb_mol_per_l": self.sp_nmpc.config.target_cb_mol_per_l,
                },
                "latest_result": {
                    "optimal_cooling_valve_pct": self.latest_sp_nmpc.optimal_cooling_valve_pct,
                    "predicted_cb_trajectory": self.latest_sp_nmpc.predicted_cb_trajectory,
                    "predicted_temp_c_trajectory": self.latest_sp_nmpc.predicted_temp_c_trajectory,
                    "predicted_selectivity_trajectory": self.latest_sp_nmpc.predicted_selectivity_trajectory,
                    "planned_control_moves": self.latest_sp_nmpc.planned_control_moves,
                    "cost": self.latest_sp_nmpc.cost,
                    "iterations": self.latest_sp_nmpc.iterations,
                    "solve_time_ms": self.latest_sp_nmpc.solve_time_ms,
                    "converged": self.latest_sp_nmpc.converged,
                } if self.latest_sp_nmpc else None,
            },
        }
