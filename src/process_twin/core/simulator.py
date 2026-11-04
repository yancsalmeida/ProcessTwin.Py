import math
import random
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional
from process_twin.core.models import (
    PlantTelemetry,
    AlarmEvent,
    ActuatorCommand,
    PlantStatus,
)
from process_twin.core.alarms import evaluate_safety_alarms


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

        # Alarms & events
        self.active_alarms: List[AlarmEvent] = []
        self.interlock_tripped = False
        self.status = PlantStatus.NORMAL

        # Sensor noise settings (standard deviation)
        self.level_noise_std = 0.02   # ± 2 cm noise
        self.temp_noise_std = 0.15    # ± 0.15 °C noise
        self.press_noise_std = 0.015  # ± 0.015 bar noise

    def apply_actuator_command(self, cmd: ActuatorCommand) -> None:
        """Applies manual operator overrides from SCADA/HMI interface."""
        if cmd.time_scale is not None:
            self.time_scale = max(1.0, min(20.0, cmd.time_scale))

        if cmd.cooling_valve_pct is not None:
            self.cooling_valve_pct = max(0.0, min(100.0, cmd.cooling_valve_pct))

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
        # Rate of temperature rise ~ feed sensible heat + non-linear Arrhenius heat gen - cooling removal
        # Calibrated around nominal operating point T = 85 °C (358.15 K)
        arrhenius_factor = math.exp(-4200.0 * (1.0 / (self.reactor_temp_c + 273.15) - 1.0 / 358.15))
        heat_generation = 0.1503 * arrhenius_factor  # °C/s heat generation
        cooling_duty = (self.cooling_valve_pct / 100.0) * 0.004 * (self.reactor_temp_c - self.coolant_temp_c)
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
        )

        # Safety Interlock & Alarm Evaluation (SIS)
        status, new_alarms, should_trip = evaluate_safety_alarms(frame)
        self.status = status

        if should_trip and not self.interlock_tripped:
            # Automated safety response: shut inlet feed valve and flood cooling jacket!
            self.interlock_tripped = True
            self.feed_valve_open = False
            self.cooling_valve_pct = 100.0
            frame.interlock_tripped = True
            frame.feed_valve_open = False
            frame.cooling_valve_pct = 100.0

        frame.status = self.status

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
