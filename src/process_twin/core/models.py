from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Literal, List
from pydantic import BaseModel, Field


class AlarmSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PlantStatus(str, Enum):
    NORMAL = "NORMAL"
    HIGH_ALARM = "HIGH_ALARM"
    CRITICAL_SHUTDOWN = "CRITICAL_SHUTDOWN"


class TransportProperties(BaseModel):
    """Rigorous fluid transport properties calculated via ChemFlow.NET (Chung or Ely-Hanley TRAPP)."""
    dynamic_viscosity_cp: float = Field(..., description="Dynamic viscosity in centipoise (cP)")
    kinematic_viscosity_cst: float = Field(..., description="Kinematic viscosity in centistokes (cSt)")
    thermal_conductivity_w_m_k: float = Field(..., description="Thermal conductivity in W/(m·K)")
    prandtl_number: float = Field(..., description="Dimensionless Prandtl number Pr")
    density_kg_m3: float = Field(..., description="Fluid density in kg/m³")
    phase: str = Field(default="SubcooledLiquid", description="Thermodynamic phase")
    model_name: str = Field(default="Chung et al. (1984/1988)", description="Transport model name")


class CstrBatteryReactorState(BaseModel):
    """Telemetry state for an individual reactor in a multi-reactor parallel battery."""
    reactor_index: int = Field(..., description="Zero-based index of reactor in battery")
    ca_mol_l: float = Field(..., description="Reactant concentration (mol/L)")
    temp_c: float = Field(..., description="Reactor temperature (°C)")
    conversion_pct: float = Field(..., description="Reactant conversion percentage (%)")
    reaction_rate_mol_l_min: float = Field(..., description="Instantaneous reaction rate (mol/(L·min))")
    is_runaway: bool = Field(..., description="Thermal runaway threshold exceeded flag")
    dynamic_viscosity_cp: float = Field(..., description="Dynamic viscosity in reactor (cP)")
    thermal_conductivity_w_m_k: float = Field(..., description="Thermal conductivity in reactor (W/(m·K))")
    prandtl_number: float = Field(..., description="Prandtl number in reactor")


class CstrBatteryTelemetry(BaseModel):
    """Aggregated telemetry for a train of parallel CSTR reactors computed via SIMD AVX2."""
    battery_id: str = Field(default="CSTR-BATTERY-01")
    name: str = Field(default="Parallel CSTR Battery")
    reactor_count: int = Field(default=8)
    mean_conversion_pct: float = Field(..., description="Average conversion across battery (%)")
    mean_temp_c: float = Field(..., description="Mean effluent temperature (°C)")
    mean_ca_mol_l: float = Field(..., description="Mean outlet reactant concentration (mol/L)")
    has_thermal_runaway: bool = Field(..., description="True if any reactor exceeded safe temperature threshold")
    is_using_native_simd: bool = Field(default=True, description="True if AVX2/FMA vector instructions were used")
    energy_duty_kw: float = Field(default=0.0, description="Total battery heat exchange duty (kW)")
    reactor_states: List[CstrBatteryReactorState] = Field(default_factory=list)


class PlantTelemetry(BaseModel):
    """Represents a real-time snapshot frame from industrial plant transmitters."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tank_level_m: float = Field(..., ge=0.0, le=12.0, description="Buffer tank liquid level (m)")
    tank_volume_m3: float = Field(..., ge=0.0, description="Buffer tank volume (m³)")
    feed_flow_m3_h: float = Field(..., ge=0.0, description="Inlet feed flow rate (m³/h)")
    discharge_flow_m3_h: float = Field(..., ge=0.0, description="Outlet discharge flow rate (m³/h)")
    reactor_temp_c: float = Field(..., description="CSTR reactor core temperature (°C)")
    reactor_pressure_bar: float = Field(..., ge=0.0, description="Reactor vapor space pressure (bar)")
    cooling_valve_pct: float = Field(..., ge=0.0, le=100.0, description="Cooling jacket control valve position (%)")
    feed_valve_open: bool = Field(..., description="Automated emergency feed shutoff valve status")
    interlock_tripped: bool = Field(default=False, description="Whether safety interlock tripped automated shutdown")
    status: PlantStatus = Field(default=PlantStatus.NORMAL)
    time_scale: float = Field(default=5.0, ge=1.0, le=20.0, description="Simulation speed acceleration factor")
    vapor_fraction_beta: Optional[float] = Field(default=None, description="Rigorous VLE vapor fraction beta from ChemFlow.NET")
    thermo_phase: Optional[str] = Field(default=None, description="Equilibrium phase state from ChemFlow.NET")
    thermo_model: Optional[str] = Field(default=None, description="Thermodynamic model applied (e.g. Peng-Robinson EOS)")
    molar_enthalpy_j_mol: Optional[float] = Field(default=None, description="Fluid molar enthalpy departure in J/mol")
    # Transport properties and CSTR Battery telemetry
    transport_properties: Optional[TransportProperties] = Field(default=None, description="Rigorous fluid transport properties from ChemFlow.NET")
    cstr_battery: Optional[CstrBatteryTelemetry] = Field(default=None, description="Parallel CSTR Battery simulation and state ensemble")
    # Extended Kalman Filter (EKF) state estimation fields
    estimated_ca_mol_l: Optional[float] = Field(default=None, description="Unmeasured reactant concentration Ca estimated via EKF")
    estimated_temp_c: Optional[float] = Field(default=None, description="Filtered temperature state from EKF")
    estimated_conversion_pct: Optional[float] = Field(default=None, description="Estimated conversion percentage X_A from EKF")
    estimated_reaction_rate: Optional[float] = Field(default=None, description="Estimated instantaneous reaction rate from EKF")
    ekf_temp_residual_c: Optional[float] = Field(default=None, description="EKF temperature innovation residual y - y_hat (°C)")
    ekf_nis: Optional[float] = Field(default=None, description="Normalized Innovation Squared for sensor drift/anomaly detection")
    # Non-linear Model Predictive Control (NMPC) fields
    nmpc_enabled: bool = Field(default=False, description="Whether NMPC closed-loop control is actively modulating cooling valve")
    nmpc_target_temp_c: Optional[float] = Field(default=None, description="Active NMPC reactor temperature setpoint (°C)")
    nmpc_optimal_cooling_pct: Optional[float] = Field(default=None, description="Optimal cooling valve position computed by NMPC optimizer (%)")
    nmpc_solve_time_ms: Optional[float] = Field(default=None, description="Execution time of NMPC optimizer in milliseconds")
    nmpc_cost: Optional[float] = Field(default=None, description="Objective function cost value of optimal trajectory")
    nmpc_predicted_temp_trajectory: Optional[List[float]] = Field(default=None, description="Forward predicted temperature trajectory across prediction horizon")


class AlarmEvent(BaseModel):
    """Represents an annunciated safety alarm or process interlock event."""
    id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlarmSeverity
    tag: str  # e.g., "LT-101", "TT-102", "PT-103"
    message: str
    acknowledged: bool = False
    resolved: bool = False


class ActuatorCommand(BaseModel):
    """Operator supervisory control command (SCADA setpoint override)."""
    cooling_valve_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    feed_valve_open: Optional[bool] = None
    target_feed_flow_m3_h: Optional[float] = Field(None, ge=0.0, le=60.0)
    target_discharge_flow_m3_h: Optional[float] = Field(None, ge=0.0, le=60.0)
    time_scale: Optional[float] = Field(None, ge=1.0, le=20.0)
    nmpc_enabled: Optional[bool] = Field(None, description="Toggle NMPC closed-loop control")
    nmpc_target_temp_c: Optional[float] = Field(None, ge=20.0, le=150.0, description="NMPC reactor temperature setpoint (°C)")
    nmpc_target_ca: Optional[float] = Field(None, ge=0.01, le=2.0, description="NMPC reactant concentration setpoint (mol/L)")


class NmpcToggleCommand(BaseModel):
    """Command to enable or disable NMPC closed-loop control."""
    enabled: bool = Field(..., description="Enable or disable NMPC closed-loop control")
    temp_setpoint_c: Optional[float] = Field(None, ge=20.0, le=150.0, description="Optional target reactor temperature (°C)")
    ca_setpoint: Optional[float] = Field(None, ge=0.01, le=2.0, description="Optional target concentration Ca (mol/L)")


class NmpcSetpointsCommand(BaseModel):
    """Command to update NMPC tracking setpoints."""
    temp_setpoint_c: Optional[float] = Field(None, ge=20.0, le=150.0, description="Target reactor temperature (°C)")
    ca_setpoint: Optional[float] = Field(None, ge=0.01, le=2.0, description="Optional target concentration Ca (mol/L)")


class TransportCalculationRequest(BaseModel):
    """Request payload to compute fluid transport properties via ChemFlow.NET."""
    temperature_c: float = Field(default=85.0, description="Temperature (°C)")
    pressure_bar: float = Field(default=1.5, description="Pressure (bar)")
    compound_ids: Optional[List[str]] = Field(default=None, description="Compound IDs (e.g. ['C6H6', 'C7H8'])")
    composition_z: Optional[List[float]] = Field(default=None, description="Mole fractions summing to 1.0")
    package: str = Field(default="Chung", description="Transport package ('Chung' or 'ElyHanley')")


class CstrBatterySimulationRequest(BaseModel):
    """Request payload to simulate a parallel CSTR battery ensemble."""
    reactor_count: int = Field(default=8, ge=1, le=64, description="Number of parallel CSTR reactors")
    total_feed_flow_mol_s: float = Field(default=100.0, ge=1.0, description="Total feed molar flow rate (mol/s)")
    feed_temp_c: float = Field(default=76.85, description="Feed temperature (°C)")
    feed_pressure_bar: float = Field(default=1.5, description="Feed pressure (bar)")
    reactant_id: str = Field(default="C6H6", description="Reactant compound identifier")
    product_id: str = Field(default="C7H8", description="Product compound identifier")
    composition_z: Optional[List[float]] = Field(default=None, description="Feed mole fractions [z_A, z_B]")
    reactor_volume_l: float = Field(default=100.0, ge=1.0, description="Volume per reactor in Liters")
    coolant_temp_c: float = Field(default=26.85, description="Cooling jacket inlet temperature (°C)")
    cooling_ua_w_k: float = Field(default=666.67, ge=0.0, description="Overall heat transfer coefficient UA (W/K)")
    max_safe_temp_c: float = Field(default=126.85, description="Maximum allowable temperature before runaway trip (°C)")
    package: str = Field(default="Chung", description="Transport property package ('Chung' or 'ElyHanley')")


