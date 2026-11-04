from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


class AlarmSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class PlantStatus(str, Enum):
    NORMAL = "NORMAL"
    HIGH_ALARM = "HIGH_ALARM"
    CRITICAL_SHUTDOWN = "CRITICAL_SHUTDOWN"


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
