import uuid
from typing import List, Tuple
from process_twin.core.models import (
    PlantTelemetry,
    AlarmEvent,
    AlarmSeverity,
    PlantStatus,
)


class AlarmThresholds:
    LEVEL_HIGH_WARN = 8.0          # m (Warning)
    LEVEL_CRITICAL_TRIP = 9.5      # m (Automatic interlock: shut feed)
    TEMP_HIGH_WARN = 115.0         # °C (Warning)
    TEMP_CRITICAL_TRIP = 135.0     # °C (Automatic interlock: flood cooling, shut feed)
    PRESSURE_HIGH_WARN = 4.0       # bar (Warning)
    PRESSURE_CRITICAL_TRIP = 5.5   # bar (Overpressure relief)
    BATTERY_TEMP_HIGH_WARN = 105.0 # °C (Warning across battery)
    BATTERY_TEMP_CRITICAL_TRIP = 126.85 # °C (400 K Runaway trip)


def evaluate_safety_alarms(telemetry: PlantTelemetry) -> Tuple[PlantStatus, List[AlarmEvent], bool]:
    """
    Evaluates sensor readings against critical safety thresholds (HAZOP / Safety Instrumented Systems).
    Returns:
        (updated_status, list_of_new_alarms, interlock_tripped)
    """
    alarms: List[AlarmEvent] = []
    interlock_tripped = False
    has_critical = False
    has_warning = False

    # 1. Level Transmitters (LT-101)
    if telemetry.tank_level_m >= AlarmThresholds.LEVEL_CRITICAL_TRIP:
        has_critical = True
        interlock_tripped = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.CRITICAL,
            tag="LT-101",
            message=f"CRITICAL HIGH LEVEL ({telemetry.tank_level_m:.2f} m >= {AlarmThresholds.LEVEL_CRITICAL_TRIP} m): Emergency feed shutoff tripped!",
        ))
    elif telemetry.tank_level_m >= AlarmThresholds.LEVEL_HIGH_WARN:
        has_warning = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.WARNING,
            tag="LT-101",
            message=f"High buffer tank level ({telemetry.tank_level_m:.2f} m >= {AlarmThresholds.LEVEL_HIGH_WARN} m)",
        ))

    # 2. Temperature Transmitters (TT-102)
    if telemetry.reactor_temp_c >= AlarmThresholds.TEMP_CRITICAL_TRIP:
        has_critical = True
        interlock_tripped = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.CRITICAL,
            tag="TT-102",
            message=f"CRITICAL REACTOR RUNAWAY ({telemetry.reactor_temp_c:.1f} °C >= {AlarmThresholds.TEMP_CRITICAL_TRIP} °C): Emergency cooling flood tripped!",
        ))
    elif telemetry.reactor_temp_c >= AlarmThresholds.TEMP_HIGH_WARN:
        has_warning = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.WARNING,
            tag="TT-102",
            message=f"High reactor temperature warning ({telemetry.reactor_temp_c:.1f} °C >= {AlarmThresholds.TEMP_HIGH_WARN} °C)",
        ))

    # 3. Pressure Transmitters (PT-103)
    if telemetry.reactor_pressure_bar >= AlarmThresholds.PRESSURE_CRITICAL_TRIP:
        has_critical = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.CRITICAL,
            tag="PT-103",
            message=f"CRITICAL OVERPRESSURE ({telemetry.reactor_pressure_bar:.2f} bar >= {AlarmThresholds.PRESSURE_CRITICAL_TRIP} bar)",
        ))
    elif telemetry.reactor_pressure_bar >= AlarmThresholds.PRESSURE_HIGH_WARN:
        has_warning = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.WARNING,
            tag="PT-103",
            message=f"High reactor pressure warning ({telemetry.reactor_pressure_bar:.2f} bar >= {AlarmThresholds.PRESSURE_HIGH_WARN} bar)",
        ))

    # 4. CSTR Battery Transmitters (TT-BATTERY)
    if telemetry.cstr_battery:
        if telemetry.cstr_battery.has_thermal_runaway or telemetry.cstr_battery.mean_temp_c >= AlarmThresholds.BATTERY_TEMP_CRITICAL_TRIP:
            has_critical = True
            interlock_tripped = True
            alarms.append(AlarmEvent(
                id=str(uuid.uuid4())[:8],
                severity=AlarmSeverity.CRITICAL,
                tag="TT-BATTERY",
                message=f"CRITICAL BATTERY RUNAWAY: Thermal runaway detected in parallel CSTR train (Mean T = {telemetry.cstr_battery.mean_temp_c:.1f} °C)!",
            ))
        elif telemetry.cstr_battery.mean_temp_c >= AlarmThresholds.BATTERY_TEMP_HIGH_WARN:
            has_warning = True
            alarms.append(AlarmEvent(
                id=str(uuid.uuid4())[:8],
                severity=AlarmSeverity.WARNING,
                tag="TT-BATTERY",
                message=f"High battery temperature warning ({telemetry.cstr_battery.mean_temp_c:.1f} °C >= {AlarmThresholds.BATTERY_TEMP_HIGH_WARN} °C)",
            ))

    # 5. 6-State Series-Parallel CSTR Thermal Runaway (SP-TT-201)
    if telemetry.sp_control and (
        telemetry.sp_control.is_thermal_runaway
        or telemetry.sp_control.reactor_temp_c >= telemetry.sp_control.max_safe_temp_c
    ):
        has_critical = True
        interlock_tripped = True
        alarms.append(AlarmEvent(
            id=str(uuid.uuid4())[:8],
            severity=AlarmSeverity.CRITICAL,
            tag="SP-TT-201",
            message=(
                f"CRITICAL SERIES-PARALLEL RUNAWAY ({telemetry.sp_control.reactor_temp_c:.1f} °C "
                f">= {telemetry.sp_control.max_safe_temp_c:.1f} °C): Flooding SP cooling jacket!"
            ),
        ))

    # Determine aggregated plant status
    if has_critical:
        status = PlantStatus.CRITICAL_SHUTDOWN
    elif has_warning:
        status = PlantStatus.HIGH_ALARM
    else:
        status = PlantStatus.NORMAL

    return status, alarms, interlock_tripped
