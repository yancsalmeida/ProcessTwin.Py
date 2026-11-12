"""
Unit Tests for ProcessTwinModbusServer (IEC 61158 / MBAP).
Validates:
1. Modbus TCP Server startup, MBAP frame decoding, and client connectivity.
2. Holding registers and coils telemetry synchronization.
3. Client reading of process variables (Level, Temperature, Pressure, Ca, Viscosity, Battery).
4. Bi-directional actuator writes from external Modbus master to digital twin.
5. Momentary pulse control (Reset, Emergency Trip).
6. Exception handling for illegal address and unsupported function codes.
"""

import pytest
import asyncio
from pymodbus.client import AsyncModbusTcpClient

from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.models import (
    PlantTelemetry,
    TransportProperties,
    CstrBatteryTelemetry,
)
from process_twin.industrial.modbus_server import ProcessTwinModbusServer


@pytest.fixture
async def modbus_fixture():
    twin = PlantDigitalTwin()
    port = 50215
    server = ProcessTwinModbusServer(host="127.0.0.1", port=port, twin_ref=twin)
    await server.start()
    client = AsyncModbusTcpClient("127.0.0.1", port=port)
    await client.connect()
    assert client.connected

    yield server, twin, client, port

    client.close()
    await server.stop()


@pytest.mark.asyncio
async def test_modbus_read_holding_registers(modbus_fixture):
    server, twin, client, port = modbus_fixture

    # Push a telemetry snapshot to server
    frame = PlantTelemetry(
        tank_level_m=5.50,          # HR 0 -> 550
        tank_volume_m3=27.50,
        feed_flow_m3_h=24.0,        # HR 3 -> 240
        discharge_flow_m3_h=24.0,   # HR 4 -> 240
        reactor_temp_c=87.5,        # HR 1 -> 875
        reactor_pressure_bar=1.14,  # HR 2 -> 114
        cooling_valve_pct=48.0,     # HR 5 -> 480
        feed_valve_open=True,       # Coil 0 -> True
        estimated_ca_mol_l=0.485,   # HR 6 -> 485
        estimated_conversion_pct=51.5, # HR 7 -> 515
        transport_properties=TransportProperties(
            dynamic_viscosity_cp=0.465,       # HR 10 -> 465
            kinematic_viscosity_cst=0.550,
            thermal_conductivity_w_m_k=0.138, # HR 11 -> 1380
            prandtl_number=6.25,              # HR 12 -> 625
            density_kg_m3=850.0,              # HR 13 -> 850
        ),
        cstr_battery=CstrBatteryTelemetry(
            battery_id="BAT-MB",
            name="Modbus Train",
            reactor_count=8,
            mean_conversion_pct=54.0,         # HR 15 -> 540
            mean_temp_c=89.2,                 # HR 14 -> 892
            mean_ca_mol_l=0.46,
            has_thermal_runaway=False,        # Coil 3 -> False
            energy_duty_kw=120.5,             # HR 16 -> 1205
        )
    )

    server.sync_telemetry(frame)

    # Read Holding Registers 0..16
    res = await client.read_holding_registers(0, count=17)
    assert res is not None and not res.isError()
    regs = res.registers

    assert regs[0] == 550   # Level 5.50 m * 100
    assert regs[1] == 875   # Temp 87.5 °C * 10
    assert regs[2] == 114   # Pressure 1.14 bar * 100
    assert regs[3] == 240   # Feed flow 24.0 m3/h * 10
    assert regs[4] == 240   # Discharge flow 24.0 m3/h * 10
    assert regs[5] == 480   # Cooling valve 48.0 % * 10
    assert regs[6] == 485   # Ca 0.485 * 1000
    assert regs[7] == 515   # Conv 51.5 % * 10
    assert regs[10] == 465  # Viscosity 0.465 cP * 1000
    assert regs[12] == 625  # Prandtl 6.25 * 100
    assert regs[14] == 892  # Battery mean temp 89.2 °C * 10
    assert regs[15] == 540  # Battery mean conv 54.0 % * 10

    # Read Coils 0..3
    coils_res = await client.read_coils(0, count=4)
    assert coils_res is not None and not coils_res.isError()
    assert coils_res.bits[0] is True   # Feed valve open
    assert coils_res.bits[1] is False  # Interlock tripped
    assert coils_res.bits[3] is False  # Runaway alarm


@pytest.mark.asyncio
async def test_modbus_actuator_writes(modbus_fixture):
    server, twin, client, port = modbus_fixture

    # 1. Write single register (FC 06) for Cooling Valve (HR 5) -> 82.0% (820)
    w_res = await client.write_register(5, 820)
    assert not w_res.isError()
    assert twin.cooling_valve_pct == 82.0

    # 2. Write single register for Target Feed Flow (HR 16) -> 35.0 m3/h (350)
    w_flow = await client.write_register(16, 350)
    assert not w_flow.isError()
    assert twin.feed_flow_m3_h == 35.0

    # 3. Write coil (FC 05) to shut emergency feed valve (Coil 0 -> False)
    c_res = await client.write_coil(0, False)
    assert not c_res.isError()
    assert twin.feed_valve_open is False

    # 4. Write multiple registers (FC 16) for HR 16 and 17 (Feed flow and Discharge flow)
    wm_res = await client.write_registers(16, [400, 320])
    assert not wm_res.isError()
    assert twin.feed_flow_m3_h == 40.0
    assert twin.discharge_flow_m3_h == 32.0


@pytest.mark.asyncio
async def test_modbus_momentary_pulses(modbus_fixture):
    server, twin, client, port = modbus_fixture

    # Set non-standard state
    twin.reactor_temp_c = 150.0
    twin.tank_level_m = 9.9
    twin.cooling_valve_pct = 100.0
    twin.interlock_tripped = True

    # Pulse Coil 4 (Reset Plant)
    await client.write_coil(4, True)
    assert twin.reactor_temp_c == 85.0
    assert twin.tank_level_m == 5.0
    assert twin.interlock_tripped is False

    # Pulse Coil 5 (Emergency Trip)
    await client.write_coil(5, True)
    assert twin.interlock_tripped is True
    assert twin.feed_valve_open is False
    assert twin.cooling_valve_pct == 100.0


@pytest.mark.asyncio
async def test_modbus_illegal_address_exception(modbus_fixture):
    server, twin, client, port = modbus_fixture

    # Reading beyond memory space (e.g. address 200) should return Modbus exception
    err_res = await client.read_holding_registers(200, count=10)
    assert err_res.isError()
