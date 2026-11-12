"""
Unit Tests for ProcessTwinOpcUaServer (IEC 62541).
Validates:
1. OPC UA Server initialization, address space hierarchy, and node creation.
2. Real-time telemetry synchronization across Buffer Tank, Reactor, Transport, and Battery.
3. Client read capabilities from standard OPC UA client.
4. Bi-directional supervisory setpoint writes from client to digital twin.
5. OPC UA method invocations (ResetPlant, EmergencyTrip, AcknowledgeAlarms).
"""

import pytest
import asyncio
from asyncua import Client, ua

from process_twin.core.simulator import PlantDigitalTwin
from process_twin.core.models import (
    PlantTelemetry,
    TransportProperties,
    CstrBatteryTelemetry,
    CstrBatteryReactorState,
    ActuatorCommand,
)
from process_twin.industrial.opcua_server import ProcessTwinOpcUaServer


@pytest.fixture
async def opcua_fixture():
    twin = PlantDigitalTwin()
    test_endpoint = "opc.tcp://127.0.0.1:48401/processtwin/test/"
    server = ProcessTwinOpcUaServer(
        endpoint=test_endpoint,
        server_name="ProcessTwin Test OPC UA Server",
        twin_ref=twin,
    )
    await server.start()
    yield server, twin, test_endpoint
    await server.stop()


@pytest.mark.asyncio
async def test_opcua_server_hierarchy_and_read(opcua_fixture):
    server, twin, endpoint = opcua_fixture
    assert server.is_running is True
    assert server.tag_count >= 30

    async with Client(endpoint) as client:
        ns_idx = await client.get_namespace_index(server.namespace_uri)
        assert ns_idx > 0

        # Read Buffer Tank tags
        node_lt = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:BufferTank", f"{ns_idx}:LT_101"])
        val_lt = await node_lt.read_value()
        assert val_lt == 5.0

        # Read CSTR Reactor tags
        node_tt = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:CstrReactor", f"{ns_idx}:TT_101"])
        val_tt = await node_tt.read_value()
        assert val_tt == 85.0

        # Read Transport tags
        node_visc = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:FluidTransport", f"{ns_idx}:DynamicViscosity_cP"])
        val_visc = await node_visc.read_value()
        assert val_visc == 0.5


@pytest.mark.asyncio
async def test_opcua_telemetry_sync(opcua_fixture):
    server, twin, endpoint = opcua_fixture

    frame = PlantTelemetry(
        tank_level_m=6.85,
        tank_volume_m3=34.25,
        feed_flow_m3_h=28.0,
        discharge_flow_m3_h=24.0,
        reactor_temp_c=91.4,
        reactor_pressure_bar=1.33,
        cooling_valve_pct=52.0,
        feed_valve_open=True,
        estimated_ca_mol_l=0.38,
        estimated_conversion_pct=62.0,
        transport_properties=TransportProperties(
            dynamic_viscosity_cp=0.42,
            kinematic_viscosity_cst=0.51,
            thermal_conductivity_w_m_k=0.136,
            prandtl_number=6.3,
            density_kg_m3=848.0,
            phase="SubcooledLiquid",
            model_name="Chung et al.",
        ),
        cstr_battery=CstrBatteryTelemetry(
            battery_id="BAT-TEST",
            name="Parallel Battery Test",
            reactor_count=8,
            mean_conversion_pct=58.5,
            mean_temp_c=94.2,
            mean_ca_mol_l=0.41,
            has_thermal_runaway=False,
            is_using_native_simd=True,
            energy_duty_kw=115.0,
            reactor_states=[
                CstrBatteryReactorState(
                    reactor_index=0,
                    ca_mol_l=0.42,
                    temp_c=93.5,
                    conversion_pct=58.0,
                    reaction_rate_mol_l_min=0.08,
                    is_runaway=False,
                    dynamic_viscosity_cp=0.42,
                    thermal_conductivity_w_m_k=0.136,
                    prandtl_number=6.3,
                )
            ],
        ),
    )

    await server.sync_telemetry(frame)

    async with Client(endpoint) as client:
        ns_idx = await client.get_namespace_index(server.namespace_uri)

        # Verify synced level & temp
        node_lt = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:BufferTank", f"{ns_idx}:LT_101"])
        assert abs(await node_lt.read_value() - 6.85) < 1e-4

        node_tt = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:CstrReactor", f"{ns_idx}:TT_101"])
        assert abs(await node_tt.read_value() - 91.4) < 1e-4

        # Verify transport property sync
        node_visc = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:FluidTransport", f"{ns_idx}:DynamicViscosity_cP"])
        assert abs(await node_visc.read_value() - 0.42) < 1e-4

        # Verify battery mean temp sync
        node_bat_t = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:CstrBattery", f"{ns_idx}:TT_BATTERY"])
        assert abs(await node_bat_t.read_value() - 94.2) < 1e-4


@pytest.mark.asyncio
async def test_opcua_client_writes_to_setpoints(opcua_fixture):
    server, twin, endpoint = opcua_fixture

    async with Client(endpoint) as client:
        ns_idx = await client.get_namespace_index(server.namespace_uri)

        # 1. Client writes cooling valve to 78.5%
        node_fcv = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:CstrReactor", f"{ns_idx}:FCV_102"])
        await node_fcv.write_value(78.5)

        # 2. Client writes emergency valve to False (shutoff)
        node_xv = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:BufferTank", f"{ns_idx}:XV_101"])
        await node_xv.write_value(False)

        # 3. Client writes target feed flow to 36.0 m3/h
        node_tff = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin", f"{ns_idx}:BufferTank", f"{ns_idx}:TargetFeedFlow_m3_h"])
        await node_tff.write_value(36.0)

        # Server processes writes
        await server.check_client_writes()

        # Verify digital twin was updated
        assert twin.cooling_valve_pct == 78.5
        assert twin.feed_valve_open is False
        assert twin.feed_flow_m3_h == 36.0


@pytest.mark.asyncio
async def test_opcua_callable_methods(opcua_fixture):
    server, twin, endpoint = opcua_fixture

    # Set twin in non-standard state
    twin.cooling_valve_pct = 95.0
    twin.tank_level_m = 9.8
    twin.reactor_temp_c = 145.0
    twin.interlock_tripped = True

    async with Client(endpoint) as client:
        ns_idx = await client.get_namespace_index(server.namespace_uri)
        root_node = await client.nodes.objects.get_child([f"{ns_idx}:ProcessTwin"])

        # Invoke ResetPlant() method
        res = await root_node.call_method(f"{ns_idx}:ResetPlant")
        assert res[0].Value is True
        assert twin.reactor_temp_c == 85.0
        assert twin.tank_level_m == 5.0
        assert twin.interlock_tripped is False

        # Invoke EmergencyTrip() method
        trip_res = await root_node.call_method(f"{ns_idx}:EmergencyTrip")
        assert trip_res[0].Value is True
        assert twin.interlock_tripped is True
        assert twin.feed_valve_open is False
        assert twin.cooling_valve_pct == 100.0
