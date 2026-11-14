"""
Industrial OPC UA Server (IEC 62541) for ProcessTwin.Py.
Exposes real-time chemical process telemetry, fluid transport properties,
parallel CSTR battery states, and bi-directional supervisory setpoints to
external PLCs, DCS systems, and enterprise SCADA hosts (WinCC, Ignition, Kepware).
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from asyncua import Server, ua, uamethod

from process_twin.core.models import (
    PlantTelemetry,
    ActuatorCommand,
    SeriesParallelControlConfigCommand,
)

logger = logging.getLogger("process_twin.opcua")


class ProcessTwinOpcUaServer:
    """
    Industrial OPC UA Server exposing ProcessTwin.Py's address space.
    """

    def __init__(
        self,
        endpoint: str = "opc.tcp://0.0.0.0:4840/processtwin/server/",
        server_name: str = "ProcessTwin OPC UA Server",
        namespace_uri: str = "urn:processtwin:chemical:plant",
        twin_ref: Any = None,
    ):
        self.endpoint = endpoint
        self.server_name = server_name
        self.namespace_uri = namespace_uri
        self.twin = twin_ref
        self.server: Optional[Server] = None
        self.is_running = False
        self.ns_idx: int = 0
        self.nodes: Dict[str, Any] = {}
        self.tag_count: int = 0
        self.last_sync_utc: Optional[str] = None
        self._last_known_setpoints: Dict[str, Any] = {}

    async def start(self) -> None:
        """Initializes the OPC UA information model and binds the listening endpoint."""
        if self.is_running:
            return

        self.server = Server()
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name(self.server_name)

        # Register custom chemical plant namespace
        self.ns_idx = await self.server.register_namespace(self.namespace_uri)
        idx = self.ns_idx

        # Root ProcessTwin object
        root_obj = await self.server.nodes.objects.add_object(idx, "ProcessTwin")

        # 1. Buffer Tank Subsystem
        tank_obj = await root_obj.add_object(idx, "BufferTank")
        self.nodes["tank_level_m"] = await tank_obj.add_variable(idx, "LT_101", 5.0, ua.VariantType.Double)
        self.nodes["tank_volume_m3"] = await tank_obj.add_variable(idx, "V_101", 25.0, ua.VariantType.Double)
        self.nodes["feed_flow_m3_h"] = await tank_obj.add_variable(idx, "FT_101", 24.0, ua.VariantType.Double)
        self.nodes["discharge_flow_m3_h"] = await tank_obj.add_variable(idx, "FT_102", 24.0, ua.VariantType.Double)

        # Writable Actuators
        self.nodes["feed_valve_open"] = await tank_obj.add_variable(idx, "XV_101", True, ua.VariantType.Boolean)
        await self.nodes["feed_valve_open"].set_writable()

        self.nodes["target_feed_flow"] = await tank_obj.add_variable(idx, "TargetFeedFlow_m3_h", 24.0, ua.VariantType.Double)
        await self.nodes["target_feed_flow"].set_writable()

        self.nodes["target_discharge_flow"] = await tank_obj.add_variable(idx, "TargetDischargeFlow_m3_h", 24.0, ua.VariantType.Double)
        await self.nodes["target_discharge_flow"].set_writable()

        # 2. CSTR Reactor Subsystem
        cstr_obj = await root_obj.add_object(idx, "CstrReactor")
        self.nodes["reactor_temp_c"] = await cstr_obj.add_variable(idx, "TT_101", 85.0, ua.VariantType.Double)
        self.nodes["reactor_pressure_bar"] = await cstr_obj.add_variable(idx, "PT_101", 1.013, ua.VariantType.Double)
        self.nodes["interlock_tripped"] = await cstr_obj.add_variable(idx, "SIS_Tripped", False, ua.VariantType.Boolean)
        self.nodes["plant_status"] = await cstr_obj.add_variable(idx, "PlantStatus", "NORMAL", ua.VariantType.String)

        # Writable Cooling Valve
        self.nodes["cooling_valve_pct"] = await cstr_obj.add_variable(idx, "FCV_102", 45.0, ua.VariantType.Double)
        await self.nodes["cooling_valve_pct"].set_writable()

        # State Estimation (EKF)
        self.nodes["ca_mol_l"] = await cstr_obj.add_variable(idx, "AT_101_EST", 0.5, ua.VariantType.Double)
        self.nodes["conversion_pct"] = await cstr_obj.add_variable(idx, "XA_101_EST", 50.0, ua.VariantType.Double)
        self.nodes["reaction_rate"] = await cstr_obj.add_variable(idx, "RA_101_EST", 0.005, ua.VariantType.Double)
        self.nodes["ekf_nis"] = await cstr_obj.add_variable(idx, "EKF_NIS", 1.5, ua.VariantType.Double)

        # 3. Rigorous Fluid Transport Properties
        trans_obj = await root_obj.add_object(idx, "FluidTransport")
        self.nodes["dynamic_viscosity_cp"] = await trans_obj.add_variable(idx, "DynamicViscosity_cP", 0.5, ua.VariantType.Double)
        self.nodes["kinematic_viscosity_cst"] = await trans_obj.add_variable(idx, "KinematicViscosity_cSt", 0.6, ua.VariantType.Double)
        self.nodes["thermal_conductivity_w_m_k"] = await trans_obj.add_variable(idx, "ThermalConductivity_W_m_K", 0.14, ua.VariantType.Double)
        self.nodes["prandtl_number"] = await trans_obj.add_variable(idx, "PrandtlNumber", 6.0, ua.VariantType.Double)
        self.nodes["density_kg_m3"] = await trans_obj.add_variable(idx, "Density_kg_m3", 850.0, ua.VariantType.Double)
        self.nodes["transport_phase"] = await trans_obj.add_variable(idx, "Phase", "SubcooledLiquid", ua.VariantType.String)
        self.nodes["transport_model"] = await trans_obj.add_variable(idx, "ModelName", "Chung et al.", ua.VariantType.String)

        # 4. Parallel CSTR Battery Subsystem (SIMD AVX2)
        bat_obj = await root_obj.add_object(idx, "CstrBattery")
        self.nodes["battery_mean_temp_c"] = await bat_obj.add_variable(idx, "TT_BATTERY", 85.0, ua.VariantType.Double)
        self.nodes["battery_mean_conv_pct"] = await bat_obj.add_variable(idx, "MeanConversion_Pct", 50.0, ua.VariantType.Double)
        self.nodes["battery_mean_ca_mol_l"] = await bat_obj.add_variable(idx, "MeanCa_mol_L", 0.5, ua.VariantType.Double)
        self.nodes["battery_runaway"] = await bat_obj.add_variable(idx, "HasThermalRunaway", False, ua.VariantType.Boolean)
        self.nodes["battery_simd"] = await bat_obj.add_variable(idx, "SimdAccelerated", True, ua.VariantType.Boolean)
        self.nodes["battery_duty_kw"] = await bat_obj.add_variable(idx, "HeatDuty_kW", 0.0, ua.VariantType.Double)

        # Individual reactor train nodes R01..R08
        train_folder = await bat_obj.add_folder(idx, "Reactors")
        for r_i in range(8):
            r_node = await train_folder.add_object(idx, f"Reactor_{r_i + 1:02d}")
            self.nodes[f"r_{r_i}_temp"] = await r_node.add_variable(idx, "Temperature_C", 85.0, ua.VariantType.Double)
            self.nodes[f"r_{r_i}_conv"] = await r_node.add_variable(idx, "Conversion_Pct", 50.0, ua.VariantType.Double)
            self.nodes[f"r_{r_i}_ca"] = await r_node.add_variable(idx, "Ca_mol_L", 0.5, ua.VariantType.Double)
            self.nodes[f"r_{r_i}_runaway"] = await r_node.add_variable(idx, "IsRunaway", False, ua.VariantType.Boolean)

        # 5. Non-Linear Model Predictive Controller (NMPC)
        nmpc_obj = await root_obj.add_object(idx, "NmpcController")
        self.nodes["nmpc_enabled"] = await nmpc_obj.add_variable(idx, "Enabled", False, ua.VariantType.Boolean)
        await self.nodes["nmpc_enabled"].set_writable()

        self.nodes["nmpc_target_temp"] = await nmpc_obj.add_variable(idx, "TargetTemp_C", 85.0, ua.VariantType.Double)
        await self.nodes["nmpc_target_temp"].set_writable()

        self.nodes["nmpc_target_ca"] = await nmpc_obj.add_variable(idx, "TargetCa_mol_L", 0.5, ua.VariantType.Double)
        await self.nodes["nmpc_target_ca"].set_writable()

        self.nodes["nmpc_optimal_cooling"] = await nmpc_obj.add_variable(idx, "OptimalCooling_Pct", 45.0, ua.VariantType.Double)
        self.nodes["nmpc_solve_time_ms"] = await nmpc_obj.add_variable(idx, "SolveTime_ms", 0.0, ua.VariantType.Double)
        self.nodes["nmpc_cost"] = await nmpc_obj.add_variable(idx, "Cost", 0.0, ua.VariantType.Double)

        # 5b. 6-State Series-Parallel CSTR + EKF + Economic NMPC
        sp_obj = await root_obj.add_object(idx, "SeriesParallelCstr")
        self.nodes["sp_ca_mol_l"] = await sp_obj.add_variable(idx, "SP_CA_MolPerL", 0.48, ua.VariantType.Double)
        self.nodes["sp_cb_mol_l"] = await sp_obj.add_variable(idx, "SP_CB_MolPerL", 0.28, ua.VariantType.Double)
        self.nodes["sp_cc_mol_l"] = await sp_obj.add_variable(idx, "SP_CC_MolPerL", 0.21, ua.VariantType.Double)
        self.nodes["sp_cd_mol_l"] = await sp_obj.add_variable(idx, "SP_CD_MolPerL", 0.03, ua.VariantType.Double)
        self.nodes["sp_reactor_temp_c"] = await sp_obj.add_variable(idx, "SP_TT_201", 62.0, ua.VariantType.Double)
        self.nodes["sp_jacket_temp_c"] = await sp_obj.add_variable(idx, "SP_TT_202_Jacket", 36.0, ua.VariantType.Double)
        self.nodes["sp_conversion_a_pct"] = await sp_obj.add_variable(idx, "SP_ConversionA_Pct", 52.0, ua.VariantType.Double)
        self.nodes["sp_yield_b_pct"] = await sp_obj.add_variable(idx, "SP_YieldB_Pct", 28.0, ua.VariantType.Double)
        self.nodes["sp_selectivity_b_pct"] = await sp_obj.add_variable(idx, "SP_SelectivityB_Pct", 53.8, ua.VariantType.Double)
        self.nodes["sp_est_cb_mol_l"] = await sp_obj.add_variable(idx, "SP_EKF_CB_MolPerL", 0.28, ua.VariantType.Double)
        self.nodes["sp_ekf_nis"] = await sp_obj.add_variable(idx, "SP_EKF_NIS", 0.0, ua.VariantType.Double)
        self.nodes["sp_runaway_alarm"] = await sp_obj.add_variable(idx, "SP_RunawayAlarm", False, ua.VariantType.Boolean)
        self.nodes["sp_nmpc_solve_time_ms"] = await sp_obj.add_variable(idx, "SP_NMPC_SolveTime_ms", 0.0, ua.VariantType.Double)
        self.nodes["sp_nmpc_cost"] = await sp_obj.add_variable(idx, "SP_NMPC_Cost", 0.0, ua.VariantType.Double)

        # Writable SP supervisory / manual tags
        self.nodes["sp_nmpc_enabled"] = await sp_obj.add_variable(idx, "SP_NmpcEnabled", False, ua.VariantType.Boolean)
        await self.nodes["sp_nmpc_enabled"].set_writable()

        self.nodes["sp_cooling_valve_pct"] = await sp_obj.add_variable(idx, "SP_FCV_201", 55.0, ua.VariantType.Double)
        await self.nodes["sp_cooling_valve_pct"].set_writable()

        self.nodes["sp_feed_temp_c"] = await sp_obj.add_variable(idx, "SP_FeedTemp_C", 51.85, ua.VariantType.Double)
        await self.nodes["sp_feed_temp_c"].set_writable()

        self.nodes["sp_target_cb"] = await sp_obj.add_variable(idx, "SP_TargetCb_MolPerL", 0.0, ua.VariantType.Double)
        await self.nodes["sp_target_cb"].set_writable()

        # 6. Callable Methods on Root Object
        twin = self.twin

        @uamethod
        def reset_plant(parent):
            if twin:
                twin.reset()
            return [ua.Variant(True, ua.VariantType.Boolean)]

        @uamethod
        def emergency_trip(parent):
            if twin:
                twin.apply_actuator_command(ActuatorCommand(feed_valve_open=False, cooling_valve_pct=100.0))
                twin.interlock_tripped = True
            return [ua.Variant(True, ua.VariantType.Boolean)]

        @uamethod
        def acknowledge_alarms(parent):
            if twin:
                for a in twin.active_alarms:
                    a.acknowledged = True
            return [ua.Variant(True, ua.VariantType.Boolean)]

        await root_obj.add_method(idx, "ResetPlant", reset_plant, [], [ua.VariantType.Boolean])
        await root_obj.add_method(idx, "EmergencyTrip", emergency_trip, [], [ua.VariantType.Boolean])
        await root_obj.add_method(idx, "AcknowledgeAlarms", acknowledge_alarms, [], [ua.VariantType.Boolean])

        # Cache initial setpoints for write detection
        self._last_known_setpoints = {
            "cooling_valve_pct": 45.0,
            "feed_valve_open": True,
            "target_feed_flow": 24.0,
            "target_discharge_flow": 24.0,
            "nmpc_enabled": False,
            "nmpc_target_temp": 85.0,
            "nmpc_target_ca": 0.5,
            "sp_nmpc_enabled": False,
            "sp_cooling_valve_pct": 55.0,
            "sp_feed_temp_c": 51.85,
            "sp_target_cb": 0.0,
        }

        self.tag_count = len(self.nodes)

        # Start listening
        await self.server.start()
        self.is_running = True
        logger.info(f"ProcessTwin OPC UA Server running at {self.endpoint} ({self.tag_count} nodes published)")

    async def stop(self) -> None:
        """Stops the OPC UA server and releases network sockets."""
        if self.server and self.is_running:
            await self.server.stop()
            self.is_running = False
            logger.info("ProcessTwin OPC UA Server stopped.")

    async def sync_telemetry(self, frame: PlantTelemetry) -> None:
        """
        Synchronizes a 5 Hz plant telemetry snapshot into the OPC UA address space.
        Executes in < 50 microseconds.
        """
        if not self.is_running:
            return

        # Buffer Tank
        await self.nodes["tank_level_m"].write_value(float(frame.tank_level_m))
        await self.nodes["tank_volume_m3"].write_value(float(frame.tank_volume_m3))
        await self.nodes["feed_flow_m3_h"].write_value(float(frame.feed_flow_m3_h))
        await self.nodes["discharge_flow_m3_h"].write_value(float(frame.discharge_flow_m3_h))

        # CSTR Reactor
        await self.nodes["reactor_temp_c"].write_value(float(frame.reactor_temp_c))
        await self.nodes["reactor_pressure_bar"].write_value(float(frame.reactor_pressure_bar))
        await self.nodes["interlock_tripped"].write_value(bool(frame.interlock_tripped))
        await self.nodes["plant_status"].write_value(str(frame.status.value))

        # EKF State Estimates
        if frame.estimated_ca_mol_l is not None:
            await self.nodes["ca_mol_l"].write_value(float(frame.estimated_ca_mol_l))
        if frame.estimated_conversion_pct is not None:
            await self.nodes["conversion_pct"].write_value(float(frame.estimated_conversion_pct))
        if frame.estimated_reaction_rate is not None:
            await self.nodes["reaction_rate"].write_value(float(frame.estimated_reaction_rate))
        if frame.ekf_nis is not None:
            await self.nodes["ekf_nis"].write_value(float(frame.ekf_nis))

        # Fluid Transport Properties
        if frame.transport_properties:
            tp = frame.transport_properties
            await self.nodes["dynamic_viscosity_cp"].write_value(float(tp.dynamic_viscosity_cp))
            await self.nodes["kinematic_viscosity_cst"].write_value(float(tp.kinematic_viscosity_cst))
            await self.nodes["thermal_conductivity_w_m_k"].write_value(float(tp.thermal_conductivity_w_m_k))
            await self.nodes["prandtl_number"].write_value(float(tp.prandtl_number))
            await self.nodes["density_kg_m3"].write_value(float(tp.density_kg_m3))
            await self.nodes["transport_phase"].write_value(str(tp.phase))
            await self.nodes["transport_model"].write_value(str(tp.model_name))

        # Parallel CSTR Battery Telemetry
        if frame.cstr_battery:
            bat = frame.cstr_battery
            await self.nodes["battery_mean_temp_c"].write_value(float(bat.mean_temp_c))
            await self.nodes["battery_mean_conv_pct"].write_value(float(bat.mean_conversion_pct))
            await self.nodes["battery_mean_ca_mol_l"].write_value(float(bat.mean_ca_mol_l))
            await self.nodes["battery_runaway"].write_value(bool(bat.has_thermal_runaway))
            await self.nodes["battery_simd"].write_value(bool(bat.is_using_native_simd))
            await self.nodes["battery_duty_kw"].write_value(float(bat.energy_duty_kw))

            for s in bat.reactor_states:
                idx = s.reactor_index
                if f"r_{idx}_temp" in self.nodes:
                    await self.nodes[f"r_{idx}_temp"].write_value(float(s.temp_c))
                    await self.nodes[f"r_{idx}_conv"].write_value(float(s.conversion_pct))
                    await self.nodes[f"r_{idx}_ca"].write_value(float(s.ca_mol_l))
                    await self.nodes[f"r_{idx}_runaway"].write_value(bool(s.is_runaway))

        # NMPC Status
        await self.nodes["nmpc_enabled"].write_value(bool(frame.nmpc_enabled))
        if frame.nmpc_optimal_cooling_pct is not None:
            await self.nodes["nmpc_optimal_cooling"].write_value(float(frame.nmpc_optimal_cooling_pct))
        if frame.nmpc_solve_time_ms is not None:
            await self.nodes["nmpc_solve_time_ms"].write_value(float(frame.nmpc_solve_time_ms))
        if frame.nmpc_cost is not None:
            await self.nodes["nmpc_cost"].write_value(float(frame.nmpc_cost))

        # 6-State Series-Parallel CSTR + EKF + Economic NMPC
        if frame.sp_control:
            sp = frame.sp_control
            await self.nodes["sp_ca_mol_l"].write_value(float(sp.ca_mol_per_l))
            await self.nodes["sp_cb_mol_l"].write_value(float(sp.cb_mol_per_l))
            await self.nodes["sp_cc_mol_l"].write_value(float(sp.cc_mol_per_l))
            await self.nodes["sp_cd_mol_l"].write_value(float(sp.cd_mol_per_l))
            await self.nodes["sp_reactor_temp_c"].write_value(float(sp.reactor_temp_c))
            await self.nodes["sp_jacket_temp_c"].write_value(float(sp.jacket_temp_c))
            await self.nodes["sp_conversion_a_pct"].write_value(float(sp.conversion_a_pct))
            await self.nodes["sp_yield_b_pct"].write_value(float(sp.yield_b_pct))
            await self.nodes["sp_selectivity_b_pct"].write_value(float(sp.selectivity_b_pct))
            await self.nodes["sp_est_cb_mol_l"].write_value(float(sp.est_cb_mol_per_l))
            await self.nodes["sp_ekf_nis"].write_value(float(sp.ekf_nis))
            await self.nodes["sp_runaway_alarm"].write_value(bool(sp.is_thermal_runaway))
            if sp.nmpc_solve_time_ms is not None:
                await self.nodes["sp_nmpc_solve_time_ms"].write_value(float(sp.nmpc_solve_time_ms))
            if sp.nmpc_cost is not None:
                await self.nodes["sp_nmpc_cost"].write_value(float(sp.nmpc_cost))
            if sp.nmpc_enabled:
                await self.nodes["sp_cooling_valve_pct"].write_value(float(sp.cooling_valve_pct))
                self._last_known_setpoints["sp_cooling_valve_pct"] = float(sp.cooling_valve_pct)

    async def check_client_writes(self) -> None:
        """
        Polls writable OPC UA setpoint variables. If an external client (SCADA/PLC)
        wrote a new value, immediately applies the command to the digital twin.
        """
        if not self.is_running or not self.twin:
            return

        # 1. Cooling valve FCV-102
        val_cooling = await self.nodes["cooling_valve_pct"].read_value()
        if abs(val_cooling - self._last_known_setpoints["cooling_valve_pct"]) > 0.05:
            self._last_known_setpoints["cooling_valve_pct"] = val_cooling
            self.twin.apply_actuator_command(ActuatorCommand(cooling_valve_pct=val_cooling))

        # 2. Emergency Feed Valve XV-101
        val_valve = await self.nodes["feed_valve_open"].read_value()
        if val_valve != self._last_known_setpoints["feed_valve_open"]:
            self._last_known_setpoints["feed_valve_open"] = val_valve
            self.twin.apply_actuator_command(ActuatorCommand(feed_valve_open=val_valve))

        # 3. Target Feed Flow
        val_ff = await self.nodes["target_feed_flow"].read_value()
        if abs(val_ff - self._last_known_setpoints["target_feed_flow"]) > 0.05:
            self._last_known_setpoints["target_feed_flow"] = val_ff
            self.twin.apply_actuator_command(ActuatorCommand(target_feed_flow_m3_h=val_ff))

        # 4. Target Discharge Flow
        val_df = await self.nodes["target_discharge_flow"].read_value()
        if abs(val_df - self._last_known_setpoints["target_discharge_flow"]) > 0.05:
            self._last_known_setpoints["target_discharge_flow"] = val_df
            self.twin.apply_actuator_command(ActuatorCommand(target_discharge_flow_m3_h=val_df))

        # 5. NMPC Controller Toggle
        val_nmpc_en = await self.nodes["nmpc_enabled"].read_value()
        if val_nmpc_en != self._last_known_setpoints["nmpc_enabled"]:
            self._last_known_setpoints["nmpc_enabled"] = val_nmpc_en
            self.twin.enable_nmpc(enabled=val_nmpc_en)

        # 6. NMPC Target Temperature Setpoint
        val_nmpc_sp = await self.nodes["nmpc_target_temp"].read_value()
        if abs(val_nmpc_sp - self._last_known_setpoints["nmpc_target_temp"]) > 0.05:
            self._last_known_setpoints["nmpc_target_temp"] = val_nmpc_sp
            self.twin.update_nmpc_setpoints(temp_setpoint_c=val_nmpc_sp)

        # 7. Series-Parallel Economic NMPC Supervisory Setpoints
        val_sp_en = await self.nodes["sp_nmpc_enabled"].read_value()
        if val_sp_en != self._last_known_setpoints["sp_nmpc_enabled"]:
            self._last_known_setpoints["sp_nmpc_enabled"] = val_sp_en
            self.twin.update_series_parallel_control(
                SeriesParallelControlConfigCommand(nmpc_enabled=bool(val_sp_en))
            )

        val_sp_cool = await self.nodes["sp_cooling_valve_pct"].read_value()
        if abs(val_sp_cool - self._last_known_setpoints["sp_cooling_valve_pct"]) > 0.05:
            self._last_known_setpoints["sp_cooling_valve_pct"] = val_sp_cool
            self.twin.update_series_parallel_control(
                SeriesParallelControlConfigCommand(cooling_valve_pct=float(val_sp_cool))
            )

        val_sp_feed_t = await self.nodes["sp_feed_temp_c"].read_value()
        if abs(val_sp_feed_t - self._last_known_setpoints["sp_feed_temp_c"]) > 0.05:
            self._last_known_setpoints["sp_feed_temp_c"] = val_sp_feed_t
            self.twin.update_series_parallel_control(
                SeriesParallelControlConfigCommand(feed_temp_c=float(val_sp_feed_t))
            )

        val_sp_target_cb = await self.nodes["sp_target_cb"].read_value()
        if abs(val_sp_target_cb - self._last_known_setpoints["sp_target_cb"]) > 0.002:
            self._last_known_setpoints["sp_target_cb"] = val_sp_target_cb
            if val_sp_target_cb <= 0.0:
                self.twin.update_series_parallel_control(
                    SeriesParallelControlConfigCommand(clear_target_cb=True)
                )
            else:
                self.twin.update_series_parallel_control(
                    SeriesParallelControlConfigCommand(target_cb_mol_per_l=float(val_sp_target_cb))
                )
