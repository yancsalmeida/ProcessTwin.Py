"""
Industrial Modbus TCP Server (IEC 61158 / MBAP) for ProcessTwin.Py.
Provides a high-speed, lightweight asynchronous Modbus TCP gateway
enabling legacy and modern PLCs (Siemens S7, Schneider Modicon, Rockwell)
to poll process telemetry and command actuator setpoints over standard Modbus TCP.
"""

import asyncio
import struct
import logging
from typing import Optional, Dict, Any

from process_twin.core.models import PlantTelemetry, ActuatorCommand

logger = logging.getLogger("process_twin.modbus")


class ProcessTwinModbusServer:
    """
    High-performance asynchronous Modbus TCP server (MBAP port 5020).
    Maps physical chemical process telemetry to standard Holding Registers and Coils.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5020, twin_ref: Any = None):
        self.host = host
        self.port = port
        self.twin = twin_ref
        self.server: Optional[asyncio.Server] = None
        self.is_running = False

        # Modbus Address Memory Space (0-based internal indexing)
        self.holding_registers = [0] * 128
        self.coils = [False] * 64

        # Track external writes to trigger digital twin actuator updates
        self._last_synced_reg_cooling = 450
        self._last_synced_reg_feed_flow = 240
        self._last_synced_reg_disch_flow = 240
        self._last_synced_reg_nmpc_sp = 850
        self._last_synced_coil_valve = True
        self._last_synced_coil_nmpc = False

        # Metrics
        self.requests_served = 0
        self.writes_received = 0

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handles incoming Modbus TCP MBAP protocol frames."""
        try:
            while self.is_running:
                # MBAP Header: 7 bytes
                # Transaction ID (2B) + Protocol ID (2B) + Length (2B) + Unit ID (1B)
                header = await reader.read(7)
                if not header or len(header) < 7:
                    break

                tid, pid, length, uid = struct.unpack(">HHHB", header)
                pdu_len = length - 1
                if pdu_len <= 0:
                    break

                pdu = await reader.read(pdu_len)
                if not pdu or len(pdu) < pdu_len:
                    break

                self.requests_served += 1
                fc = pdu[0]
                resp_pdu = b""

                if fc == 3:  # Read Holding Registers (FC 0x03)
                    start_addr, count = struct.unpack(">HH", pdu[1:5])
                    if start_addr + count <= len(self.holding_registers):
                        vals = self.holding_registers[start_addr:start_addr + count]
                        byte_count = count * 2
                        data_bytes = b"".join(struct.pack(">H", v & 0xFFFF) for v in vals)
                        resp_pdu = struct.pack(">BB", fc, byte_count) + data_bytes
                    else:
                        resp_pdu = struct.pack(">BB", fc | 0x80, 0x02)  # Illegal Data Address

                elif fc == 6:  # Write Single Holding Register (FC 0x06)
                    reg_addr, reg_val = struct.unpack(">HH", pdu[1:5])
                    if reg_addr < len(self.holding_registers):
                        self.holding_registers[reg_addr] = reg_val
                        self.writes_received += 1
                        self._handle_register_write(reg_addr, reg_val)
                        resp_pdu = pdu[:5]
                    else:
                        resp_pdu = struct.pack(">BB", fc | 0x80, 0x02)

                elif fc == 16:  # Write Multiple Registers (FC 0x10)
                    start_addr, count, byte_count = struct.unpack(">HHB", pdu[1:6])
                    if start_addr + count <= len(self.holding_registers) and byte_count == count * 2:
                        for i in range(count):
                            val = struct.unpack(">H", pdu[6 + i * 2:8 + i * 2])[0]
                            self.holding_registers[start_addr + i] = val
                            self._handle_register_write(start_addr + i, val)
                        self.writes_received += count
                        resp_pdu = struct.pack(">BHH", fc, start_addr, count)
                    else:
                        resp_pdu = struct.pack(">BB", fc | 0x80, 0x02)

                elif fc == 1:  # Read Coils (FC 0x01)
                    start_addr, count = struct.unpack(">HH", pdu[1:5])
                    if start_addr + count <= len(self.coils):
                        byte_count = (count + 7) // 8
                        bytes_list = [0] * byte_count
                        for i in range(count):
                            if self.coils[start_addr + i]:
                                bytes_list[i // 8] |= (1 << (i % 8))
                        resp_pdu = struct.pack(">BB", fc, byte_count) + bytes(bytes_list)
                    else:
                        resp_pdu = struct.pack(">BB", fc | 0x80, 0x02)

                elif fc == 5:  # Write Single Coil (FC 0x05)
                    coil_addr, coil_val = struct.unpack(">HH", pdu[1:5])
                    if coil_addr < len(self.coils):
                        state = (coil_val == 0xFF00)
                        self.coils[coil_addr] = state
                        self.writes_received += 1
                        self._handle_coil_write(coil_addr, state)
                        resp_pdu = pdu[:5]
                    else:
                        resp_pdu = struct.pack(">BB", fc | 0x80, 0x02)

                else:
                    # Illegal Function Exception
                    resp_pdu = struct.pack(">BB", fc | 0x80, 0x01)

                # Send MBAP Response
                resp_header = struct.pack(">HHHB", tid, pid, len(resp_pdu) + 1, uid)
                writer.write(resp_header + resp_pdu)
                await writer.drain()

        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.warning(f"Modbus client handling exception: {e}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _handle_register_write(self, addr: int, val: int) -> None:
        """Dispatches external PLC holding register writes directly to the digital twin."""
        if not self.twin:
            return

        if addr == 5:  # Cooling Valve % * 10
            valve_pct = val / 10.0
            self._last_synced_reg_cooling = val
            self.twin.apply_actuator_command(ActuatorCommand(cooling_valve_pct=valve_pct))

        elif addr == 16:  # Target Feed Flow * 10
            ff = val / 10.0
            self._last_synced_reg_feed_flow = val
            self.twin.apply_actuator_command(ActuatorCommand(target_feed_flow_m3_h=ff))

        elif addr == 17:  # Target Discharge Flow * 10
            df = val / 10.0
            self._last_synced_reg_disch_flow = val
            self.twin.apply_actuator_command(ActuatorCommand(target_discharge_flow_m3_h=df))

        elif addr == 18:  # NMPC Target Temp * 10
            sp = val / 10.0
            self._last_synced_reg_nmpc_sp = val
            self.twin.update_nmpc_setpoints(temp_setpoint_c=sp)

    def _handle_coil_write(self, addr: int, state: bool) -> None:
        """Dispatches external PLC coil writes directly to the digital twin."""
        if not self.twin:
            return

        if addr == 0:  # Emergency Feed Valve XV-101
            self._last_synced_coil_valve = state
            self.twin.apply_actuator_command(ActuatorCommand(feed_valve_open=state))

        elif addr == 2:  # NMPC Controller Enable/Disable
            self._last_synced_coil_nmpc = state
            self.twin.enable_nmpc(enabled=state)

        elif addr == 4 and state:  # Pulse: Reset Plant
            self.twin.reset()
            self.coils[4] = False  # Auto-clear momentary pulse

        elif addr == 5 and state:  # Pulse: Emergency SIS Trip
            self.twin.apply_actuator_command(ActuatorCommand(feed_valve_open=False, cooling_valve_pct=100.0))
            self.twin.interlock_tripped = True
            self.coils[5] = False  # Auto-clear momentary pulse

    async def start(self) -> None:
        """Binds the Modbus TCP socket and starts serving requests."""
        if self.is_running:
            return

        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self.is_running = True
        logger.info(f"ProcessTwin Modbus TCP Server running on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stops the Modbus TCP server and closes client connections."""
        if self.server and self.is_running:
            self.is_running = False
            self.server.close()
            await self.server.wait_closed()
            logger.info("ProcessTwin Modbus TCP Server stopped.")

    def sync_telemetry(self, frame: PlantTelemetry) -> None:
        """
        Maps the latest PlantTelemetry snapshot into standard Modbus registers and coils.
        Zero allocation, sub-microsecond in-memory update.
        """
        if not self.is_running:
            return

        # Holding Registers (Scaled Integers)
        # 0: Tank Level (m * 100)
        self.holding_registers[0] = int(round(frame.tank_level_m * 100)) & 0xFFFF
        # 1: Reactor Temp (°C * 10)
        self.holding_registers[1] = int(round(frame.reactor_temp_c * 10)) & 0xFFFF
        # 2: Reactor Pressure (bar * 100)
        self.holding_registers[2] = int(round(frame.reactor_pressure_bar * 100)) & 0xFFFF
        # 3: Feed Flow (m³/h * 10)
        self.holding_registers[3] = int(round(frame.feed_flow_m3_h * 10)) & 0xFFFF
        # 4: Discharge Flow (m³/h * 10)
        self.holding_registers[4] = int(round(frame.discharge_flow_m3_h * 10)) & 0xFFFF
        # 5: Cooling Valve (% * 10)
        self.holding_registers[5] = int(round(frame.cooling_valve_pct * 10)) & 0xFFFF
        # 6: Reactant Ca (mol/L * 1000)
        if frame.estimated_ca_mol_l is not None:
            self.holding_registers[6] = int(round(frame.estimated_ca_mol_l * 1000)) & 0xFFFF
        # 7: Conversion XA (% * 10)
        if frame.estimated_conversion_pct is not None:
            self.holding_registers[7] = int(round(frame.estimated_conversion_pct * 10)) & 0xFFFF
        # 8: Reaction Rate (mol/(L·s) * 10000)
        if frame.estimated_reaction_rate is not None:
            self.holding_registers[8] = int(round(frame.estimated_reaction_rate * 10000)) & 0xFFFF
        # 9: EKF NIS (* 100)
        if frame.ekf_nis is not None:
            self.holding_registers[9] = int(round(frame.ekf_nis * 100)) & 0xFFFF

        # Transport Properties
        if frame.transport_properties:
            tp = frame.transport_properties
            # 10: Dynamic Viscosity (cP * 1000)
            self.holding_registers[10] = int(round(tp.dynamic_viscosity_cp * 1000)) & 0xFFFF
            # 11: Thermal Conductivity (W/(m·K) * 10000)
            self.holding_registers[11] = int(round(tp.thermal_conductivity_w_m_k * 10000)) & 0xFFFF
            # 12: Prandtl Number (* 100)
            self.holding_registers[12] = int(round(tp.prandtl_number * 100)) & 0xFFFF
            # 13: Density (kg/m³)
            self.holding_registers[13] = int(round(tp.density_kg_m3)) & 0xFFFF

        # Parallel CSTR Battery
        if frame.cstr_battery:
            bat = frame.cstr_battery
            # 14: Mean Temp (°C * 10)
            self.holding_registers[14] = int(round(bat.mean_temp_c * 10)) & 0xFFFF
            # 15: Mean Conversion (% * 10)
            self.holding_registers[15] = int(round(bat.mean_conversion_pct * 10)) & 0xFFFF
            # 16: Heat Duty (kW * 10)
            self.holding_registers[16] = int(round(bat.energy_duty_kw * 10)) & 0xFFFF

        # NMPC
        if frame.nmpc_target_temp_c is not None:
            self.holding_registers[18] = int(round(frame.nmpc_target_temp_c * 10)) & 0xFFFF
        if frame.nmpc_optimal_cooling_pct is not None:
            self.holding_registers[20] = int(round(frame.nmpc_optimal_cooling_pct * 10)) & 0xFFFF

        # Coils (Discrete Booleans)
        self.coils[0] = bool(frame.feed_valve_open)
        self.coils[1] = bool(frame.interlock_tripped)
        self.coils[2] = bool(frame.nmpc_enabled)
        self.coils[3] = bool(frame.cstr_battery.has_thermal_runaway) if frame.cstr_battery else False
