"""
Industrial Gateway orchestrating OPC UA and Modbus TCP protocol servers.
Provides a unified industrial communication bridge between ProcessTwin.Py
and external industrial hardware/software (PLCs, HMIs, DCS, Historians).
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Any

from process_twin.core.models import PlantTelemetry, IndustrialGatewayStatus
from process_twin.industrial.opcua_server import ProcessTwinOpcUaServer
from process_twin.industrial.modbus_server import ProcessTwinModbusServer

logger = logging.getLogger("process_twin.gateway")


class IndustrialGateway:
    """
    Central industrial communication coordinator for ProcessTwin.Py.
    Manages OPC UA Server and Modbus TCP Server lifecycle and real-time state synchronization.
    """

    def __init__(
        self,
        twin_ref: Any = None,
        opcua_endpoint: Optional[str] = None,
        modbus_host: str = "0.0.0.0",
        modbus_port: Optional[int] = None,
        opcua_enabled: bool = True,
        modbus_enabled: bool = True,
    ):
        self.twin = twin_ref
        self.opcua_enabled = opcua_enabled
        self.modbus_enabled = modbus_enabled

        # Configure endpoints from arguments or environment variables
        self.opcua_endpoint = opcua_endpoint or os.getenv(
            "OPCUA_ENDPOINT", "opc.tcp://0.0.0.0:4840/processtwin/server/"
        )
        self.modbus_host = modbus_host
        self.modbus_port = modbus_port or int(os.getenv("MODBUS_PORT", "5020"))

        self.opcua_server = ProcessTwinOpcUaServer(
            endpoint=self.opcua_endpoint,
            twin_ref=self.twin,
        )
        self.modbus_server = ProcessTwinModbusServer(
            host=self.modbus_host,
            port=self.modbus_port,
            twin_ref=self.twin,
        )

        self.last_sync_utc: Optional[str] = None
        self.sync_count: int = 0

    async def start(self) -> None:
        """Starts both OPC UA and Modbus TCP servers asynchronously."""
        if self.opcua_enabled:
            try:
                await self.opcua_server.start()
            except Exception as e:
                logger.error(f"Failed to start OPC UA server: {e}")

        if self.modbus_enabled:
            try:
                await self.modbus_server.start()
            except Exception as e:
                logger.error(f"Failed to start Modbus TCP server: {e}")

    async def stop(self) -> None:
        """Stops all running industrial communication servers."""
        if self.opcua_server.is_running:
            await self.opcua_server.stop()
        if self.modbus_server.is_running:
            await self.modbus_server.stop()

    async def sync_telemetry(self, frame: PlantTelemetry) -> None:
        """
        Dispatches the 5 Hz plant telemetry frame to all active industrial protocol servers.
        """
        if self.opcua_server.is_running:
            try:
                await self.opcua_server.sync_telemetry(frame)
            except Exception as e:
                logger.warning(f"Error syncing telemetry to OPC UA: {e}")

        if self.modbus_server.is_running:
            try:
                self.modbus_server.sync_telemetry(frame)
            except Exception as e:
                logger.warning(f"Error syncing telemetry to Modbus: {e}")

        self.last_sync_utc = datetime.now(timezone.utc).isoformat()
        self.sync_count += 1

    async def check_client_writes(self) -> None:
        """Checks for supervisory writes from OPC UA clients or Modbus masters."""
        if self.opcua_server.is_running:
            try:
                await self.opcua_server.check_client_writes()
            except Exception as e:
                logger.warning(f"Error reading OPC UA client writes: {e}")

    def get_status(self) -> IndustrialGatewayStatus:
        """Returns the real-time operational status of the industrial gateway."""
        return IndustrialGatewayStatus(
            opcua_enabled=self.opcua_enabled,
            opcua_running=self.opcua_server.is_running,
            opcua_endpoint=self.opcua_endpoint,
            opcua_tags_count=self.opcua_server.tag_count,
            modbus_enabled=self.modbus_enabled,
            modbus_running=self.modbus_server.is_running,
            modbus_host=self.modbus_host,
            modbus_port=self.modbus_port,
            modbus_requests_served=self.modbus_server.requests_served,
            modbus_writes_received=self.modbus_server.writes_received,
            last_sync_utc=self.last_sync_utc,
        )
