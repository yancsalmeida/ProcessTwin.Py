import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, Field

from process_twin.core.models import PlantTelemetry


class BridgeStatus(BaseModel):
    enabled: bool = True
    connected: bool = False
    engine_url: str = "http://localhost:5100"
    ingestion_mode: str = "transactional"
    active_batch_id: Optional[str] = None
    active_batch_code: Optional[str] = None
    frames_sent: int = 0
    measurements_ingested: int = 0
    alerts_triggered: int = 0
    last_sync_utc: Optional[str] = None
    last_error: Optional[str] = None


class IndustrialTelemetryBridge:
    """
    Asynchronous client bridging the ProcessTwin simulation engine with
    the enterprise IndustrialTelemetry.Engine (.NET 8 + PostgreSQL 16) microservice.
    """

    def __init__(
        self,
        engine_url: Optional[str] = None,
        enabled: bool = True,
        ingestion_mode: Optional[str] = None
    ) -> None:
        self.engine_url = (
            engine_url
            or os.getenv("TELEMETRY_ENGINE_URL")
            or "http://localhost:5100"
        ).rstrip("/")
        self.ingestion_mode = (
            ingestion_mode
            or os.getenv("TELEMETRY_INGESTION_MODE")
            or "transactional"
        ).lower()
        self.enabled = enabled
        self.connected = False
        self.active_batch_id: Optional[str] = None
        self.active_batch_code: Optional[str] = None
        self.frames_sent = 0
        self.measurements_ingested = 0
        self.alerts_triggered = 0
        self.last_sync_utc: Optional[str] = None
        self.last_error: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=3.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def check_health(self) -> bool:
        """Pings the .NET health endpoint."""
        client = self._get_client()
        try:
            resp = await client.get(f"{self.engine_url}/health")
            self.connected = (resp.status_code == 200)
            if self.connected:
                self.last_error = None
            return self.connected
        except Exception as e:
            self.connected = False
            self.last_error = f"Health check failed: {str(e)}"
            return False

    async def ensure_active_batch(self) -> Optional[str]:
        """
        Discovers or creates an active batch in 'InProgress' state
        within the .NET IndustrialTelemetry.Engine.
        """
        if not self.enabled:
            return None

        client = self._get_client()

        # 1. If we already have a batch_id, check if it's still InProgress
        if self.active_batch_id:
            try:
                resp = await client.get(f"{self.engine_url}/api/batches/{self.active_batch_id}")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "InProgress":
                        self.active_batch_code = data.get("code")
                        return self.active_batch_id
            except Exception:
                pass
            self.active_batch_id = None
            self.active_batch_code = None

        # 2. Query for existing InProgress batch
        try:
            resp = await client.get(f"{self.engine_url}/api/batches?status=InProgress&limit=1")
            if resp.status_code == 200:
                batches = resp.json()
                if batches and len(batches) > 0:
                    batch = batches[0]
                    self.active_batch_id = batch["id"]
                    self.active_batch_code = batch["code"]
                    self.connected = True
                    return self.active_batch_id
        except Exception as e:
            self.connected = False
            self.last_error = f"Query batches failed: {str(e)}"
            return None

        # 3. None found, create a new batch and start it
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        batch_code = f"BATCH-TWIN-{now_str}"
        create_payload = {
            "code": batch_code,
            "productRecipe": "CSTR-Polystyrene-Emulsion-Grade-A",
            "reactorId": "R-101",
            "targetVolumeM3": 20.0,
            "operatorNotes": "Automated batch session initiated by ProcessTwin.Py simulation engine."
        }

        try:
            create_resp = await client.post(f"{self.engine_url}/api/batches", json=create_payload)
            if create_resp.status_code in (200, 201):
                batch_data = create_resp.json()
                batch_id = batch_data["id"]

                # Start the newly scheduled batch
                start_resp = await client.post(f"{self.engine_url}/api/batches/{batch_id}/start", json={})
                if start_resp.status_code == 200:
                    self.active_batch_id = batch_id
                    self.active_batch_code = batch_code
                    self.connected = True
                    return self.active_batch_id
                else:
                    self.last_error = f"Failed to start batch: {start_resp.text}"
            else:
                self.last_error = f"Failed to create batch: {create_resp.text}"
        except Exception as e:
            self.connected = False
            self.last_error = f"Batch creation error: {str(e)}"

        return None

    async def send_frame(self, frame: PlantTelemetry) -> Optional[Dict[str, Any]]:
        """
        Translates a ProcessTwin plant telemetry snapshot into .NET IngestTelemetryBatchDto
        and transmits it to POST /api/telemetry/ingest.
        """
        if not self.enabled:
            return None

        batch_id = await self.ensure_active_batch()
        if not batch_id:
            return None

        client = self._get_client()
        ts_iso = frame.timestamp.isoformat()

        # Dynamic sensor quality evaluation from Extended Kalman Filter (EKF) NIS
        # NIS follows chi-square distribution with 1 DOF (temperature residual).
        # Normal bounds: NIS <= 3.84 (95% conf) -> Good
        # Moderate deviation: 3.84 < NIS <= 6.63 (99% conf) -> Uncertain
        # Severe anomaly/sensor drift/trip: NIS > 6.63 -> Bad
        quality = "Good"
        if frame.ekf_nis is not None:
            if frame.ekf_nis > 6.63:
                quality = "Bad"
            elif frame.ekf_nis > 3.84:
                quality = "Uncertain"

        measurements = [
            {
                "timestampUtc": ts_iso,
                "sensorTag": "TT-101",
                "value": round(float(frame.reactor_temp_c), 2),
                "unit": "Celsius",
                "quality": quality
            },
            {
                "timestampUtc": ts_iso,
                "sensorTag": "PT-101",
                "value": round(float(frame.reactor_pressure_bar), 2),
                "unit": "Bar",
                "quality": "Good"
            },
            {
                "timestampUtc": ts_iso,
                "sensorTag": "LT-101",
                "value": round(float(frame.tank_level_m), 2),
                "unit": "Meters",
                "quality": "Good"
            },
            {
                "timestampUtc": ts_iso,
                "sensorTag": "FT-101",
                "value": round(float(frame.feed_flow_m3_h), 2),
                "unit": "M3PerHour",
                "quality": "Good"
            },
            {
                "timestampUtc": ts_iso,
                "sensorTag": "FT-102",
                "value": round(float(frame.discharge_flow_m3_h), 2),
                "unit": "M3PerHour",
                "quality": "Good"
            },
            {
                "timestampUtc": ts_iso,
                "sensorTag": "TCV-101",
                "value": round(float(frame.cooling_valve_pct), 2),
                "unit": "Percent",
                "quality": "Good"
            }
        ]

        # Extended Kalman Filter (EKF) Estimated States
        if frame.estimated_temp_c is not None:
            measurements.append({
                "timestampUtc": ts_iso,
                "sensorTag": "TIT-101-EST",
                "value": round(float(frame.estimated_temp_c), 2),
                "unit": "Celsius",
                "quality": quality
            })
        if frame.estimated_ca_mol_l is not None:
            measurements.append({
                "timestampUtc": ts_iso,
                "sensorTag": "AT-101-EST",
                "value": round(float(frame.estimated_ca_mol_l), 4),
                "unit": "MolPerLiter",
                "quality": quality
            })
        if frame.estimated_conversion_pct is not None:
            measurements.append({
                "timestampUtc": ts_iso,
                "sensorTag": "X-101-EST",
                "value": round(float(frame.estimated_conversion_pct), 2),
                "unit": "Percent",
                "quality": quality
            })

        # Non-linear Model Predictive Control (NMPC) Outputs & Setpoints
        if frame.nmpc_optimal_cooling_pct is not None:
            measurements.append({
                "timestampUtc": ts_iso,
                "sensorTag": "TCV-101-NMPC",
                "value": round(float(frame.nmpc_optimal_cooling_pct), 2),
                "unit": "Percent",
                "quality": "Good"
            })
        if frame.nmpc_target_temp_c is not None:
            measurements.append({
                "timestampUtc": ts_iso,
                "sensorTag": "TIC-101-SP",
                "value": round(float(frame.nmpc_target_temp_c), 2),
                "unit": "Celsius",
                "quality": "Good"
            })

        # 6-State Series-Parallel CSTR + EKF + Economic NMPC Tags
        if frame.sp_control is not None:
            sp = frame.sp_control
            sp_quality = "Good" if sp.ekf_nis <= 5.99 else ("Uncertain" if sp.ekf_nis <= 9.21 else "Bad")
            measurements.extend([
                {
                    "timestampUtc": ts_iso,
                    "sensorTag": "SP-TT-201",
                    "value": round(float(sp.reactor_temp_c), 2),
                    "unit": "Celsius",
                    "quality": sp_quality,
                },
                {
                    "timestampUtc": ts_iso,
                    "sensorTag": "SP-TT-202",
                    "value": round(float(sp.jacket_temp_c), 2),
                    "unit": "Celsius",
                    "quality": sp_quality,
                },
                {
                    "timestampUtc": ts_iso,
                    "sensorTag": "SP-AT-201-CB",
                    "value": round(float(sp.est_cb_mol_per_l), 4),
                    "unit": "MolPerLiter",
                    "quality": sp_quality,
                },
                {
                    "timestampUtc": ts_iso,
                    "sensorTag": "SP-SB-201",
                    "value": round(float(sp.selectivity_b_pct), 2),
                    "unit": "Percent",
                    "quality": "Good",
                },
                {
                    "timestampUtc": ts_iso,
                    "sensorTag": "SP-TCV-201-NMPC",
                    "value": round(float(sp.cooling_valve_pct), 2),
                    "unit": "Percent",
                    "quality": "Good",
                },
            ])

        # High-performance wire protocol transmission
        if self.ingestion_mode == "wire":
            wire_lines = [
                f"{m['timestampUtc']}|{m['sensorTag']}|{m['value']}|{m['unit']}|{m['quality']}"
                for m in measurements
            ]
            wire_payload = "\n".join(wire_lines)
            try:
                resp = await client.post(
                    f"{self.engine_url}/api/telemetry/batch/{batch_id}/ingest/wire?mode=Channel",
                    content=wire_payload,
                    headers={"Content-Type": "text/plain"}
                )
                if resp.status_code in (200, 202):
                    result = resp.json()
                    self.frames_sent += 1
                    self.measurements_ingested += result.get(
                        "validMeasurements",
                        result.get("totalLinesParsed", len(measurements))
                    )
                    self.alerts_triggered += result.get("alertsTriggered", 0)
                    self.last_sync_utc = datetime.now(timezone.utc).isoformat()
                    self.connected = True
                    self.last_error = None
                    return result
                elif resp.status_code == 400:
                    self.last_error = f"Ingestion rejected (HTTP 400): {resp.text}"
                    self.active_batch_id = None
                    self.active_batch_code = None
                else:
                    self.last_error = f"Ingestion failed (HTTP {resp.status_code}): {resp.text}"
            except Exception as e:
                self.connected = False
                self.last_error = f"Ingestion transport error: {str(e)}"
            return None

        payload = {
            "batchId": batch_id,
            "measurements": measurements
        }

        endpoint = (
            "/api/telemetry/ingest/channel"
            if self.ingestion_mode == "channel"
            else "/api/telemetry/ingest"
        )

        try:
            resp = await client.post(f"{self.engine_url}{endpoint}", json=payload)
            if resp.status_code in (200, 202):
                result = resp.json()
                self.frames_sent += 1
                self.measurements_ingested += result.get(
                    "measurementsIngested",
                    result.get("acceptedCount", result.get("validMeasurements", len(measurements)))
                )
                self.alerts_triggered += result.get("alertsTriggered", 0)
                self.last_sync_utc = datetime.now(timezone.utc).isoformat()
                self.connected = True
                self.last_error = None
                return result
            elif resp.status_code == 400:
                # If batch is no longer InProgress (e.g. aborted/completed), clear active batch
                self.last_error = f"Ingestion rejected (HTTP 400): {resp.text}"
                self.active_batch_id = None
                self.active_batch_code = None
            else:
                self.last_error = f"Ingestion failed (HTTP {resp.status_code}): {resp.text}"
        except Exception as e:
            self.connected = False
            self.last_error = f"Ingestion transport error: {str(e)}"

        return None

    async def abort_batch(self, reason: str) -> Optional[Dict[str, Any]]:
        """Invokes POST /api/batches/{id}/abort with mandatory regulatory reason."""
        if not self.active_batch_id:
            return None

        client = self._get_client()
        batch_id = self.active_batch_id
        try:
            resp = await client.post(
                f"{self.engine_url}/api/batches/{batch_id}/abort",
                json={"reason": reason}
            )
            self.active_batch_id = None
            self.active_batch_code = None
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            self.last_error = f"Abort batch error: {str(e)}"

        return None

    async def complete_batch(self) -> Optional[Dict[str, Any]]:
        """Completes active batch run and retrieves its analytics summary."""
        if not self.active_batch_id:
            return None

        client = self._get_client()
        batch_id = self.active_batch_id
        try:
            complete_resp = await client.post(
                f"{self.engine_url}/api/batches/{batch_id}/complete",
                json={}
            )
            summary_resp = await client.get(f"{self.engine_url}/api/batches/{batch_id}/summary")
            self.active_batch_id = None
            self.active_batch_code = None
            if summary_resp.status_code == 200:
                return summary_resp.json()
            elif complete_resp.status_code == 200:
                return complete_resp.json()
        except Exception as e:
            self.last_error = f"Complete batch error: {str(e)}"

        return None

    async def get_summary(self, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves analytics summary for the specified or active batch."""
        target_id = batch_id or self.active_batch_id
        if not target_id:
            return None

        client = self._get_client()
        try:
            resp = await client.get(f"{self.engine_url}/api/batches/{target_id}/summary")
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            self.last_error = f"Get summary error: {str(e)}"

        return None

    def get_status(self) -> BridgeStatus:
        return BridgeStatus(
            enabled=self.enabled,
            connected=self.connected,
            engine_url=self.engine_url,
            ingestion_mode=self.ingestion_mode,
            active_batch_id=self.active_batch_id,
            active_batch_code=self.active_batch_code,
            frames_sent=self.frames_sent,
            measurements_ingested=self.measurements_ingested,
            alerts_triggered=self.alerts_triggered,
            last_sync_utc=self.last_sync_utc,
            last_error=self.last_error,
        )
