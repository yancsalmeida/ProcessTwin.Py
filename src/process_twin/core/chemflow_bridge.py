import os
import math
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, Field


class ChemFlowStatus(BaseModel):
    enabled: bool = True
    connected: bool = False
    chemflow_url: str = "http://localhost:5080"
    active_model: str = "Peng-Robinson EOS (1976)"
    requests_count: int = 0
    last_beta: Optional[float] = None
    last_phase: Optional[str] = None
    last_enthalpy_j_mol: Optional[float] = None
    last_sync_utc: Optional[str] = None
    last_error: Optional[str] = None


class ChemFlowThermodynamicsBridge:
    """
    Asynchronous client bridging ProcessTwin.Py with ChemFlow.NET (.NET 8 Process Simulator).
    Queries rigorous cubic Peng-Robinson (1976) equation of state and isothermal VLE flash calculations
    to obtain real-fluid vapor fractions, fugacities, and enthalpy departures.
    Includes automated local analytical fallback when ChemFlow.NET is offline.
    """

    def __init__(
        self,
        chemflow_url: Optional[str] = None,
        enabled: bool = True,
        model: str = "Peng-Robinson"
    ) -> None:
        self.chemflow_url = (
            chemflow_url
            or os.getenv("CHEMFLOW_ENGINE_URL")
            or os.getenv("CHEMFLOW_URL")
            or "http://localhost:5080"
        ).rstrip("/")
        self.enabled = enabled
        self.model = model
        self.connected = False
        self.requests_count = 0
        self.last_beta: Optional[float] = None
        self.last_phase: Optional[str] = None
        self.last_enthalpy_j_mol: Optional[float] = None
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
        """Pings ChemFlow.NET compounds endpoint to verify service reachability."""
        if not self.enabled:
            self.connected = False
            return False

        client = self._get_client()
        try:
            resp = await client.get(f"{self.chemflow_url}/api/compounds")
            self.connected = (resp.status_code == 200)
            if self.connected:
                self.last_error = None
            return self.connected
        except Exception as e:
            self.connected = False
            self.last_error = f"ChemFlow health check failed: {str(e)}"
            return False

    async def calculate_flash(
        self,
        temperature_c: float,
        pressure_bar: float,
        compound_ids: Optional[List[str]] = None,
        composition_z: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Executes isothermal flash (T, P) against ChemFlow.NET using Peng-Robinson EOS.
        Converts units: T (°C -> K), P (bar -> kPa).
        Falls back gracefully to local Raoult/Antoine calculation if ChemFlow.NET is offline.
        """
        temp_k = temperature_c + 273.15
        press_kpa = max(10.0, pressure_bar * 100.0)
        c_ids = compound_ids or ["C6H6", "C7H8"]
        z_fracs = composition_z or [0.5, 0.5]

        if not self.enabled:
            return self._local_fallback_flash(temp_k, press_kpa, c_ids, z_fracs)

        client = self._get_client()
        payload = {
            "compoundIds": c_ids,
            "compositionZ": z_fracs,
            "temperatureKelvin": round(temp_k, 2),
            "pressureKpa": round(press_kpa, 2),
            "model": self.model
        }

        try:
            resp = await client.post(
                f"{self.chemflow_url}/api/thermo/flash",
                json=payload
            )

            if resp.status_code == 200:
                data = resp.json()
                self.connected = True
                self.requests_count += 1
                self.last_beta = data.get("vaporFractionBeta")
                self.last_phase = data.get("phase")
                self.last_enthalpy_j_mol = data.get("molarEnthalpyJPerMol")
                self.last_sync_utc = datetime.now(timezone.utc).isoformat()
                self.last_error = None
                return data
            else:
                self.last_error = f"HTTP {resp.status_code}: {resp.text}"
                return self._local_fallback_flash(temp_k, press_kpa, c_ids, z_fracs)

        except Exception as e:
            self.connected = False
            self.last_error = f"Flash request failed: {str(e)}"
            return self._local_fallback_flash(temp_k, press_kpa, c_ids, z_fracs)

    def _local_fallback_flash(
        self,
        temp_k: float,
        press_kpa: float,
        compound_ids: List[str],
        composition_z: List[float]
    ) -> Dict[str, Any]:
        """
        Local analytical fallback using Antoine vapor pressure correlation
        when ChemFlow.NET microservice is unavailable.
        """
        # Benzene Antoine: log10(P_sat/bar) = 4.72583 - 1660.652 / (T - 1.461)
        # T in Kelvin, P in bar => kPa
        t_c = temp_k - 273.15
        p_sat_bar = math.exp(11.68 - 3816.44 / (t_c + 227.02))
        p_sat_kpa = p_sat_bar * 100.0

        if press_kpa > p_sat_kpa * 1.05:
            phase = "SubcooledLiquid"
            beta = 0.0
        elif press_kpa < p_sat_kpa * 0.95:
            phase = "SuperheatedVapor"
            beta = 1.0
        else:
            phase = "VaporLiquidEquilibrium"
            beta = max(0.0, min(1.0, (p_sat_kpa - press_kpa * 0.95) / (p_sat_kpa * 0.1)))

        self.last_beta = round(beta, 4)
        self.last_phase = phase
        self.last_sync_utc = datetime.now(timezone.utc).isoformat()

        return {
            "thermodynamicModel": "Antoine Local Fallback (Raoult)",
            "phase": phase,
            "temperatureKelvin": round(temp_k, 2),
            "temperatureCelsius": round(t_c, 2),
            "pressureKpa": round(press_kpa, 2),
            "vaporFractionBeta": self.last_beta,
            "compoundIds": compound_ids,
            "overallCompositionZ": composition_z,
            "liquidCompositionX": composition_z,
            "vaporCompositionY": composition_z,
            "kValues": [round(p_sat_kpa / press_kpa, 4)] * len(compound_ids),
            "molarEnthalpyJPerMol": round(31500.0 * beta, 2),
            "iterations": 1
        }

    def get_status(self) -> ChemFlowStatus:
        return ChemFlowStatus(
            enabled=self.enabled,
            connected=self.connected,
            chemflow_url=self.chemflow_url,
            active_model=self.model,
            requests_count=self.requests_count,
            last_beta=self.last_beta,
            last_phase=self.last_phase,
            last_enthalpy_j_mol=self.last_enthalpy_j_mol,
            last_sync_utc=self.last_sync_utc,
            last_error=self.last_error
        )
