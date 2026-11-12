import os
import math
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import httpx
from pydantic import BaseModel, Field

from process_twin.core.models import (
    TransportProperties,
    CstrBatteryTelemetry,
    CstrBatteryReactorState,
)
from process_twin.core.native_reactor import NativeReactorCore, NativeCstrParams


class ChemFlowStatus(BaseModel):
    enabled: bool = True
    connected: bool = False
    chemflow_url: str = "http://localhost:5080"
    active_model: str = "Peng-Robinson EOS (1976)"
    requests_count: int = 0
    last_beta: Optional[float] = None
    last_phase: Optional[str] = None
    last_enthalpy_j_mol: Optional[float] = None
    last_transport: Optional[TransportProperties] = None
    last_battery: Optional[CstrBatteryTelemetry] = None
    last_sync_utc: Optional[str] = None
    last_error: Optional[str] = None


class ChemFlowThermodynamicsBridge:
    """
    Asynchronous client bridging ProcessTwin.Py with ChemFlow.NET (.NET 8 Process Simulator).
    Queries rigorous cubic Peng-Robinson (1976) equation of state, isothermal VLE flash calculations,
    multicomponent fluid transport properties (Chung / Ely-Hanley TRAPP), and parallel CSTR battery simulations.
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
        self.last_transport: Optional[TransportProperties] = None
        self.last_battery: Optional[CstrBatteryTelemetry] = None
        self.last_sync_utc: Optional[str] = None
        self.last_error: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._native_reactor = NativeReactorCore()

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

    async def calculate_transport_properties(
        self,
        temperature_c: float,
        pressure_bar: float,
        compound_ids: Optional[List[str]] = None,
        composition_z: Optional[List[float]] = None,
        package: str = "Chung"
    ) -> TransportProperties:
        """
        Calculates rigorous fluid transport properties (viscosity, thermal conductivity, Prandtl number)
        via ChemFlow.NET microservice using Chung et al. (1984/1988) or Ely & Hanley TRAPP.
        Falls back seamlessly to local analytical correlation when ChemFlow.NET is unreachable.
        """
        temp_k = temperature_c + 273.15
        press_kpa = max(10.0, pressure_bar * 100.0)
        c_ids = compound_ids or ["C6H6", "C7H8"]
        z_fracs = composition_z or [0.5, 0.5]

        if not self.enabled:
            tp = self._local_fallback_transport_properties(temp_k, press_kpa, c_ids, z_fracs, package)
            self.last_transport = tp
            return tp

        client = self._get_client()
        payload = {
            "compoundIds": c_ids,
            "compositionZ": z_fracs,
            "temperatureKelvin": round(temp_k, 2),
            "pressureKpa": round(press_kpa, 2),
            "package": package
        }

        try:
            resp = await client.post(
                f"{self.chemflow_url}/api/transport/properties",
                json=payload
            )

            if resp.status_code == 200:
                data = resp.json()
                self.connected = True
                self.requests_count += 1
                tp = TransportProperties(
                    dynamic_viscosity_cp=data.get("dynamicViscosityCp", 0.5),
                    kinematic_viscosity_cst=data.get("kinematicViscosityCSt", 0.6),
                    thermal_conductivity_w_m_k=data.get("thermalConductivityWPerMK", 0.14),
                    prandtl_number=data.get("prandtlNumber", 6.5),
                    density_kg_m3=data.get("densityKgPerM3", 850.0),
                    phase=data.get("phase", "SubcooledLiquid"),
                    model_name=data.get("modelName", f"{package} Transport Model")
                )
                self.last_transport = tp
                self.last_sync_utc = datetime.now(timezone.utc).isoformat()
                self.last_error = None
                return tp
            else:
                self.last_error = f"HTTP {resp.status_code}: {resp.text}"
                tp = self._local_fallback_transport_properties(temp_k, press_kpa, c_ids, z_fracs, package)
                self.last_transport = tp
                return tp

        except Exception as e:
            self.connected = False
            self.last_error = f"Transport calculation failed: {str(e)}"
            tp = self._local_fallback_transport_properties(temp_k, press_kpa, c_ids, z_fracs, package)
            self.last_transport = tp
            return tp

    def _local_fallback_transport_properties(
        self,
        temp_k: float,
        press_kpa: float,
        compound_ids: List[str],
        composition_z: List[float],
        package: str = "Chung"
    ) -> TransportProperties:
        """
        Local analytical fallback using Letsou & Stiel (1973) corresponding states for liquid viscosity
        and Latini (1984) for thermal conductivity.
        """
        # Parameters for standard species (Benzene, Toluene)
        # Tc (K), Pc (bar), M (g/mol), omega
        species_db = {
            "C6H6": {"tc": 562.2, "pc_bar": 48.98, "mw": 78.11, "omega": 0.210},
            "C7H8": {"tc": 591.8, "pc_bar": 41.06, "mw": 92.14, "omega": 0.264},
            "n-C6H14": {"tc": 507.6, "pc_bar": 30.25, "mw": 86.18, "omega": 0.301},
            "n-C7H16": {"tc": 540.2, "pc_bar": 27.40, "mw": 100.21, "omega": 0.350},
        }

        pure_mu_cp = []
        pure_lambda = []
        total_mw = 0.0

        for c_id, z in zip(compound_ids, composition_z):
            prop = species_db.get(c_id, {"tc": 570.0, "pc_bar": 45.0, "mw": 80.0, "omega": 0.23})
            tc = prop["tc"]
            pc_bar = prop["pc_bar"]
            mw = prop["mw"]
            omega = prop["omega"]
            total_mw += z * mw

            tr = min(0.98, max(0.3, temp_k / tc))

            # Letsou-Stiel (1973)
            xi = math.pow(tc, 1.0 / 6.0) / (math.sqrt(mw) * math.pow(pc_bar, 2.0 / 3.0))
            f0 = 0.015174 - 0.02135 * tr + 0.0075 * tr * tr
            f1 = 0.042552 - 0.07674 * tr + 0.0340 * tr * tr
            eta_xi = max(0.001, f0 + omega * f1)
            mu_cp = eta_xi / xi
            pure_mu_cp.append(mu_cp)

            # Latini (1984) in SI W/(m·K)
            a_latini = 0.359 * math.pow(tc, 0.216) / math.sqrt(mw)
            lam = max(0.05, (a_latini * math.pow(1.0 - tr, 0.38)) / math.pow(tr, 1.0 / 6.0))
            pure_lambda.append(lam)

        # Logarithmic mixing rule for viscosity
        ln_mu = sum(z * math.log(max(1e-4, mu)) for z, mu in zip(composition_z, pure_mu_cp))
        mix_mu_cp = math.exp(ln_mu)

        # Mass-fraction mixing rule for thermal conductivity
        mix_lambda = sum(z * lam for z, lam in zip(composition_z, pure_lambda))

        # Density approx for organic liquids ~850-870 kg/m³
        density = 865.0
        kinematic_cst = (mix_mu_cp / density) * 1000.0  # cP / (kg/m³) * 1000 = mm²/s (cSt)
        cp_j_kg_k = 1850.0  # approximate liquid heat capacity
        prandtl = (cp_j_kg_k * (mix_mu_cp * 1e-3)) / mix_lambda

        return TransportProperties(
            dynamic_viscosity_cp=round(mix_mu_cp, 4),
            kinematic_viscosity_cst=round(kinematic_cst, 4),
            thermal_conductivity_w_m_k=round(mix_lambda, 5),
            prandtl_number=round(prandtl, 3),
            density_kg_m3=round(density, 2),
            phase="SubcooledLiquid",
            model_name=f"{package} Analytical Fallback"
        )

    async def simulate_cstr_battery(
        self,
        reactor_count: int = 8,
        total_feed_flow_mol_s: float = 100.0,
        feed_temp_c: float = 76.85,
        feed_pressure_bar: float = 1.5,
        reactant_id: str = "C6H6",
        product_id: str = "C7H8",
        composition_z: Optional[List[float]] = None,
        reactor_volume_l: float = 100.0,
        coolant_temp_c: float = 26.85,
        cooling_ua_w_k: float = 666.67,
        max_safe_temp_c: float = 126.85,
        package: str = "Chung"
    ) -> CstrBatteryTelemetry:
        """
        Executes a high-speed parallel CSTR battery simulation across N reactors simultaneously
        via ChemFlow.NET (.NET 8 SIMD AVX2 engine).
        Falls back seamlessly to local native C11 / Python batch integration when offline.
        """
        feed_temp_k = feed_temp_c + 273.15
        coolant_temp_k = coolant_temp_c + 273.15
        max_safe_temp_k = max_safe_temp_c + 273.15
        feed_press_kpa = max(10.0, feed_pressure_bar * 100.0)
        comp_z = composition_z or [0.80, 0.20]

        if not self.enabled:
            battery = self._local_fallback_cstr_battery(
                reactor_count, total_feed_flow_mol_s, feed_temp_k, feed_press_kpa,
                reactant_id, product_id, comp_z, reactor_volume_l,
                coolant_temp_k, cooling_ua_w_k, max_safe_temp_k, package
            )
            self.last_battery = battery
            return battery

        client = self._get_client()
        payload = {
            "reactorCount": reactor_count,
            "totalFeedFlowMolPerSec": total_feed_flow_mol_s,
            "feedTemperatureKelvin": round(feed_temp_k, 2),
            "feedPressureKpa": round(feed_press_kpa, 2),
            "reactantCompoundId": reactant_id,
            "productCompoundId": product_id,
            "compositionZ": comp_z,
            "reactorVolumeLiters": reactor_volume_l,
            "coolantTemperatureKelvin": round(coolant_temp_k, 2),
            "coolingUaWattsPerKelvin": cooling_ua_w_k,
            "maxSafeTemperatureKelvin": round(max_safe_temp_k, 2),
            "transportPackage": package
        }

        try:
            resp = await client.post(
                f"{self.chemflow_url}/api/telemetry/cstr/battery",
                json=payload
            )

            if resp.status_code == 200:
                data = resp.json()
                self.connected = True
                self.requests_count += 1

                states: List[CstrBatteryReactorState] = []
                for s in data.get("reactorStates", []):
                    states.append(CstrBatteryReactorState(
                        reactor_index=s.get("reactorIndex", 0),
                        ca_mol_l=s.get("reactantConcentrationMolPerL", 0.5),
                        temp_c=s.get("temperatureCelsius", 80.0),
                        conversion_pct=s.get("conversionPct", 0.0),
                        reaction_rate_mol_l_min=s.get("reactionRateMolPerLMin", 0.0),
                        is_runaway=s.get("isRunaway", False),
                        dynamic_viscosity_cp=s.get("dynamicViscosityCp", 0.5),
                        thermal_conductivity_w_m_k=s.get("thermalConductivityWPerMK", 0.14),
                        prandtl_number=s.get("prandtlNumber", 6.0)
                    ))

                battery = CstrBatteryTelemetry(
                    battery_id=data.get("batteryId", "CSTR-BATTERY-SIM"),
                    name=data.get("name", f"Parallel CSTR Battery ({reactor_count} Reactors)"),
                    reactor_count=data.get("reactorCount", reactor_count),
                    mean_conversion_pct=data.get("meanConversionPct", 0.0),
                    mean_temp_c=data.get("meanTemperatureCelsius", feed_temp_c),
                    mean_ca_mol_l=data.get("meanOutletConcentrationMolPerL", 0.5),
                    has_thermal_runaway=data.get("hasThermalRunaway", False),
                    is_using_native_simd=data.get("isUsingNativeSimd", True),
                    energy_duty_kw=data.get("energyDutyKilowatts", 0.0),
                    reactor_states=states
                )
                self.last_battery = battery
                self.last_sync_utc = datetime.now(timezone.utc).isoformat()
                self.last_error = None
                return battery
            else:
                self.last_error = f"HTTP {resp.status_code}: {resp.text}"
                battery = self._local_fallback_cstr_battery(
                    reactor_count, total_feed_flow_mol_s, feed_temp_k, feed_press_kpa,
                    reactant_id, product_id, comp_z, reactor_volume_l,
                    coolant_temp_k, cooling_ua_w_k, max_safe_temp_k, package
                )
                self.last_battery = battery
                return battery

        except Exception as e:
            self.connected = False
            self.last_error = f"Battery simulation request failed: {str(e)}"
            battery = self._local_fallback_cstr_battery(
                reactor_count, total_feed_flow_mol_s, feed_temp_k, feed_press_kpa,
                reactant_id, product_id, comp_z, reactor_volume_l,
                coolant_temp_k, cooling_ua_w_k, max_safe_temp_k, package
            )
            self.last_battery = battery
            return battery

    def _local_fallback_cstr_battery(
        self,
        reactor_count: int,
        total_feed_flow_mol_s: float,
        feed_temp_k: float,
        feed_press_kpa: float,
        reactant_id: str,
        product_id: str,
        composition_z: List[float],
        reactor_volume_l: float,
        coolant_temp_k: float,
        cooling_ua_w_k: float,
        max_safe_temp_k: float,
        package: str
    ) -> CstrBatteryTelemetry:
        """
        Local fallback solving parallel CSTR battery via native C11 / Python batch integration.
        """
        # Feed setup: q = flow per reactor
        flow_per_reactor_mol_s = total_feed_flow_mol_s / max(1, reactor_count)
        avg_mw = 0.8 * 78.11 + 0.2 * 92.14
        mass_flow_kg_s = (flow_per_reactor_mol_s * avg_mw) / 1000.0
        q_l_min = (mass_flow_kg_s / 1.0) * 60.0  # rho approx 1.0 kg/L
        if q_l_min <= 0.0:
            q_l_min = 100.0

        tau_min = reactor_volume_l / q_l_min
        ca_inlet = max(0.01, composition_z[0] * 1.0)

        params = NativeCstrParams(
            volumetric_flow_l_per_min=q_l_min,
            volume_liters=reactor_volume_l,
            k0_per_min=7.08e10,
            ea_over_r_kelvin=8330.1,
            delta_h_j_per_mol=200000.0,
            rho_cp_j_per_l_k=4000.0,
            ua_j_per_min_k=cooling_ua_w_k * 60.0,
            ca_feed_mol_per_l=ca_inlet,
            t_feed_kelvin=feed_temp_k,
            t_coolant_kelvin=coolant_temp_k,
        )

        ca_list = [ca_inlet] * reactor_count
        temp_list = [feed_temp_k] * reactor_count

        dt = min(0.005, tau_min / 200.0)
        steps = int(math.ceil(tau_min / dt))
        if steps < 10:
            steps = 10
        dt = tau_min / steps

        # Solve across tau
        curr_ca = list(ca_list)
        curr_temp = list(temp_list)
        for _ in range(steps):
            curr_ca, curr_temp = self._native_reactor.step_batch_rk4(params, curr_ca, curr_temp, dt)

        raw_states = self._native_reactor.step_batch_detailed(
            params, curr_ca, curr_temp, dt, max_safe_temp_k
        )

        states: List[CstrBatteryReactorState] = []
        any_runaway = False
        sum_conv = 0.0
        sum_temp_c = 0.0
        sum_ca = 0.0

        for i, s in enumerate(raw_states):
            t_c = s.temp_kelvin - 273.15
            if s.is_runaway:
                any_runaway = True
            sum_conv += s.conversion_pct
            sum_temp_c += t_c
            sum_ca += s.ca_mol_per_l

            # Approximate local transport properties
            tp = self._local_fallback_transport_properties(
                s.temp_kelvin, feed_press_kpa, [reactant_id, product_id], composition_z, package
            )

            states.append(CstrBatteryReactorState(
                reactor_index=i,
                ca_mol_l=round(s.ca_mol_per_l, 4),
                temp_c=round(t_c, 2),
                conversion_pct=round(s.conversion_pct, 2),
                reaction_rate_mol_l_min=round(s.reaction_rate_mol_per_l_min, 4),
                is_runaway=s.is_runaway,
                dynamic_viscosity_cp=round(tp.dynamic_viscosity_cp, 4),
                thermal_conductivity_w_m_k=round(tp.thermal_conductivity_w_m_k, 5),
                prandtl_number=round(tp.prandtl_number, 3),
            ))

        mean_conv = sum_conv / max(1, reactor_count)
        mean_t_c = sum_temp_c / max(1, reactor_count)
        mean_ca = sum_ca / max(1, reactor_count)

        return CstrBatteryTelemetry(
            battery_id="CSTR-BATTERY-LOCAL",
            name=f"Parallel CSTR Battery ({reactor_count} Reactors - Local Solver)",
            reactor_count=reactor_count,
            mean_conversion_pct=round(mean_conv, 2),
            mean_temp_c=round(mean_t_c, 2),
            mean_ca_mol_l=round(mean_ca, 4),
            has_thermal_runaway=any_runaway,
            is_using_native_simd=self._native_reactor.has_simd(),
            energy_duty_kw=round((cooling_ua_w_k * (mean_t_c - (coolant_temp_k - 273.15))) / 1000.0, 2),
            reactor_states=states,
        )

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
            last_transport=self.last_transport,
            last_battery=self.last_battery,
            last_sync_utc=self.last_sync_utc,
            last_error=self.last_error
        )
