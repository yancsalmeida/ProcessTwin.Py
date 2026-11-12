"""
Automated Unit Tests for ChemFlowThermodynamicsBridge (ProcessTwin.Py <-> ChemFlow.NET).
Validates:
1. Bridge initialization, configuration, and default parameters.
2. Health check probing and error tracking.
3. Flash calculation against ChemFlow.NET Peng-Robinson API with proper unit conversion.
4. Graceful analytical fallback when ChemFlow.NET service is unreachable.
5. FastAPI endpoints /api/thermo/current and /api/thermo/status.
6. PlantTelemetry schema serialization with thermodynamic fields.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from process_twin.core.chemflow_bridge import ChemFlowThermodynamicsBridge, ChemFlowStatus
from process_twin.core.models import PlantTelemetry, PlantStatus
from process_twin.api.app import app


@pytest.fixture
def bridge() -> ChemFlowThermodynamicsBridge:
    return ChemFlowThermodynamicsBridge(
        chemflow_url="http://localhost:5080",
        enabled=True,
        model="Peng-Robinson"
    )


def test_bridge_defaults(bridge: ChemFlowThermodynamicsBridge):
    assert bridge.chemflow_url == "http://localhost:5080"
    assert bridge.enabled is True
    assert bridge.model == "Peng-Robinson"
    assert bridge.connected is False
    assert bridge.requests_count == 0

    status = bridge.get_status()
    assert isinstance(status, ChemFlowStatus)
    assert status.enabled is True
    assert status.connected is False


@pytest.mark.asyncio
async def test_health_check_success(bridge: ChemFlowThermodynamicsBridge):
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200)

        ok = await bridge.check_health()
        assert ok is True
        assert bridge.connected is True
        assert bridge.last_error is None


@pytest.mark.asyncio
async def test_health_check_failure(bridge: ChemFlowThermodynamicsBridge):
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        ok = await bridge.check_health()
        assert ok is False
        assert bridge.connected is False
        assert "Connection refused" in (bridge.last_error or "")


@pytest.mark.asyncio
async def test_calculate_flash_success(bridge: ChemFlowThermodynamicsBridge):
    mock_response_data = {
        "thermodynamicModel": "Peng-Robinson EOS (1976)",
        "phase": "VaporLiquidEquilibrium",
        "temperatureKelvin": 368.15,
        "temperatureCelsius": 95.0,
        "pressureKpa": 101.325,
        "vaporFractionBeta": 0.4403,
        "compoundIds": ["C6H6", "C7H8"],
        "overallCompositionZ": [0.5, 0.5],
        "liquidCompositionX": [0.4022, 0.5978],
        "vaporCompositionY": [0.6243, 0.3757],
        "kValues": [1.552, 0.628],
        "molarEnthalpyJPerMol": 16719.2,
        "iterations": 8
    }

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response_data)

        res = await bridge.calculate_flash(temperature_c=95.0, pressure_bar=1.01325)

        assert res["vaporFractionBeta"] == 0.4403
        assert res["phase"] == "VaporLiquidEquilibrium"
        assert bridge.connected is True
        assert bridge.requests_count == 1
        assert bridge.last_beta == 0.4403
        assert bridge.last_phase == "VaporLiquidEquilibrium"
        assert bridge.last_enthalpy_j_mol == 16719.2

        # Verify payload sent to ChemFlow.NET
        mock_post.assert_called_once()
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["temperatureKelvin"] == 368.15  # 95 + 273.15
        assert sent_payload["pressureKpa"] == 101.33      # 1.01325 * 100 rounded
        assert sent_payload["model"] == "Peng-Robinson"


@pytest.mark.asyncio
async def test_calculate_flash_local_fallback(bridge: ChemFlowThermodynamicsBridge):
    # When ChemFlow returns 500 or network fails, fallback should provide analytical VLE
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectTimeout("Timeout")

        res = await bridge.calculate_flash(temperature_c=25.0, pressure_bar=2.0)

        assert "Fallback" in res["thermodynamicModel"]
        assert res["phase"] == "SubcooledLiquid"
        assert res["vaporFractionBeta"] == 0.0
        assert bridge.last_beta == 0.0
        assert bridge.connected is False
        assert "Timeout" in (bridge.last_error or "")


def test_api_thermo_endpoints():
    with TestClient(app) as client:
        # GET /api/thermo/current
        resp_curr = client.get("/api/thermo/current")
        assert resp_curr.status_code == 200
        curr_data = resp_curr.json()
        assert "status" in curr_data
        assert "latest_flash" in curr_data

        # GET /api/thermo/status
        resp_stat = client.get("/api/thermo/status")
        assert resp_stat.status_code == 200
        stat_data = resp_stat.json()
        assert "chemflow_url" in stat_data
        assert "active_model" in stat_data


def test_telemetry_model_serialization_with_thermo():
    telemetry = PlantTelemetry(
        tank_level_m=5.0,
        tank_volume_m3=25.0,
        feed_flow_m3_h=24.0,
        discharge_flow_m3_h=24.0,
        reactor_temp_c=85.0,
        reactor_pressure_bar=1.013,
        cooling_valve_pct=45.0,
        feed_valve_open=True,
        interlock_tripped=False,
        status=PlantStatus.NORMAL,
        time_scale=5.0,
        vapor_fraction_beta=0.42,
        thermo_phase="VaporLiquidEquilibrium",
        thermo_model="Peng-Robinson EOS (1976)",
        molar_enthalpy_j_mol=14200.5,
    )

    data = telemetry.model_dump()
    assert data["vapor_fraction_beta"] == 0.42
    assert data["thermo_phase"] == "VaporLiquidEquilibrium"
    assert data["thermo_model"] == "Peng-Robinson EOS (1976)"
    assert data["molar_enthalpy_j_mol"] == 14200.5


@pytest.mark.asyncio
async def test_calculate_transport_properties_success(bridge: ChemFlowThermodynamicsBridge):
    mock_tp = {
        "dynamicViscosityCp": 0.452,
        "kinematicViscosityCSt": 0.531,
        "thermalConductivityWPerMK": 0.138,
        "prandtlNumber": 6.2,
        "densityKgPerM3": 851.0,
        "phase": "SubcooledLiquid",
        "modelName": "Chung et al. (1984/1988) Transport Model"
    }

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_tp)

        tp = await bridge.calculate_transport_properties(temperature_c=85.0, pressure_bar=1.5)

        assert tp.dynamic_viscosity_cp == 0.452
        assert tp.kinematic_viscosity_cst == 0.531
        assert tp.thermal_conductivity_w_m_k == 0.138
        assert tp.prandtl_number == 6.2
        assert tp.density_kg_m3 == 851.0
        assert bridge.last_transport is not None
        assert bridge.last_transport.dynamic_viscosity_cp == 0.452


@pytest.mark.asyncio
async def test_calculate_transport_properties_fallback(bridge: ChemFlowThermodynamicsBridge):
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        tp = await bridge.calculate_transport_properties(temperature_c=85.0, pressure_bar=1.5)

        assert "Fallback" in tp.model_name
        assert 0.1 < tp.dynamic_viscosity_cp < 2.0
        assert 0.05 < tp.thermal_conductivity_w_m_k < 0.5
        assert tp.prandtl_number > 0.0
        assert tp.density_kg_m3 > 700.0


@pytest.mark.asyncio
async def test_simulate_cstr_battery_fallback(bridge: ChemFlowThermodynamicsBridge):
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        battery = await bridge.simulate_cstr_battery(reactor_count=8, feed_temp_c=76.85)

        assert battery.reactor_count == 8
        assert len(battery.reactor_states) == 8
        assert battery.mean_conversion_pct >= 0.0
        assert battery.mean_temp_c >= 70.0
        assert bridge.last_battery is not None

