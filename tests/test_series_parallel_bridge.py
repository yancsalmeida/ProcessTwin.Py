"""
Automated Tests for Series-Parallel CSTR Plant & Thermal Runaway Integration
in ProcessTwin.Py (Bridge + NativeReactorCore Fallback + FastAPI endpoints).
"""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from process_twin.api.app import app, twin
from process_twin.core.chemflow_bridge import ChemFlowThermodynamicsBridge
from process_twin.core.models import (
    SeriesParallelPlantRequest,
    SeriesParallelRecyclePlantSummary,
    SeriesParallelRunawayProfile,
    SeriesParallelRunawayRequest,
)


@pytest.mark.asyncio
async def test_series_parallel_plant_offline_fallback():
    bridge = ChemFlowThermodynamicsBridge(chemflow_url="http://localhost:5080", enabled=False)
    summary = await bridge.simulate_series_parallel_plant(SeriesParallelPlantRequest())

    assert isinstance(summary, SeriesParallelRecyclePlantSummary)
    assert summary.source == "NativeReactorCore Fallback"
    assert summary.has_recycle is True
    assert summary.recycle_iterations >= 1
    assert 10.0 <= summary.single_pass_conversion_a_pct <= 95.0
    assert summary.overall_plant_conversion_a_pct > summary.single_pass_conversion_a_pct
    assert summary.selectivity_b_pct > 0.0
    assert summary.yield_b_pct > 0.0
    assert summary.product_b_purity_mole_fraction > 0.70
    assert summary.is_thermal_runaway is False


@pytest.mark.asyncio
async def test_series_parallel_plant_online_mock():
    bridge = ChemFlowThermodynamicsBridge(chemflow_url="http://localhost:5080", enabled=True)
    mock_payload = {
        "savedSimulationRunId": 42,
        "flowsheetName": "Series-Parallel CSTR Recycle Plant",
        "thermodynamicModel": "Ideal (Raoult's Law)",
        "hasRecycle": True,
        "recycleIterations": 11,
        "singlePassConversionAPercent": 46.8,
        "overallPlantConversionAPercent": 82.4,
        "yieldBPercent": 38.2,
        "selectivityBPercent": 81.6,
        "reactorTemperatureCelsius": 78.4,
        "jacketOutletTemperatureCelsius": 73.1,
        "outletCaMolPerL": 4.92,
        "outletCbMolPerL": 3.55,
        "outletCcMolPerL": 0.42,
        "outletCdMolPerL": 0.31,
        "heatGeneratedWatts": 18400.0,
        "heatRemovedWatts": 18350.0,
        "isThermalRunaway": False,
        "isMassConserved": True,
        "isEnergyConserved": True,
        "productBPurityMoleFraction": 0.912,
        "purifiedBenzeneFlowMolPerSec": 6.45,
        "isUsingNativeCore": True,
    }

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_payload)
        summary = await bridge.simulate_series_parallel_plant()

    assert summary.source == "ChemFlow.NET API"
    assert summary.saved_simulation_run_id == 42
    assert summary.overall_plant_conversion_a_pct == pytest.approx(82.4)
    assert summary.product_b_purity_mole_fraction == pytest.approx(0.912)
    assert bridge.connected is True


@pytest.mark.asyncio
async def test_series_parallel_runaway_profile_offline_fallback():
    bridge = ChemFlowThermodynamicsBridge(chemflow_url="http://localhost:5080", enabled=False)
    req = SeriesParallelRunawayRequest(steps=80)
    profile = await bridge.simulate_series_parallel_runaway_profile(req)

    assert isinstance(profile, SeriesParallelRunawayProfile)
    assert profile.source == "NativeReactorCore Fallback"
    assert len(profile.nominal_trajectory) == 80
    assert len(profile.runaway_trajectory) == 80
    assert profile.nominal_has_runaway is False
    assert profile.runaway_triggered is True
    assert profile.runaway_peak_temp_c > profile.nominal_peak_temp_c
    assert profile.runaway_peak_temp_c > profile.max_safe_temp_c
    assert profile.nominal_trajectory[-1].jacket_temp_c > req.jacket_inlet_temp_c
    assert profile.nominal_trajectory[-1].cb_mol_per_l > 0.0
    assert profile.runaway_trajectory[-1].status == "CRITICAL_RUNAWAY"


@pytest.mark.asyncio
async def test_series_parallel_runaway_profile_online_mock():
    bridge = ChemFlowThermodynamicsBridge(chemflow_url="http://localhost:5080", enabled=True)
    mock_frame = {
        "frameIndex": 1,
        "timeSeconds": 1.2,
        "caMolPerL": 0.92,
        "cbMolPerL": 0.07,
        "ccMolPerL": 0.005,
        "cdMolPerL": 0.005,
        "temperatureCelsius": 54.0,
        "jacketTemperatureCelsius": 23.5,
        "conversionAPct": 8.0,
        "yieldBPct": 7.0,
        "selectivityBPct": 87.5,
        "isThermalRunaway": False,
        "status": "NORMAL",
    }
    mock_runaway_frame = {
        **mock_frame,
        "frameIndex": 80,
        "temperatureCelsius": 195.0,
        "isThermalRunaway": True,
        "status": "CRITICAL_RUNAWAY",
    }
    mock_profile = {
        "engineMode": "Native C11 (v1.0)",
        "maxSafeTemperatureCelsius": 146.85,
        "nominalPeakTemperatureCelsius": 54.0,
        "nominalHasRunaway": False,
        "runawayPeakTemperatureCelsius": 195.0,
        "runawayTriggered": True,
        "nominalTrajectory": [mock_frame],
        "runawayTrajectory": [mock_runaway_frame],
    }

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_profile)
        profile = await bridge.simulate_series_parallel_runaway_profile()

    assert profile.source == "ChemFlow.NET API"
    assert profile.nominal_has_runaway is False
    assert profile.runaway_triggered is True
    assert profile.runaway_peak_temp_c == pytest.approx(195.0)
    assert profile.runaway_trajectory[-1].status == "CRITICAL_RUNAWAY"


def test_fastapi_series_parallel_endpoints():
    twin.latest_series_parallel_plant = None
    twin.latest_series_parallel_profile = None

    with TestClient(app) as client:
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("offline in test")):
            plant_res = client.post("/api/series-parallel/plant", json={})
            assert plant_res.status_code == 200
            plant_data = plant_res.json()
            assert plant_data["source"] == "NativeReactorCore Fallback"
            assert plant_data["overall_plant_conversion_a_pct"] > plant_data["single_pass_conversion_a_pct"]

            latest_res = client.get("/api/series-parallel/plant/latest")
            assert latest_res.status_code == 200
            assert latest_res.json()["selectivity_b_pct"] == pytest.approx(plant_data["selectivity_b_pct"])

            prof_res = client.post("/api/series-parallel/runaway-profile", json={"steps": 60})
            assert prof_res.status_code == 200
            prof_data = prof_res.json()
            assert prof_data["nominal_has_runaway"] is False
            assert prof_data["runaway_triggered"] is True
            assert len(prof_data["nominal_trajectory"]) == 60
            assert len(prof_data["runaway_trajectory"]) == 60
