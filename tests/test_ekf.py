import math
import random
import pytest
from httpx import AsyncClient, ASGITransport

from process_twin.core.ekf import CstrExtendedKalmanFilter
from process_twin.core.simulator import PlantDigitalTwin
from process_twin.api.app import app


def test_ekf_initialization():
    ekf = CstrExtendedKalmanFilter(initial_ca=0.5, initial_temp_c=85.0)
    assert ekf.ca == 0.5
    assert ekf.temp_c == 85.0
    assert ekf.p11 > 0.0
    assert ekf.p22 > 0.0
    assert ekf.p12 == 0.0
    # Determinant of initial covariance is strictly positive
    det = ekf.p11 * ekf.p22 - ekf.p12 * ekf.p12
    assert det > 0.0


def test_ekf_convergence_from_perturbed_initial_ca():
    """
    Core EKF validation:
    True reactor state is Ca = 0.5 mol/L and T = 85.0 °C.
    The filter is initialized with a deliberately large error (Ca_hat = 0.90 mol/L, an 80% error).
    Feeding noisy temperature measurements must drive the estimated Ca back towards 0.5 mol/L.
    """
    random.seed(42)
    twin = PlantDigitalTwin()
    twin.time_scale = 1.0
    twin.temp_noise_std = 0.10  # realistic transmitter noise

    # Initialize EKF with perturbed estimate
    ekf = CstrExtendedKalmanFilter(
        initial_ca=0.90,  # Deliberately perturbed
        initial_temp_c=85.0,
        r_temp=0.10 ** 2,
    )

    initial_error = abs(ekf.ca - 0.5)
    assert initial_error == pytest.approx(0.40, abs=1e-3)

    # Run for 200 simulation steps (~40 seconds simulated)
    for _ in range(200):
        frame = twin.step(dt_seconds=0.2)
        ekf.step(
            dt_seconds=0.2,
            measured_temp_c=frame.reactor_temp_c,
            flow_m3_h=twin.discharge_flow_m3_h,
            cooling_valve_pct=twin.cooling_valve_pct,
            feed_active=twin.feed_valve_open,
        )

    final_error = abs(ekf.ca - 0.5)
    # Estimate must significantly reduce error towards steady-state
    assert final_error < 0.08
    assert ekf.conversion_pct == pytest.approx(50.0, abs=8.0)


def test_ekf_covariance_symmetry_and_positive_definiteness():
    ekf = CstrExtendedKalmanFilter(initial_ca=0.5, initial_temp_c=85.0)

    for i in range(50):
        noisy_t = 85.0 + math.sin(i * 0.1) * 2.0
        ekf.step(dt_seconds=0.2, measured_temp_c=noisy_t)

        # Symmetry
        assert ekf.p12 == pytest.approx(ekf.p21, abs=1e-9)
        # Positive variances
        assert ekf.p11 > 0.0
        assert ekf.p22 > 0.0
        # Positive determinant
        det = ekf.p11 * ekf.p22 - ekf.p12 * ekf.p12
        assert det > 0.0


def test_ekf_nis_sensor_anomaly_detection():
    """
    When an unexpected sensor spike or transmitter bias occurs,
    Normalized Innovation Squared (NIS) spikes above the chi-squared threshold (e.g. > 9.0).
    """
    ekf = CstrExtendedKalmanFilter(initial_ca=0.5, initial_temp_c=85.0, r_temp=0.15 ** 2)

    # Run normal steady-state steps
    for _ in range(20):
        ekf.step(dt_seconds=0.2, measured_temp_c=85.0)

    assert ekf.last_nis < 3.84  # Normal 95% chi2 bound

    # Inject sudden large sensor glitch (+12 °C transmitter surge)
    estimate = ekf.step(dt_seconds=0.2, measured_temp_c=97.0)
    assert estimate.nis > 10.0
    assert abs(estimate.temp_residual) > 5.0


def test_simulator_embeds_ekf_in_telemetry_stream():
    twin = PlantDigitalTwin()
    frame = twin.step(dt_seconds=0.2)

    assert frame.estimated_ca_mol_l is not None
    assert frame.estimated_temp_c is not None
    assert frame.estimated_conversion_pct is not None
    assert frame.estimated_reaction_rate is not None
    assert frame.ekf_temp_residual_c is not None
    assert frame.ekf_nis is not None

    # Steady-state sanity bounds
    assert 0.2 <= frame.estimated_ca_mol_l <= 0.8
    assert 80.0 <= frame.estimated_temp_c <= 90.0
    assert 30.0 <= frame.estimated_conversion_pct <= 70.0


@pytest.mark.asyncio
async def test_ekf_diagnostics_api_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/ekf/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "estimate" in data
        assert "diagnostics" in data
        assert data["estimate"]["ca_mol_per_l"] is not None
        assert data["diagnostics"]["sensor_healthy"] is True
        assert data["diagnostics"]["p_ca"] > 0
        assert data["diagnostics"]["p_temp"] > 0
