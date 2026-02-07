"""
Integration tests for API endpoints.

Uses httpx AsyncClient with ASGITransport to test FastAPI endpoints
without spinning up a live server.
"""

import os
import tempfile

# Ensure config env vars are set BEFORE importing create_app,
# since Pydantic settings validation occurs at import time.
os.environ.setdefault("PTPD_GCP_PROJECT_ID", "test-project")
os.environ.setdefault("PTPD_GCS_BUCKET", "test-bucket")
os.environ.setdefault("PTPD_GCP_REGION", "us-central1")
# Force local filesystem backend to avoid hitting real GCS:
os.environ["PTPD_FORCE_LOCAL_STORAGE"] = "true"
_staging_dir = tempfile.mkdtemp(prefix="ptpd_test_")
os.environ["PTPD_STAGING_DIR"] = _staging_dir

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from ptpd_calibration.api.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    """Create a FastAPI app instance for the test module."""
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """Create an httpx AsyncClient bound to the ASGI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DENSITIES = [0.08, 0.15, 0.28, 0.42, 0.58, 0.72, 0.88, 1.02, 1.18,
                     1.32, 1.45, 1.55, 1.65, 1.72, 1.78, 1.82, 1.86, 1.89,
                     1.91, 1.93, 1.95]

SAMPLE_CURVE_INPUTS = [round(i / 20, 2) for i in range(21)]
SAMPLE_CURVE_OUTPUTS = [round((i / 20) ** 0.9, 4) for i in range(21)]


# ---------------------------------------------------------------------------
# Health & Root
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    """Tests for health and root endpoints."""

    @pytest.mark.asyncio
    async def test_root(self, client: httpx.AsyncClient):
        """GET / returns API info."""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health(self, client: httpx.AsyncClient):
        """GET /api/health returns healthy status."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    """Tests for the density analysis endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_densities(self, client: httpx.AsyncClient):
        """POST /api/analyze returns density metrics."""
        resp = await client.post("/api/analyze", json={
            "densities": SAMPLE_DENSITIES,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert "dmin" in data
        assert "dmax" in data
        assert "range" in data
        assert "is_monotonic" in data
        assert "max_error" in data
        assert "rms_error" in data
        assert "suggestions" in data

        assert data["dmin"] == pytest.approx(min(SAMPLE_DENSITIES))
        assert data["dmax"] == pytest.approx(max(SAMPLE_DENSITIES))
        assert data["range"] == pytest.approx(data["dmax"] - data["dmin"])


# ---------------------------------------------------------------------------
# Curves — Generate, Retrieve, Modify, Smooth, Export
# ---------------------------------------------------------------------------

class TestCurveEndpoints:
    """Tests for curve CRUD and manipulation endpoints."""

    @pytest.mark.asyncio
    async def test_generate_curve(self, client: httpx.AsyncClient):
        """POST /api/curves/generate creates a curve and returns its ID."""
        resp = await client.post("/api/curves/generate", json={
            "densities": SAMPLE_DENSITIES,
            "name": "Integration Test Curve",
            "curve_type": "linear",
            "paper_type": "Platine Rag 310",
            "chemistry": "Pt/Pd 50/50",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert "curve_id" in data
        assert data["name"] == "Integration Test Curve"
        assert data["num_points"] > 0
        assert len(data["input_values"]) > 0
        assert len(data["output_values"]) > 0

    @pytest.mark.asyncio
    async def test_modify_curve_brightness(self, client: httpx.AsyncClient):
        """POST /api/curves/modify applies brightness adjustment."""
        resp = await client.post("/api/curves/modify", json={
            "input_values": SAMPLE_CURVE_INPUTS,
            "output_values": SAMPLE_CURVE_OUTPUTS,
            "name": "Brightness Adjusted",
            "adjustment_type": "brightness",
            "amount": 0.1,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["adjustment_applied"] == "brightness"
        assert len(data["output_values"]) == len(SAMPLE_CURVE_INPUTS)

    @pytest.mark.asyncio
    async def test_modify_curve_contrast(self, client: httpx.AsyncClient):
        """POST /api/curves/modify applies contrast adjustment."""
        resp = await client.post("/api/curves/modify", json={
            "input_values": SAMPLE_CURVE_INPUTS,
            "output_values": SAMPLE_CURVE_OUTPUTS,
            "name": "Contrast Adjusted",
            "adjustment_type": "contrast",
            "amount": 0.15,
            "pivot": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["adjustment_applied"] == "contrast"

    @pytest.mark.asyncio
    async def test_modify_curve_gamma(self, client: httpx.AsyncClient):
        """POST /api/curves/modify applies gamma adjustment."""
        resp = await client.post("/api/curves/modify", json={
            "input_values": SAMPLE_CURVE_INPUTS,
            "output_values": SAMPLE_CURVE_OUTPUTS,
            "name": "Gamma Adjusted",
            "adjustment_type": "gamma",
            "amount": 0.2,
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @pytest.mark.asyncio
    async def test_smooth_curve_gaussian(self, client: httpx.AsyncClient):
        """POST /api/curves/smooth applies Gaussian smoothing."""
        resp = await client.post("/api/curves/smooth", json={
            "input_values": SAMPLE_CURVE_INPUTS,
            "output_values": SAMPLE_CURVE_OUTPUTS,
            "name": "Smoothed Curve",
            "method": "gaussian",
            "strength": 0.3,
            "preserve_endpoints": True,
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["method_applied"] == "gaussian"
        assert len(data["output_values"]) == len(SAMPLE_CURVE_INPUTS)

    @pytest.mark.asyncio
    async def test_export_curve_csv(self, client: httpx.AsyncClient):
        """POST /api/curves/export returns a downloadable file."""
        densities_str = ",".join(str(d) for d in SAMPLE_DENSITIES)
        resp = await client.post("/api/curves/export", data={
            "densities": SAMPLE_DENSITIES,
            "name": "export_test",
            "format": "csv",
        })
        # Export uses Form fields; httpx sends them as form-encoded
        assert resp.status_code == 200
        # Should return file content (not JSON)
        assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_modify_unknown_type_returns_400(self, client: httpx.AsyncClient):
        """POST /api/curves/modify with unknown type returns 400."""
        resp = await client.post("/api/curves/modify", json={
            "input_values": SAMPLE_CURVE_INPUTS,
            "output_values": SAMPLE_CURVE_OUTPUTS,
            "adjustment_type": "nonexistent",
            "amount": 0.1,
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Calibrations — CRUD
# ---------------------------------------------------------------------------

class TestCalibrationEndpoints:
    """Tests for calibration record CRUD."""

    @pytest.mark.asyncio
    async def test_create_and_list_calibrations(self, client: httpx.AsyncClient):
        """POST /api/calibrations creates a record, GET lists it."""
        # Create
        create_resp = await client.post("/api/calibrations", json={
            "paper_type": "Platine Rag 310",
            "exposure_time": 12.5,
            "metal_ratio": 0.6,
            "contrast_agent": "none",
            "developer": "potassium_oxalate",
            "chemistry_type": "platinum_palladium",
            "densities": SAMPLE_DENSITIES,
            "notes": "Integration test calibration",
        })
        assert create_resp.status_code == 200
        create_data = create_resp.json()
        assert create_data["success"] is True
        cal_id = create_data["id"]

        # List
        list_resp = await client.get("/api/calibrations")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["count"] >= 1

        # Verify our record appears in the list
        record_ids = [r["id"] for r in list_data["records"]]
        assert cal_id in record_ids

    @pytest.mark.asyncio
    async def test_get_calibration_by_id(self, client: httpx.AsyncClient):
        """GET /api/calibrations/{id} returns a specific record."""
        # Create first
        create_resp = await client.post("/api/calibrations", json={
            "paper_type": "Bergger COT 320",
            "exposure_time": 15.0,
            "metal_ratio": 0.5,
            "contrast_agent": "none",
            "developer": "potassium_oxalate",
            "chemistry_type": "platinum_palladium",
            "densities": SAMPLE_DENSITIES[:11],
        })
        cal_id = create_resp.json()["id"]

        # Get by ID
        get_resp = await client.get(f"/api/calibrations/{cal_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["paper_type"] == "Bergger COT 320"
        assert data["exposure_time"] == 15.0

    @pytest.mark.asyncio
    async def test_get_nonexistent_calibration_returns_404(self, client: httpx.AsyncClient):
        """GET /api/calibrations/{bad_id} returns 404."""
        resp = await client.get("/api/calibrations/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_calibrations_with_filter(self, client: httpx.AsyncClient):
        """GET /api/calibrations?paper_type=X filters results."""
        resp = await client.get("/api/calibrations", params={"paper_type": "Nonexistent Paper"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatisticsEndpoint:
    """Tests for the statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, client: httpx.AsyncClient):
        """GET /api/statistics returns a statistics object."""
        resp = await client.get("/api/statistics")
        assert resp.status_code == 200
        data = resp.json()
        # Statistics structure may vary; just ensure it's a valid dict
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Chat (with mocked LLM)
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Tests for the chat endpoint with mocked LLM."""

    @pytest.mark.asyncio
    async def test_chat_with_mocked_assistant(self, client: httpx.AsyncClient):
        """POST /api/chat returns a response when LLM is mocked."""
        mock_assistant = MagicMock()
        mock_assistant.chat = AsyncMock(return_value="This is a test response.")

        with patch("ptpd_calibration.llm.create_assistant", return_value=mock_assistant):
            resp = await client.post("/api/chat", json={
                "message": "What is platinum/palladium printing?",
                "include_history": False,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "This is a test response."
