"""
Comprehensive unit tests for MCTS API router and agent health module.

Tests:
1. MCTS API router endpoints (search, evaluate, train, export, status, feedback, recommendations)
2. Agent health monitoring system
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Check FastAPI availability
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None

# Always import health module (no FastAPI dependency)
from ptpd_calibration.agents.health import (
    AgentHealthReport,
    DependencyHealth,
    DependencyType,
    HealthChecker,
    HealthCheckResult,
    HealthCheckSettings,
    HealthStatus,
    check_agent_health,
    get_health_checker,
)

# Conditionally import MCTS router
if FASTAPI_AVAILABLE:
    from ptpd_calibration.api.mcts_router import (
        create_mcts_router,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mcts_app():
    """Create minimal FastAPI app with MCTS router."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    app = FastAPI()
    router = create_mcts_router()
    app.include_router(router)
    return app


@pytest.fixture
def mcts_client(mcts_app):
    """Create test client for MCTS router."""
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")

    return TestClient(mcts_app)


@pytest.fixture
def sample_parameters():
    """Sample calibration parameters."""
    return {
        "exposure_time": 180.0,
        "metal_ratio": 0.5,
        "contrast_amount": 5.0,
        "developer_temp": 21.0,
        "coating_thickness": 0.1,
    }


@pytest.fixture
def sample_density_curve():
    """Sample density curve (21 steps)."""
    return [0.1 + i * 0.1 for i in range(21)]


@pytest.fixture
def mock_simulator():
    """Mock ExtendedProcessSimulator."""
    mock = Mock()
    mock_result = Mock()
    mock_result.density_curve = [0.1 + i * 0.1 for i in range(21)]
    mock_result.dmin = 0.1
    mock_result.dmax = 2.1
    mock_result.density_range = 2.0
    mock_result.gamma = 0.85
    mock.simulate.return_value = mock_result
    return mock


@pytest.fixture
def mock_coordinator():
    """Mock CalibrationCoordinatorSubagent."""
    mock = AsyncMock()
    mock_result = Mock()
    mock_result.success = True
    mock_result.error = None
    mock_result.result = {
        "full_parameters": {
            "exposure_time": 180.0,
            "metal_ratio": 0.5,
            "contrast_amount": 5.0,
        },
        "evaluation": {
            "predicted_curve": [0.1 + i * 0.1 for i in range(21)],
            "quality_score": 0.85,
        },
    }
    mock.run.return_value = mock_result
    return mock


@pytest.fixture
def health_settings():
    """Create health check settings with low thresholds for testing."""
    return HealthCheckSettings(
        check_interval_seconds=10.0,
        check_timeout_seconds=5.0,
        memory_warning_mb=100.0,
        memory_critical_mb=200.0,
        queue_warning_depth=50,
        queue_critical_depth=100,
        max_active_workflows=5,
        response_time_warning_ms=500.0,
        response_time_critical_ms=1000.0,
        failure_rate_warning=0.2,
        failure_rate_critical=0.4,
    )


# =============================================================================
# MCTS Router Tests
# =============================================================================


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestMCTSRouter:
    """Test MCTS API router endpoints."""

    # -------------------------------------------------------------------------
    # Status Endpoint
    # -------------------------------------------------------------------------

    def test_get_mcts_status_success(self, mcts_client):
        """Test getting MCTS status returns engine info."""
        response = mcts_client.get("/api/mcts/status")

        assert response.status_code == 200
        data = response.json()
        assert "engine_ready" in data
        assert "torch_available" in data
        assert "parameter_ranges" in data

    @patch("ptpd_calibration.mcts.config.DEFAULT_PARAMETER_RANGES", {})
    def test_mcts_status_with_empty_ranges(self, mcts_client):
        """Test status endpoint with empty parameter ranges (works around model issue)."""
        response = mcts_client.get("/api/mcts/status")

        # With empty ranges, validation should pass
        assert response.status_code == 200
        data = response.json()

        assert "engine_ready" in data
        assert "networks_loaded" in data
        assert "torch_available" in data
        assert "parameter_ranges" in data
        assert isinstance(data["parameter_ranges"], dict)

    # -------------------------------------------------------------------------
    # Search Endpoint
    # -------------------------------------------------------------------------

    @patch("ptpd_calibration.mcts.agents.CalibrationCoordinatorSubagent")
    def test_run_mcts_search_minimal(self, mock_coord_class, mcts_client, mock_coordinator):
        """Test MCTS search with minimal parameters."""
        mock_coord_class.return_value = mock_coordinator

        request_data = {}
        response = mcts_client.post("/api/mcts/search", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "search_id" in data
        assert "best_parameters" in data
        assert "predicted_curve" in data
        assert "quality_score" in data
        assert "alternatives" in data
        assert "search_time_seconds" in data
        assert "num_simulations" in data

        # Validate data types
        assert isinstance(data["search_id"], str)
        assert isinstance(data["best_parameters"], dict)
        assert isinstance(data["predicted_curve"], list)
        assert isinstance(data["quality_score"], int | float)
        assert isinstance(data["alternatives"], list)

    @patch("ptpd_calibration.mcts.agents.CalibrationCoordinatorSubagent")
    def test_run_mcts_search_with_fixed_parameters(
        self, mock_coord_class, mcts_client, mock_coordinator, sample_parameters
    ):
        """Test MCTS search with fixed parameters."""
        mock_coord_class.return_value = mock_coordinator

        request_data = {
            "fixed_parameters": {"exposure_time": 180.0},
            "num_simulations": 100,
        }
        response = mcts_client.post("/api/mcts/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["num_simulations"] == 100

    @patch("ptpd_calibration.mcts.agents.CalibrationCoordinatorSubagent")
    def test_run_mcts_search_with_target_aesthetics(
        self, mock_coord_class, mcts_client, mock_coordinator
    ):
        """Test MCTS search with target aesthetics."""
        mock_coord_class.return_value = mock_coordinator

        request_data = {"target_aesthetics": {"contrast": 0.8, "warmth": 0.6, "tonal_range": 0.9}}
        response = mcts_client.post("/api/mcts/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "best_parameters" in data

    @patch("ptpd_calibration.mcts.agents.CalibrationCoordinatorSubagent")
    def test_run_mcts_search_failure(self, mock_coord_class, mcts_client):
        """Test MCTS search when coordinator fails."""
        mock_coordinator = AsyncMock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Simulation failed"
        mock_coordinator.run.return_value = mock_result
        mock_coord_class.return_value = mock_coordinator

        request_data = {}
        response = mcts_client.post("/api/mcts/search", json=request_data)

        assert response.status_code == 500
        assert "Search failed" in response.json()["detail"]

    def test_run_mcts_search_invalid_simulations(self, mcts_client):
        """Test search with invalid num_simulations."""
        # Too low
        response = mcts_client.post("/api/mcts/search", json={"num_simulations": 10})
        assert response.status_code == 422  # Validation error

        # Too high
        response = mcts_client.post("/api/mcts/search", json={"num_simulations": 20000})
        assert response.status_code == 422

    # -------------------------------------------------------------------------
    # Evaluate Endpoint
    # -------------------------------------------------------------------------

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    @patch("ptpd_calibration.mcts.quality.QualityScorer")
    def test_evaluate_parameters_success(
        self, mock_scorer_class, mock_sim_class, mcts_client, mock_simulator, sample_parameters
    ):
        """Test evaluating parameters successfully."""
        mock_sim_class.return_value = mock_simulator
        mock_scorer = Mock()
        mock_scorer.score.return_value = 0.85
        mock_scorer_class.return_value = mock_scorer

        request_data = {"parameters": sample_parameters}
        response = mcts_client.post("/api/mcts/evaluate", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Validate response structure
        assert "density_curve" in data
        assert "dmin" in data
        assert "dmax" in data
        assert "density_range" in data
        assert "gamma" in data
        assert "quality_score" in data

        # Validate values
        assert isinstance(data["density_curve"], list)
        assert len(data["density_curve"]) > 0
        assert data["dmin"] == 0.1
        assert data["dmax"] == 2.1
        assert data["gamma"] == 0.85
        assert data["quality_score"] == 0.85

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    @patch("ptpd_calibration.mcts.quality.QualityScorer")
    def test_evaluate_parameters_empty_uses_defaults(
        self, mock_scorer_class, mock_sim_class, mcts_client, mock_simulator
    ):
        """Test evaluating with empty parameters uses defaults."""
        mock_sim_class.return_value = mock_simulator
        mock_scorer = Mock()
        mock_scorer.score.return_value = 0.85
        mock_scorer_class.return_value = mock_scorer

        request_data = {"parameters": {}}
        response = mcts_client.post("/api/mcts/evaluate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "density_curve" in data

    def test_evaluate_parameters_simulator_error(self, mcts_client):
        """Test evaluation when simulator raises error."""
        with patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator") as mock_sim_class:
            mock_simulator = Mock()
            mock_simulator.simulate.side_effect = ValueError("Invalid parameters")
            mock_sim_class.return_value = mock_simulator

            request_data = {"parameters": {"invalid_param": 123}}
            response = mcts_client.post("/api/mcts/evaluate", json=request_data)

            # Should get 500 error
            assert response.status_code == 500
            assert "Evaluation failed" in response.json()["detail"]

    # -------------------------------------------------------------------------
    # Train Endpoint
    # -------------------------------------------------------------------------

    def test_start_training_success(self, mcts_client):
        """Test starting training session successfully."""
        request_data = {"num_episodes": 100}
        response = mcts_client.post("/api/mcts/train", json=request_data)

        # Should return immediately with session info
        assert response.status_code in [200, 503]  # 503 if torch not available

        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "status" in data
            assert "message" in data
            assert data["status"] == "starting"

    def test_start_training_with_custom_episodes(self, mcts_client):
        """Test training with custom episode count."""
        request_data = {"num_episodes": 50}
        response = mcts_client.post("/api/mcts/train", json=request_data)

        if response.status_code == 200:
            data = response.json()
            assert "50 episodes" in data["message"]

    def test_start_training_invalid_episodes(self, mcts_client):
        """Test training with invalid episode counts."""
        # Too low
        response = mcts_client.post("/api/mcts/train", json={"num_episodes": 5})
        assert response.status_code == 422

        # Too high
        response = mcts_client.post("/api/mcts/train", json={"num_episodes": 20000})
        assert response.status_code == 422

    def test_get_training_status_not_found(self, mcts_client):
        """Test getting status for non-existent training session."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = mcts_client.get(f"/api/mcts/train/{fake_id}/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_training_status_success(self, mcts_client):
        """Test getting training status for existing session."""
        # First start a training session
        train_response = mcts_client.post("/api/mcts/train", json={"num_episodes": 10})

        if train_response.status_code == 200:
            session_id = train_response.json()["session_id"]

            # Get status
            status_response = mcts_client.get(f"/api/mcts/train/{session_id}/status")
            assert status_response.status_code == 200

            data = status_response.json()
            assert "session_id" in data
            assert "status" in data
            assert "episodes_completed" in data
            assert "num_episodes" in data
            assert data["session_id"] == session_id

    # -------------------------------------------------------------------------
    # Export Endpoint
    # -------------------------------------------------------------------------

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    def test_export_json_format(
        self, mock_sim_class, mcts_client, mock_simulator, sample_parameters
    ):
        """Test exporting result in JSON format."""
        mock_sim_class.return_value = mock_simulator

        response = mcts_client.post(
            "/api/mcts/export", params={"format": "json"}, json=sample_parameters
        )

        assert response.status_code == 200
        data = response.json()

        assert "parameters" in data
        assert "density_curve" in data
        assert "dmin" in data
        assert "dmax" in data
        assert "gamma" in data

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    def test_export_csv_format(
        self, mock_sim_class, mcts_client, mock_simulator, sample_parameters
    ):
        """Test exporting result in CSV format."""
        mock_sim_class.return_value = mock_simulator

        response = mcts_client.post(
            "/api/mcts/export", params={"format": "csv"}, json=sample_parameters
        )

        assert response.status_code == 200
        data = response.json()

        assert "csv" in data
        csv_content = data["csv"]
        assert "input,output" in csv_content
        assert "\n" in csv_content

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    def test_export_recipe_format(
        self, mock_sim_class, mcts_client, mock_simulator, sample_parameters
    ):
        """Test exporting result as recipe."""
        mock_sim_class.return_value = mock_simulator

        response = mcts_client.post(
            "/api/mcts/export", params={"format": "recipe"}, json=sample_parameters
        )

        assert response.status_code == 200
        data = response.json()

        assert "title" in data
        assert "parameters" in data
        assert "predicted_results" in data
        assert "MCTS" in data["title"]

    @patch("ptpd_calibration.mcts.simulator.ExtendedProcessSimulator")
    def test_export_invalid_format(
        self, mock_sim_class, mcts_client, mock_simulator, sample_parameters
    ):
        """Test export with invalid format."""
        mock_sim_class.return_value = mock_simulator

        response = mcts_client.post(
            "/api/mcts/export", params={"format": "invalid"}, json=sample_parameters
        )

        assert response.status_code == 500
        assert "Unknown format" in response.json()["detail"]

    # -------------------------------------------------------------------------
    # Feedback Endpoint
    # -------------------------------------------------------------------------

    def test_submit_feedback_success(self, mcts_client, sample_parameters, sample_density_curve):
        """Test submitting feedback successfully."""
        request_data = {
            "parameters": sample_parameters,
            "measured_curve": sample_density_curve,
            "quality_rating": 0.85,
        }
        response = mcts_client.post("/api/mcts/feedback", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data
        assert "recorded" in data["message"].lower()

    def test_submit_feedback_invalid_quality_rating(
        self, mcts_client, sample_parameters, sample_density_curve
    ):
        """Test feedback with invalid quality rating."""
        # Rating too high
        request_data = {
            "parameters": sample_parameters,
            "measured_curve": sample_density_curve,
            "quality_rating": 1.5,
        }
        response = mcts_client.post("/api/mcts/feedback", json=request_data)
        assert response.status_code == 422

        # Rating negative
        request_data["quality_rating"] = -0.1
        response = mcts_client.post("/api/mcts/feedback", json=request_data)
        assert response.status_code == 422

    def test_submit_feedback_missing_fields(self, mcts_client):
        """Test feedback with missing required fields."""
        # Missing parameters
        response = mcts_client.post(
            "/api/mcts/feedback", json={"measured_curve": [], "quality_rating": 0.5}
        )
        assert response.status_code == 422

        # Missing measured_curve
        response = mcts_client.post(
            "/api/mcts/feedback", json={"parameters": {}, "quality_rating": 0.5}
        )
        assert response.status_code == 422

        # Missing quality_rating
        response = mcts_client.post(
            "/api/mcts/feedback", json={"parameters": {}, "measured_curve": []}
        )
        assert response.status_code == 422

    # -------------------------------------------------------------------------
    # Recommendations Endpoint
    # -------------------------------------------------------------------------

    def test_get_recommendations_default(self, mcts_client):
        """Test getting recommendations with default parameters."""
        response = mcts_client.get("/api/mcts/recommendations")

        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= 5  # Default limit

        if data["recommendations"]:
            rec = data["recommendations"][0]
            assert "parameters" in rec
            assert "predicted_quality" in rec
            assert "rationale" in rec

    def test_get_recommendations_with_paper_type(self, mcts_client):
        """Test recommendations filtered by paper type."""
        response = mcts_client.get("/api/mcts/recommendations?paper_type=Arches+Platine")

        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        if data["recommendations"]:
            # Check that rationale mentions the paper type
            rationale = data["recommendations"][0]["rationale"]
            assert "Arches Platine" in rationale or "general" in rationale

    def test_get_recommendations_with_limit(self, mcts_client):
        """Test recommendations with custom limit."""
        response = mcts_client.get("/api/mcts/recommendations?limit=2")

        assert response.status_code == 200
        data = response.json()

        assert len(data["recommendations"]) <= 2

    def test_get_recommendations_zero_limit(self, mcts_client):
        """Test recommendations with zero limit."""
        response = mcts_client.get("/api/mcts/recommendations?limit=0")

        assert response.status_code == 200
        data = response.json()

        assert len(data["recommendations"]) == 0


# =============================================================================
# Agent Health Tests
# =============================================================================


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestDependencyType:
    """Test DependencyType enum."""

    def test_dependency_types(self):
        """Test dependency type enum values."""
        assert DependencyType.LLM_SERVICE.value == "llm_service"
        assert DependencyType.MESSAGE_BUS.value == "message_bus"
        assert DependencyType.MEMORY_SYSTEM.value == "memory_system"
        assert DependencyType.TOOL_REGISTRY.value == "tool_registry"
        assert DependencyType.SUBAGENT_REGISTRY.value == "subagent_registry"


class TestHealthCheckSettings:
    """Test HealthCheckSettings configuration."""

    def test_default_settings(self):
        """Test default health check settings."""
        settings = HealthCheckSettings()

        assert settings.check_interval_seconds == 30.0
        assert settings.check_timeout_seconds == 10.0
        assert settings.memory_warning_mb == 500.0
        assert settings.memory_critical_mb == 1000.0
        assert settings.queue_warning_depth == 100
        assert settings.queue_critical_depth == 500

    def test_custom_settings(self):
        """Test custom health check settings."""
        settings = HealthCheckSettings(
            check_interval_seconds=60.0,
            memory_warning_mb=200.0,
            queue_warning_depth=50,
        )

        assert settings.check_interval_seconds == 60.0
        assert settings.memory_warning_mb == 200.0
        assert settings.queue_warning_depth == 50

    def test_settings_validation(self):
        """Test settings validation with invalid values."""
        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            HealthCheckSettings(check_interval_seconds=-1.0)

        with pytest.raises(Exception):
            HealthCheckSettings(memory_warning_mb=50.0)  # Below minimum


class TestDependencyHealth:
    """Test DependencyHealth model."""

    def test_create_dependency_health(self):
        """Test creating dependency health instance."""
        dep_health = DependencyHealth(
            name="test_service",
            dependency_type=DependencyType.LLM_SERVICE,
            status=HealthStatus.HEALTHY,
            latency_ms=50.0,
            message="All good",
        )

        assert dep_health.name == "test_service"
        assert dep_health.dependency_type == DependencyType.LLM_SERVICE
        assert dep_health.status == HealthStatus.HEALTHY
        assert dep_health.latency_ms == 50.0
        assert dep_health.message == "All good"

    def test_dependency_health_defaults(self):
        """Test dependency health with default values."""
        dep_health = DependencyHealth(name="test", dependency_type=DependencyType.DATABASE)

        assert dep_health.status == HealthStatus.UNKNOWN
        assert dep_health.latency_ms is None
        assert dep_health.message is None
        assert isinstance(dep_health.last_check, datetime)


class TestAgentHealthReport:
    """Test AgentHealthReport model."""

    def test_create_health_report(self):
        """Test creating health report."""
        report = AgentHealthReport(
            status=HealthStatus.HEALTHY,
            llm_connected=True,
            message_bus_active=True,
            memory_system_active=True,
            memory_usage_mb=100.0,
            active_workflows=2,
        )

        assert report.status == HealthStatus.HEALTHY
        assert report.llm_connected is True
        assert report.message_bus_active is True
        assert report.memory_usage_mb == 100.0

    def test_health_report_defaults(self):
        """Test health report with default values."""
        report = AgentHealthReport()

        assert report.status == HealthStatus.UNKNOWN
        assert report.llm_connected is False
        assert report.memory_usage_mb == 0.0
        assert len(report.dependencies) == 0
        assert len(report.issues) == 0


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_create_check_result(self):
        """Test creating health check result."""
        result = HealthCheckResult(name="test_check", healthy=True, latency_ms=25.0, message="OK")

        assert result.name == "test_check"
        assert result.healthy is True
        assert result.latency_ms == 25.0
        assert result.message == "OK"

    def test_check_result_with_metadata(self):
        """Test check result with metadata."""
        result = HealthCheckResult(
            name="test",
            healthy=True,
            latency_ms=10.0,
            metadata={"version": "1.0", "count": 5},
        )

        assert result.metadata["version"] == "1.0"
        assert result.metadata["count"] == 5


class TestHealthChecker:
    """Test HealthChecker class."""

    def test_health_checker_initialization(self, health_settings):
        """Test health checker initialization."""
        checker = HealthChecker(settings=health_settings)

        assert checker.settings == health_settings
        assert checker._last_report is None
        assert len(checker._check_history) == 0

    def test_health_checker_default_settings(self):
        """Test health checker with default settings."""
        checker = HealthChecker()

        assert checker.settings is not None
        assert checker.settings.check_interval_seconds == 30.0

    # -------------------------------------------------------------------------
    # LLM Connectivity Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_success(self):
        """Test LLM connectivity check when successful."""
        checker = HealthChecker()

        with patch("ptpd_calibration.config.get_settings") as mock_settings:
            mock_llm = Mock()
            mock_llm.get_active_api_key.return_value = "test-key"
            mock_settings.return_value.llm = mock_llm
            mock_settings.return_value.llm.provider.value = "anthropic"

            with patch("ptpd_calibration.llm.client.create_client") as mock_client:
                mock_client.return_value = Mock()

                result = await checker.check_llm_connectivity()

                assert result.name == "llm_service"
                assert result.healthy is True
                assert result.latency_ms > 0
                assert "initialized" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_no_api_key(self):
        """Test LLM connectivity check with no API key."""
        checker = HealthChecker()

        with patch("ptpd_calibration.config.get_settings") as mock_settings:
            mock_llm = Mock()
            mock_llm.get_active_api_key.return_value = None
            mock_settings.return_value.llm = mock_llm

            result = await checker.check_llm_connectivity()

            assert result.name == "llm_service"
            assert result.healthy is False
            assert "No API key" in result.message

    @pytest.mark.asyncio
    async def test_check_llm_connectivity_import_error(self):
        """Test LLM connectivity check with import error."""
        checker = HealthChecker()

        with patch("ptpd_calibration.config.get_settings", side_effect=ImportError("No module")):
            result = await checker.check_llm_connectivity()

            assert result.healthy is False
            assert "not available" in result.message

    # -------------------------------------------------------------------------
    # Message Bus Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_message_bus_active(self, health_settings):
        """Test message bus check when active."""
        checker = HealthChecker(settings=health_settings)

        with patch("ptpd_calibration.agents.communication.get_message_bus") as mock_bus:
            mock_instance = Mock()
            mock_instance.get_queue_size.return_value = 10
            mock_bus.return_value = mock_instance

            result = await checker.check_message_bus()

            assert result.name == "message_bus"
            assert result.healthy is True
            assert result.metadata["queue_depth"] == 10

    @pytest.mark.asyncio
    async def test_check_message_bus_high_queue(self, health_settings):
        """Test message bus check with high queue depth."""
        checker = HealthChecker(settings=health_settings)

        with patch("ptpd_calibration.agents.communication.get_message_bus") as mock_bus:
            mock_instance = Mock()
            mock_instance.get_queue_size.return_value = 75  # Above warning threshold
            mock_bus.return_value = mock_instance

            result = await checker.check_message_bus()

            assert result.healthy is True  # Still healthy but warned
            assert "queue high" in result.message

    @pytest.mark.asyncio
    async def test_check_message_bus_overloaded(self, health_settings):
        """Test message bus check when overloaded."""
        checker = HealthChecker(settings=health_settings)

        with patch("ptpd_calibration.agents.communication.get_message_bus") as mock_bus:
            mock_instance = Mock()
            mock_instance.get_queue_size.return_value = 150  # Above critical
            mock_bus.return_value = mock_instance

            result = await checker.check_message_bus()

            assert result.healthy is False
            assert "overloaded" in result.message

    @pytest.mark.asyncio
    async def test_check_message_bus_not_initialized(self):
        """Test message bus check when not initialized."""
        checker = HealthChecker()

        with patch("ptpd_calibration.agents.communication.get_message_bus", return_value=None):
            result = await checker.check_message_bus()

            assert result.healthy is False
            assert "not initialized" in result.message

    # -------------------------------------------------------------------------
    # Memory System Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_memory_system_normal(self, health_settings):
        """Test memory system check with normal usage."""
        checker = HealthChecker(settings=health_settings)

        import sys

        with patch.dict(sys.modules, {"psutil": Mock()}):
            import psutil

            mock_process = Mock()
            mock_memory = Mock()
            mock_memory.rss = 50 * 1024 * 1024  # 50 MB
            mock_process.memory_info.return_value = mock_memory
            psutil.Process.return_value = mock_process

            result = await checker.check_memory_system()

            assert result.name == "memory_system"
            assert result.healthy is True
            assert result.metadata["memory_mb"] < health_settings.memory_warning_mb

    @pytest.mark.asyncio
    async def test_check_memory_system_high(self, health_settings):
        """Test memory system check with high usage."""
        checker = HealthChecker(settings=health_settings)

        import sys

        with patch.dict(sys.modules, {"psutil": Mock()}):
            import psutil

            mock_process = Mock()
            mock_memory = Mock()
            mock_memory.rss = 150 * 1024 * 1024  # 150 MB (above warning)
            mock_process.memory_info.return_value = mock_memory
            psutil.Process.return_value = mock_process

            result = await checker.check_memory_system()

            assert result.healthy is True
            assert "High memory" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_system_critical(self, health_settings):
        """Test memory system check with critical usage."""
        checker = HealthChecker(settings=health_settings)

        import sys

        with patch.dict(sys.modules, {"psutil": Mock()}):
            import psutil

            mock_process = Mock()
            mock_memory = Mock()
            mock_memory.rss = 250 * 1024 * 1024  # 250 MB (above critical)
            mock_process.memory_info.return_value = mock_memory
            psutil.Process.return_value = mock_process

            result = await checker.check_memory_system()

            assert result.healthy is False
            assert "Critical memory" in result.message

    @pytest.mark.asyncio
    async def test_check_memory_system_psutil_not_installed(self):
        """Test memory check when psutil not installed."""
        checker = HealthChecker()

        import sys

        # Remove psutil from modules if it exists
        original_psutil = sys.modules.pop("psutil", None)
        try:
            # Make import fail
            with patch.dict(sys.modules, {"psutil": None}):
                result = await checker.check_memory_system()

                assert result.healthy is True  # Degrades gracefully
                assert "unavailable" in result.message
        finally:
            if original_psutil is not None:
                sys.modules["psutil"] = original_psutil

    # -------------------------------------------------------------------------
    # Tool Registry Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_tool_registry_success(self):
        """Test tool registry check success."""
        checker = HealthChecker()

        with patch("ptpd_calibration.agents.tools.ToolRegistry") as mock_registry:
            mock_instance = Mock()
            mock_instance.list_tools.return_value = ["tool1", "tool2", "tool3"]
            mock_registry.return_value = mock_instance

            result = await checker.check_tool_registry()

            assert result.name == "tool_registry"
            assert result.healthy is True
            assert result.metadata["tool_count"] == 3
            assert "3 tools" in result.message

    @pytest.mark.asyncio
    async def test_check_tool_registry_empty(self):
        """Test tool registry check with no tools."""
        checker = HealthChecker()

        with patch("ptpd_calibration.agents.tools.ToolRegistry") as mock_registry:
            mock_instance = Mock()
            mock_instance.list_tools.return_value = []
            mock_registry.return_value = mock_instance

            result = await checker.check_tool_registry()

            assert result.healthy is True  # Still healthy, just empty
            assert result.metadata["tool_count"] == 0

    @pytest.mark.asyncio
    async def test_check_tool_registry_error(self):
        """Test tool registry check when error occurs."""
        checker = HealthChecker()

        with patch(
            "ptpd_calibration.agents.tools.ToolRegistry", side_effect=Exception("Registry error")
        ):
            result = await checker.check_tool_registry()

            assert result.healthy is False
            assert "failed" in result.message.lower()

    # -------------------------------------------------------------------------
    # Subagent Registry Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_subagent_registry_success(self):
        """Test subagent registry check success."""
        checker = HealthChecker()

        with patch("ptpd_calibration.agents.subagents.base.get_subagent_registry") as mock_registry:
            mock_instance = Mock()
            mock_instance.list_agent_types.return_value = ["agent1", "agent2"]
            mock_registry.return_value = mock_instance

            result = await checker.check_subagent_registry()

            assert result.name == "subagent_registry"
            assert result.healthy is True
            assert result.metadata["subagent_count"] == 2

    # -------------------------------------------------------------------------
    # Performance Tracking
    # -------------------------------------------------------------------------

    def test_record_request(self):
        """Test recording request performance."""
        checker = HealthChecker()

        checker.record_request(duration_ms=100.0, success=True)
        checker.record_request(duration_ms=200.0, success=True)
        checker.record_request(duration_ms=150.0, success=False)

        assert len(checker._request_times) == 3
        assert len(checker._request_successes) == 3

    def test_record_request_limits_samples(self):
        """Test request recording limits sample size."""
        checker = HealthChecker()
        checker._max_samples = 10

        # Record more than max
        for i in range(15):
            checker.record_request(duration_ms=float(i), success=True)

        assert len(checker._request_times) == 10
        assert len(checker._request_successes) == 10

    def test_calculate_performance_metrics(self):
        """Test calculating performance metrics."""
        checker = HealthChecker()

        checker.record_request(100.0, True)
        checker.record_request(200.0, True)
        checker.record_request(300.0, False)

        avg_time, success_rate = checker._calculate_performance_metrics()

        assert avg_time == 200.0  # (100 + 200 + 300) / 3
        assert success_rate == 2 / 3  # 2 successes out of 3

    def test_calculate_performance_metrics_empty(self):
        """Test performance metrics with no data."""
        checker = HealthChecker()

        avg_time, success_rate = checker._calculate_performance_metrics()

        assert avg_time is None
        assert success_rate is None

    # -------------------------------------------------------------------------
    # Overall Health Status
    # -------------------------------------------------------------------------

    def test_determine_overall_status_healthy(self, health_settings):
        """Test determining overall status when healthy."""
        checker = HealthChecker(settings=health_settings)

        results = [
            HealthCheckResult("test1", healthy=True, latency_ms=10.0),
            HealthCheckResult("test2", healthy=True, latency_ms=15.0),
        ]

        status, issues, warnings = checker._determine_overall_status(
            results, memory_mb=50.0, avg_response_time=100.0, success_rate=0.95
        )

        assert status == HealthStatus.HEALTHY
        assert len(issues) == 0
        assert len(warnings) == 0

    def test_determine_overall_status_degraded(self, health_settings):
        """Test determining degraded status."""
        checker = HealthChecker(settings=health_settings)

        results = [
            HealthCheckResult("test1", healthy=True, latency_ms=10.0),
        ]

        # High memory triggers warning
        status, issues, warnings = checker._determine_overall_status(
            results, memory_mb=150.0, avg_response_time=100.0, success_rate=0.95
        )

        assert status == HealthStatus.DEGRADED
        assert len(issues) == 0
        assert len(warnings) > 0

    def test_determine_overall_status_unhealthy(self, health_settings):
        """Test determining unhealthy status."""
        checker = HealthChecker(settings=health_settings)

        results = [
            HealthCheckResult("test1", healthy=False, latency_ms=10.0, message="Failed"),
        ]

        status, issues, warnings = checker._determine_overall_status(
            results, memory_mb=50.0, avg_response_time=100.0, success_rate=0.95
        )

        assert status == HealthStatus.UNHEALTHY
        assert len(issues) > 0

    def test_determine_overall_status_high_response_time(self, health_settings):
        """Test status with high response time."""
        checker = HealthChecker(settings=health_settings)

        results = [HealthCheckResult("test1", healthy=True, latency_ms=10.0)]

        status, issues, warnings = checker._determine_overall_status(
            results, memory_mb=50.0, avg_response_time=1500.0, success_rate=0.95
        )

        assert status == HealthStatus.UNHEALTHY
        assert any("response time" in issue.lower() for issue in issues)

    def test_determine_overall_status_high_failure_rate(self, health_settings):
        """Test status with high failure rate."""
        checker = HealthChecker(settings=health_settings)

        results = [HealthCheckResult("test1", healthy=True, latency_ms=10.0)]

        status, issues, warnings = checker._determine_overall_status(
            results, memory_mb=50.0, avg_response_time=100.0, success_rate=0.5
        )

        assert status == HealthStatus.UNHEALTHY
        assert any("failure rate" in issue.lower() for issue in issues)

    # -------------------------------------------------------------------------
    # Comprehensive Health Check
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_health_comprehensive(self, health_settings):
        """Test comprehensive health check."""
        checker = HealthChecker(settings=health_settings)

        # Mock all checks to return healthy
        with (
            patch.object(
                checker,
                "check_llm_connectivity",
                return_value=HealthCheckResult("llm_service", True, 10.0),
            ),
            patch.object(
                checker,
                "check_message_bus",
                return_value=HealthCheckResult("message_bus", True, 5.0),
            ),
            patch.object(
                checker,
                "check_memory_system",
                return_value=HealthCheckResult(
                    "memory_system", True, 3.0, metadata={"memory_mb": 50.0}
                ),
            ),
            patch.object(
                checker,
                "check_tool_registry",
                return_value=HealthCheckResult(
                    "tool_registry", True, 2.0, metadata={"tool_count": 5}
                ),
            ),
            patch.object(
                checker,
                "check_subagent_registry",
                return_value=HealthCheckResult(
                    "subagent_registry", True, 2.0, metadata={"subagent_count": 3}
                ),
            ),
        ):
            report = await checker.check_health()

            assert isinstance(report, AgentHealthReport)
            assert report.status == HealthStatus.HEALTHY
            assert len(report.dependencies) == 5
            assert report.memory_usage_mb == 50.0
            assert report.registered_tools == 5
            assert report.registered_subagents == 3

    @pytest.mark.asyncio
    async def test_check_health_stores_history(self):
        """Test that health checks are stored in history."""
        checker = HealthChecker()

        # Perform multiple health checks
        with (
            patch.object(
                checker,
                "check_llm_connectivity",
                return_value=HealthCheckResult("llm_service", True, 10.0),
            ),
            patch.object(
                checker,
                "check_message_bus",
                return_value=HealthCheckResult("message_bus", True, 5.0),
            ),
            patch.object(
                checker,
                "check_memory_system",
                return_value=HealthCheckResult(
                    "memory_system", True, 3.0, metadata={"memory_mb": 50.0}
                ),
            ),
            patch.object(
                checker,
                "check_tool_registry",
                return_value=HealthCheckResult("tool_registry", True, 2.0),
            ),
            patch.object(
                checker,
                "check_subagent_registry",
                return_value=HealthCheckResult("subagent_registry", True, 2.0),
            ),
        ):
            await checker.check_health()
            await checker.check_health()

            history = checker.get_history()
            assert len(history) == 2

    @pytest.mark.asyncio
    async def test_check_health_handles_exceptions(self):
        """Test health check handles exceptions gracefully."""
        checker = HealthChecker()

        # Make one check raise exception
        with (
            patch.object(checker, "check_llm_connectivity", side_effect=Exception("Test error")),
            patch.object(
                checker,
                "check_message_bus",
                return_value=HealthCheckResult("message_bus", True, 5.0),
            ),
            patch.object(
                checker,
                "check_memory_system",
                return_value=HealthCheckResult(
                    "memory_system", True, 3.0, metadata={"memory_mb": 50.0}
                ),
            ),
            patch.object(
                checker,
                "check_tool_registry",
                return_value=HealthCheckResult("tool_registry", True, 2.0),
            ),
            patch.object(
                checker,
                "check_subagent_registry",
                return_value=HealthCheckResult("subagent_registry", True, 2.0),
            ),
        ):
            report = await checker.check_health()

            # Should still produce a report
            assert isinstance(report, AgentHealthReport)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def test_get_last_report(self):
        """Test getting last health report."""
        checker = HealthChecker()

        assert checker.get_last_report() is None

        # After a check
        checker._last_report = AgentHealthReport(status=HealthStatus.HEALTHY)
        assert checker.get_last_report() is not None

    def test_get_history_with_limit(self):
        """Test getting health history with limit."""
        checker = HealthChecker()

        # Add some reports
        for i in range(15):
            checker._check_history.append(AgentHealthReport(status=HealthStatus.HEALTHY))

        history = checker.get_history(limit=5)
        assert len(history) == 5

    def test_is_healthy(self):
        """Test is_healthy convenience method."""
        checker = HealthChecker()

        # No report yet - assume healthy
        assert checker.is_healthy() is True

        # After healthy check
        checker._last_report = AgentHealthReport(status=HealthStatus.HEALTHY)
        assert checker.is_healthy() is True

        # After unhealthy check
        checker._last_report = AgentHealthReport(status=HealthStatus.UNHEALTHY)
        assert checker.is_healthy() is False


class TestHealthCheckerGlobalInstance:
    """Test global health checker instance."""

    def test_get_health_checker_singleton(self):
        """Test get_health_checker returns singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

    @pytest.mark.asyncio
    async def test_check_agent_health_convenience(self):
        """Test check_agent_health convenience function."""
        # Mock all dependencies to avoid import errors
        with (
            patch("ptpd_calibration.config.get_settings", side_effect=ImportError),
            patch("ptpd_calibration.agents.communication.get_message_bus", return_value=None),
            patch("ptpd_calibration.agents.tools.ToolRegistry", side_effect=ImportError),
            patch(
                "ptpd_calibration.agents.subagents.base.get_subagent_registry",
                side_effect=ImportError,
            ),
        ):
            import sys

            # Temporarily remove psutil if it's loaded
            original_psutil = sys.modules.pop("psutil", None)
            try:
                with patch.dict(sys.modules, {"psutil": None}):
                    report = await check_agent_health()
                    assert isinstance(report, AgentHealthReport)
            finally:
                if original_psutil is not None:
                    sys.modules["psutil"] = original_psutil


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_health_check_with_all_failures(self, health_settings):
        """Test health check when all dependencies fail."""
        checker = HealthChecker(settings=health_settings)

        with (
            patch.object(
                checker,
                "check_llm_connectivity",
                return_value=HealthCheckResult("llm_service", False, 10.0),
            ),
            patch.object(
                checker,
                "check_message_bus",
                return_value=HealthCheckResult("message_bus", False, 5.0),
            ),
            patch.object(
                checker,
                "check_memory_system",
                return_value=HealthCheckResult(
                    "memory_system", False, 3.0, metadata={"memory_mb": 500.0}
                ),
            ),
            patch.object(
                checker,
                "check_tool_registry",
                return_value=HealthCheckResult("tool_registry", False, 2.0),
            ),
            patch.object(
                checker,
                "check_subagent_registry",
                return_value=HealthCheckResult("subagent_registry", False, 2.0),
            ),
        ):
            report = await checker.check_health()

            assert report.status == HealthStatus.UNHEALTHY
            assert len(report.issues) > 0

    @pytest.mark.asyncio
    async def test_health_checker_max_history_limit(self):
        """Test health checker respects max history limit."""
        checker = HealthChecker()
        checker._max_history = 5

        # Mock all checks to run successfully
        with (
            patch.object(
                checker,
                "check_llm_connectivity",
                return_value=HealthCheckResult("llm_service", True, 10.0),
            ),
            patch.object(
                checker,
                "check_message_bus",
                return_value=HealthCheckResult("message_bus", True, 5.0),
            ),
            patch.object(
                checker,
                "check_memory_system",
                return_value=HealthCheckResult(
                    "memory_system", True, 3.0, metadata={"memory_mb": 50.0}
                ),
            ),
            patch.object(
                checker,
                "check_tool_registry",
                return_value=HealthCheckResult("tool_registry", True, 2.0),
            ),
            patch.object(
                checker,
                "check_subagent_registry",
                return_value=HealthCheckResult("subagent_registry", True, 2.0),
            ),
        ):
            # Run more checks than max history
            for _ in range(10):
                await checker.check_health()

            assert len(checker._check_history) == 5

    @pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
    @patch("ptpd_calibration.api.mcts_router.TORCH_AVAILABLE", False)
    def test_train_without_torch(self, mcts_client):
        """Test training endpoint when PyTorch not available."""
        response = mcts_client.post("/api/mcts/train", json={"num_episodes": 100})

        assert response.status_code == 503
        assert "PyTorch" in response.json()["detail"]
