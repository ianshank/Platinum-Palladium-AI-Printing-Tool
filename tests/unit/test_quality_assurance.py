"""
Comprehensive tests for quality assurance module.

Tests coverage:
- NegativeDensityValidator
- ChemistryFreshnessTracker
- PaperHumidityChecker
- UVLightMeterIntegration
- QualityReport
- AlertSystem
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from ptpd_calibration.config import QASettings
from ptpd_calibration.qa.quality_assurance import (
    Alert,
    AlertSeverity,
    AlertSystem,
    AlertType,
    ChemistryFreshnessTracker,
    ChemistrySolution,
    DensityAnalysis,
    HumidityReading,
    NegativeDensityValidator,
    PaperHumidityChecker,
    QualityReport,
    ReportFormat,
    SolutionType,
    UVLightMeterIntegration,
    UVReading,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def qa_settings():
    """Default QA settings for testing."""
    return QASettings()


@pytest.fixture
def custom_qa_settings():
    """Custom QA settings with non-default values."""
    return QASettings(
        min_density=0.05,
        max_density=3.0,
        highlight_warning_threshold=0.15,
        shadow_warning_threshold=2.5,
        ideal_humidity_min=45.0,
        ideal_humidity_max=55.0,
        uv_intensity_target=150.0,
        ferric_oxalate_shelf_life=90,
        expiration_warning_days=14,
    )


@pytest.fixture
def sample_image_low_density():
    """Create a test image with low density (bright)."""
    # High pixel values = low density
    img = np.ones((100, 100), dtype=np.uint8) * 200
    return img


@pytest.fixture
def sample_image_high_density():
    """Create a test image with high density (dark)."""
    # Low pixel values = high density
    img = np.ones((100, 100), dtype=np.uint8) * 10
    return img


@pytest.fixture
def sample_image_good_range():
    """Create a test image with good density range."""
    # Create gradient from white to black
    img = np.linspace(255, 30, 10000).reshape(100, 100).astype(np.uint8)
    return img


@pytest.fixture
def sample_image_blocked_highlights():
    """Create image with blocked highlights (very low Dmin)."""
    img = np.ones((100, 100), dtype=np.uint8) * 250
    return img


@pytest.fixture
def sample_image_blocked_shadows():
    """Create image with blocked shadows (very high Dmax)."""
    # Very low pixel values = very high density (> 2.0 threshold)
    img = np.ones((100, 100), dtype=np.uint8) * 2
    return img


@pytest.fixture
def density_validator(qa_settings):
    """NegativeDensityValidator instance."""
    return NegativeDensityValidator(qa_settings)


@pytest.fixture
def chemistry_tracker(qa_settings):
    """ChemistryFreshnessTracker instance."""
    return ChemistryFreshnessTracker(qa_settings)


@pytest.fixture
def humidity_checker(qa_settings):
    """PaperHumidityChecker instance."""
    return PaperHumidityChecker(qa_settings)


@pytest.fixture
def uv_meter(qa_settings):
    """UVLightMeterIntegration instance."""
    return UVLightMeterIntegration(qa_settings)


@pytest.fixture
def quality_report(qa_settings):
    """QualityReport instance."""
    return QualityReport(qa_settings)


@pytest.fixture
def alert_system(qa_settings):
    """AlertSystem instance."""
    return AlertSystem(qa_settings)


# ============================================================================
# NegativeDensityValidator Tests
# ============================================================================


class TestNegativeDensityValidator:
    """Tests for NegativeDensityValidator class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        validator = NegativeDensityValidator()
        assert validator.settings is not None
        assert isinstance(validator.settings, QASettings)

    def test_init_custom_settings(self, custom_qa_settings):
        """Test initialization with custom settings."""
        validator = NegativeDensityValidator(custom_qa_settings)
        assert validator.settings == custom_qa_settings

    @pytest.mark.parametrize(
        "image_fixture,expected_min_range,expected_max_range",
        [
            ("sample_image_low_density", 0.0, 0.2),  # Bright image, low density
            ("sample_image_high_density", 1.2, 2.0),  # Dark image, high density
            ("sample_image_good_range", 0.0, 2.0),  # Good range (wider tolerance)
        ],
    )
    def test_validate_density_range_values(
        self, density_validator, request, image_fixture, expected_min_range, expected_max_range
    ):
        """Test density range validation with different images."""
        image = request.getfixturevalue(image_fixture)
        analysis = density_validator.validate_density_range(image)

        assert isinstance(analysis, DensityAnalysis)
        assert expected_min_range <= analysis.min_density <= expected_max_range
        assert analysis.density_range >= 0

    def test_validate_density_range_with_pil_image(
        self, density_validator, sample_image_good_range
    ):
        """Test validation with PIL Image input."""
        pil_image = Image.fromarray(sample_image_good_range)
        analysis = density_validator.validate_density_range(pil_image)

        assert isinstance(analysis, DensityAnalysis)
        assert analysis.min_density >= 0
        assert analysis.max_density > analysis.min_density

    def test_validate_density_range_with_rgb_image(self, density_validator):
        """Test validation with RGB image (converted to grayscale)."""
        rgb_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        analysis = density_validator.validate_density_range(rgb_image)

        assert isinstance(analysis, DensityAnalysis)
        assert 0.2 <= analysis.mean_density <= 0.4  # ~128 gray should be around 0.3 density

    def test_validate_density_range_blocked_highlights(
        self, density_validator, sample_image_blocked_highlights
    ):
        """Test detection of blocked highlights."""
        analysis = density_validator.validate_density_range(sample_image_blocked_highlights)

        assert analysis.highlight_blocked is True
        assert any("highlight" in w.lower() for w in analysis.warnings)

    def test_validate_density_range_blocked_shadows(
        self, density_validator, sample_image_blocked_shadows
    ):
        """Test detection of blocked shadows."""
        analysis = density_validator.validate_density_range(sample_image_blocked_shadows)

        assert analysis.shadow_blocked is True
        assert any("shadow" in w.lower() for w in analysis.warnings)

    def test_validate_density_range_low_contrast(self, density_validator):
        """Test detection of low contrast images."""
        # Image with very narrow density range
        low_contrast = np.ones((100, 100), dtype=np.uint8) * 128
        analysis = density_validator.validate_density_range(low_contrast)

        assert analysis.density_range < 1.5
        assert any("contrast" in w.lower() or "range" in w.lower() for w in analysis.warnings)

    def test_validate_density_range_suggestions(self, density_validator, sample_image_good_range):
        """Test that suggestions are generated."""
        analysis = density_validator.validate_density_range(sample_image_good_range)

        assert isinstance(analysis.suggestions, list)
        # Suggestions might be empty for a good image or contain helpful advice

    def test_validate_density_range_zone_distribution(
        self, density_validator, sample_image_good_range
    ):
        """Test zone distribution calculation."""
        analysis = density_validator.validate_density_range(sample_image_good_range)

        assert isinstance(analysis.zone_distribution, dict)
        assert len(analysis.zone_distribution) == 11  # Zones 0-10
        # All zones should sum to approximately 1.0
        total = sum(analysis.zone_distribution.values())
        assert 0.99 <= total <= 1.01

    def test_validate_density_range_histogram(self, density_validator, sample_image_good_range):
        """Test histogram generation."""
        analysis = density_validator.validate_density_range(sample_image_good_range)

        assert isinstance(analysis.histogram, np.ndarray)
        assert len(analysis.histogram) == 100  # Default bins

    def test_check_highlight_detail_good(self, density_validator, sample_image_good_range):
        """Test highlight detail check with good image."""
        has_detail, message = density_validator.check_highlight_detail(sample_image_good_range)

        # May or may not have detail depending on image content
        assert isinstance(has_detail, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_check_highlight_detail_blocked(
        self, density_validator, sample_image_blocked_highlights
    ):
        """Test highlight detail check with blocked highlights."""
        has_detail, message = density_validator.check_highlight_detail(
            sample_image_blocked_highlights
        )

        assert has_detail is False
        assert "blocked" in message.lower()

    def test_check_shadow_detail_good(self, density_validator, sample_image_good_range):
        """Test shadow detail check with good image."""
        has_detail, message = density_validator.check_shadow_detail(sample_image_good_range)

        assert isinstance(has_detail, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_check_shadow_detail_blocked(self, density_validator, sample_image_blocked_shadows):
        """Test shadow detail check with blocked shadows."""
        has_detail, message = density_validator.check_shadow_detail(sample_image_blocked_shadows)

        assert has_detail is False
        assert "blocked" in message.lower()

    def test_get_density_histogram_default_bins(self, density_validator, sample_image_good_range):
        """Test histogram generation with default bins."""
        hist, edges = density_validator.get_density_histogram(sample_image_good_range)

        assert isinstance(hist, np.ndarray)
        assert isinstance(edges, np.ndarray)
        assert len(hist) == 256  # Default bins
        assert len(edges) == 257  # Bin edges is bins + 1

    @pytest.mark.parametrize("bins", [64, 128, 512])
    def test_get_density_histogram_custom_bins(
        self, density_validator, sample_image_good_range, bins
    ):
        """Test histogram generation with custom bin counts."""
        hist, edges = density_validator.get_density_histogram(sample_image_good_range, bins=bins)

        assert len(hist) == bins
        assert len(edges) == bins + 1

    @pytest.mark.parametrize(
        "min_d,max_d,range_d,h_blocked,s_blocked,expected_count",
        [
            (0.1, 2.0, 1.9, False, False, 1),  # Good density, should get 1+ suggestion
            (0.3, 1.5, 1.2, False, False, 1),  # Low range
            (0.02, 2.5, 2.48, True, False, 1),  # Highlight blocked
            (0.15, 2.8, 2.65, False, True, 1),  # Shadow blocked
            (0.05, 3.5, 3.45, True, True, 2),  # Both blocked
        ],
    )
    def test_suggest_corrections(
        self, density_validator, min_d, max_d, range_d, h_blocked, s_blocked, expected_count
    ):
        """Test correction suggestions for various scenarios."""
        suggestions = density_validator.suggest_corrections(
            min_d, max_d, range_d, h_blocked, s_blocked
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) >= expected_count

    def test_density_analysis_to_dict(self, density_validator, sample_image_good_range):
        """Test DensityAnalysis to_dict conversion."""
        analysis = density_validator.validate_density_range(sample_image_good_range)
        data = analysis.to_dict()

        assert isinstance(data, dict)
        assert "min_density" in data
        assert "max_density" in data
        assert "density_range" in data
        assert "highlight_blocked" in data
        assert "shadow_blocked" in data
        assert "zone_distribution" in data
        assert "warnings" in data
        assert "suggestions" in data


# ============================================================================
# ChemistryFreshnessTracker Tests
# ============================================================================


class TestChemistryFreshnessTracker:
    """Tests for ChemistryFreshnessTracker class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        tracker = ChemistryFreshnessTracker()
        assert tracker.settings is not None
        assert len(tracker.solutions) == 0

    def test_init_custom_settings(self, custom_qa_settings):
        """Test initialization with custom settings."""
        tracker = ChemistryFreshnessTracker(custom_qa_settings)
        assert tracker.settings == custom_qa_settings

    @pytest.mark.parametrize(
        "solution_type,expected_shelf_life",
        [
            (SolutionType.FERRIC_OXALATE_1, 180),
            (SolutionType.PALLADIUM, 365),
            (SolutionType.PLATINUM, 365),
            (SolutionType.DEVELOPER, 90),
        ],
    )
    def test_register_solution(self, chemistry_tracker, solution_type, expected_shelf_life):
        """Test solution registration with different types."""
        date_mixed = datetime.now()
        volume = 100.0

        solution_id = chemistry_tracker.register_solution(
            solution_type=solution_type,
            date_mixed=date_mixed,
            volume_ml=volume,
            notes="Test solution",
        )

        assert solution_id in chemistry_tracker.solutions
        solution = chemistry_tracker.solutions[solution_id]
        assert solution.solution_type == solution_type
        assert solution.initial_volume_ml == volume
        assert solution.current_volume_ml == volume
        assert solution.shelf_life_days == expected_shelf_life

    def test_register_solution_custom_id(self, chemistry_tracker):
        """Test solution registration with custom ID."""
        custom_id = "test_solution_123"

        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
            solution_id=custom_id,
        )

        assert solution_id == custom_id
        assert custom_id in chemistry_tracker.solutions

    def test_check_freshness_fresh_solution(self, chemistry_tracker):
        """Test freshness check for fresh solution."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        is_fresh, message = chemistry_tracker.check_freshness(solution_id)

        assert is_fresh is True
        assert "fresh" in message.lower() or "remaining" in message.lower()

    def test_check_freshness_expired_solution(self, chemistry_tracker):
        """Test freshness check for expired solution."""
        # Solution mixed 400 days ago (expired for palladium with 365 day shelf life)
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now() - timedelta(days=400),
            volume_ml=100.0,
        )

        is_fresh, message = chemistry_tracker.check_freshness(solution_id)

        assert is_fresh is False
        assert "expired" in message.lower()

    def test_check_freshness_expiring_soon(self, chemistry_tracker):
        """Test freshness check for solution expiring soon."""
        # Solution mixed such that it expires in 5 days (critical threshold is 7)
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now() - timedelta(days=360),  # 365 - 360 = 5 days left
            volume_ml=100.0,
        )

        is_fresh, message = chemistry_tracker.check_freshness(solution_id)

        assert is_fresh is True
        assert "critical" in message.lower() or "expires" in message.lower()

    def test_check_freshness_not_found(self, chemistry_tracker):
        """Test freshness check for non-existent solution."""
        is_fresh, message = chemistry_tracker.check_freshness("nonexistent_id")

        assert is_fresh is False
        assert "not found" in message.lower()

    def test_get_expiration_date(self, chemistry_tracker):
        """Test getting expiration date."""
        date_mixed = datetime.now()
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=date_mixed,
            volume_ml=100.0,
        )

        exp_date = chemistry_tracker.get_expiration_date(solution_id)

        assert exp_date is not None
        expected = date_mixed + timedelta(days=365)
        assert abs((exp_date - expected).total_seconds()) < 1

    def test_get_expiration_date_not_found(self, chemistry_tracker):
        """Test getting expiration date for non-existent solution."""
        exp_date = chemistry_tracker.get_expiration_date("nonexistent_id")
        assert exp_date is None

    def test_log_usage_success(self, chemistry_tracker):
        """Test successful usage logging."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        success = chemistry_tracker.log_usage(solution_id, 10.0)

        assert success is True
        assert chemistry_tracker.solutions[solution_id].current_volume_ml == 90.0
        assert len(chemistry_tracker.solutions[solution_id].usage_log) == 1

    def test_log_usage_insufficient_volume(self, chemistry_tracker):
        """Test usage logging with insufficient volume."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=50.0,
        )

        success = chemistry_tracker.log_usage(solution_id, 100.0)  # Try to use more than available

        assert success is False
        assert chemistry_tracker.solutions[solution_id].current_volume_ml == 50.0  # Unchanged

    def test_log_usage_not_found(self, chemistry_tracker):
        """Test usage logging for non-existent solution."""
        success = chemistry_tracker.log_usage("nonexistent_id", 10.0)
        assert success is False

    def test_log_usage_custom_timestamp(self, chemistry_tracker):
        """Test usage logging with custom timestamp."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        custom_time = datetime.now() - timedelta(hours=5)
        chemistry_tracker.log_usage(solution_id, 10.0, timestamp=custom_time)

        log_entry = chemistry_tracker.solutions[solution_id].usage_log[0]
        assert abs((log_entry[0] - custom_time).total_seconds()) < 1

    def test_get_alerts_expired(self, chemistry_tracker):
        """Test alerts for expired solution."""
        chemistry_tracker.register_solution(
            solution_type=SolutionType.DEVELOPER,
            date_mixed=datetime.now() - timedelta(days=100),  # Expired (90 day shelf life)
            volume_ml=100.0,
        )

        alerts = chemistry_tracker.get_alerts()

        assert len(alerts) > 0
        assert any(a["type"] == "expired" for a in alerts)
        assert any(a["severity"] == "critical" for a in alerts)

    def test_get_alerts_low_volume(self, chemistry_tracker):
        """Test alerts for low volume."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )
        # Use 85ml, leaving 15% (below 20% warning threshold)
        chemistry_tracker.log_usage(solution_id, 85.0)

        alerts = chemistry_tracker.get_alerts()

        assert len(alerts) > 0
        assert any(a["type"] == "low_volume" for a in alerts)

    def test_get_alerts_critical_volume(self, chemistry_tracker):
        """Test alerts for critical volume."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )
        # Use 95ml, leaving 5% (below 10% critical threshold)
        chemistry_tracker.log_usage(solution_id, 95.0)

        alerts = chemistry_tracker.get_alerts()

        critical_alerts = [
            a for a in alerts if a["severity"] == "error" and a["type"] == "low_volume"
        ]
        assert len(critical_alerts) > 0

    def test_recommend_replenishment_sufficient_data(self, chemistry_tracker):
        """Test replenishment recommendation with usage data."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        # Log multiple usages
        chemistry_tracker.log_usage(solution_id, 10.0, timestamp=datetime.now() - timedelta(days=5))
        chemistry_tracker.log_usage(solution_id, 10.0, timestamp=datetime.now() - timedelta(days=3))
        chemistry_tracker.log_usage(solution_id, 10.0, timestamp=datetime.now())

        recommendation = chemistry_tracker.recommend_replenishment(solution_id)

        assert recommendation is not None
        assert "days" in recommendation.lower()

    def test_recommend_replenishment_insufficient_data(self, chemistry_tracker):
        """Test replenishment recommendation with insufficient data."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        # Log only one usage
        chemistry_tracker.log_usage(solution_id, 10.0)

        recommendation = chemistry_tracker.recommend_replenishment(solution_id)

        assert "insufficient" in recommendation.lower()

    def test_recommend_replenishment_not_found(self, chemistry_tracker):
        """Test replenishment recommendation for non-existent solution."""
        recommendation = chemistry_tracker.recommend_replenishment("nonexistent_id")
        assert recommendation is None

    def test_get_solution_info(self, chemistry_tracker):
        """Test getting solution information."""
        solution_id = chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        info = chemistry_tracker.get_solution_info(solution_id)

        assert info is not None
        assert isinstance(info, dict)
        assert "solution_id" in info
        assert "solution_type" in info
        assert "current_volume_ml" in info

    def test_list_all_solutions(self, chemistry_tracker):
        """Test listing all solutions."""
        # Register multiple solutions
        chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM, date_mixed=datetime.now(), volume_ml=100.0
        )
        chemistry_tracker.register_solution(
            solution_type=SolutionType.PLATINUM, date_mixed=datetime.now(), volume_ml=50.0
        )

        solutions = chemistry_tracker.list_all_solutions()

        assert len(solutions) == 2
        assert all(isinstance(s, dict) for s in solutions)

    def test_chemistry_solution_properties(self):
        """Test ChemistrySolution property calculations."""
        solution = ChemistrySolution(
            solution_id="test",
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now() - timedelta(days=100),
            initial_volume_ml=100.0,
            current_volume_ml=75.0,
            shelf_life_days=365,
        )

        # Allow for timing differences (264-265 days)
        assert 264 <= solution.days_until_expiration <= 265
        assert solution.is_expired is False
        assert solution.volume_percent_remaining == 75.0


# ============================================================================
# PaperHumidityChecker Tests
# ============================================================================


class TestPaperHumidityChecker:
    """Tests for PaperHumidityChecker class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        checker = PaperHumidityChecker()
        assert checker.settings is not None
        assert len(checker.readings) == 0

    def test_measure_paper_humidity(self, humidity_checker):
        """Test measuring paper humidity."""
        reading = humidity_checker.measure_paper_humidity(
            humidity_percent=50.0,
            temperature_celsius=20.0,
            paper_type="Arches Platine",
            notes="Test reading",
        )

        assert isinstance(reading, HumidityReading)
        assert reading.humidity_percent == 50.0
        assert reading.temperature_celsius == 20.0
        assert len(humidity_checker.readings) == 1

    def test_measure_paper_humidity_custom_timestamp(self, humidity_checker):
        """Test measuring humidity with custom timestamp."""
        custom_time = datetime.now() - timedelta(hours=2)

        reading = humidity_checker.measure_paper_humidity(
            humidity_percent=50.0, timestamp=custom_time
        )

        assert abs((reading.timestamp - custom_time).total_seconds()) < 1

    @pytest.mark.parametrize(
        "humidity,expected_ready",
        [
            (50.0, True),  # Ideal
            (45.0, True),  # Within range
            (55.0, True),  # Within range
            (30.0, False),  # Too dry
            (70.0, False),  # Too humid
        ],
    )
    def test_is_paper_ready(self, humidity_checker, humidity, expected_ready):
        """Test paper readiness check with various humidity levels."""
        humidity_checker.measure_paper_humidity(humidity_percent=humidity)

        is_ready, message = humidity_checker.is_paper_ready()

        assert is_ready == expected_ready
        assert isinstance(message, str)
        assert len(message) > 0

    def test_is_paper_ready_no_readings(self, humidity_checker):
        """Test paper readiness check with no readings."""
        is_ready, message = humidity_checker.is_paper_ready()

        assert is_ready is False
        assert "no" in message.lower() and "reading" in message.lower()

    def test_is_paper_ready_custom_range(self, humidity_checker):
        """Test paper readiness check with custom target range."""
        humidity_checker.measure_paper_humidity(humidity_percent=48.0)

        is_ready, message = humidity_checker.is_paper_ready(
            target_humidity_min=45.0, target_humidity_max=55.0
        )

        assert is_ready is True

    @pytest.mark.parametrize(
        "current,target,expected_action",
        [
            (60.0, 50.0, "drying"),
            (40.0, 50.0, "humidifying"),
        ],
    )
    def test_estimate_drying_time(self, humidity_checker, current, target, expected_action):
        """Test drying time estimation."""
        hours, message = humidity_checker.estimate_drying_time(
            current_humidity=current, target_humidity=target
        )

        assert hours > 0
        assert expected_action in message.lower()

    def test_estimate_drying_time_with_temperature(self, humidity_checker):
        """Test drying time estimation with temperature adjustment."""
        hours_20c, _ = humidity_checker.estimate_drying_time(
            current_humidity=60.0, target_humidity=50.0, room_temperature=20.0
        )

        hours_25c, _ = humidity_checker.estimate_drying_time(
            current_humidity=60.0, target_humidity=50.0, room_temperature=25.0
        )

        # Higher temperature should result in shorter drying time
        assert hours_25c < hours_20c

    def test_log_ambient_conditions(self, humidity_checker):
        """Test logging ambient conditions."""
        humidity_checker.log_ambient_conditions(humidity_percent=55.0, temperature_celsius=22.0)

        assert len(humidity_checker.ambient_conditions) == 1
        timestamp, humidity, temp = humidity_checker.ambient_conditions[0]
        assert humidity == 55.0
        assert temp == 22.0

    @pytest.mark.parametrize(
        "humidity,expected_pattern",
        [
            (30.0, "humidif"),  # Too dry, need to humidify
            (70.0, "dehumidif"),  # Too humid, need to dehumidify
            (50.0, "acceptable"),  # Just right
        ],
    )
    def test_recommend_humidity_adjustment(self, humidity_checker, humidity, expected_pattern):
        """Test humidity adjustment recommendations."""
        humidity_checker.measure_paper_humidity(humidity_percent=humidity)

        recommendation = humidity_checker.recommend_humidity_adjustment()

        assert recommendation is not None
        assert expected_pattern in recommendation.lower()

    def test_recommend_humidity_adjustment_no_readings(self, humidity_checker):
        """Test humidity recommendation with no readings."""
        recommendation = humidity_checker.recommend_humidity_adjustment()

        assert "no" in recommendation.lower() and "reading" in recommendation.lower()

    def test_get_latest_reading(self, humidity_checker):
        """Test getting latest reading."""
        humidity_checker.measure_paper_humidity(humidity_percent=45.0)
        humidity_checker.measure_paper_humidity(humidity_percent=50.0)

        latest = humidity_checker.get_latest_reading()

        assert latest is not None
        assert latest.humidity_percent == 50.0

    def test_get_latest_reading_empty(self, humidity_checker):
        """Test getting latest reading with no data."""
        latest = humidity_checker.get_latest_reading()
        assert latest is None

    def test_get_readings_history(self, humidity_checker):
        """Test getting readings history."""
        # Add reading from 2 hours ago
        humidity_checker.measure_paper_humidity(
            humidity_percent=45.0, timestamp=datetime.now() - timedelta(hours=2)
        )
        # Add reading from 30 hours ago (should be excluded from 24h window)
        humidity_checker.measure_paper_humidity(
            humidity_percent=50.0, timestamp=datetime.now() - timedelta(hours=30)
        )
        # Add current reading
        humidity_checker.measure_paper_humidity(humidity_percent=55.0)

        recent = humidity_checker.get_readings_history(hours=24)

        assert len(recent) == 2  # Only 2 readings in last 24 hours

    def test_humidity_reading_to_dict(self):
        """Test HumidityReading to_dict conversion."""
        reading = HumidityReading(
            timestamp=datetime.now(),
            humidity_percent=50.5,
            temperature_celsius=20.3,
            paper_type="Test Paper",
            notes="Test",
        )

        data = reading.to_dict()

        assert isinstance(data, dict)
        assert data["humidity_percent"] == 50.5
        assert data["temperature_celsius"] == 20.3
        assert data["paper_type"] == "Test Paper"


# ============================================================================
# UVLightMeterIntegration Tests
# ============================================================================


class TestUVLightMeterIntegration:
    """Tests for UVLightMeterIntegration class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        meter = UVLightMeterIntegration()
        assert meter.settings is not None
        assert len(meter.readings) == 0
        assert meter.calibration_factor == 1.0

    def test_calibrate_meter_no_reference(self, uv_meter):
        """Test meter calibration without reference."""
        message = uv_meter.calibrate_meter()

        assert uv_meter.calibration_date is not None
        assert "calibrated" in message.lower()

    def test_calibrate_meter_with_reference(self, uv_meter):
        """Test meter calibration with reference intensity."""
        # Add a reading first
        uv_meter.read_intensity(80.0)

        # Calibrate with reference of 100
        message = uv_meter.calibrate_meter(reference_intensity=100.0)

        assert uv_meter.calibration_factor == 100.0 / 80.0
        assert "calibrated" in message.lower()

    def test_read_intensity_basic(self, uv_meter):
        """Test reading UV intensity."""
        reading = uv_meter.read_intensity(
            intensity=95.0, wavelength=365.0, bulb_hours=100.0, notes="Test"
        )

        assert isinstance(reading, UVReading)
        assert reading.intensity == 95.0  # No calibration applied yet
        assert reading.wavelength == 365.0
        assert reading.bulb_hours == 100.0
        assert len(uv_meter.readings) == 1

    def test_read_intensity_with_calibration(self, uv_meter):
        """Test that calibration factor is applied."""
        uv_meter.calibration_factor = 1.25

        reading = uv_meter.read_intensity(intensity=80.0)

        assert reading.intensity == 100.0  # 80 * 1.25

    def test_read_intensity_custom_timestamp(self, uv_meter):
        """Test reading with custom timestamp."""
        custom_time = datetime.now() - timedelta(hours=3)

        reading = uv_meter.read_intensity(intensity=100.0, timestamp=custom_time)

        assert abs((reading.timestamp - custom_time).total_seconds()) < 1

    def test_log_reading_alias(self, uv_meter):
        """Test log_reading as alias for read_intensity."""
        reading = uv_meter.log_reading(intensity=100.0, wavelength=365.0)

        assert isinstance(reading, UVReading)
        assert len(uv_meter.readings) == 1

    @pytest.mark.parametrize(
        "target,actual,expected_factor_range",
        [
            (100.0, 100.0, (0.95, 1.05)),  # Perfect match
            (100.0, 80.0, (1.2, 1.3)),  # Need more exposure
            (100.0, 120.0, (0.8, 0.9)),  # Need less exposure
        ],
    )
    def test_calculate_exposure_adjustment(self, uv_meter, target, actual, expected_factor_range):
        """Test exposure adjustment calculation."""
        adjustment, message = uv_meter.calculate_exposure_adjustment(
            target_intensity=target, actual_intensity=actual
        )

        assert expected_factor_range[0] <= adjustment <= expected_factor_range[1]
        assert isinstance(message, str)

    def test_calculate_exposure_adjustment_from_reading(self, uv_meter):
        """Test exposure adjustment using latest reading."""
        uv_meter.read_intensity(intensity=70.0)  # Lower intensity to trigger increase message

        adjustment, message = uv_meter.calculate_exposure_adjustment(target_intensity=100.0)

        assert adjustment > 1.0  # Should need more exposure
        assert "increase" in message.lower()

    def test_calculate_exposure_adjustment_no_readings(self, uv_meter):
        """Test exposure adjustment with no readings."""
        adjustment, message = uv_meter.calculate_exposure_adjustment()

        assert adjustment == 1.0
        assert "no" in message.lower()

    def test_check_bulb_degradation_insufficient_data(self, uv_meter):
        """Test bulb degradation check with insufficient data."""
        uv_meter.read_intensity(100.0)

        needs_replacement, message = uv_meter.check_bulb_degradation()

        assert needs_replacement is False
        assert "insufficient" in message.lower()

    def test_check_bulb_degradation_significant(self, uv_meter):
        """Test bulb degradation detection with significant decline."""
        # Add readings showing degradation
        base_time = datetime.now() - timedelta(days=7)

        # Earlier readings at 100
        for i in range(5):
            uv_meter.read_intensity(intensity=100.0, timestamp=base_time + timedelta(days=i))

        # Recent readings at 80 (20% decline)
        for i in range(5):
            uv_meter.read_intensity(intensity=80.0, timestamp=base_time + timedelta(days=5 + i))

        needs_replacement, message = uv_meter.check_bulb_degradation(readings_window_hours=168)

        assert needs_replacement is True
        assert "degradation" in message.lower()

    def test_check_bulb_degradation_normal(self, uv_meter):
        """Test bulb degradation check with stable readings."""
        # Add consistent readings
        for i in range(10):
            uv_meter.read_intensity(intensity=100.0, timestamp=datetime.now() - timedelta(hours=i))

        needs_replacement, message = uv_meter.check_bulb_degradation()

        assert needs_replacement is False
        assert "normal" in message.lower()

    def test_recommend_bulb_replacement_by_hours(self, uv_meter):
        """Test bulb replacement recommendation based on hours."""
        uv_meter.read_intensity(intensity=100.0, bulb_hours=1100.0)  # Over 1000 hour limit

        recommendation = uv_meter.recommend_bulb_replacement()

        assert "replace" in recommendation.lower()
        assert "1100" in recommendation

    def test_recommend_bulb_replacement_hours_ok(self, uv_meter):
        """Test bulb replacement recommendation with acceptable hours."""
        uv_meter.read_intensity(intensity=100.0, bulb_hours=500.0)

        recommendation = uv_meter.recommend_bulb_replacement()

        assert "ok" in recommendation.lower() or "remaining" in recommendation.lower()

    def test_recommend_bulb_replacement_no_data(self, uv_meter):
        """Test bulb replacement recommendation with no data."""
        recommendation = uv_meter.recommend_bulb_replacement()

        assert "not currently needed" in recommendation.lower()

    def test_get_latest_reading(self, uv_meter):
        """Test getting latest reading."""
        uv_meter.read_intensity(intensity=90.0)
        uv_meter.read_intensity(intensity=95.0)

        latest = uv_meter.get_latest_reading()

        assert latest is not None
        assert latest.intensity == 95.0

    def test_get_latest_reading_empty(self, uv_meter):
        """Test getting latest reading with no data."""
        latest = uv_meter.get_latest_reading()
        assert latest is None

    def test_get_readings_history(self, uv_meter):
        """Test getting readings history."""
        # Add reading from 2 hours ago
        uv_meter.read_intensity(intensity=90.0, timestamp=datetime.now() - timedelta(hours=2))
        # Add reading from 30 hours ago
        uv_meter.read_intensity(intensity=95.0, timestamp=datetime.now() - timedelta(hours=30))
        # Add current reading
        uv_meter.read_intensity(intensity=100.0)

        recent = uv_meter.get_readings_history(hours=24)

        assert len(recent) == 2  # Only 2 in last 24 hours

    def test_uv_reading_to_dict(self):
        """Test UVReading to_dict conversion."""
        reading = UVReading(
            timestamp=datetime.now(),
            intensity=100.5,
            wavelength=365.0,
            bulb_hours=500.5,
            notes="Test",
        )

        data = reading.to_dict()

        assert isinstance(data, dict)
        assert data["intensity"] == 100.5
        assert data["wavelength"] == 365.0
        assert data["bulb_hours"] == 500.5


# ============================================================================
# QualityReport Tests
# ============================================================================


class TestQualityReport:
    """Tests for QualityReport class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        report = QualityReport()
        assert report.settings is not None

    def test_generate_pre_print_checklist_empty(self, quality_report):
        """Test pre-print checklist with no inputs."""
        checklist = quality_report.generate_pre_print_checklist()

        assert isinstance(checklist, dict)
        assert "timestamp" in checklist
        assert "checks" in checklist
        assert "ready_to_print" in checklist
        assert checklist["ready_to_print"] is True  # No checks, so default ready

    def test_generate_pre_print_checklist_with_image(self, quality_report, sample_image_good_range):
        """Test pre-print checklist with image validation."""
        checklist = quality_report.generate_pre_print_checklist(image=sample_image_good_range)

        assert "negative_density" in checklist["checks"]
        assert "density_range" in checklist["checks"]["negative_density"]
        assert "status" in checklist["checks"]["negative_density"]

    def test_generate_pre_print_checklist_with_blocked_image(
        self, quality_report, sample_image_blocked_highlights
    ):
        """Test pre-print checklist with blocked highlights."""
        checklist = quality_report.generate_pre_print_checklist(
            image=sample_image_blocked_highlights
        )

        assert checklist["ready_to_print"] is False
        assert len(checklist["errors"]) > 0

    def test_generate_pre_print_checklist_with_chemistry(self, quality_report, chemistry_tracker):
        """Test pre-print checklist with chemistry tracker."""
        # Add expired solution
        chemistry_tracker.register_solution(
            solution_type=SolutionType.DEVELOPER,
            date_mixed=datetime.now() - timedelta(days=100),
            volume_ml=100.0,
        )

        checklist = quality_report.generate_pre_print_checklist(chemistry_tracker=chemistry_tracker)

        assert "chemistry" in checklist["checks"]
        assert checklist["ready_to_print"] is False  # Expired solution

    def test_generate_pre_print_checklist_with_humidity(self, quality_report, humidity_checker):
        """Test pre-print checklist with humidity checker."""
        humidity_checker.measure_paper_humidity(humidity_percent=50.0)

        checklist = quality_report.generate_pre_print_checklist(humidity_checker=humidity_checker)

        assert "paper_humidity" in checklist["checks"]
        assert checklist["checks"]["paper_humidity"]["status"] == "pass"

    def test_generate_pre_print_checklist_with_bad_humidity(self, quality_report, humidity_checker):
        """Test pre-print checklist with bad humidity."""
        humidity_checker.measure_paper_humidity(humidity_percent=80.0)  # Too humid

        checklist = quality_report.generate_pre_print_checklist(humidity_checker=humidity_checker)

        assert checklist["ready_to_print"] is False

    def test_generate_pre_print_checklist_with_uv(self, quality_report, uv_meter):
        """Test pre-print checklist with UV meter."""
        uv_meter.read_intensity(intensity=100.0)

        checklist = quality_report.generate_pre_print_checklist(uv_meter=uv_meter)

        assert "uv_light" in checklist["checks"]
        assert "exposure_adjustment" in checklist["checks"]["uv_light"]

    def test_generate_pre_print_checklist_complete(
        self,
        quality_report,
        sample_image_good_range,
        chemistry_tracker,
        humidity_checker,
        uv_meter,
    ):
        """Test pre-print checklist with all components."""
        # Setup good conditions
        chemistry_tracker.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )
        humidity_checker.measure_paper_humidity(humidity_percent=50.0)
        uv_meter.read_intensity(intensity=100.0)

        checklist = quality_report.generate_pre_print_checklist(
            image=sample_image_good_range,
            chemistry_tracker=chemistry_tracker,
            humidity_checker=humidity_checker,
            uv_meter=uv_meter,
        )

        assert len(checklist["checks"]) == 4
        assert "negative_density" in checklist["checks"]
        assert "chemistry" in checklist["checks"]
        assert "paper_humidity" in checklist["checks"]
        assert "uv_light" in checklist["checks"]

    def test_generate_post_print_analysis(self, quality_report, sample_image_good_range):
        """Test post-print analysis."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)

        assert isinstance(analysis, dict)
        assert "timestamp" in analysis
        assert "density_analysis" in analysis
        assert "quality_assessment" in analysis
        assert "quality_score" in analysis
        assert "grade" in analysis
        assert "recommendations" in analysis

    def test_generate_post_print_analysis_with_expected_range(
        self, quality_report, sample_image_good_range
    ):
        """Test post-print analysis with expected density range."""
        analysis = quality_report.generate_post_print_analysis(
            sample_image_good_range, expected_density_range=(0.1, 2.0)
        )

        # Should have recommendations if density deviates
        assert isinstance(analysis["recommendations"], list)

    @pytest.mark.parametrize(
        "score,expected_grade",
        [
            (95.0, "A"),
            (85.0, "B"),
            (75.0, "C"),
            (65.0, "D"),
            (55.0, "F"),
        ],
    )
    def test_get_grade(self, quality_report, score, expected_grade):
        """Test grade calculation."""
        grade = quality_report._get_grade(score)
        assert grade == expected_grade

    def test_export_report_json(self, quality_report, sample_image_good_range, tmp_path):
        """Test report export as JSON."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)
        output_path = tmp_path / "report.json"

        result = quality_report.export_report(analysis, ReportFormat.JSON, output_path)

        assert output_path.exists()
        assert result == str(output_path)

        # Verify JSON is valid
        import json

        with open(output_path) as f:
            data = json.load(f)
        assert "quality_score" in data

    def test_export_report_markdown(self, quality_report, sample_image_good_range):
        """Test report export as Markdown."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)

        content = quality_report.export_report(analysis, ReportFormat.MARKDOWN)

        assert isinstance(content, str)
        assert "# Quality Assurance Report" in content
        assert "Quality Assessment" in content

    def test_export_report_html(self, quality_report, sample_image_good_range):
        """Test report export as HTML."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)

        content = quality_report.export_report(analysis, ReportFormat.HTML)

        assert isinstance(content, str)
        assert "<html>" in content
        assert "Quality Assurance Report" in content

    def test_export_report_pdf_placeholder(self, quality_report, sample_image_good_range):
        """Test report export as PDF (placeholder)."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)

        content = quality_report.export_report(analysis, ReportFormat.PDF)

        assert "PDF export requires conversion tool" in content

    def test_export_report_invalid_format(self, quality_report, sample_image_good_range):
        """Test report export with invalid format."""
        analysis = quality_report.generate_post_print_analysis(sample_image_good_range)

        with pytest.raises(ValueError):
            quality_report.export_report(analysis, "invalid_format")


# ============================================================================
# AlertSystem Tests
# ============================================================================


class TestAlertSystem:
    """Tests for AlertSystem class."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        system = AlertSystem()
        assert system.settings is not None
        assert len(system.alerts) == 0

    def test_add_alert_basic(self, alert_system):
        """Test adding a basic alert."""
        alert_id = alert_system.add_alert(
            alert_type=AlertType.DENSITY,
            message="Test alert",
            severity=AlertSeverity.WARNING,
        )

        assert alert_id in alert_system.alerts
        assert alert_system.alerts[alert_id].message == "Test alert"
        assert not alert_system.alerts[alert_id].dismissed

    def test_add_alert_custom_id(self, alert_system):
        """Test adding alert with custom ID."""
        custom_id = "test_alert_123"

        alert_id = alert_system.add_alert(
            alert_type=AlertType.CHEMISTRY,
            message="Custom alert",
            severity=AlertSeverity.ERROR,
            alert_id=custom_id,
        )

        assert alert_id == custom_id

    def test_add_alert_custom_timestamp(self, alert_system):
        """Test adding alert with custom timestamp."""
        custom_time = datetime.now() - timedelta(hours=5)

        alert_id = alert_system.add_alert(
            alert_type=AlertType.GENERAL,
            message="Old alert",
            severity=AlertSeverity.INFO,
            timestamp=custom_time,
        )

        alert = alert_system.alerts[alert_id]
        assert abs((alert.timestamp - custom_time).total_seconds()) < 1

    @pytest.mark.parametrize(
        "severity",
        [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL],
    )
    def test_add_alert_all_severities(self, alert_system, severity):
        """Test adding alerts with different severities."""
        alert_id = alert_system.add_alert(
            alert_type=AlertType.GENERAL, message=f"Test {severity}", severity=severity
        )

        assert alert_system.alerts[alert_id].severity == severity

    def test_get_active_alerts_all(self, alert_system):
        """Test getting all active alerts."""
        # Add multiple alerts
        alert_system.add_alert(AlertType.DENSITY, "Alert 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.CHEMISTRY, "Alert 2", AlertSeverity.ERROR)
        alert_system.add_alert(AlertType.HUMIDITY, "Alert 3", AlertSeverity.INFO)

        active = alert_system.get_active_alerts()

        assert len(active) == 3
        # Should be sorted by severity
        assert active[0].severity == AlertSeverity.ERROR
        assert active[-1].severity == AlertSeverity.INFO

    def test_get_active_alerts_filter_by_severity(self, alert_system):
        """Test getting active alerts filtered by severity."""
        alert_system.add_alert(AlertType.GENERAL, "Warning 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Error 1", AlertSeverity.ERROR)
        alert_system.add_alert(AlertType.GENERAL, "Warning 2", AlertSeverity.WARNING)

        warnings = alert_system.get_active_alerts(severity=AlertSeverity.WARNING)

        assert len(warnings) == 2
        assert all(a.severity == AlertSeverity.WARNING for a in warnings)

    def test_get_active_alerts_filter_by_type(self, alert_system):
        """Test getting active alerts filtered by type."""
        alert_system.add_alert(AlertType.DENSITY, "Density 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.CHEMISTRY, "Chem 1", AlertSeverity.ERROR)
        alert_system.add_alert(AlertType.DENSITY, "Density 2", AlertSeverity.INFO)

        density_alerts = alert_system.get_active_alerts(alert_type=AlertType.DENSITY)

        assert len(density_alerts) == 2
        assert all(a.alert_type == AlertType.DENSITY for a in density_alerts)

    def test_get_active_alerts_excludes_dismissed(self, alert_system):
        """Test that dismissed alerts are excluded."""
        id1 = alert_system.add_alert(AlertType.GENERAL, "Alert 1", AlertSeverity.WARNING)
        id2 = alert_system.add_alert(AlertType.GENERAL, "Alert 2", AlertSeverity.ERROR)

        # Dismiss one alert
        alert_system.dismiss_alert(id1)

        active = alert_system.get_active_alerts()

        assert len(active) == 1
        assert active[0].alert_id == id2

    def test_dismiss_alert_success(self, alert_system):
        """Test successful alert dismissal."""
        alert_id = alert_system.add_alert(AlertType.GENERAL, "Test alert", AlertSeverity.WARNING)

        success = alert_system.dismiss_alert(alert_id)

        assert success is True
        assert alert_system.alerts[alert_id].dismissed is True
        assert alert_system.alerts[alert_id].dismissed_at is not None

    def test_dismiss_alert_not_found(self, alert_system):
        """Test dismissing non-existent alert."""
        success = alert_system.dismiss_alert("nonexistent_id")
        assert success is False

    def test_get_alert_history_all(self, alert_system):
        """Test getting full alert history."""
        # Add alerts
        id1 = alert_system.add_alert(AlertType.GENERAL, "Alert 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Alert 2", AlertSeverity.ERROR)

        # Dismiss one
        alert_system.dismiss_alert(id1)

        history = alert_system.get_alert_history()

        assert len(history) == 2  # Both active and dismissed

    def test_get_alert_history_exclude_dismissed(self, alert_system):
        """Test getting history excluding dismissed alerts."""
        id1 = alert_system.add_alert(AlertType.GENERAL, "Alert 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Alert 2", AlertSeverity.ERROR)

        alert_system.dismiss_alert(id1)

        history = alert_system.get_alert_history(include_dismissed=False)

        assert len(history) == 1

    def test_get_alert_history_time_filter(self, alert_system):
        """Test getting history with time filter."""
        # Add old alert
        alert_system.add_alert(
            AlertType.GENERAL,
            "Old alert",
            AlertSeverity.WARNING,
            timestamp=datetime.now() - timedelta(hours=30),
        )

        # Add recent alert
        alert_system.add_alert(AlertType.GENERAL, "Recent alert", AlertSeverity.ERROR)

        history = alert_system.get_alert_history(hours=24)

        assert len(history) == 1

    def test_get_alert_history_sorted(self, alert_system):
        """Test that history is sorted by timestamp (newest first)."""
        old_time = datetime.now() - timedelta(hours=5)
        new_time = datetime.now()

        alert_system.add_alert(AlertType.GENERAL, "Old", AlertSeverity.WARNING, timestamp=old_time)
        alert_system.add_alert(AlertType.GENERAL, "New", AlertSeverity.ERROR, timestamp=new_time)

        history = alert_system.get_alert_history()

        assert history[0].message == "New"
        assert history[1].message == "Old"

    def test_clear_old_alerts(self, alert_system):
        """Test clearing old dismissed alerts."""
        # Add and dismiss old alert
        old_time = datetime.now() - timedelta(days=100)
        id1 = alert_system.add_alert(
            AlertType.GENERAL, "Old", AlertSeverity.WARNING, timestamp=old_time
        )
        alert_system.alerts[id1].dismissed = True
        alert_system.alerts[id1].dismissed_at = old_time

        # Add and dismiss recent alert
        recent_time = datetime.now() - timedelta(days=10)
        id2 = alert_system.add_alert(
            AlertType.GENERAL, "Recent", AlertSeverity.ERROR, timestamp=recent_time
        )
        alert_system.alerts[id2].dismissed = True
        alert_system.alerts[id2].dismissed_at = recent_time

        # Add active alert
        alert_system.add_alert(AlertType.GENERAL, "Active", AlertSeverity.INFO)

        cleared = alert_system.clear_old_alerts()

        assert cleared == 1  # Only old dismissed alert cleared
        assert id1 not in alert_system.alerts
        assert id2 in alert_system.alerts

    def test_get_alert_summary(self, alert_system):
        """Test getting alert summary."""
        alert_system.add_alert(AlertType.GENERAL, "Critical 1", AlertSeverity.CRITICAL)
        alert_system.add_alert(AlertType.GENERAL, "Error 1", AlertSeverity.ERROR)
        alert_system.add_alert(AlertType.GENERAL, "Warning 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Warning 2", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Info 1", AlertSeverity.INFO)

        summary = alert_system.get_alert_summary()

        assert summary["total"] == 5
        assert summary["critical"] == 1
        assert summary["error"] == 1
        assert summary["warning"] == 2
        assert summary["info"] == 1

    def test_get_alert_summary_excludes_dismissed(self, alert_system):
        """Test that summary excludes dismissed alerts."""
        id1 = alert_system.add_alert(AlertType.GENERAL, "Alert 1", AlertSeverity.WARNING)
        alert_system.add_alert(AlertType.GENERAL, "Alert 2", AlertSeverity.ERROR)

        alert_system.dismiss_alert(id1)

        summary = alert_system.get_alert_summary()

        assert summary["total"] == 1
        assert summary["warning"] == 0
        assert summary["error"] == 1

    def test_get_alert(self, alert_system):
        """Test getting specific alert by ID."""
        alert_id = alert_system.add_alert(AlertType.GENERAL, "Test alert", AlertSeverity.WARNING)

        alert = alert_system.get_alert(alert_id)

        assert alert is not None
        assert alert.message == "Test alert"

    def test_get_alert_not_found(self, alert_system):
        """Test getting non-existent alert."""
        alert = alert_system.get_alert("nonexistent_id")
        assert alert is None

    def test_alert_to_dict(self):
        """Test Alert to_dict conversion."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.CHEMISTRY,
            severity=AlertSeverity.ERROR,
            message="Test alert message",
            timestamp=datetime.now(),
        )

        data = alert.to_dict()

        assert isinstance(data, dict)
        assert data["alert_id"] == "test_123"
        assert data["alert_type"] == "chemistry"
        assert data["severity"] == "error"
        assert data["message"] == "Test alert message"
        assert data["dismissed"] is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestQualityAssuranceIntegration:
    """Integration tests combining multiple QA components."""

    def test_complete_workflow(
        self,
        sample_image_good_range,
        qa_settings,
    ):
        """Test complete QA workflow from measurement to report."""
        # Setup all components
        NegativeDensityValidator(qa_settings)
        chemistry = ChemistryFreshnessTracker(qa_settings)
        humidity = PaperHumidityChecker(qa_settings)
        uv = UVLightMeterIntegration(qa_settings)
        report_gen = QualityReport(qa_settings)

        # Register chemistry
        chemistry.register_solution(
            solution_type=SolutionType.PALLADIUM,
            date_mixed=datetime.now(),
            volume_ml=100.0,
        )

        # Measure conditions
        humidity.measure_paper_humidity(humidity_percent=50.0, temperature_celsius=20.0)
        uv.read_intensity(intensity=100.0, wavelength=365.0)

        # Generate pre-print checklist
        checklist = report_gen.generate_pre_print_checklist(
            image=sample_image_good_range,
            chemistry_tracker=chemistry,
            humidity_checker=humidity,
            uv_meter=uv,
        )

        # Verify checklist completeness - all checks should be present
        assert len(checklist["checks"]) == 4
        assert "negative_density" in checklist["checks"]
        assert "chemistry" in checklist["checks"]
        assert "paper_humidity" in checklist["checks"]
        assert "uv_light" in checklist["checks"]

        # Chemistry and humidity should pass with our setup
        assert checklist["checks"]["chemistry"]["status"] in ["pass", "warning"]
        assert checklist["checks"]["paper_humidity"]["status"] == "pass"

        # Checklist should have timestamp
        assert "timestamp" in checklist
        assert isinstance(checklist["ready_to_print"], bool)

        # Generate post-print analysis
        analysis = report_gen.generate_post_print_analysis(sample_image_good_range)

        assert analysis["quality_score"] > 0
        assert analysis["grade"] in ["A", "B", "C", "D", "F"]

    def test_alert_workflow(self, qa_settings):
        """Test alert generation and management workflow."""
        chemistry = ChemistryFreshnessTracker(qa_settings)
        alerts = AlertSystem(qa_settings)

        # Register expired solution
        chemistry.register_solution(
            solution_type=SolutionType.DEVELOPER,
            date_mixed=datetime.now() - timedelta(days=100),
            volume_ml=100.0,
        )

        # Get chemistry alerts
        chem_alerts = chemistry.get_alerts()

        # Add to alert system
        for chem_alert in chem_alerts:
            alerts.add_alert(
                alert_type=AlertType.CHEMISTRY,
                message=chem_alert["message"],
                severity=AlertSeverity.CRITICAL
                if chem_alert["severity"] == "critical"
                else AlertSeverity.WARNING,
            )

        # Check alerts
        active = alerts.get_active_alerts()
        assert len(active) > 0

        # Dismiss alerts
        for alert in active:
            alerts.dismiss_alert(alert.alert_id)

        # Verify dismissal
        remaining = alerts.get_active_alerts()
        assert len(remaining) == 0
