"""
Tests for print analyzer module.

Tests print analysis, density measurements, calibration sessions, and target densities.
"""

import pytest
import numpy as np
from datetime import datetime
from uuid import UUID

from ptpd_calibration.calibration.print_analyzer import (
    PrintAnalysis,
    PrintAnalyzer,
    CalibrationSession,
    CalibrationIteration,
    TargetDensities,
)


# =============================================================================
# TargetDensities Tests
# =============================================================================


class TestTargetDensities:
    """Tests for TargetDensities dataclass."""

    def test_default_targets(self):
        """Default targets should be for Pt/Pd."""
        targets = TargetDensities()
        assert targets.highlight == 0.12
        assert targets.midtone == 0.65
        assert targets.shadow == 1.55
        assert targets.tonal_range == 1.40

    def test_platinum_palladium_targets(self):
        """Should create Pt/Pd targets."""
        targets = TargetDensities.for_platinum_palladium()
        assert targets.highlight == 0.12
        assert targets.midtone == 0.65
        assert targets.shadow == 1.55

    def test_cyanotype_targets(self):
        """Should create cyanotype targets."""
        targets = TargetDensities.for_cyanotype()
        assert targets.highlight == 0.15
        assert targets.midtone == 0.80
        assert targets.shadow == 1.90

    def test_silver_gelatin_targets(self):
        """Should create silver gelatin targets."""
        targets = TargetDensities.for_silver_gelatin()
        assert targets.highlight == 0.08
        assert targets.midtone == 0.55
        assert targets.shadow == 2.10

    def test_target_ordering(self):
        """Target densities should be in correct order."""
        targets = TargetDensities.for_platinum_palladium()
        assert targets.highlight < targets.midtone < targets.shadow


# =============================================================================
# PrintAnalysis Tests
# =============================================================================


class TestPrintAnalysis:
    """Tests for PrintAnalysis dataclass."""

    def test_default_analysis(self):
        """Default analysis should have zero values."""
        analysis = PrintAnalysis()
        assert analysis.highlight_density == 0.0
        assert analysis.midtone_density == 0.0
        assert analysis.shadow_density == 0.0
        assert analysis.id is not None
        assert isinstance(analysis.id, UUID)

    def test_custom_analysis(self):
        """Custom analysis should accept all parameters."""
        analysis = PrintAnalysis(
            highlight_density=0.15,
            midtone_density=0.70,
            shadow_density=1.60,
            tonal_range=1.45,
            midtone_separation=0.20,
            recommended_highlight_adj=0.02,
            recommended_midtone_adj=0.08,
            recommended_shadow_adj=0.05,
            notes=["Good highlights", "Midtones slightly muted"],
        )
        assert analysis.highlight_density == 0.15
        assert analysis.midtone_density == 0.70
        assert analysis.shadow_density == 1.60
        assert len(analysis.notes) == 2

    def test_analysis_summary(self):
        """Should generate summary string."""
        analysis = PrintAnalysis(
            highlight_density=0.15,
            midtone_density=0.70,
            shadow_density=1.60,
            tonal_range=1.45,
            recommended_highlight_adj=0.02,
            recommended_midtone_adj=0.08,
            recommended_shadow_adj=0.05,
            notes=["Test note"],
        )
        summary = analysis.summary()

        assert "PRINT ANALYSIS SUMMARY" in summary
        assert "0.15" in summary  # Highlight density
        assert "0.70" in summary  # Midtone density
        assert "1.60" in summary  # Shadow density
        assert "Test note" in summary

    def test_analysis_to_dict(self):
        """Should serialize to dictionary."""
        analysis = PrintAnalysis(
            highlight_density=0.15,
            midtone_density=0.70,
            shadow_density=1.60,
        )
        d = analysis.to_dict()

        assert d['highlight_density'] == 0.15
        assert d['midtone_density'] == 0.70
        assert d['shadow_density'] == 1.60
        assert 'id' in d
        assert 'timestamp' in d


# =============================================================================
# PrintAnalyzer Tests
# =============================================================================


class TestPrintAnalyzer:
    """Tests for PrintAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create default analyzer."""
        return PrintAnalyzer()

    @pytest.fixture
    def cyanotype_analyzer(self):
        """Create analyzer with cyanotype targets."""
        return PrintAnalyzer(targets=TargetDensities.for_cyanotype())

    @pytest.fixture
    def gradient_image(self):
        """Create a gradient test image (white to black)."""
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(255 * (1 - i / 100))  # White on left, black on right
        return img

    @pytest.fixture
    def uniform_gray_image(self):
        """Create a uniform gray test image."""
        return np.full((100, 100), 128, dtype=np.uint8)

    @pytest.fixture
    def high_contrast_image(self):
        """Create a high-contrast test image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 240  # Light top half
        img[50:, :] = 20   # Dark bottom half
        return img

    @pytest.fixture
    def rgb_gradient_image(self):
        """Create an RGB gradient test image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            val = int(255 * (1 - i / 100))
            img[:, i, :] = val
        return img

    def test_analyzer_default_targets(self, analyzer):
        """Analyzer should have default Pt/Pd targets."""
        assert analyzer.targets.highlight == 0.12
        assert analyzer.targets.midtone == 0.65
        assert analyzer.targets.shadow == 1.55

    def test_analyzer_custom_targets(self, cyanotype_analyzer):
        """Analyzer should use custom targets."""
        assert cyanotype_analyzer.targets.shadow == 1.90

    def test_analyze_gradient_image(self, analyzer, gradient_image):
        """Should analyze gradient image correctly."""
        analysis = analyzer.analyze_print_scan(gradient_image, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)
        assert analysis.highlight_density >= 0
        assert analysis.midtone_density >= 0
        assert analysis.shadow_density >= 0
        assert analysis.tonal_range > 0

    def test_analyze_rgb_image(self, analyzer, rgb_gradient_image):
        """Should handle RGB images."""
        analysis = analyzer.analyze_print_scan(rgb_gradient_image, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)
        assert analysis.tonal_range > 0

    def test_analyze_uniform_gray(self, analyzer, uniform_gray_image):
        """Should analyze uniform gray image."""
        analysis = analyzer.analyze_print_scan(uniform_gray_image, exclude_borders=False)

        # All zones should have similar density for uniform image
        assert isinstance(analysis, PrintAnalysis)
        # Tonal range should be very small for uniform image
        assert analysis.tonal_range < 0.5

    def test_analyze_high_contrast(self, analyzer, high_contrast_image):
        """Should detect high contrast correctly."""
        analysis = analyzer.analyze_print_scan(high_contrast_image, exclude_borders=False)

        # Should have good tonal range for high contrast image
        assert analysis.tonal_range > 0.5

    def test_analyze_generates_recommendations(self, analyzer, gradient_image):
        """Should generate adjustment recommendations."""
        analysis = analyzer.analyze_print_scan(gradient_image, exclude_borders=False)

        # Recommendations should be within valid range
        assert -0.20 <= analysis.recommended_highlight_adj <= 0.20
        assert -0.20 <= analysis.recommended_midtone_adj <= 0.20
        assert -0.20 <= analysis.recommended_shadow_adj <= 0.20

    def test_analyze_generates_notes(self, analyzer, gradient_image):
        """Should generate analysis notes."""
        analysis = analyzer.analyze_print_scan(gradient_image, exclude_borders=False)

        assert isinstance(analysis.notes, list)
        # Should have some notes about the analysis
        assert len(analysis.notes) >= 0

    def test_analyze_with_mask(self, analyzer):
        """Should respect provided mask."""
        # Create image with bright border, dark center
        img = np.full((100, 100), 240, dtype=np.uint8)  # Bright background
        img[20:80, 20:80] = 10  # Very dark center (density ~1.4)

        # Create mask for center only
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        analysis = analyzer.analyze_print_scan(img, mask=mask, exclude_borders=False)

        # Shadow density should be significant (dark center was analyzed)
        # Value 10/255 ≈ 0.039 reflectance, density = -log10(0.039) ≈ 1.4
        assert analysis.shadow_density > 1.0

    def test_zone_histogram(self, analyzer, gradient_image):
        """Should generate zone histogram."""
        analysis = analyzer.analyze_print_scan(gradient_image, exclude_borders=False)

        assert analysis.zone_histogram is not None
        assert isinstance(analysis.zone_histogram, dict)
        assert len(analysis.zone_histogram) > 0

        # Histogram values should sum to approximately 1.0
        total = sum(analysis.zone_histogram.values())
        assert 0.99 <= total <= 1.01

    def test_exclude_borders(self, analyzer, gradient_image):
        """Should auto-exclude borders when requested."""
        analysis_with_border = analyzer.analyze_print_scan(
            gradient_image, exclude_borders=True
        )
        analysis_no_border = analyzer.analyze_print_scan(
            gradient_image, exclude_borders=False
        )

        # Both should produce valid analysis
        assert isinstance(analysis_with_border, PrintAnalysis)
        assert isinstance(analysis_no_border, PrintAnalysis)

    def test_reflectance_to_density(self, analyzer):
        """Should convert reflectance to density correctly."""
        # Pure white (R=1.0) should have density ~0
        # Pure black (R~0.01) should have high density
        reflectances = np.array([1.0, 0.1, 0.01])
        densities = analyzer._reflectance_to_density(reflectances)

        assert densities[0] < 0.01  # White has ~0 density
        assert densities[1] > densities[0]  # Gray has more density than white
        assert densities[2] > densities[1]  # Black has most density

    def test_analyze_from_file(self, analyzer, tmp_path):
        """Should analyze from file path."""
        from PIL import Image

        # Create and save test image
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(255 * (1 - i / 100))

        image_path = tmp_path / "test_scan.png"
        Image.fromarray(img).save(image_path)

        analysis = analyzer.analyze_from_file(str(image_path))

        assert isinstance(analysis, PrintAnalysis)
        assert analysis.source_image_path == str(image_path)


# =============================================================================
# CalibrationIteration Tests
# =============================================================================


class TestCalibrationIteration:
    """Tests for CalibrationIteration dataclass."""

    def test_default_iteration(self):
        """Default iteration should have base values."""
        iteration = CalibrationIteration(
            iteration_number=1,
            exposure_time="3:00",
            curve_version="v1"
        )
        assert iteration.iteration_number == 1
        assert iteration.exposure_time == "3:00"
        assert iteration.curve_version == "v1"

    def test_iteration_with_adjustments(self):
        """Iteration should track applied adjustments."""
        iteration = CalibrationIteration(
            iteration_number=2,
            exposure_time="3:15",
            curve_version="v2",
            adjustments_applied={
                'highlight': 0.02,
                'midtone': 0.08,
                'shadow': 0.05
            }
        )
        assert iteration.adjustments_applied['midtone'] == 0.08

    def test_iteration_with_analysis(self):
        """Iteration should include analysis results."""
        analysis = PrintAnalysis(
            highlight_density=0.15,
            midtone_density=0.70,
            shadow_density=1.55
        )
        iteration = CalibrationIteration(
            iteration_number=1,
            exposure_time="3:00",
            curve_version="v1",
            analysis=analysis
        )
        assert iteration.analysis is not None
        assert iteration.analysis.midtone_density == 0.70


# =============================================================================
# CalibrationSession Tests
# =============================================================================


class TestCalibrationSession:
    """Tests for CalibrationSession class."""

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return CalibrationSession(
            paper_type="Arches Platine",
            chemistry="6Pd:2Pt",
            notes="Test calibration session"
        )

    @pytest.fixture
    def gradient_image(self):
        """Create a gradient test image."""
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(255 * (1 - i / 100))
        return img

    def test_session_creation(self, session):
        """Session should be created with metadata."""
        assert session.paper_type == "Arches Platine"
        assert session.chemistry == "6Pd:2Pt"
        assert session.notes == "Test calibration session"
        assert isinstance(session.id, UUID)
        assert isinstance(session.created_at, datetime)

    def test_session_empty_iterations(self, session):
        """New session should have no iterations."""
        assert len(session.iterations) == 0

    def test_add_iteration(self, session, gradient_image):
        """Should add iteration with analysis."""
        analysis = session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1",
            adjustments_applied={'midtone': 0.05}
        )

        assert isinstance(analysis, PrintAnalysis)
        assert len(session.iterations) == 1
        assert session.iterations[0].exposure_time == "3:00"
        assert session.iterations[0].curve_version == "v1"

    def test_add_multiple_iterations(self, session, gradient_image):
        """Should track multiple iterations."""
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1"
        )
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:15",
            curve_version="v2",
            adjustments_applied={'midtone': 0.08}
        )
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:15",
            curve_version="v3",
            adjustments_applied={'midtone': 0.05}
        )

        assert len(session.iterations) == 3
        assert session.iterations[0].iteration_number == 1
        assert session.iterations[1].iteration_number == 2
        assert session.iterations[2].iteration_number == 3

    def test_cumulative_adjustment(self, session, gradient_image):
        """Should calculate cumulative adjustments."""
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1",
            adjustments_applied={'highlight': 0.02, 'midtone': 0.05, 'shadow': 0.03}
        )
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:15",
            curve_version="v2",
            adjustments_applied={'highlight': 0.01, 'midtone': 0.03, 'shadow': 0.02}
        )

        cumulative = session.get_cumulative_adjustment()

        assert cumulative['highlight'] == 0.03
        assert cumulative['midtone'] == 0.08
        assert cumulative['shadow'] == 0.05

    def test_latest_recommendations(self, session, gradient_image):
        """Should get latest recommendations."""
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1"
        )

        recommendations = session.get_latest_recommendations()

        assert recommendations is not None
        assert 'highlight' in recommendations
        assert 'midtone' in recommendations
        assert 'shadow' in recommendations

    def test_latest_recommendations_empty(self, session):
        """Should return None if no iterations."""
        recommendations = session.get_latest_recommendations()
        assert recommendations is None

    def test_session_summary(self, session, gradient_image):
        """Should generate session summary."""
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1",
            adjustments_applied={'midtone': 0.05},
            notes="First test print"
        )

        summary = session.summary()

        assert "CALIBRATION SESSION SUMMARY" in summary
        assert "Arches Platine" in summary
        assert "6Pd:2Pt" in summary
        assert "3:00" in summary

    def test_session_to_dict(self, session, gradient_image):
        """Should serialize session to dictionary."""
        session.add_iteration(
            scan=gradient_image,
            exposure_time="3:00",
            curve_version="v1"
        )

        d = session.to_dict()

        assert d['paper_type'] == "Arches Platine"
        assert d['chemistry'] == "6Pd:2Pt"
        assert len(d['iterations']) == 1
        assert 'cumulative_adjustment' in d

    def test_add_iteration_from_file(self, session, tmp_path):
        """Should add iteration from image file."""
        from PIL import Image

        # Create and save test image
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[:, i] = int(255 * (1 - i / 100))

        image_path = tmp_path / "test_scan.png"
        Image.fromarray(img).save(image_path)

        analysis = session.add_iteration_from_file(
            filepath=str(image_path),
            exposure_time="3:00",
            curve_version="v1"
        )

        assert isinstance(analysis, PrintAnalysis)
        assert analysis.source_image_path == str(image_path)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def analyzer(self):
        return PrintAnalyzer()

    def test_very_small_image(self, analyzer):
        """Should handle very small images."""
        tiny_img = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        analysis = analyzer.analyze_print_scan(tiny_img, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)

    def test_single_pixel_image(self, analyzer):
        """Should handle single-pixel image."""
        single_pixel = np.array([[128]], dtype=np.uint8)
        analysis = analyzer.analyze_print_scan(single_pixel, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)

    def test_all_black_image(self, analyzer):
        """Should handle all-black image."""
        black_img = np.zeros((100, 100), dtype=np.uint8)
        analysis = analyzer.analyze_print_scan(black_img, exclude_borders=False)

        assert analysis.shadow_density > 1.5  # High density for black

    def test_all_white_image(self, analyzer):
        """Should handle all-white image."""
        white_img = np.full((100, 100), 255, dtype=np.uint8)
        analysis = analyzer.analyze_print_scan(white_img, exclude_borders=False)

        assert analysis.highlight_density < 0.1  # Low density for white

    def test_float_image(self, analyzer):
        """Should handle float images (0-1 range)."""
        float_img = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        analysis = analyzer.analyze_print_scan(float_img, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)

    def test_normalized_image(self, analyzer):
        """Should handle already normalized images."""
        # Image already in 0-1 range
        norm_img = np.random.rand(100, 100).astype(np.float32)
        analysis = analyzer.analyze_print_scan(norm_img, exclude_borders=False)

        assert isinstance(analysis, PrintAnalysis)

    def test_session_with_no_adjustments(self):
        """Session should handle iterations without adjustments."""
        session = CalibrationSession(paper_type="Test Paper")
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        session.add_iteration(
            scan=img,
            exposure_time="3:00",
            curve_version="v1"
            # No adjustments_applied
        )

        cumulative = session.get_cumulative_adjustment()
        assert cumulative['highlight'] == 0.0
        assert cumulative['midtone'] == 0.0
        assert cumulative['shadow'] == 0.0

    def test_analysis_metadata(self, analyzer):
        """Analysis should include metadata fields."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        analysis = analyzer.analyze_print_scan(img, exclude_borders=False)

        analysis.paper_type = "Test Paper"
        analysis.chemistry = "Pt/Pd"

        d = analysis.to_dict()
        assert d['paper_type'] == "Test Paper"
        assert d['chemistry'] == "Pt/Pd"
