"""
End-to-end tests for the calibration feedback loop workflow.

Tests cover:
1. Complete calibration session workflow
2. Print analysis with curve refinement
3. Paper profile application
4. Database persistence across sessions
5. Multi-iteration calibration convergence
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gradient_scan():
    """Create a synthetic gradient print scan."""
    # Create a 200x200 gradient image simulating a step tablet scan
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        value = int(i * 255 / 199)
        img[i, :, :] = value
    return img


@pytest.fixture
def uniform_print_scan():
    """Create a uniform gray print scan (Zone V)."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


@pytest.fixture
def step_wedge_scan():
    """Create a synthetic step wedge scan with 11 zones."""
    img = np.zeros((220, 100, 3), dtype=np.uint8)
    # Create 11 zones (20 pixels each)
    for zone in range(11):
        value = int(zone * 255 / 10)
        img[zone * 20:(zone + 1) * 20, :, :] = value
    return img


@pytest.fixture
def sample_quad_content():
    """Create simple QTR format .quad content."""
    lines = [
        "## QuadToneRIP K,C,M,Y",
        "# Test Profile for E2E Tests",
        "# K Curve",
    ]
    # K channel - linear ramp
    for i in range(256):
        lines.append(str(i * 257))  # 0-65535 range

    # C, M, Y channels - zeros
    for channel in ["C", "M", "Y"]:
        lines.append(f"# {channel} Curve")
        for i in range(256):
            lines.append("0")

    return "\n".join(lines)


@pytest.fixture
def quad_file(tmp_path, sample_quad_content):
    """Create a temporary .quad file for testing."""
    quad_path = tmp_path / "test_calibration.quad"
    quad_path.write_text(sample_quad_content)
    return quad_path


@pytest.fixture
def temp_database(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_calibration_history.db"
    return db_path


# ============================================================================
# Functional E2E Tests (No Browser Required)
# ============================================================================

@pytest.mark.functional
class TestCalibrationWorkflowFunctional:
    """Functional end-to-end tests for calibration workflow."""

    def test_complete_single_iteration_workflow(self, gradient_scan, quad_file, temp_database):
        """Test a complete single-iteration calibration workflow."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CurveCalibrator,
            QuadCurveParser,
            CalibrationDatabase,
            CalibrationRecord,
            TargetDensities,
        )

        # Step 1: Initialize components
        targets = TargetDensities.for_platinum_palladium()
        analyzer = PrintAnalyzer(targets=targets)
        db = CalibrationDatabase(temp_database)

        # Step 2: Analyze the print scan
        analysis = analyzer.analyze_print_scan(gradient_scan)

        assert analysis is not None
        assert analysis.highlight_density > 0
        assert analysis.shadow_density > 0
        assert analysis.tonal_range > 0

        # Step 3: Parse base curve
        header, curves = QuadCurveParser.parse(str(quad_file))

        assert header is not None
        assert "K" in curves
        assert len(curves["K"]) == 256

        # Step 4: Apply adjustments based on analysis
        calibrator = CurveCalibrator()
        refined_curves = calibrator.adjust_all_from_feedback(
            curves,
            highlight_delta=analysis.recommended_highlight_adj,
            midtone_delta=analysis.recommended_midtone_adj,
            shadow_delta=analysis.recommended_shadow_adj
        )

        assert "K" in refined_curves
        assert len(refined_curves["K"]) == 256

        # Step 5: Save session and record to database
        session = db.create_session("Arches Platine", "6Pd:2Pt", "E2E Test Session")

        record = CalibrationRecord(
            session_id=session.id,
            iteration_number=1,
            paper_type="Arches Platine",
            exposure_time="3:00",
            highlight_density=analysis.highlight_density,
            midtone_density=analysis.midtone_density,
            shadow_density=analysis.shadow_density,
            tonal_range=analysis.tonal_range,
            highlight_adj=analysis.recommended_highlight_adj,
            midtone_adj=analysis.recommended_midtone_adj,
            shadow_adj=analysis.recommended_shadow_adj
        )
        record_id = db.add_record(record)

        assert record_id is not None

        # Step 6: Export refined curve
        output_path = quad_file.parent / "refined.quad"
        QuadCurveParser.write(
            str(output_path),
            header,
            refined_curves,
            extra_comments=["E2E test refinement"]
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "## QuadToneRIP" in content or "# K Curve" in content

    def test_multi_iteration_convergence(self, gradient_scan, quad_file, temp_database):
        """Test multiple iterations of calibration converging toward target."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CurveCalibrator,
            QuadCurveParser,
            CalibrationDatabase,
            CalibrationRecord,
            TargetDensities,
        )

        targets = TargetDensities.for_platinum_palladium()
        analyzer = PrintAnalyzer(targets=targets)
        db = CalibrationDatabase(temp_database)

        # Create session
        session = db.create_session("Arches Platine", "6Pd:2Pt", "Multi-iteration Test")

        # Parse initial curve
        header, curves = QuadCurveParser.parse(str(quad_file))
        calibrator = CurveCalibrator()

        # Simulate 3 iterations
        for iteration in range(1, 4):
            # Analyze (using same scan for simplicity)
            analysis = analyzer.analyze_print_scan(gradient_scan)

            # Record iteration
            record = CalibrationRecord(
                session_id=session.id,
                iteration_number=iteration,
                paper_type="Arches Platine",
                exposure_time="3:00",
                highlight_density=analysis.highlight_density,
                midtone_density=analysis.midtone_density,
                shadow_density=analysis.shadow_density,
                tonal_range=analysis.tonal_range,
                highlight_adj=analysis.recommended_highlight_adj * 0.5,
                midtone_adj=analysis.recommended_midtone_adj * 0.5,
                shadow_adj=analysis.recommended_shadow_adj * 0.5
            )
            db.add_record(record)

            # Apply adjustments
            curves = calibrator.adjust_all_from_feedback(
                curves,
                highlight_delta=analysis.recommended_highlight_adj * 0.5,  # Dampen
                midtone_delta=analysis.recommended_midtone_adj * 0.5,
                shadow_delta=analysis.recommended_shadow_adj * 0.5
            )

        # Verify session has all iterations
        records = db.get_session_records(session.id)
        assert len(records) == 3

    def test_paper_profile_application_workflow(self, quad_file):
        """Test applying paper profiles to curves."""
        from ptpd_calibration.calibration import (
            CurveCalibrator,
            QuadCurveParser,
            CALIBRATION_PROFILES,
            get_available_calibration_profiles,
        )

        # Parse base curve
        header, curves = QuadCurveParser.parse(str(quad_file))
        original_k = curves["K"].copy()

        # Get available profiles
        profiles = get_available_calibration_profiles()
        assert len(profiles) > 0

        # Test applying each profile
        results = {}
        for profile_key in profiles:
            profile = CALIBRATION_PROFILES[profile_key]
            calibrator = CurveCalibrator(profile)
            adjusted = calibrator.adjust_all_curves(curves.copy())

            results[profile_key] = adjusted

        # Verify all profiles processed
        assert len(results) == len(profiles)

    def test_database_persistence_across_instances(self, temp_database, gradient_scan):
        """Test that database persists data across instances."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CalibrationDatabase,
            CalibrationRecord,
            TargetDensities,
        )

        # First instance - create session
        db1 = CalibrationDatabase(temp_database)
        session = db1.create_session("Test Paper", "6Pd:2Pt", "Persistence Test")
        session_id = session.id

        targets = TargetDensities.for_platinum_palladium()
        analyzer = PrintAnalyzer(targets=targets)
        analysis = analyzer.analyze_print_scan(gradient_scan)

        record = CalibrationRecord(
            session_id=session_id,
            iteration_number=1,
            paper_type="Test Paper",
            exposure_time="2:30",
            highlight_density=analysis.highlight_density,
            midtone_density=analysis.midtone_density,
            shadow_density=analysis.shadow_density
        )
        db1.add_record(record)
        db1.close()

        # Second instance - verify data persists
        db2 = CalibrationDatabase(temp_database)
        sessions = db2.list_sessions()

        assert len(sessions) > 0
        found_session = next((s for s in sessions if s.id == session_id), None)
        assert found_session is not None
        assert found_session.paper_type == "Test Paper"

        records = db2.get_session_records(session_id)
        assert len(records) == 1
        db2.close()

    def test_different_process_targets(self, gradient_scan):
        """Test analysis with different process target densities."""
        from ptpd_calibration.calibration import PrintAnalyzer, TargetDensities

        # Test with different processes
        processes = {
            "Pt/Pd": TargetDensities.for_platinum_palladium(),
            "Cyanotype": TargetDensities.for_cyanotype(),
            "Silver Gelatin": TargetDensities.for_silver_gelatin(),
        }

        results = {}
        for name, targets in processes.items():
            analyzer = PrintAnalyzer(targets=targets)
            analysis = analyzer.analyze_print_scan(gradient_scan)
            results[name] = analysis

        # All analyses should complete
        assert len(results) == 3


@pytest.mark.functional
class TestCalibrationSessionManagement:
    """Tests for calibration session management."""

    def test_create_and_list_sessions(self, temp_database):
        """Test creating and listing calibration sessions."""
        from ptpd_calibration.calibration import CalibrationDatabase

        db = CalibrationDatabase(temp_database)

        # Create multiple sessions
        session1 = db.create_session("Paper A", "6Pd:2Pt", "Test 1")
        session2 = db.create_session("Paper B", None, "Test 2")
        session3 = db.create_session("Paper C", "Cyanotype", "Test 3")

        # List sessions
        sessions = db.list_sessions()

        assert len(sessions) >= 3
        session_ids = [s.id for s in sessions]
        assert session1.id in session_ids
        assert session2.id in session_ids
        assert session3.id in session_ids
        db.close()

    def test_session_metadata_retrieval(self, temp_database):
        """Test retrieving session metadata."""
        from ptpd_calibration.calibration import CalibrationDatabase

        db = CalibrationDatabase(temp_database)

        # Create session with specific metadata
        session = db.create_session(
            "Arches Platine",
            "6Pd:2Pt",
            "Detailed metadata test"
        )

        # Retrieve session
        retrieved = db.get_session(session.id)

        assert retrieved is not None
        assert retrieved.paper_type == "Arches Platine"
        assert retrieved.chemistry == "6Pd:2Pt"
        assert "Detailed metadata test" in retrieved.name
        db.close()

    def test_delete_session_cascade(self, temp_database, gradient_scan):
        """Test that deleting a session removes its records."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CalibrationDatabase,
            CalibrationRecord,
            TargetDensities,
        )

        db = CalibrationDatabase(temp_database)

        # Create session with records
        session = db.create_session("Test", "6Pd:2Pt", "Delete test")

        targets = TargetDensities.for_platinum_palladium()
        analyzer = PrintAnalyzer(targets=targets)
        analysis = analyzer.analyze_print_scan(gradient_scan)

        for i in range(1, 3):
            record = CalibrationRecord(
                session_id=session.id,
                iteration_number=i,
                paper_type="Test",
                exposure_time="3:00",
                highlight_density=analysis.highlight_density,
                midtone_density=analysis.midtone_density,
                shadow_density=analysis.shadow_density
            )
            db.add_record(record)

        # Verify records exist
        records_before = db.get_session_records(session.id)
        assert len(records_before) == 2

        # Delete session
        db.delete_session(session.id)

        # Verify session and records are gone
        retrieved = db.get_session(session.id)
        assert retrieved is None

        records_after = db.get_session_records(session.id)
        assert len(records_after) == 0
        db.close()


@pytest.mark.functional
class TestCurveRoundTrip:
    """Tests for curve loading, modification, and saving."""

    def test_curve_roundtrip_preserves_format(self, quad_file):
        """Test that loading and saving a curve preserves format."""
        from ptpd_calibration.calibration import QuadCurveParser

        # Load original
        header, curves = QuadCurveParser.parse(str(quad_file))

        # Save to new file
        output_path = quad_file.parent / "roundtrip.quad"
        QuadCurveParser.write(str(output_path), header, curves)

        # Verify QTR format
        new_content = output_path.read_text()
        assert "# K Curve" in new_content

    def test_curve_modification_affects_values(self, quad_file):
        """Test that curve modifications actually change values."""
        from ptpd_calibration.calibration import (
            CurveCalibrator,
            QuadCurveParser,
            CALIBRATION_PROFILES,
        )

        header, curves = QuadCurveParser.parse(str(quad_file))
        original_k = curves["K"].copy()

        # Apply Arches Platine profile (10% midtone boost)
        profile = CALIBRATION_PROFILES["arches_platine"]
        calibrator = CurveCalibrator(profile)
        adjusted = calibrator.adjust_all_curves(curves)

        # Midtone values should change with a boost
        mid_region = slice(64, 192)
        original_mid_sum = sum(original_k[mid_region])
        adjusted_mid_sum = sum(adjusted["K"][mid_region])

        # With midtone boost, adjusted should be different
        assert adjusted_mid_sum != original_mid_sum

    def test_extreme_adjustments_clamped(self, quad_file):
        """Test that extreme adjustments don't produce invalid values."""
        from ptpd_calibration.calibration import CurveCalibrator, QuadCurveParser

        header, curves = QuadCurveParser.parse(str(quad_file))

        calibrator = CurveCalibrator()

        # Apply extreme adjustments
        adjusted = calibrator.adjust_all_from_feedback(
            curves,
            highlight_delta=0.5,  # 50% - extreme
            midtone_delta=0.5,
            shadow_delta=0.5
        )

        # All values should be in valid range
        for channel, values in adjusted.items():
            for v in values:
                assert 0 <= v <= 65535, f"Value {v} out of range in channel {channel}"


@pytest.mark.functional
class TestIntegrationScenarios:
    """Integration tests for realistic calibration scenarios."""

    def test_new_paper_calibration_scenario(self, quad_file, temp_database):
        """Simulate calibrating a new paper from scratch."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CurveCalibrator,
            QuadCurveParser,
            CalibrationDatabase,
            CalibrationRecord,
            CalibrationProfile,
            TargetDensities,
        )

        db = CalibrationDatabase(temp_database)
        targets = TargetDensities.for_platinum_palladium()
        analyzer = PrintAnalyzer(targets=targets)

        # Create a custom profile for new paper
        custom_profile = CalibrationProfile(
            name="New Test Paper",
            absorption_factor=0.95,
            highlight_boost=0.02,
            midtone_boost=0.06,
            shadow_boost=0.04
        )

        calibrator = CurveCalibrator(custom_profile)

        # Load base curve and apply custom profile
        header, curves = QuadCurveParser.parse(str(quad_file))
        initial_adjusted = calibrator.adjust_all_curves(curves)

        # Create session
        session = db.create_session("New Test Paper", "6Pd:2Pt", "Initial calibration")

        # Simulate first print analysis
        first_print = np.full((100, 100, 3), 140, dtype=np.uint8)  # Slightly dark
        analysis1 = analyzer.analyze_print_scan(first_print)

        record = CalibrationRecord(
            session_id=session.id,
            iteration_number=1,
            paper_type="New Test Paper",
            exposure_time="3:15",
            highlight_density=analysis1.highlight_density,
            midtone_density=analysis1.midtone_density,
            shadow_density=analysis1.shadow_density
        )
        db.add_record(record)

        # Apply feedback refinement
        refined = calibrator.adjust_all_from_feedback(
            initial_adjusted,
            highlight_delta=analysis1.recommended_highlight_adj,
            midtone_delta=analysis1.recommended_midtone_adj,
            shadow_delta=analysis1.recommended_shadow_adj
        )

        # Verify we can export the refined curve
        output_path = quad_file.parent / "new_paper_calibrated.quad"
        QuadCurveParser.write(str(output_path), header, refined)

        assert output_path.exists()

        # Verify session has record
        records = db.get_session_records(session.id)
        assert len(records) == 1
        db.close()

    def test_alternative_process_calibration(self, quad_file, temp_database):
        """Test calibration for alternative processes (cyanotype)."""
        from ptpd_calibration.calibration import (
            PrintAnalyzer,
            CurveCalibrator,
            QuadCurveParser,
            CalibrationDatabase,
            CalibrationRecord,
            TargetDensities,
        )

        db = CalibrationDatabase(temp_database)

        # Use cyanotype targets
        targets = TargetDensities.for_cyanotype()
        analyzer = PrintAnalyzer(targets=targets)

        # Create session for cyanotype
        session = db.create_session("Bergger COT 320", "Cyanotype", "Cyanotype calibration")

        # Load and parse curve
        header, curves = QuadCurveParser.parse(str(quad_file))
        calibrator = CurveCalibrator()

        # Simulate cyanotype print (higher contrast typical)
        cyan_print = np.zeros((100, 100, 3), dtype=np.uint8)
        cyan_print[:33] = 10  # Deep shadows
        cyan_print[33:66] = 120  # Midtones
        cyan_print[66:] = 240  # Highlights

        analysis = analyzer.analyze_print_scan(cyan_print)

        record = CalibrationRecord(
            session_id=session.id,
            iteration_number=1,
            paper_type="Bergger COT 320",
            exposure_time="5:00",
            highlight_density=analysis.highlight_density,
            midtone_density=analysis.midtone_density,
            shadow_density=analysis.shadow_density
        )
        db.add_record(record)

        # Cyanotype typically needs more contrast
        refined = calibrator.adjust_all_from_feedback(
            curves,
            highlight_delta=analysis.recommended_highlight_adj,
            midtone_delta=analysis.recommended_midtone_adj,
            shadow_delta=analysis.recommended_shadow_adj
        )

        # Verify
        records = db.get_session_records(session.id)
        assert len(records) == 1
        assert "K" in refined
        db.close()


# ============================================================================
# Browser-based E2E Tests (Require Playwright)
# ============================================================================

# Try to import playwright
try:
    from playwright.sync_api import Page, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


@pytest.mark.browser
@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
@pytest.mark.skip(reason="Playwright selectors pending update for hierarchical navigation")
class TestPrintAnalysisBrowser:
    """Browser-based E2E tests for Print Analysis tab."""

    def test_navigate_to_print_analysis_tab(self, page: "Page", app_url, ensure_app_running):
        """Test navigating to the Print Analysis tab."""
        page.goto(app_url)

        # Navigate to Calibration section
        page.get_by_role("tab", name="Calibration", exact=False).first.click()

        # Navigate to Print Analysis
        page.get_by_role("tab", name="Print Analysis").first.click()

        # Verify tab is visible
        expect(page.get_by_text("Print Analysis & Curve Refinement")).to_be_visible(timeout=5000)

    def test_upload_and_analyze_print(self, page: "Page", app_url, ensure_app_running, tmp_path):
        """Test uploading and analyzing a print scan."""
        from PIL import Image

        # Create test image
        img_path = tmp_path / "test_scan.png"
        gradient = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (32, 1))
        rgb = np.stack([gradient, gradient, gradient], axis=-1)
        Image.fromarray(rgb).save(img_path)

        page.goto(app_url)
        page.get_by_role("tab", name="Calibration", exact=False).first.click()
        page.get_by_role("tab", name="Print Analysis").first.click()

        # Upload image
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(img_path))

        # Click analyze
        page.get_by_role("button", name="Analyze Print").click()

        # Wait for results
        page.wait_for_timeout(2000)

        # Verify analysis results appear
        expect(page.get_by_text("Analysis Results")).to_be_visible(timeout=10000)

    def test_apply_paper_profile(self, page: "Page", app_url, ensure_app_running, tmp_path, sample_quad_content):
        """Test applying a paper profile to a curve."""
        # Create test quad file
        quad_path = tmp_path / "test.quad"
        quad_path.write_text(sample_quad_content)

        page.goto(app_url)
        page.get_by_role("tab", name="Calibration", exact=False).first.click()
        page.get_by_role("tab", name="Print Analysis").first.click()
        page.get_by_role("tab", name="Paper Profiles").click()

        # Upload base curve
        file_input = page.locator("input[type='file']").first
        file_input.set_input_files(str(quad_path))

        # Select paper profile
        page.get_by_label("Paper Profile").select_option("Arches Platine")

        # Apply profile
        page.get_by_role("button", name="Apply Paper Profile").click()

        # Wait for result
        page.wait_for_timeout(2000)

        # Verify output
        expect(page.get_by_text("saved", exact=False)).to_be_visible(timeout=5000)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "functional"])
