"""
End-to-end tests for the Vertex AI integration.

These tests verify complete user workflows through the system,
simulating realistic usage scenarios without requiring live
Google Cloud services.

All Google Cloud API calls are mocked to allow offline testing.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.e2e
class TestDarkroomAssistantWorkflow:
    """E2E test simulating a complete darkroom assistant session."""

    def test_new_user_first_calibration(self):
        """Simulate a new user's first calibration session end-to-end."""
        from ptpd_calibration.vertex.agents import (
            calculate_chemistry_recipe,
            calculate_uv_exposure,
        )
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryBankClient(storage_path=tmpdir)

            # Step 1: New user arrives
            profile = memory.get_profile("new-photographer")
            assert profile.preferences == {}

            # Step 2: Set initial preferences
            profile.display_name = "Alex Photographer"
            profile.update_preference("paper_type", "Hahnemühle Platinum Rag")
            profile.update_preference("default_ratio", "50:50")
            profile.update_preference("uv_source", "LED 365nm")

            # Step 3: Calculate chemistry recipe
            recipe_json = calculate_chemistry_recipe(
                print_size_inches="8x10",
                pt_pd_ratio="50:50",
                method="traditional",
                paper_type="Hahnemühle Platinum Rag",
            )
            recipe = json.loads(recipe_json)
            assert recipe["status"] == "success"
            assert recipe["total_ml"] > 0

            # Step 4: Calculate UV exposure
            exposure_json = calculate_uv_exposure(
                uv_source="LED 365nm",
                negative_dr=1.5,
                paper_type="Hahnemühle Platinum Rag",
            )
            exposure = json.loads(exposure_json)
            assert exposure["status"] == "success"

            # Step 5: Record calibration result
            profile.add_calibration(
                CalibrationSnapshot(
                    paper_type="Hahnemühle Platinum Rag",
                    pt_pd_ratio="50:50",
                    exposure_seconds=exposure["recommended_seconds"],
                    dmin=0.04,
                    dmax=1.78,
                    density_range=1.74,
                    notes="First print - good results",
                )
            )
            profile.add_successful_recipe(recipe)
            memory.save_profile(profile)

            # Step 6: Verify session context for next session
            context = memory.get_context_for_session("new-photographer")
            assert "Alex Photographer" in context
            assert "Hahnemühle Platinum Rag" in context

    def test_returning_user_with_drift(self):
        """Simulate a returning user whose calibration has drifted."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryBankClient(storage_path=tmpdir)

            # Set up existing user with calibration history
            profile = memory.get_profile("experienced-user")
            profile.display_name = "Bob Darkroom"
            profile.update_preference("paper_type", "Arches Platine")

            # Historical calibration (good)
            profile.add_calibration(
                CalibrationSnapshot(
                    paper_type="Arches Platine",
                    pt_pd_ratio="70:30",
                    exposure_seconds=240,
                    dmin=0.04,
                    dmax=1.90,
                    density_range=1.86,
                )
            )

            # Recent calibration (drift detected)
            profile.add_calibration(
                CalibrationSnapshot(
                    paper_type="Arches Platine",
                    pt_pd_ratio="70:30",
                    exposure_seconds=240,
                    dmin=0.12,
                    dmax=1.60,
                    density_range=1.48,
                )
            )
            memory.save_profile(profile)

            # Check session context reports drift
            context = memory.get_context_for_session("experienced-user")
            assert "Bob Darkroom" in context
            # Should detect both Dmax and Dmin drift
            has_drift = any(
                word in context.lower() for word in ["drift", "dmax", "dmin", "re-calibrat"]
            )
            assert has_drift, f"Expected drift warning in context: {context}"

            # Calculate new exposure with iterative refinement
            new_exposure = json.loads(
                calculate_uv_exposure(
                    uv_source="LED 365nm",
                    negative_dr=1.5,
                    previous_time_seconds=240,
                )
            )
            assert new_exposure["status"] == "success"
            # Test strip should center on previous time
            center = next(t for t in new_exposure["test_strip_times"] if t["label"] == "100%")
            assert center["seconds"] == 240


@pytest.mark.e2e
class TestCorpusToSearchWorkflow:
    """E2E test for corpus preparation -> upload -> search workflow."""

    def test_corpus_prepare_and_format(self):
        """Prepare corpus, then format as search context."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator
        from ptpd_calibration.vertex.search import PtPdSearchClient, SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare a corpus
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text(
                "# Pt/Pd Calibration Studio\n\n"
                "Use 50:50 Pt/Pd ratio for neutral tones.\n"
                "Use higher Pd for warmer tones.\n"
            )

            output = Path(tmpdir) / "corpus"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_all()
            assert count >= 1

            # Simulate search results from the prepared corpus
            search_client = PtPdSearchClient(project_id="test", data_store_id="test")
            results = [
                SearchResult(
                    title="README",
                    snippet="Use 50:50 Pt/Pd ratio for neutral tones.",
                    document_id="readme-1",
                    relevance_score=0.92,
                ),
            ]

            # Format for LLM
            context = search_client.format_context_for_llm(results)
            assert "50:50 Pt/Pd" in context
            assert "knowledge base" in context.lower()


@pytest.mark.e2e
class TestMultiSessionPersistence:
    """E2E test verifying data persists across multiple simulated sessions."""

    def test_three_session_progression(self):
        """Simulate 3 sessions: setup, calibrate, refine."""
        from ptpd_calibration.vertex.agents import (
            calculate_chemistry_recipe,
            calculate_uv_exposure,
        )
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Setup
            mem1 = MemoryBankClient(storage_path=tmpdir)
            p = mem1.get_profile("session-test")
            p.display_name = "Session Tester"
            p.update_preference("paper", "Bergger COT320")
            mem1.save_profile(p)

            # Session 2: First calibration
            mem2 = MemoryBankClient(storage_path=tmpdir)
            p = mem2.get_profile("session-test")
            assert p.display_name == "Session Tester"

            recipe = json.loads(
                calculate_chemistry_recipe(
                    print_size_inches="11x14",
                    pt_pd_ratio="40:60",
                )
            )
            exposure = json.loads(
                calculate_uv_exposure(
                    uv_source="Metal Halide",
                )
            )

            p.add_calibration(
                CalibrationSnapshot(
                    paper_type="Bergger COT320",
                    pt_pd_ratio="40:60",
                    exposure_seconds=exposure["recommended_seconds"],
                    dmin=0.06,
                    dmax=1.65,
                )
            )
            p.add_successful_recipe(recipe)
            p.add_note("First session with Metal Halide - good results")
            mem2.save_profile(p)

            # Session 3: Verify all data persisted
            mem3 = MemoryBankClient(storage_path=tmpdir)
            p = mem3.get_profile("session-test")

            assert p.display_name == "Session Tester"
            assert p.preferences["paper"] == "Bergger COT320"
            assert len(p.calibration_history) == 1
            assert len(p.successful_recipes) == 1
            assert len(p.notes) == 1
            assert "Metal Halide" in p.notes[0]

            summary = p.get_summary()
            assert "Session Tester" in summary
            assert "Bergger COT320" in summary


@pytest.mark.e2e
class TestVisionAnalysisWorkflow:
    """E2E test for vision analysis with mocked Gemini API."""

    def test_step_tablet_analysis_workflow(self):
        """Complete step tablet analysis: load image -> analyze -> parse."""
        from ptpd_calibration.vertex.vision import (
            _load_image,
            _parse_vision_response,
        )

        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()
            image_path = f.name

        import os

        try:
            # Step 1: Load image
            data, mime = _load_image(image_path)
            assert len(data) > 0
            assert mime == "image/png"

            # Step 2: Simulate Gemini analysis
            simulated_response = json.dumps(
                {
                    "steps": [
                        {"step": i, "density": i * 0.08, "quality": "good"} for i in range(1, 22)
                    ],
                    "overall_quality": {"score": 8.0, "description": "Good calibration"},
                    "dmin": 0.05,
                    "dmax": 1.68,
                    "density_range": 1.63,
                    "issues": ["Slight uneven coating in step 15"],
                    "recommendations": [
                        "Increase exposure by 10%",
                        "Check coating rod for defects",
                    ],
                }
            )

            # Step 3: Parse response
            result = _parse_vision_response(simulated_response, "step_tablet_analysis")

            assert result.analysis_type == "step_tablet_analysis"
            assert result.confidence == 0.8  # 8.0 / 10.0
            assert len(result.recommendations) == 2
            assert result.structured_data["dmax"] == 1.68
            assert len(result.structured_data["steps"]) == 21
        finally:
            os.unlink(image_path)

    def test_print_quality_evaluation_workflow(self):
        """Complete print quality evaluation: load -> evaluate -> parse."""
        from ptpd_calibration.vertex.vision import _load_image, _parse_vision_response

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()
            image_path = f.name

        import os

        try:
            data, mime = _load_image(image_path)
            assert mime == "image/jpeg"

            simulated_response = json.dumps(
                {
                    "scores": {
                        "tonal_range": 9.0,
                        "highlight_quality": 8.5,
                        "shadow_quality": 7.0,
                        "midtone_smoothness": 8.0,
                        "coating_quality": 9.5,
                        "overall_impression": 8.5,
                    },
                    "overall_score": 8.4,
                    "strengths": ["Excellent tonal range", "Beautiful coating"],
                    "improvements": ["Shadow detail could be richer"],
                    "next_steps": ["Increase exposure 5% for deeper shadows"],
                }
            )

            result = _parse_vision_response(simulated_response, "print_quality_evaluation")
            assert result.analysis_type == "print_quality_evaluation"
            assert len(result.recommendations) >= 1
        finally:
            os.unlink(image_path)
