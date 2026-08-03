"""
Integration tests for the Vertex AI module.

Tests verify that components work together correctly:
- Memory Bank + User Profile workflow
- Corpus preparation full pipeline
- Search client + context formatting
- Vision pipeline (load -> analyze -> parse)
- Agent tool wrappers producing valid outputs
- Configuration propagation across modules
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.integration
class TestMemoryBankWorkflow:
    """Integration tests for the full Memory Bank workflow."""

    def test_full_user_session_workflow(self):
        """Complete user workflow: create -> calibrate -> save -> recall -> drift."""
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            # Session 1: New user sets preferences
            profile = client.get_profile("photographer-1")
            profile.display_name = "Jane Smith"
            profile.update_preference("paper_type", "Hahnemühle Platinum Rag")
            profile.update_preference("default_pt_pd_ratio", "60:40")
            profile.update_preference("uv_source", "LED 365nm")
            profile.add_calibration(
                CalibrationSnapshot(
                    paper_type="Hahnemühle Platinum Rag",
                    pt_pd_ratio="60:40",
                    exposure_seconds=210,
                    dmin=0.04,
                    dmax=1.82,
                    density_range=1.78,
                    notes="First calibration, excellent results",
                )
            )
            profile.add_successful_recipe(
                {
                    "name": "Standard 8x10",
                    "pt_pd": "60:40",
                    "total_ml": 2.5,
                }
            )
            client.save_profile(profile)

            # Session 2: Load existing profile (fresh client simulates restart)
            client2 = MemoryBankClient(storage_path=tmpdir)
            recalled = client2.get_profile("photographer-1")
            assert recalled.display_name == "Jane Smith"
            assert recalled.preferences["paper_type"] == "Hahnemühle Platinum Rag"
            assert len(recalled.calibration_history) == 1
            assert len(recalled.successful_recipes) == 1

            # Session 3: Add more calibrations showing drift
            recalled.add_calibration(
                CalibrationSnapshot(
                    paper_type="Hahnemühle Platinum Rag",
                    pt_pd_ratio="60:40",
                    exposure_seconds=210,
                    dmin=0.04,
                    dmax=1.55,
                    density_range=1.51,
                    notes="Dmax dropped - chemistry aging?",
                )
            )
            client2.save_profile(recalled)

            # Verify drift detection
            context = client2.get_context_for_session("photographer-1")
            assert "Jane Smith" in context
            assert any(word in context.lower() for word in ["drift", "warning", "dmax"])

    def test_multiple_users_isolated(self):
        """Different users should have isolated profiles."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            p1 = client.get_profile("user-1")
            p1.update_preference("paper", "Arches Platine")
            client.save_profile(p1)

            p2 = client.get_profile("user-2")
            p2.update_preference("paper", "Revere Platinum")
            client.save_profile(p2)

            profiles = client.list_profiles()
            assert len(profiles) == 2

            assert client.get_profile("user-1").preferences["paper"] == "Arches Platine"
            assert client.get_profile("user-2").preferences["paper"] == "Revere Platinum"


@pytest.mark.integration
class TestCorpusPipeline:
    """Integration tests for the corpus preparation pipeline."""

    def test_full_corpus_preparation(self):
        """Full pipeline: prepare repo docs + code docs + domain knowledge."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up a realistic repo structure
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("# Pt/Pd Calibration\nOverview of the system.")
            (repo / "ARCHITECTURE.md").write_text("# Architecture\nComponent design.")

            src = repo / "src" / "ptpd_calibration"
            (src / "detection").mkdir(parents=True)
            (src / "detection" / "detector.py").write_text(
                '"""Step tablet detector."""\nclass StepTabletDetector:\n    pass'
            )
            (src / "curves").mkdir(parents=True)
            (src / "curves" / "generator.py").write_text(
                '"""Curve generator."""\nclass CurveGenerator:\n    pass'
            )

            output = Path(tmpdir) / "corpus"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            total = preparator.prepare_all()

            # Should have: 2 repo docs + 2 code docs + domain knowledge files
            assert total >= 4

            # Verify output files are enriched
            output_files = list(preparator.output_dir.iterdir())
            assert len(output_files) >= 4

            # Check enrichment headers
            for f in output_files:
                if f.name.startswith("repo_doc__"):
                    content = f.read_text()
                    assert "Pt/Pd Calibration Studio" in content

    def test_corpus_code_docs_enrichment(self):
        """Code docs should be enriched with metadata."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            src = repo / "src" / "ptpd_calibration" / "chemistry"
            src.mkdir(parents=True)
            (src / "calculator.py").write_text(
                '"""Chemistry calculator."""\n\ndef calculate_recipe():\n    pass'
            )

            output = Path(tmpdir) / "corpus"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_code_docs()

            assert count == 1
            out_files = list(preparator.output_dir.iterdir())
            code_doc = out_files[0].read_text()
            assert "chemistry" in code_doc.lower()
            assert "```python" in code_doc


@pytest.mark.integration
class TestAgentToolIntegration:
    """Integration tests for agent tool wrappers working together."""

    def test_chemistry_and_exposure_workflow(self):
        """Chemistry recipe + UV exposure should produce consistent results."""
        from ptpd_calibration.vertex.agents import (
            calculate_chemistry_recipe,
            calculate_uv_exposure,
        )

        recipe = json.loads(
            calculate_chemistry_recipe(
                print_size_inches="8x10",
                pt_pd_ratio="50:50",
                method="traditional",
                paper_type="Arches Platine",
            )
        )

        exposure = json.loads(
            calculate_uv_exposure(
                uv_source="LED 365nm",
                negative_dr=1.5,
                paper_type="Arches Platine",
            )
        )

        assert recipe["status"] == "success"
        assert exposure["status"] == "success"
        assert recipe["paper"] == "Arches Platine"
        assert exposure["paper_type"] == "Arches Platine"

    def test_exposure_iterative_refinement(self):
        """UV exposure should support iterative refinement via previous_time."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        # First exposure
        first = json.loads(calculate_uv_exposure(uv_source="LED 365nm"))
        initial_time = first["recommended_seconds"]

        # Refined with previous result
        refined = json.loads(
            calculate_uv_exposure(
                uv_source="LED 365nm",
                previous_time_seconds=initial_time * 1.1,  # Slightly over
            )
        )

        # Bracket should center on new time
        center = next(t for t in refined["test_strip_times"] if t["label"] == "100%")
        assert center["seconds"] == round(initial_time * 1.1)

    def test_all_contrast_goals_produce_recipes(self):
        """All contrast goal levels should produce valid recipes."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        for goal in ["low", "normal", "high", "very high"]:
            result = json.loads(calculate_chemistry_recipe(contrast_goal=goal))
            assert result["status"] == "success"
            assert result["contrast_agent"]  # Non-empty

    def test_all_uv_sources_produce_results(self):
        """All known UV sources should produce valid exposure data."""
        from ptpd_calibration.vertex.agents import UV_BASE_TIMES, calculate_uv_exposure

        for source in UV_BASE_TIMES:
            result = json.loads(calculate_uv_exposure(uv_source=source))
            assert result["status"] == "success"
            assert result["recommended_seconds"] > 0


@pytest.mark.integration
class TestSearchFormatting:
    """Integration tests for search + context formatting pipeline."""

    def test_search_results_to_llm_context(self):
        """Search results should format into useful LLM context."""
        from ptpd_calibration.vertex.search import PtPdSearchClient, SearchResult

        client = PtPdSearchClient(project_id="test", data_store_id="test")

        results = [
            SearchResult(
                title="Chemistry Fundamentals",
                snippet="Platinum and palladium salts are sensitized with ferric oxalate.",
                document_id="chem-1",
                relevance_score=0.95,
            ),
            SearchResult(
                title="Paper Selection Guide",
                snippet="Choose papers with internal sizing for best results.",
                document_id="paper-1",
                relevance_score=0.88,
            ),
            SearchResult(
                title="Troubleshooting",
                snippet="Low Dmax often indicates insufficient exposure.",
                document_id="trouble-1",
                relevance_score=0.75,
            ),
        ]

        context = client.format_context_for_llm(results)

        # Should have header
        assert "knowledge base" in context.lower()

        # All sources should be included
        assert "Source 1" in context
        assert "Source 2" in context
        assert "Source 3" in context

        # Content should be present
        assert "ferric oxalate" in context
        assert "internal sizing" in context
        assert "insufficient exposure" in context


@pytest.mark.integration
class TestConfigPropagation:
    """Integration tests verifying config propagates correctly."""

    def test_vertex_settings_propagate_to_search_client(self):
        """VertexAI settings should propagate to search client."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient()
        # Client should pick up settings from config
        assert client.location is not None

    def test_vertex_settings_propagate_to_vision_analyzer(self):
        """VertexAI settings should propagate to vision analyzer."""
        from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

        analyzer = GeminiVisionAnalyzer()
        assert analyzer.model is not None
        assert analyzer.location is not None

    def test_vertex_settings_propagate_to_memory_client(self):
        """VertexAI settings should propagate to memory client."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)
            assert client.storage_path.exists()


@pytest.mark.integration
class TestVisionPipeline:
    """Integration tests for the vision analysis pipeline."""

    def test_load_and_parse_pipeline(self):
        """Image loading + response parsing should work end-to-end."""
        from ptpd_calibration.vertex.vision import _load_image, _parse_vision_response

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()

            data, mime = _load_image(f.name)
            assert mime == "image/png"

        import os

        os.unlink(f.name)

        # Simulated Gemini response
        response = json.dumps(
            {
                "steps": [
                    {"step": 1, "density": 0.05, "quality": "good"},
                    {"step": 21, "density": 1.75, "quality": "good"},
                ],
                "overall_quality": {"score": 8.5, "description": "Good print"},
                "recommendations": ["Consider 5% more exposure"],
            }
        )

        result = _parse_vision_response(response, "step_tablet_analysis")
        assert result.analysis_type == "step_tablet_analysis"
        assert result.confidence == 0.85  # 8.5/10
        assert len(result.recommendations) == 1
        assert result.structured_data["steps"][0]["density"] == 0.05
