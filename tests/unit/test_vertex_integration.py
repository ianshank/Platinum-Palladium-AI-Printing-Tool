"""
Unit tests for Vertex AI integration.

Tests cover:
- Configuration (VertexAISettings, LLMProvider.VERTEX_AI)
- VertexAIClient creation
- Memory Bank (UserProfile, MemoryBankClient)
- Corpus preparation (CorpusPreparator)
- Vision helpers (_load_image, _parse_vision_response)
- ADK agent tool wrappers
- Search client construction
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── Configuration Tests ───


@pytest.mark.unit
class TestVertexAIConfig:
    """Tests for Vertex AI configuration."""

    def test_vertex_ai_provider_in_enum(self):
        """VERTEX_AI should be a valid LLMProvider value."""
        from ptpd_calibration.config import LLMProvider

        assert LLMProvider.VERTEX_AI == "vertex_ai"
        assert LLMProvider("vertex_ai") == LLMProvider.VERTEX_AI

    def test_vertex_ai_settings_defaults(self):
        """VertexAISettings should have sensible defaults."""
        from ptpd_calibration.config import VertexAISettings

        settings = VertexAISettings()
        assert settings.location == "us-central1"
        assert settings.vision_model == "gemini-2.5-flash"
        assert settings.coordinator_model == "gemini-2.5-pro"
        assert settings.specialist_model == "gemini-2.5-flash"
        assert settings.enable_memory_bank is True
        assert settings.search_max_results == 10
        assert settings.project_id is None
        assert settings.search_data_store_id is None

    def test_llm_settings_vertex_model(self):
        """LLMSettings should include vertex_model field."""
        from ptpd_calibration.config import LLMSettings

        settings = LLMSettings()
        assert settings.vertex_model == "gemini-2.5-flash"
        assert settings.vertex_project is None
        assert settings.vertex_location == "us-central1"

    def test_llm_settings_get_active_api_key_vertex(self):
        """Vertex AI provider should return None for API key (uses ADC)."""
        from ptpd_calibration.config import LLMProvider, LLMSettings

        settings = LLMSettings(provider=LLMProvider.VERTEX_AI)
        assert settings.get_active_api_key() is None

    def test_main_settings_has_vertex(self):
        """Main Settings should include vertex subsettings."""
        from ptpd_calibration.config import Settings

        settings = Settings()
        assert hasattr(settings, "vertex")
        assert settings.vertex.location == "us-central1"


# ─── LLM Client Tests ───


@pytest.mark.unit
class TestVertexAIClient:
    """Tests for the VertexAIClient."""

    def test_create_client_vertex_ai(self):
        """create_client should return VertexAIClient for VERTEX_AI provider."""
        from ptpd_calibration.config import LLMProvider, LLMSettings
        from ptpd_calibration.llm.client import VertexAIClient, create_client

        settings = LLMSettings(provider=LLMProvider.VERTEX_AI)
        client = create_client(settings)
        assert isinstance(client, VertexAIClient)

    def test_vertex_client_stores_settings(self):
        """VertexAIClient should store settings reference."""
        from ptpd_calibration.config import LLMProvider, LLMSettings
        from ptpd_calibration.llm.client import VertexAIClient

        settings = LLMSettings(
            provider=LLMProvider.VERTEX_AI,
            vertex_project="test-project",
            vertex_location="us-east1",
            vertex_model="gemini-2.5-pro",
        )
        client = VertexAIClient(settings)
        assert client.settings.vertex_project == "test-project"
        assert client.settings.vertex_location == "us-east1"
        assert client.settings.vertex_model == "gemini-2.5-pro"

    def test_convert_messages_to_gemini(self):
        """Message conversion should map roles correctly."""
        from ptpd_calibration.llm.client import _convert_messages_to_gemini

        # Create mock types module
        mock_types = MagicMock()
        mock_types.Content = MagicMock()
        mock_types.Part.from_text = MagicMock(side_effect=lambda text: f"Part({text})")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Help me"},
        ]

        result = _convert_messages_to_gemini(messages, mock_types)
        assert len(result) == 3

        # Check role mapping
        calls = mock_types.Content.call_args_list
        assert calls[0][1]["role"] == "user"
        assert calls[1][1]["role"] == "model"
        assert calls[2][1]["role"] == "user"


# ─── Memory Bank Tests ───


@pytest.mark.unit
class TestMemoryBank:
    """Tests for the Memory Bank local client."""

    def test_user_profile_creation(self):
        """UserProfile should initialize with defaults."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="test-user")
        assert profile.user_id == "test-user"
        assert profile.preferences == {}
        assert profile.calibration_history == []
        assert profile.successful_recipes == []

    def test_user_profile_update_preference(self):
        """update_preference should store key-value pairs."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="test-user")
        profile.update_preference("paper_type", "Hahnemühle Platinum Rag")
        profile.update_preference("default_pt_pd_ratio", "50:50")

        assert profile.preferences["paper_type"] == "Hahnemühle Platinum Rag"
        assert profile.preferences["default_pt_pd_ratio"] == "50:50"

    def test_user_profile_add_calibration(self):
        """add_calibration should append snapshots."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")
        snapshot = CalibrationSnapshot(
            paper_type="Arches Platine",
            pt_pd_ratio="80:20",
            exposure_seconds=240,
            dmin=0.05,
            dmax=1.85,
            density_range=1.80,
        )
        profile.add_calibration(snapshot)

        assert len(profile.calibration_history) == 1
        assert profile.calibration_history[0].paper_type == "Arches Platine"
        assert profile.calibration_history[0].dmax == 1.85

    def test_user_profile_detect_drift(self):
        """detect_drift should flag significant density changes."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")

        # Add two calibrations with drift
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.85, dmin=0.05)
        )
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.60, dmin=0.05)
        )

        warnings = profile.detect_drift(threshold=0.1)
        assert len(warnings) >= 1
        assert "drift" in warnings[0].lower() or "Dmax" in warnings[0]

    def test_user_profile_no_drift_when_stable(self):
        """detect_drift should return empty list when values are stable."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.85, dmin=0.05)
        )
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.87, dmin=0.06)
        )

        warnings = profile.detect_drift(threshold=0.1)
        assert len(warnings) == 0

    def test_user_profile_get_summary(self):
        """get_summary should produce readable output."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="test-user", display_name="Test Photographer")
        profile.update_preference("paper_type", "Hahnemühle Platinum Rag")

        summary = profile.get_summary()
        assert "Test Photographer" in summary
        assert "Hahnemühle Platinum Rag" in summary

    def test_user_profile_summary_new_user(self):
        """get_summary for new user should indicate no history."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="new-user")
        summary = profile.get_summary()
        assert "no history" in summary.lower() or "New user" in summary

    def test_memory_bank_client_crud(self):
        """MemoryBankClient should support create/read/update/delete."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            # Create
            profile = client.get_profile("user-1")
            assert profile.user_id == "user-1"

            # Update
            profile.update_preference("paper", "Arches Platine")
            client.save_profile(profile)

            # Read back (clear cache to force file read)
            client._cache.clear()
            loaded = client.get_profile("user-1")
            assert loaded.preferences["paper"] == "Arches Platine"

            # List
            profiles = client.list_profiles()
            assert "user-1" in profiles

            # Delete
            assert client.delete_profile("user-1") is True
            assert client.delete_profile("nonexistent") is False

    def test_memory_bank_context_for_session(self):
        """get_context_for_session should produce LLM-ready context."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            profile = client.get_profile("user-2")
            profile.display_name = "Alice"
            profile.update_preference("paper_type", "Revere Platinum")
            client.save_profile(profile)

            context = client.get_context_for_session("user-2")
            assert "Alice" in context
            assert "Revere Platinum" in context

    def test_user_profile_serialization(self):
        """UserProfile should serialize/deserialize cleanly."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="ser-test", display_name="Serialization Test")
        profile.update_preference("key", "value")
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Test Paper", dmax=1.5)
        )
        profile.add_successful_recipe({"name": "Test Recipe", "pt_pd": "50:50"})
        profile.add_note("This is a test note")

        # Serialize and deserialize
        data = json.loads(profile.model_dump_json())
        restored = UserProfile(**data)

        assert restored.user_id == "ser-test"
        assert restored.preferences["key"] == "value"
        assert len(restored.calibration_history) == 1
        assert len(restored.successful_recipes) == 1
        assert len(restored.notes) == 1


# ─── Corpus Preparation Tests ───


@pytest.mark.unit
class TestCorpusPreparation:
    """Tests for corpus preparation."""

    def test_preparator_creates_output_dir(self):
        """CorpusPreparator should create output directory."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_output"
            preparator = CorpusPreparator(
                repo_path=".",
                output_dir=str(output),
            )
            assert preparator.output_dir.exists()

    def test_prepare_domain_knowledge(self):
        """prepare_domain_knowledge should copy knowledge files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_output"
            preparator = CorpusPreparator(
                repo_path=".",
                output_dir=str(output),
            )
            count = preparator.prepare_domain_knowledge()
            # Should find the knowledge files we created
            assert count >= 4  # chemistry, paper, exposure, troubleshooting

    def test_prepare_repo_docs(self):
        """prepare_repo_docs should process existing markdown files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake repo with one doc
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("# Test Readme\nSome content.")

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_repo_docs()

            assert count == 1
            output_files = list(preparator.output_dir.iterdir())
            assert len(output_files) == 1
            content = output_files[0].read_text()
            assert "Test Readme" in content
            assert "Pt/Pd Calibration Studio" in content  # Enrichment header

    def test_prepare_code_docs(self):
        """prepare_code_docs should process Python source files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake source structure
            repo = Path(tmpdir) / "repo"
            src = repo / "src" / "ptpd_calibration" / "detection"
            src.mkdir(parents=True)
            (src / "detector.py").write_text('"""Step tablet detector."""\nclass Detector: pass')

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_code_docs()

            assert count == 1


# ─── Vision Helpers Tests ───


@pytest.mark.unit
class TestVisionHelpers:
    """Tests for vision module helper functions."""

    def test_load_image_png(self):
        """_load_image should handle PNG files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            f.flush()

            data, mime = _load_image(f.name)
            assert mime == "image/png"
            assert len(data) > 0

        os.unlink(f.name)

    def test_load_image_jpeg(self):
        """_load_image should handle JPEG files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()

            data, mime = _load_image(f.name)
            assert mime == "image/jpeg"

        os.unlink(f.name)

    def test_load_image_tiff(self):
        """_load_image should handle TIFF files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(b"II\x2a\x00" + b"\x00" * 100)
            f.flush()

            data, mime = _load_image(f.name)
            assert mime == "image/tiff"

        os.unlink(f.name)

    def test_load_image_not_found(self):
        """_load_image should raise FileNotFoundError for missing files."""
        from ptpd_calibration.vertex.vision import _load_image

        with pytest.raises(FileNotFoundError):
            _load_image("/nonexistent/image.png")

    def test_load_image_unsupported_format(self):
        """_load_image should raise ValueError for unsupported formats."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"data")
            f.flush()

            with pytest.raises(ValueError, match="Unsupported image format"):
                _load_image(f.name)

        os.unlink(f.name)

    def test_parse_vision_response_json(self):
        """_parse_vision_response should parse valid JSON."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        json_response = json.dumps({
            "confidence": 0.85,
            "recommendations": ["Increase exposure by 20%", "Check clearing baths"],
            "diagnosis": "Weak Dmax",
        })

        result = _parse_vision_response(json_response, "test_analysis")
        assert result.analysis_type == "test_analysis"
        assert result.confidence == 0.85
        assert len(result.recommendations) == 2
        assert result.structured_data["diagnosis"] == "Weak Dmax"

    def test_parse_vision_response_markdown_json(self):
        """_parse_vision_response should handle JSON in markdown code blocks."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        response = '```json\n{"confidence": 0.9, "fix_steps": ["Step 1"]}\n```'

        result = _parse_vision_response(response, "test")
        assert result.confidence == 0.9
        assert len(result.recommendations) == 1

    def test_parse_vision_response_invalid_json(self):
        """_parse_vision_response should handle non-JSON gracefully."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        result = _parse_vision_response("This is not JSON at all.", "test")
        assert result.raw_response == "This is not JSON at all."
        assert result.structured_data == {}
        assert result.confidence == 0.0


# ─── ADK Agent Tool Wrapper Tests ───


@pytest.mark.unit
class TestADKToolWrappers:
    """Tests for ADK agent tool wrapper functions."""

    def test_calculate_chemistry_recipe(self):
        """calculate_chemistry_recipe should return valid recipe JSON."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        result = json.loads(calculate_chemistry_recipe(
            print_size_inches="8x10",
            pt_pd_ratio="50:50",
            method="traditional",
        ))

        assert result["status"] == "success"
        assert result["print_size"] == "8x10"
        assert result["pt_pd_ratio"] == "50:50"
        assert result["total_ml"] > 0
        assert result["total_drops"] > 0
        assert "ferric_oxalate" in result
        assert "platinum" in result
        assert "palladium" in result
        assert len(result["instructions"]) >= 5

    def test_calculate_chemistry_recipe_pure_palladium(self):
        """Pure palladium recipe should have zero platinum."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        result = json.loads(calculate_chemistry_recipe(
            print_size_inches="8x10",
            pt_pd_ratio="0:100",
        ))

        assert result["status"] == "success"
        assert result["platinum"]["drops"] == 0
        assert result["palladium"]["drops"] > 0

    def test_calculate_chemistry_recipe_large_print(self):
        """Larger prints should require more chemistry."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        small = json.loads(calculate_chemistry_recipe(print_size_inches="5x7"))
        large = json.loads(calculate_chemistry_recipe(print_size_inches="16x20"))

        assert large["total_ml"] > small["total_ml"]

    def test_calculate_uv_exposure(self):
        """calculate_uv_exposure should return valid exposure data."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        result = json.loads(calculate_uv_exposure(
            uv_source="LED 365nm",
            negative_dr=1.5,
        ))

        assert result["status"] == "success"
        assert result["recommended_seconds"] > 0
        assert ":" in result["recommended_formatted"]
        assert len(result["test_strip_times"]) == 5

    def test_calculate_uv_exposure_different_sources(self):
        """Different UV sources should give different exposure times."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        led = json.loads(calculate_uv_exposure(uv_source="LED 365nm"))
        halide = json.loads(calculate_uv_exposure(uv_source="Metal Halide"))

        assert halide["recommended_seconds"] > led["recommended_seconds"]

    def test_get_contrast_agent(self):
        """_get_contrast_agent should return appropriate recommendations."""
        from ptpd_calibration.vertex.agents import _get_contrast_agent

        assert "None" in _get_contrast_agent("low")
        assert "H2O2" in _get_contrast_agent("normal")
        assert "H2O2" in _get_contrast_agent("high") or "chlorate" in _get_contrast_agent("high")


# ─── Search Client Tests ───


@pytest.mark.unit
class TestSearchClient:
    """Tests for the Vertex AI Search client."""

    def test_search_client_construction(self):
        """PtPdSearchClient should initialize with config."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(
            project_id="test-project",
            data_store_id="test-store",
        )
        assert client.project_id == "test-project"
        assert client.data_store_id == "test-store"
        assert client.location == "global"

    def test_serving_config_path(self):
        """_serving_config_path should build correct resource path."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(
            project_id="my-project",
            data_store_id="ptpd-knowledge",
            serving_config="default_search",
        )

        path = client._serving_config_path
        assert "projects/my-project" in path
        assert "dataStores/ptpd-knowledge" in path
        assert "servingConfigs/default_search" in path

    def test_format_context_for_llm(self):
        """format_context_for_llm should produce formatted context."""
        from ptpd_calibration.vertex.search import PtPdSearchClient, SearchResult

        client = PtPdSearchClient(project_id="test", data_store_id="test")

        results = [
            SearchResult(
                title="Chemistry Guide",
                snippet="Use 50:50 Pt/Pd for neutral tones.",
                document_id="doc1",
                relevance_score=0.95,
            ),
            SearchResult(
                title="Paper Profiles",
                snippet="Hahnemühle Platinum Rag is the gold standard.",
                document_id="doc2",
                relevance_score=0.85,
            ),
        ]

        context = client.format_context_for_llm(results)
        assert "Chemistry Guide" in context
        assert "Paper Profiles" in context
        assert "50:50 Pt/Pd" in context
        assert "knowledge base" in context.lower()

    def test_format_context_max_length(self):
        """format_context_for_llm should respect max_context_length."""
        from ptpd_calibration.vertex.search import PtPdSearchClient, SearchResult

        client = PtPdSearchClient(project_id="test", data_store_id="test")

        results = [
            SearchResult(
                title=f"Doc {i}",
                snippet="x" * 200,
                document_id=f"doc{i}",
            )
            for i in range(20)
        ]

        context = client.format_context_for_llm(results, max_context_length=500)
        assert len(context) <= 600  # Allow some overhead for headers

    def test_format_context_empty_results(self):
        """format_context_for_llm should handle empty results."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test", data_store_id="test")
        context = client.format_context_for_llm([])
        assert context == ""


# ─── SearchResult Tests ───


@pytest.mark.unit
class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_defaults(self):
        """SearchResult should have sensible defaults."""
        from ptpd_calibration.vertex.search import SearchResult

        result = SearchResult(
            title="Test",
            snippet="Test snippet",
            document_id="id1",
        )
        assert result.relevance_score == 0.0
        assert result.metadata == {}

    def test_search_result_with_metadata(self):
        """SearchResult should store arbitrary metadata."""
        from ptpd_calibration.vertex.search import SearchResult

        result = SearchResult(
            title="Test",
            snippet="Test snippet",
            document_id="id1",
            relevance_score=0.9,
            metadata={"category": "chemistry", "source": "Bostick-Sullivan"},
        )
        assert result.metadata["category"] == "chemistry"


# ─── VisionAnalysisResult Tests ───


@pytest.mark.unit
class TestVisionAnalysisResult:
    """Tests for VisionAnalysisResult dataclass."""

    def test_result_defaults(self):
        """VisionAnalysisResult should have sensible defaults."""
        from ptpd_calibration.vertex.vision import VisionAnalysisResult

        result = VisionAnalysisResult(
            analysis_type="test",
            raw_response="raw text",
        )
        assert result.structured_data == {}
        assert result.confidence == 0.0
        assert result.recommendations == []

    def test_result_with_data(self):
        """VisionAnalysisResult should store structured data."""
        from ptpd_calibration.vertex.vision import VisionAnalysisResult

        result = VisionAnalysisResult(
            analysis_type="step_tablet",
            raw_response="json...",
            structured_data={"dmax": 1.85, "steps": 21},
            confidence=0.92,
            recommendations=["Increase exposure"],
        )
        assert result.structured_data["dmax"] == 1.85
        assert result.confidence == 0.92
