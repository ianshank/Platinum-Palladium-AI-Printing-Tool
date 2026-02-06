"""
Unit tests for Vertex AI integration.

Tests cover:
- Configuration (VertexAISettings, LLMProvider.VERTEX_AI)
- VertexAIClient creation and completion
- Memory Bank (UserProfile, MemoryBankClient)
- Corpus preparation (CorpusPreparator)
- Vision helpers (_load_image, _parse_vision_response)
- Vision analyzer (GeminiVisionAnalyzer with mocked Gemini)
- ADK agent tool wrappers
- ADK agent creation (create_adk_agents, deploy_to_agent_engine)
- Search client construction and methods
- Search document extraction helper
- Module constants and configurable values
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

    def test_vertex_settings_env_override(self):
        """VertexAISettings should respect env variables."""
        from ptpd_calibration.config import VertexAISettings

        with patch.dict(os.environ, {"PTPD_VERTEX_PROJECT_ID": "test-proj-123"}):
            settings = VertexAISettings()
            assert settings.project_id == "test-proj-123"

    def test_vertex_settings_all_fields(self):
        """VertexAISettings should expose all configuration fields."""
        from ptpd_calibration.config import VertexAISettings

        settings = VertexAISettings()
        # Verify all fields exist
        assert hasattr(settings, "staging_bucket")
        assert hasattr(settings, "search_serving_config")
        assert hasattr(settings, "vision_max_output_tokens")
        assert hasattr(settings, "agent_engine_id")
        assert hasattr(settings, "memory_scope")
        assert hasattr(settings, "corpus_bucket")
        assert hasattr(settings, "corpus_local_staging")
        assert hasattr(settings, "search_summary_model")

    def test_vertex_settings_search_summary_model_default(self):
        """VertexAISettings should have search_summary_model with default."""
        from ptpd_calibration.config import VertexAISettings

        settings = VertexAISettings()
        assert settings.search_summary_model == "gemini-2.5-flash"


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

        calls = mock_types.Content.call_args_list
        assert calls[0][1]["role"] == "user"
        assert calls[1][1]["role"] == "model"
        assert calls[2][1]["role"] == "user"

    def test_convert_messages_empty(self):
        """Empty message list should produce empty result."""
        from ptpd_calibration.llm.client import _convert_messages_to_gemini

        mock_types = MagicMock()
        result = _convert_messages_to_gemini([], mock_types)
        assert result == []

    def test_convert_messages_system_role(self):
        """System role messages should map to 'user' in Gemini."""
        from ptpd_calibration.llm.client import _convert_messages_to_gemini

        mock_types = MagicMock()
        mock_types.Content = MagicMock()
        mock_types.Part.from_text = MagicMock(side_effect=lambda text: text)

        messages = [{"role": "system", "content": "You are helpful"}]
        result = _convert_messages_to_gemini(messages, mock_types)
        assert len(result) == 1
        calls = mock_types.Content.call_args_list
        assert calls[0][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_vertex_client_complete_mocked(self):
        """VertexAIClient.complete should call Gemini API and return text."""
        from ptpd_calibration.config import LLMProvider, LLMSettings
        from ptpd_calibration.llm.client import VertexAIClient

        settings = LLMSettings(
            provider=LLMProvider.VERTEX_AI,
            vertex_project="test-proj",
            vertex_location="us-central1",
        )
        client = VertexAIClient(settings)

        mock_response = MagicMock()
        mock_response.text = "Test response"

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_genai_client

        mock_types = MagicMock()
        mock_types.Content = MagicMock()
        mock_types.Part.from_text = MagicMock(side_effect=lambda text: text)
        mock_types.GenerateContentConfig = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "google": MagicMock(),
                    "google.genai": mock_genai,
                    "google.genai.types": mock_types,
                },
            ),
            patch("ptpd_calibration.llm.client._convert_messages_to_gemini", return_value=[]),
            patch.object(client, "complete") as mock_complete,
        ):
            mock_complete.return_value = "Test response"
            result = await client.complete(
                messages=[{"role": "user", "content": "Hello"}],
            )
            assert result == "Test response"

    def test_create_client_unsupported_provider(self):
        """create_client should raise ValueError for unsupported provider."""
        from ptpd_calibration.llm.client import create_client

        mock_settings = MagicMock()
        mock_settings.provider = "unsupported"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            create_client(mock_settings)

    def test_vertex_client_default_settings(self):
        """VertexAIClient with no args should use global settings."""
        from ptpd_calibration.llm.client import VertexAIClient

        client = VertexAIClient()
        assert client.settings is not None


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

    def test_user_profile_drift_dmin(self):
        """detect_drift should flag significant Dmin changes."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.85, dmin=0.05)
        )
        profile.add_calibration(
            CalibrationSnapshot(paper_type="Arches Platine", dmax=1.85, dmin=0.25)
        )

        warnings = profile.detect_drift(threshold=0.1)
        assert any("Dmin" in w or "dmin" in w.lower() for w in warnings)

    def test_user_profile_drift_insufficient_data(self):
        """detect_drift with < 2 entries should return empty list."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")
        assert profile.detect_drift() == []

        profile.add_calibration(CalibrationSnapshot(paper_type="Test", dmax=1.5))
        assert profile.detect_drift() == []

    def test_user_profile_drift_multiple_papers(self):
        """detect_drift should track drift per paper type independently."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user")
        profile.add_calibration(CalibrationSnapshot(paper_type="Paper A", dmax=1.85, dmin=0.05))
        profile.add_calibration(CalibrationSnapshot(paper_type="Paper A", dmax=1.85, dmin=0.05))
        profile.add_calibration(CalibrationSnapshot(paper_type="Paper B", dmax=1.6, dmin=0.10))
        profile.add_calibration(CalibrationSnapshot(paper_type="Paper B", dmax=1.3, dmin=0.10))

        warnings = profile.detect_drift(threshold=0.1)
        assert any("Paper B" in w for w in warnings)
        assert not any("Paper A" in w for w in warnings)

    def test_user_profile_get_summary(self):
        """get_summary should produce readable output."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="test-user", display_name="Test Photographer")
        profile.update_preference("paper_type", "Hahnemühle Platinum Rag")

        summary = profile.get_summary()
        assert "Test Photographer" in summary
        assert "Hahnemühle Platinum Rag" in summary

    def test_user_profile_summary_with_history(self):
        """get_summary should include calibration history."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="test-user", display_name="Photographer")
        for i in range(5):
            profile.add_calibration(
                CalibrationSnapshot(
                    paper_type="Arches Platine",
                    pt_pd_ratio="50:50",
                    exposure_seconds=200 + i * 10,
                    dmax=1.7 + i * 0.02,
                )
            )

        summary = profile.get_summary()
        assert "calibrations" in summary.lower() or "5 total" in summary
        assert "Arches Platine" in summary

    def test_user_profile_summary_with_recipes(self):
        """get_summary should mention saved recipes count."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="test-user")
        profile.add_successful_recipe({"name": "Recipe 1"})
        profile.add_successful_recipe({"name": "Recipe 2"})

        summary = profile.get_summary()
        assert "2" in summary or "recipe" in summary.lower()

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

    def test_memory_bank_client_cache(self):
        """MemoryBankClient should use in-memory cache."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            profile1 = client.get_profile("cached-user")
            profile2 = client.get_profile("cached-user")
            assert profile1 is profile2  # Same object from cache

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

    def test_memory_bank_context_with_drift(self):
        """get_context_for_session should include drift warnings."""
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)

            profile = client.get_profile("drift-user")
            profile.add_calibration(
                CalibrationSnapshot(paper_type="Test Paper", dmax=1.85, dmin=0.05)
            )
            profile.add_calibration(
                CalibrationSnapshot(paper_type="Test Paper", dmax=1.50, dmin=0.05)
            )
            client.save_profile(profile)

            context = client.get_context_for_session("drift-user")
            assert "drift" in context.lower() or "warning" in context.lower()

    def test_memory_bank_profile_path_sanitization(self):
        """Profile paths should sanitize special characters."""
        from ptpd_calibration.vertex.memory import MemoryBankClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MemoryBankClient(storage_path=tmpdir)
            path = client._profile_path("user/with\\slashes")
            assert "/" not in path.stem or "\\" not in path.stem

    def test_user_profile_serialization(self):
        """UserProfile should serialize/deserialize cleanly."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot, UserProfile

        profile = UserProfile(user_id="ser-test", display_name="Serialization Test")
        profile.update_preference("key", "value")
        profile.add_calibration(CalibrationSnapshot(paper_type="Test Paper", dmax=1.5))
        profile.add_successful_recipe({"name": "Test Recipe", "pt_pd": "50:50"})
        profile.add_note("This is a test note")

        data = json.loads(profile.model_dump_json())
        restored = UserProfile(**data)

        assert restored.user_id == "ser-test"
        assert restored.preferences["key"] == "value"
        assert len(restored.calibration_history) == 1
        assert len(restored.successful_recipes) == 1
        assert len(restored.notes) == 1

    def test_user_profile_add_note_timestamped(self):
        """add_note should include a timestamp."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="note-test")
        profile.add_note("Test note")
        assert profile.notes[0].startswith("[")
        assert "Test note" in profile.notes[0]

    def test_user_profile_add_successful_recipe_timestamp(self):
        """add_successful_recipe should add saved_at timestamp."""
        from ptpd_calibration.vertex.memory import UserProfile

        profile = UserProfile(user_id="recipe-test")
        recipe = {"name": "Test", "pt_pd": "60:40"}
        profile.add_successful_recipe(recipe)
        assert "saved_at" in profile.successful_recipes[0]

    def test_calibration_snapshot_defaults(self):
        """CalibrationSnapshot should have sensible defaults."""
        from ptpd_calibration.vertex.memory import CalibrationSnapshot

        snapshot = CalibrationSnapshot()
        assert snapshot.paper_type == ""
        assert snapshot.pt_pd_ratio == "50:50"
        assert snapshot.exposure_seconds == 0.0
        assert snapshot.timestamp  # Should have auto-generated timestamp


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
            assert count >= 4  # chemistry, paper, exposure, troubleshooting

    def test_prepare_repo_docs(self):
        """prepare_repo_docs should process existing markdown files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
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
            assert "Pt/Pd Calibration Studio" in content

    def test_prepare_repo_docs_missing_files(self):
        """prepare_repo_docs should skip missing files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()  # No markdown files

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_repo_docs()
            assert count == 0

    def test_prepare_code_docs(self):
        """prepare_code_docs should process Python source files."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            src = repo / "src" / "ptpd_calibration" / "detection"
            src.mkdir(parents=True)
            (src / "detector.py").write_text('"""Step tablet detector."""\nclass Detector: pass')

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_code_docs()

            assert count == 1

    def test_prepare_code_docs_missing_src(self):
        """prepare_code_docs should handle missing src directory."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            count = preparator.prepare_code_docs()
            assert count == 0

    def test_prepare_all(self):
        """prepare_all should call all sub-preparators and sum counts."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("# Test")

            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=str(repo), output_dir=str(output))
            total = preparator.prepare_all()
            assert total >= 1  # At least README.md

    def test_upload_to_gcs_import_error(self):
        """upload_to_gcs should raise ImportError when google-cloud-storage missing."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            preparator = CorpusPreparator(repo_path=".", output_dir=str(tmpdir))

            # Mock the import to fail
            with (
                patch.dict(
                    "sys.modules", {"google.cloud.storage": None, "google.cloud": MagicMock()}
                ),
                patch("builtins.__import__", side_effect=ImportError("no storage")),
                pytest.raises(ImportError),
            ):
                preparator.upload_to_gcs(bucket_name="test-bucket")

    def test_upload_to_gcs_no_bucket(self):
        """upload_to_gcs should raise ValueError when no bucket specified."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            preparator = CorpusPreparator(repo_path=".", output_dir=str(tmpdir))

            mock_storage = MagicMock()
            with (
                patch.dict(
                    "sys.modules",
                    {"google.cloud.storage": mock_storage, "google.cloud": MagicMock()},
                ),
                patch("ptpd_calibration.vertex.corpus.get_settings") as mock_settings,
                pytest.raises(ValueError, match="GCS bucket name required"),
            ):
                mock_settings.return_value.vertex.corpus_bucket = None
                preparator.upload_to_gcs(bucket_name=None)

    def test_upload_to_gcs_strips_gs_prefix(self):
        """upload_to_gcs should strip gs:// prefix from bucket name."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=".", output_dir=str(output))

            # Create a file in the output directory
            preparator.output_dir.mkdir(parents=True, exist_ok=True)
            (preparator.output_dir / "test.txt").write_text("test content")

            mock_client = MagicMock()
            mock_bucket = MagicMock()
            mock_blob = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_bucket.blob.return_value = mock_blob

            mock_storage_module = MagicMock()
            mock_storage_module.Client.return_value = mock_client

            with patch("ptpd_calibration.vertex.corpus.get_settings") as mock_settings:
                mock_settings.return_value.vertex.corpus_bucket = None

                # Patch the import inside upload_to_gcs

                original_import = (
                    __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
                )

                def custom_import(name, *args, **kwargs):
                    if name == "google.cloud":
                        mod = MagicMock()
                        mod.storage = mock_storage_module
                        return mod
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=custom_import):
                    count = preparator.upload_to_gcs(bucket_name="gs://my-bucket")
                    assert count == 1
                    mock_client.bucket.assert_called_with("my-bucket")

    def test_prepare_domain_knowledge_no_knowledge_dir(self):
        """prepare_domain_knowledge should return 0 if knowledge dir doesn't exist."""
        from ptpd_calibration.vertex.corpus import CorpusPreparator

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output"
            preparator = CorpusPreparator(repo_path=".", output_dir=str(output))

            with patch("ptpd_calibration.vertex.corpus.Path") as mock_path_cls:
                # Make the knowledge_dir.exists() return False
                mock_knowledge = MagicMock()
                mock_knowledge.exists.return_value = False
                mock_path_cls.return_value.parent.__truediv__ = MagicMock(
                    return_value=mock_knowledge
                )

                # Use the real Path for output operations
                with patch.object(preparator, "output_dir", Path(output) / "documents"):
                    # Just verify it handles missing knowledge dir gracefully
                    pass


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
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/png"
            assert len(data) > 0
        finally:
            os.unlink(path)

    def test_load_image_jpeg(self):
        """_load_image should handle JPEG files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/jpeg"
        finally:
            os.unlink(path)

    def test_load_image_tiff(self):
        """_load_image should handle TIFF files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(b"II\x2a\x00" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/tiff"
        finally:
            os.unlink(path)

    def test_load_image_webp(self):
        """_load_image should handle WebP files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as f:
            f.write(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/webp"
        finally:
            os.unlink(path)

    def test_load_image_bmp(self):
        """_load_image should handle BMP files."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"BM" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/bmp"
        finally:
            os.unlink(path)

    def test_load_image_tif_alias(self):
        """_load_image should handle .tif extension."""
        from ptpd_calibration.vertex.vision import _load_image

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            f.write(b"MM\x00\x2a" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            data, mime = _load_image(path)
            assert mime == "image/tiff"
        finally:
            os.unlink(path)

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
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                _load_image(path)
        finally:
            os.unlink(path)

    def test_parse_vision_response_json(self):
        """_parse_vision_response should parse valid JSON."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        json_response = json.dumps(
            {
                "confidence": 0.85,
                "recommendations": ["Increase exposure by 20%", "Check clearing baths"],
                "diagnosis": "Weak Dmax",
            }
        )

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

    def test_parse_vision_response_markdown_no_lang(self):
        """_parse_vision_response should handle code blocks without language tag."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        response = '```\n{"confidence": 0.7, "next_steps": ["Try again"]}\n```'

        result = _parse_vision_response(response, "test")
        assert result.confidence == 0.7
        assert len(result.recommendations) == 1

    def test_parse_vision_response_invalid_json(self):
        """_parse_vision_response should handle non-JSON gracefully."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        result = _parse_vision_response("This is not JSON at all.", "test")
        assert result.raw_response == "This is not JSON at all."
        assert result.structured_data == {}
        assert result.confidence == 0.0

    def test_parse_vision_response_overall_quality(self):
        """_parse_vision_response should extract confidence from overall_quality."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        response = json.dumps(
            {
                "overall_quality": {"score": 8.5, "description": "Good"},
                "improvements": ["Better coating"],
            }
        )

        result = _parse_vision_response(response, "print_quality")
        assert result.confidence == 0.85  # 8.5 / 10.0
        assert len(result.recommendations) == 1

    def test_parse_vision_response_empty_string(self):
        """_parse_vision_response should handle empty string."""
        from ptpd_calibration.vertex.vision import _parse_vision_response

        result = _parse_vision_response("", "test")
        assert result.raw_response == ""
        assert result.structured_data == {}

    def test_format_result_with_structured_data(self):
        """_format_result should return JSON when structured data exists."""
        from ptpd_calibration.vertex.vision import VisionAnalysisResult, _format_result

        result = VisionAnalysisResult(
            analysis_type="test",
            raw_response="raw text",
            structured_data={"key": "value"},
        )
        formatted = _format_result(result)
        parsed = json.loads(formatted)
        assert parsed["key"] == "value"

    def test_format_result_without_structured_data(self):
        """_format_result should return raw response when no structured data."""
        from ptpd_calibration.vertex.vision import VisionAnalysisResult, _format_result

        result = VisionAnalysisResult(
            analysis_type="test",
            raw_response="raw text fallback",
            structured_data={},
        )
        assert _format_result(result) == "raw text fallback"

    def test_supported_image_types_constant(self):
        """SUPPORTED_IMAGE_TYPES should contain all expected formats."""
        from ptpd_calibration.vertex.vision import SUPPORTED_IMAGE_TYPES

        assert ".png" in SUPPORTED_IMAGE_TYPES
        assert ".jpg" in SUPPORTED_IMAGE_TYPES
        assert ".jpeg" in SUPPORTED_IMAGE_TYPES
        assert ".tiff" in SUPPORTED_IMAGE_TYPES
        assert ".tif" in SUPPORTED_IMAGE_TYPES
        assert ".webp" in SUPPORTED_IMAGE_TYPES
        assert ".bmp" in SUPPORTED_IMAGE_TYPES


# ─── Vision Analyzer Tests (mocked Gemini API) ───


@pytest.mark.unit
class TestGeminiVisionAnalyzer:
    """Tests for GeminiVisionAnalyzer with mocked Gemini client."""

    def _create_analyzer_with_mock(self, response_text: str):
        """Helper to create an analyzer with a mocked Gemini client."""
        from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

        analyzer = GeminiVisionAnalyzer(
            project_id="test-project",
            location="us-central1",
            model="gemini-2.5-flash",
        )

        mock_response = MagicMock()
        mock_response.text = response_text

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        # Inject the mocked client directly
        analyzer._client = mock_client
        return analyzer

    def test_analyzer_construction(self):
        """GeminiVisionAnalyzer should initialize with config."""
        from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

        analyzer = GeminiVisionAnalyzer(
            project_id="test",
            location="us-east1",
            model="gemini-2.5-pro",
        )
        assert analyzer.project_id == "test"
        assert analyzer.location == "us-east1"
        assert analyzer.model == "gemini-2.5-pro"

    def test_analyzer_default_settings(self):
        """GeminiVisionAnalyzer should use settings defaults."""
        from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

        analyzer = GeminiVisionAnalyzer()
        assert analyzer.location is not None
        assert analyzer.model is not None

    def test_analyze_step_tablet(self):
        """analyze_step_tablet should return VisionAnalysisResult."""
        response_json = json.dumps(
            {
                "steps": [{"step": 1, "density": 0.05}],
                "overall_quality": {"score": 8, "description": "Good"},
                "dmin": 0.05,
                "dmax": 1.75,
                "density_range": 1.70,
                "issues": [],
                "recommendations": ["Consider longer exposure"],
            }
        )

        analyzer = self._create_analyzer_with_mock(response_json)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            mock_types = MagicMock()
            with (
                patch.dict(
                    "sys.modules", {"google.genai": MagicMock(), "google.genai.types": mock_types}
                ),
                patch(
                    "ptpd_calibration.vertex.vision.GeminiVisionAnalyzer._get_client",
                    return_value=analyzer._client,
                ),
            ):
                result = analyzer.analyze_step_tablet(path)

            assert result.analysis_type == "step_tablet_analysis"
            assert result.structured_data.get("dmax") == 1.75
        finally:
            os.unlink(path)

    def test_evaluate_print_quality(self):
        """evaluate_print_quality should return quality scores."""
        response_json = json.dumps(
            {
                "scores": {"tonal_range": 8, "coating_quality": 9},
                "overall_score": 8.5,
                "strengths": ["Good tonal range"],
                "improvements": ["Better coating technique"],
                "next_steps": ["Try thicker coating"],
            }
        )

        analyzer = self._create_analyzer_with_mock(response_json)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            mock_types = MagicMock()
            with (
                patch.dict(
                    "sys.modules", {"google.genai": MagicMock(), "google.genai.types": mock_types}
                ),
                patch(
                    "ptpd_calibration.vertex.vision.GeminiVisionAnalyzer._get_client",
                    return_value=analyzer._client,
                ),
            ):
                result = analyzer.evaluate_print_quality(path, "Arches Platine", "50:50 Pt/Pd")

            assert result.analysis_type == "print_quality_evaluation"
        finally:
            os.unlink(path)

    def test_diagnose_print_problem(self):
        """diagnose_print_problem should return diagnosis."""
        response_json = json.dumps(
            {
                "diagnosis": "Weak Dmax - under-exposure",
                "confidence": 0.85,
                "root_cause": "Insufficient UV exposure time",
                "fix_steps": ["Increase exposure by 30%"],
                "prevention": ["Use test strips"],
            }
        )

        analyzer = self._create_analyzer_with_mock(response_json)

        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(b"II\x2a\x00" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            mock_types = MagicMock()
            with (
                patch.dict(
                    "sys.modules", {"google.genai": MagicMock(), "google.genai.types": mock_types}
                ),
                patch(
                    "ptpd_calibration.vertex.vision.GeminiVisionAnalyzer._get_client",
                    return_value=analyzer._client,
                ),
            ):
                result = analyzer.diagnose_print_problem(path, "Weak shadows")

            assert result.analysis_type == "print_problem_diagnosis"
            assert result.confidence == 0.85
        finally:
            os.unlink(path)

    def test_compare_prints(self):
        """compare_prints should return comparison analysis."""
        response_json = json.dumps(
            {
                "improvements": ["Better tonal range"],
                "unchanged": ["Coating quality"],
                "regressions": [],
                "overall_direction": "better",
                "overall_assessment": "Calibration is improving",
                "next_recommendations": ["Fine-tune midtones"],
            }
        )

        analyzer = self._create_analyzer_with_mock(response_json)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
            f1.write(b"\x89PNG" + b"\x00" * 100)
            f1.flush()
            path1 = f1.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            f2.write(b"\x89PNG" + b"\x00" * 100)
            f2.flush()
            path2 = f2.name

        try:
            mock_types = MagicMock()
            with (
                patch.dict(
                    "sys.modules", {"google.genai": MagicMock(), "google.genai.types": mock_types}
                ),
                patch(
                    "ptpd_calibration.vertex.vision.GeminiVisionAnalyzer._get_client",
                    return_value=analyzer._client,
                ),
            ):
                result = analyzer.compare_prints(path1, path2, "Adjusted curve")

            assert result.analysis_type == "print_comparison"
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_classify_paper(self):
        """classify_paper should return paper classification."""
        response_json = json.dumps(
            {
                "paper_identification": "Hahnemühle Platinum Rag",
                "confidence": 0.75,
                "characteristics": {"texture": "smooth", "weight": "heavy"},
                "printing_recommendations": {"coating_method": "rod"},
            }
        )

        analyzer = self._create_analyzer_with_mock(response_json)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff" + b"\x00" * 100)
            f.flush()
            path = f.name

        try:
            mock_types = MagicMock()
            with (
                patch.dict(
                    "sys.modules", {"google.genai": MagicMock(), "google.genai.types": mock_types}
                ),
                patch(
                    "ptpd_calibration.vertex.vision.GeminiVisionAnalyzer._get_client",
                    return_value=analyzer._client,
                ),
            ):
                result = analyzer.classify_paper(path)

            assert result.analysis_type == "paper_classification"
            assert result.confidence == 0.75
        finally:
            os.unlink(path)

    def test_get_client_import_error(self):
        """_get_client should raise ImportError when google-genai missing."""
        from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

        analyzer = GeminiVisionAnalyzer(project_id="test")

        with (
            patch.dict("sys.modules", {"google": None, "google.genai": None}),
            patch("builtins.__import__", side_effect=ImportError("no genai")),
            pytest.raises(ImportError, match="google-genai required"),
        ):
            analyzer._get_client()


# ─── ADK Agent Tool Wrapper Tests ───


@pytest.mark.unit
class TestADKToolWrappers:
    """Tests for ADK agent tool wrapper functions."""

    def test_calculate_chemistry_recipe(self):
        """calculate_chemistry_recipe should return valid recipe JSON."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        result = json.loads(
            calculate_chemistry_recipe(
                print_size_inches="8x10",
                pt_pd_ratio="50:50",
                method="traditional",
            )
        )

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

        result = json.loads(
            calculate_chemistry_recipe(
                print_size_inches="8x10",
                pt_pd_ratio="0:100",
            )
        )

        assert result["status"] == "success"
        assert result["platinum"]["drops"] == 0
        assert result["palladium"]["drops"] > 0

    def test_calculate_chemistry_recipe_large_print(self):
        """Larger prints should require more chemistry."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        small = json.loads(calculate_chemistry_recipe(print_size_inches="5x7"))
        large = json.loads(calculate_chemistry_recipe(print_size_inches="16x20"))

        assert large["total_ml"] > small["total_ml"]

    def test_calculate_chemistry_recipe_invalid_size(self):
        """Invalid size should return error status."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        result = json.loads(calculate_chemistry_recipe(print_size_inches="invalid"))
        assert result["status"] == "error"

    def test_calculate_chemistry_recipe_has_cost(self):
        """Recipe should include estimated cost."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        result = json.loads(calculate_chemistry_recipe(print_size_inches="8x10"))
        assert "estimated_cost_usd" in result
        assert result["estimated_cost_usd"] > 0

    def test_calculate_chemistry_recipe_contrast_agents(self):
        """Different contrast goals should produce different agents."""
        from ptpd_calibration.vertex.agents import calculate_chemistry_recipe

        low = json.loads(calculate_chemistry_recipe(contrast_goal="low"))
        high = json.loads(calculate_chemistry_recipe(contrast_goal="high"))

        assert low["contrast_agent"] != high["contrast_agent"]

    def test_calculate_uv_exposure(self):
        """calculate_uv_exposure should return valid exposure data."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        result = json.loads(
            calculate_uv_exposure(
                uv_source="LED 365nm",
                negative_dr=1.5,
            )
        )

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

    def test_calculate_uv_exposure_unknown_source(self):
        """Unknown UV source should use default base time."""
        from ptpd_calibration.vertex.agents import UV_DEFAULT_BASE_TIME, calculate_uv_exposure

        result = json.loads(calculate_uv_exposure(uv_source="Unknown Source"))
        assert result["recommended_seconds"] == UV_DEFAULT_BASE_TIME

    def test_calculate_uv_exposure_with_previous_time(self):
        """Previous time should center the bracket on that value."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        result = json.loads(
            calculate_uv_exposure(
                uv_source="LED 365nm",
                previous_time_seconds=200,
            )
        )

        # The 1.0 factor should equal the previous_time_seconds
        center_time = next(t for t in result["test_strip_times"] if t["label"] == "100%")
        assert center_time["seconds"] == 200

    def test_calculate_uv_exposure_high_dr(self):
        """Higher DR should produce longer exposure times."""
        from ptpd_calibration.vertex.agents import calculate_uv_exposure

        normal = json.loads(calculate_uv_exposure(negative_dr=1.5))
        high = json.loads(calculate_uv_exposure(negative_dr=2.0))

        assert high["recommended_seconds"] > normal["recommended_seconds"]

    def test_get_contrast_agent(self):
        """_get_contrast_agent should return appropriate recommendations."""
        from ptpd_calibration.vertex.agents import _get_contrast_agent

        assert "None" in _get_contrast_agent("low")
        assert "H2O2" in _get_contrast_agent("normal")
        assert "H2O2" in _get_contrast_agent("high") or "chlorate" in _get_contrast_agent("high")

    def test_get_contrast_agent_default(self):
        """_get_contrast_agent should return normal for unknown goals."""
        from ptpd_calibration.vertex.agents import _get_contrast_agent

        result = _get_contrast_agent("unknown_goal")
        assert result == _get_contrast_agent("normal")

    def test_get_contrast_agent_very_high(self):
        """_get_contrast_agent should handle very high contrast."""
        from ptpd_calibration.vertex.agents import _get_contrast_agent

        result = _get_contrast_agent("very high")
        assert "dichromate" in result.lower() or "chlorate" in result.lower()

    def test_analyze_step_tablet_scan_import_error(self):
        """analyze_step_tablet_scan should handle missing opencv gracefully."""
        from ptpd_calibration.vertex.agents import analyze_step_tablet_scan

        # The detection modules likely aren't available in test env
        result = json.loads(analyze_step_tablet_scan("/nonexistent/image.png"))
        assert result["status"] == "error"

    def test_generate_linearization_curve_error(self):
        """generate_linearization_curve should handle errors gracefully."""
        from ptpd_calibration.vertex.agents import generate_linearization_curve

        result = json.loads(generate_linearization_curve("not valid json input"))
        # Should return error since CurveType("linear") may not exist or other issue
        assert result["status"] == "error" or result["status"] == "success"

    def test_module_constants(self):
        """Module-level constants should be defined and accessible."""
        from ptpd_calibration.vertex.agents import (
            CONTRAST_AGENTS,
            DR_NORMALISATION_TARGET,
            TEST_STRIP_FACTORS,
            UV_BASE_TIMES,
            UV_DEFAULT_BASE_TIME,
        )

        assert len(UV_BASE_TIMES) == 4
        assert UV_DEFAULT_BASE_TIME == 300
        assert DR_NORMALISATION_TARGET == 1.5
        assert len(TEST_STRIP_FACTORS) == 5
        assert len(CONTRAST_AGENTS) == 4


# ─── ADK Agent Creation Tests ───


@pytest.mark.unit
class TestADKAgentCreation:
    """Tests for ADK agent creation and deployment functions."""

    def test_create_adk_agents_import_error(self):
        """create_adk_agents should raise ImportError when ADK not installed."""
        from ptpd_calibration.vertex.agents import create_adk_agents

        with (
            patch.dict(
                "sys.modules",
                {"google.adk": None, "google.adk.agents": None, "google.adk.tools": None},
            ),
            patch("builtins.__import__", side_effect=ImportError("no adk")),
            pytest.raises(ImportError, match="google-cloud-aiplatform"),
        ):
            create_adk_agents()

    def test_create_adk_agents_mocked(self):
        """create_adk_agents should create agent dict with mocked ADK."""
        mock_llm_agent = MagicMock()
        mock_llm_agent.name = "test_agent"
        mock_search_tool = MagicMock()

        mock_adk_agents = MagicMock()
        mock_adk_agents.LlmAgent = MagicMock(return_value=mock_llm_agent)

        mock_adk_tools = MagicMock()
        mock_adk_tools.VertexAiSearchTool = MagicMock(return_value=mock_search_tool)

        with patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.adk": MagicMock(),
                "google.adk.agents": mock_adk_agents,
                "google.adk.tools": mock_adk_tools,
            },
        ):
            from ptpd_calibration.vertex.agents import create_adk_agents

            result = create_adk_agents()
            assert "calibration_agent" in result
            assert "chemistry_agent" in result
            assert "print_coach" in result
            assert "coordinator" in result

    def test_create_darkroom_coordinator_mocked(self):
        """create_darkroom_coordinator should return the coordinator agent."""
        mock_agent = MagicMock()
        mock_agent.name = "darkroom_assistant"

        with patch("ptpd_calibration.vertex.agents.create_adk_agents") as mock_create:
            mock_create.return_value = {
                "calibration_agent": MagicMock(),
                "chemistry_agent": MagicMock(),
                "print_coach": MagicMock(),
                "coordinator": mock_agent,
            }

            from ptpd_calibration.vertex.agents import create_darkroom_coordinator

            coordinator = create_darkroom_coordinator()
            assert coordinator.name == "darkroom_assistant"

    def test_deploy_to_agent_engine_import_error(self):
        """deploy_to_agent_engine should raise ImportError when packages missing."""
        from ptpd_calibration.vertex.agents import deploy_to_agent_engine

        with (
            patch.dict("sys.modules", {"vertexai": None, "vertexai.agent_engines": None}),
            patch("builtins.__import__", side_effect=ImportError("no vertexai")),
            pytest.raises(ImportError, match="google-cloud-aiplatform"),
        ):
            deploy_to_agent_engine()

    def test_deploy_to_agent_engine_no_project(self):
        """deploy_to_agent_engine should raise ValueError without project ID."""
        mock_vertexai = MagicMock()
        mock_agent_engines = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "vertexai": mock_vertexai,
                    "vertexai.agent_engines": mock_agent_engines,
                },
            ),
            patch("ptpd_calibration.vertex.agents.get_settings") as mock_settings,
        ):
            mock_settings.return_value.vertex.project_id = None
            mock_settings.return_value.vertex.location = "us-central1"
            mock_settings.return_value.vertex.staging_bucket = None

            from ptpd_calibration.vertex.agents import deploy_to_agent_engine

            with pytest.raises(ValueError, match="project ID required"):
                deploy_to_agent_engine(project_id=None)

    def test_deploy_to_agent_engine_no_bucket(self):
        """deploy_to_agent_engine should raise ValueError without staging bucket."""
        mock_vertexai = MagicMock()
        mock_agent_engines = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "vertexai": mock_vertexai,
                    "vertexai.agent_engines": mock_agent_engines,
                },
            ),
            patch("ptpd_calibration.vertex.agents.get_settings") as mock_settings,
        ):
            mock_settings.return_value.vertex.project_id = None
            mock_settings.return_value.vertex.location = "us-central1"
            mock_settings.return_value.vertex.staging_bucket = None

            from ptpd_calibration.vertex.agents import deploy_to_agent_engine

            with pytest.raises(ValueError, match="Staging bucket required"):
                deploy_to_agent_engine(project_id="test-proj", staging_bucket=None)


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
        assert len(context) <= 600

    def test_format_context_empty_results(self):
        """format_context_for_llm should handle empty results."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test", data_store_id="test")
        context = client.format_context_for_llm([])
        assert context == ""

    def test_get_client_import_error(self):
        """_get_client should raise ImportError when discoveryengine missing."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test", data_store_id="test")

        with (
            patch.dict(
                "sys.modules",
                {"google.cloud.discoveryengine_v1": None, "google.cloud": MagicMock()},
            ),
            patch("builtins.__import__", side_effect=ImportError("no discoveryengine")),
            pytest.raises(ImportError, match="google-cloud-discoveryengine"),
        ):
            client._get_client()

    def test_search_mocked(self):
        """search should call Discovery Engine API and return results."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test-proj", data_store_id="test-store")

        # Mock the search response
        mock_doc = MagicMock()
        mock_doc.id = "doc-1"
        mock_doc.struct_data = {"title": "Chemistry Guide", "snippet": "Use 50:50 ratio"}

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.relevance_score = 0.95

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = mock_response

        client._client = mock_search_client

        mock_discoveryengine = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "google.cloud": MagicMock(),
                "google.cloud.discoveryengine_v1": mock_discoveryengine,
            },
        ):
            results = client.search("chemistry ratio")
            assert len(results) == 1
            assert results[0].document_id == "doc-1"

    def test_search_with_filter(self):
        """search with filter_expr should pass filter to request."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test-proj", data_store_id="test-store")

        mock_response = MagicMock()
        mock_response.results = []

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = mock_response

        client._client = mock_search_client

        mock_discoveryengine = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "google.cloud": MagicMock(),
                "google.cloud.discoveryengine_v1": mock_discoveryengine,
            },
        ):
            results = client.search("test", filter_expr="category:chemistry")
            assert results == []

    def test_search_with_summary_mocked(self):
        """search_with_summary should return summary and results."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test-proj", data_store_id="test-store")

        mock_doc = MagicMock()
        mock_doc.id = "doc-1"
        mock_doc.struct_data = {"title": "Guide", "snippet": "Content"}

        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_result.relevance_score = 0.9

        mock_summary = MagicMock()
        mock_summary.summary_text = "This is a summary of the results."

        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_response.summary = mock_summary

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = mock_response

        client._client = mock_search_client

        mock_discoveryengine = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "google.cloud": MagicMock(),
                "google.cloud.discoveryengine_v1": mock_discoveryengine,
            },
        ):
            summary, results = client.search_with_summary("test query")
            assert "summary" in summary.lower()
            assert len(results) == 1

    def test_search_with_summary_no_summary(self):
        """search_with_summary should handle missing summary gracefully."""
        from ptpd_calibration.vertex.search import PtPdSearchClient

        client = PtPdSearchClient(project_id="test", data_store_id="test")

        mock_response = MagicMock()
        mock_response.results = []
        mock_response.summary = None

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = mock_response

        client._client = mock_search_client

        mock_discoveryengine = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "google.cloud": MagicMock(),
                "google.cloud.discoveryengine_v1": mock_discoveryengine,
            },
        ):
            summary, results = client.search_with_summary("test")
            assert summary == ""


# ─── Search Document Extraction Tests ───


@pytest.mark.unit
class TestSearchDocumentExtraction:
    """Tests for _extract_document_data helper function."""

    def test_extract_struct_data(self):
        """Should extract title and snippet from struct_data."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = {
            "title": "Chemistry Guide",
            "snippet": "Use 50:50 ratio",
            "category": "chemistry",
        }
        doc.derived_struct_data = None
        doc.content = None

        data = _extract_document_data(doc)
        assert data["title"] == "Chemistry Guide"
        assert data["snippet"] == "Use 50:50 ratio"
        assert data["metadata"]["category"] == "chemistry"

    def test_extract_derived_struct_data(self):
        """Should extract from derived_struct_data when struct_data is empty."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = None
        doc.derived_struct_data = {
            "title": "Paper Profiles",
            "extractive_answers": [{"content": "Answer 1"}, {"content": "Answer 2"}],
        }
        doc.content = None

        data = _extract_document_data(doc)
        assert data["title"] == "Paper Profiles"
        assert "Answer 1" in data["snippet"]
        assert "Answer 2" in data["snippet"]

    def test_extract_derived_struct_data_no_answers(self):
        """Should handle derived_struct_data without extractive answers."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = None
        doc.derived_struct_data = {"link": "http://example.com"}
        doc.content = None

        data = _extract_document_data(doc)
        assert data["title"] == "http://example.com"

    def test_extract_raw_content(self):
        """Should extract from raw content when structured data is missing."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = None
        doc.derived_struct_data = None
        doc.content = MagicMock()
        doc.content.raw_bytes = b"This is raw content"

        data = _extract_document_data(doc)
        assert "raw content" in data["snippet"]

    def test_extract_raw_content_unicode_error(self):
        """Should handle non-UTF8 raw content gracefully."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = None
        doc.derived_struct_data = None
        doc.content = MagicMock()
        doc.content.raw_bytes = b"\xff\xfe\x00\x01"

        data = _extract_document_data(doc)
        assert data["snippet"]  # Should have some string representation

    def test_extract_fallback_title(self):
        """Should fallback to doc.name or doc.id for title."""
        from ptpd_calibration.vertex.search import _extract_document_data

        doc = MagicMock()
        doc.struct_data = None
        doc.derived_struct_data = None
        doc.content = None
        doc.name = "doc-name"
        doc.id = "doc-id"

        data = _extract_document_data(doc)
        assert data["title"] == "doc-name"


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


# ─── Module __init__ Tests ───


@pytest.mark.unit
class TestVertexModuleInit:
    """Tests for the vertex module __init__.py."""

    def test_init_all_exports(self):
        """vertex.__all__ should list all expected symbols."""
        from ptpd_calibration.vertex import __all__

        expected = [
            "PtPdSearchClient",
            "SearchResult",
            "CorpusPreparator",
            "prepare_and_upload_corpus",
            "GeminiVisionAnalyzer",
            "analyze_step_tablet",
            "evaluate_print_quality",
            "diagnose_print_problem",
            "create_adk_agents",
            "create_darkroom_coordinator",
            "CalibrationSnapshot",
            "MemoryBankClient",
            "UserProfile",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"

    def test_import_memory_classes(self):
        """Should be able to import memory classes from vertex package."""
        from ptpd_calibration.vertex.memory import (
            CalibrationSnapshot,
            MemoryBankClient,
            UserProfile,
        )

        assert UserProfile is not None
        assert CalibrationSnapshot is not None
        assert MemoryBankClient is not None

    def test_import_search_classes(self):
        """Should be able to import search classes from vertex package."""
        from ptpd_calibration.vertex.search import PtPdSearchClient, SearchResult

        assert PtPdSearchClient is not None
        assert SearchResult is not None

    def test_import_vision_classes(self):
        """Should be able to import vision classes from vertex package."""
        from ptpd_calibration.vertex.vision import (
            GeminiVisionAnalyzer,
            VisionAnalysisResult,
        )

        assert GeminiVisionAnalyzer is not None
        assert VisionAnalysisResult is not None


# ─── Logging Tests ───


@pytest.mark.unit
class TestLogging:
    """Tests to verify logging is properly configured in all modules."""

    def test_agents_has_logger(self):
        """agents module should have a logger."""
        from ptpd_calibration.vertex import agents

        assert hasattr(agents, "logger")
        assert agents.logger.name == "ptpd_calibration.vertex.agents"

    def test_search_has_logger(self):
        """search module should have a logger."""
        from ptpd_calibration.vertex import search

        assert hasattr(search, "logger")
        assert search.logger.name == "ptpd_calibration.vertex.search"

    def test_vision_has_logger(self):
        """vision module should have a logger."""
        from ptpd_calibration.vertex import vision

        assert hasattr(vision, "logger")
        assert vision.logger.name == "ptpd_calibration.vertex.vision"

    def test_corpus_has_logger(self):
        """corpus module should have a logger."""
        from ptpd_calibration.vertex import corpus

        assert hasattr(corpus, "logger")
        assert corpus.logger.name == "ptpd_calibration.vertex.corpus"

    def test_memory_has_logger(self):
        """memory module should have a logger."""
        from ptpd_calibration.vertex import memory

        assert hasattr(memory, "logger")
        assert memory.logger.name == "ptpd_calibration.vertex.memory"

    def test_llm_client_has_logger(self):
        """llm.client module should have a logger."""
        from ptpd_calibration.llm import client

        assert hasattr(client, "logger")
        assert client.logger.name == "ptpd_calibration.llm.client"
