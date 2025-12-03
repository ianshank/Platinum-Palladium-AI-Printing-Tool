"""
Tests for paper profiles module.

Tests paper characteristics, profiles, and database functionality.
"""

import pytest

from ptpd_calibration.papers.profiles import (
    CoatingBehavior,
    PaperCharacteristics,
    PaperProfile,
    PaperDatabase,
    BUILTIN_PAPERS,
)


class TestCoatingBehavior:
    """Tests for CoatingBehavior enum."""

    def test_coating_behavior_values(self):
        """All coating behaviors should be defined."""
        assert CoatingBehavior.ABSORBS_QUICKLY.value == "absorbs_quickly"
        assert CoatingBehavior.ABSORBS_SLOWLY.value == "absorbs_slowly"
        assert CoatingBehavior.POOLS.value == "pools"
        assert CoatingBehavior.SPREADS_EVENLY.value == "spreads_evenly"


class TestPaperCharacteristics:
    """Tests for PaperCharacteristics dataclass."""

    def test_default_characteristics(self):
        """Default characteristics should be sensible."""
        chars = PaperCharacteristics()
        assert chars.surface == "smooth"
        assert chars.weight_gsm == 300
        assert chars.typical_dmax > chars.typical_dmin

    def test_custom_characteristics(self):
        """Custom characteristics should be applied."""
        chars = PaperCharacteristics(
            surface="textured",
            weight_gsm=400,
            typical_dmax=1.8,
        )
        assert chars.surface == "textured"
        assert chars.weight_gsm == 400
        assert chars.typical_dmax == 1.8

    def test_to_dict(self):
        """Should serialize to dictionary."""
        chars = PaperCharacteristics()
        d = chars.to_dict()
        assert "surface" in d
        assert "weight_gsm" in d
        assert "typical_dmax" in d

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "surface": "rough",
            "weight_gsm": 500,
            "typical_dmax": 2.0,
        }
        chars = PaperCharacteristics.from_dict(data)
        assert chars.surface == "rough"
        assert chars.weight_gsm == 500


class TestPaperProfile:
    """Tests for PaperProfile dataclass."""

    def test_default_profile(self):
        """Default profile should be created."""
        profile = PaperProfile()
        assert profile.id is not None
        assert profile.characteristics is not None

    def test_custom_profile(self):
        """Custom profile should be created."""
        profile = PaperProfile(
            name="Arches Platine",
            manufacturer="Arches",
            absorbency="high",
            recommended_pt_ratio=0.5,
        )
        assert profile.name == "Arches Platine"
        assert profile.manufacturer == "Arches"
        assert profile.absorbency == "high"
        assert profile.recommended_pt_ratio == 0.5

    def test_profile_with_characteristics(self):
        """Profile should accept custom characteristics."""
        chars = PaperCharacteristics(
            surface="textured",
            weight_gsm=310,
        )
        profile = PaperProfile(
            name="Custom Paper",
            characteristics=chars,
        )
        assert profile.characteristics.surface == "textured"
        assert profile.characteristics.weight_gsm == 310

    def test_profile_to_dict(self):
        """Profile should serialize to dictionary."""
        profile = PaperProfile(name="Test Paper")
        d = profile.to_dict()
        assert "name" in d
        assert "characteristics" in d
        assert "absorbency" in d

    def test_profile_from_dict(self):
        """Profile should deserialize from dictionary."""
        data = {
            "name": "Test Paper",
            "manufacturer": "Test Mfg",
            "absorbency": "low",
        }
        profile = PaperProfile.from_dict(data)
        assert profile.name == "Test Paper"
        assert profile.manufacturer == "Test Mfg"


class TestPaperDatabase:
    """Tests for PaperDatabase class."""

    @pytest.fixture
    def database(self, tmp_path):
        """Create database with temp storage."""
        custom_file = tmp_path / "custom_papers.json"
        return PaperDatabase(custom_papers_file=custom_file)

    @pytest.fixture
    def sample_profiles(self):
        """Create sample paper profiles."""
        return [
            PaperProfile(name="Paper A", manufacturer="Mfg 1"),
            PaperProfile(name="Paper B", manufacturer="Mfg 2"),
            PaperProfile(name="Paper C", manufacturer="Mfg 1"),
        ]

    def test_database_has_builtin_papers(self, database):
        """Database should include built-in papers."""
        papers = database.list_papers()
        # Should have at least the built-in papers
        assert len(papers) >= len(BUILTIN_PAPERS)

    def test_add_custom_profile(self, database):
        """Adding profile should store it."""
        initial_count = len(database.list_papers())
        profile = PaperProfile(name="Test Custom")
        database.add_profile(profile)
        assert len(database.list_papers()) == initial_count + 1

    def test_get_profile_by_name(self, database, sample_profiles):
        """Should retrieve profile by name."""
        for p in sample_profiles:
            database.add_profile(p)

        result = database.get_by_name("Paper B")
        assert result is not None
        assert result.name == "Paper B"

    def test_get_profile_not_found(self, database):
        """Should return None for non-existent profile."""
        result = database.get_by_name("Nonexistent Paper XYZ")
        assert result is None

    def test_search_by_manufacturer(self, database, sample_profiles):
        """Should find profiles by manufacturer."""
        for p in sample_profiles:
            database.add_profile(p)

        results = database.search_papers(manufacturer="Mfg 1")
        assert len(results) == 2

    def test_remove_custom_profile(self, database, sample_profiles):
        """Should remove custom profile."""
        for p in sample_profiles:
            database.add_profile(p)

        initial_count = len(database.list_papers())
        result = database.remove_custom_paper("Paper A")
        assert result is True
        assert len(database.list_papers()) == initial_count - 1

    def test_save_and_load_custom(self, tmp_path, sample_profiles):
        """Should persist custom papers across database instances."""
        custom_file = tmp_path / "test_papers.json"

        # Create database and add custom papers
        db1 = PaperDatabase(custom_papers_file=custom_file)
        for p in sample_profiles:
            db1.add_profile(p)

        # Verify file was created
        assert custom_file.exists()

        # Create new database instance with same file
        db2 = PaperDatabase(custom_papers_file=custom_file)

        # Should have loaded the custom papers
        result = db2.get_paper("Paper A")
        assert result is not None
        assert result.name == "Paper A"


class TestBuiltinProfiles:
    """Tests for built-in paper profiles."""

    def test_builtin_papers_exist(self):
        """Should have built-in papers defined."""
        assert len(BUILTIN_PAPERS) > 0

    def test_builtin_profile_structure(self):
        """Built-in profiles should have proper structure."""
        for key, profile in BUILTIN_PAPERS.items():
            assert profile.name
            assert profile.characteristics is not None
            assert profile.characteristics.typical_dmax > 0

    def test_arches_platine_exists(self):
        """Arches Platine should be a built-in paper."""
        assert "arches_platine" in BUILTIN_PAPERS
        profile = BUILTIN_PAPERS["arches_platine"]
        assert profile.name == "Arches Platine"
        assert profile.manufacturer == "Arches"
