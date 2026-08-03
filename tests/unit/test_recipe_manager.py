"""
Comprehensive tests for recipe management and workflow automation.

Tests cover:
- PrintRecipe model validation and serialization
- RecipeManager CRUD operations, search, and comparison
- WorkflowAutomation job execution and scheduling
- RecipeDatabase persistence and querying
"""

import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
import yaml

from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
from ptpd_calibration.workflow.recipe_manager import (
    PrintRecipe,
    RecipeDatabase,
    RecipeFormat,
    RecipeManager,
    WorkflowAutomation,
    WorkflowJob,
    WorkflowStatus,
    WorkflowStep,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_recipes.db"


@pytest.fixture
def recipe_db(temp_db_path):
    """Create a fresh recipe database for testing."""
    return RecipeDatabase(db_path=temp_db_path)


@pytest.fixture
def recipe_manager(recipe_db):
    """Create a recipe manager with a test database."""
    return RecipeManager(database=recipe_db)


@pytest.fixture
def sample_recipe_data():
    """Sample recipe data for testing."""
    return {
        "name": "Test Recipe",
        "paper_type": "Arches Platine",
        "chemistry_type": ChemistryType.PLATINUM_PALLADIUM,
        "pt_pd_ratio": 0.5,
        "ferric_oxalate_1_drops": 24.0,
        "ferric_oxalate_2_drops": 0.0,
        "metal_drops": 24.0,
        "contrast_agent": ContrastAgent.NA2,
        "contrast_agent_drops": 5.0,
        "developer": DeveloperType.POTASSIUM_OXALATE,
        "developer_temperature_f": 68.0,
        "development_time_minutes": 2.0,
        "exposure_time_minutes": 10.0,
        "uv_source": "UV LED",
        "tags": ["test", "platinum"],
    }


@pytest.fixture
def sample_recipe(sample_recipe_data):
    """Create a sample PrintRecipe instance."""
    return PrintRecipe(**sample_recipe_data)


@pytest.fixture
def multiple_recipes(recipe_manager):
    """Create multiple recipes with different parameters."""
    recipes = []

    # Recipe 1: Arches Platine, 50% Pt/Pd
    recipes.append(
        recipe_manager.create_recipe(
            name="Arches Standard",
            paper_type="Arches Platine",
            pt_pd_ratio=0.5,
            exposure_time_minutes=10.0,
            tags=["arches", "standard"],
        )
    )

    # Recipe 2: Bergger COT320, 70% Pt/Pd
    recipes.append(
        recipe_manager.create_recipe(
            name="Bergger High Contrast",
            paper_type="Bergger COT320",
            pt_pd_ratio=0.7,
            exposure_time_minutes=15.0,
            tags=["bergger", "high-contrast"],
        )
    )

    # Recipe 3: Arches Platine, 30% Pt/Pd
    recipes.append(
        recipe_manager.create_recipe(
            name="Arches Warm Tone",
            paper_type="Arches Platine",
            pt_pd_ratio=0.3,
            exposure_time_minutes=8.0,
            tags=["arches", "warm"],
        )
    )

    # Recipe 4: Hahnemühle, 50% Pt/Pd
    recipes.append(
        recipe_manager.create_recipe(
            name="Hahnemühle Standard",
            paper_type="Hahnemühle Platinum Rag",
            pt_pd_ratio=0.5,
            exposure_time_minutes=12.0,
            tags=["hahnemuhle", "standard"],
        )
    )

    return recipes


@pytest.fixture
def workflow_automation():
    """Create a workflow automation instance."""
    return WorkflowAutomation()


# ============================================================================
# PRINTRECIPE MODEL TESTS
# ============================================================================


class TestPrintRecipe:
    """Tests for PrintRecipe model validation and serialization."""

    def test_create_recipe_minimal(self):
        """Test creating a recipe with minimal required fields."""
        recipe = PrintRecipe(name="Minimal", paper_type="Test Paper")

        assert recipe.name == "Minimal"
        assert recipe.paper_type == "Test Paper"
        assert isinstance(recipe.recipe_id, UUID)
        assert recipe.version == 1
        assert recipe.successful_prints == 0
        assert recipe.tags == []

    def test_create_recipe_full(self, sample_recipe_data):
        """Test creating a recipe with all fields."""
        recipe = PrintRecipe(**sample_recipe_data)

        assert recipe.name == "Test Recipe"
        assert recipe.paper_type == "Arches Platine"
        assert recipe.pt_pd_ratio == 0.5
        assert recipe.chemistry_type == ChemistryType.PLATINUM_PALLADIUM
        assert recipe.contrast_agent == ContrastAgent.NA2
        assert recipe.developer == DeveloperType.POTASSIUM_OXALATE
        assert recipe.exposure_time_minutes == 10.0
        assert "test" in recipe.tags
        assert "platinum" in recipe.tags

    def test_recipe_validation_name_length(self):
        """Test name length validation."""
        # Empty name should fail
        with pytest.raises(ValueError):
            PrintRecipe(name="", paper_type="Test Paper")

        # Very long name should fail (>256 chars)
        with pytest.raises(ValueError):
            PrintRecipe(name="x" * 257, paper_type="Test Paper")

    def test_recipe_validation_pt_pd_ratio(self):
        """Test pt_pd_ratio validation (0-1 range)."""
        # Valid ratios
        recipe1 = PrintRecipe(name="Test", paper_type="Paper", pt_pd_ratio=0.0)
        assert recipe1.pt_pd_ratio == 0.0

        recipe2 = PrintRecipe(name="Test", paper_type="Paper", pt_pd_ratio=1.0)
        assert recipe2.pt_pd_ratio == 1.0

        # Invalid ratios
        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", pt_pd_ratio=-0.1)

        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", pt_pd_ratio=1.1)

    def test_recipe_validation_exposure_time(self):
        """Test exposure time validation (0.1-120 minutes)."""
        # Valid times
        recipe1 = PrintRecipe(name="Test", paper_type="Paper", exposure_time_minutes=0.1)
        assert recipe1.exposure_time_minutes == 0.1

        recipe2 = PrintRecipe(name="Test", paper_type="Paper", exposure_time_minutes=120.0)
        assert recipe2.exposure_time_minutes == 120.0

        # Invalid times
        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", exposure_time_minutes=0.05)

        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", exposure_time_minutes=150.0)

    def test_recipe_validation_temperature_ranges(self):
        """Test temperature validation (40-90°F)."""
        # Valid temperatures
        recipe = PrintRecipe(
            name="Test",
            paper_type="Paper",
            developer_temperature_f=68.0,
            temperature_f=72.0,
        )
        assert recipe.developer_temperature_f == 68.0
        assert recipe.temperature_f == 72.0

        # Invalid temperatures
        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", developer_temperature_f=30.0)

        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", temperature_f=100.0)

    def test_recipe_validation_quality_rating(self):
        """Test quality rating validation (0-5)."""
        # Valid ratings
        recipe1 = PrintRecipe(name="Test", paper_type="Paper", quality_rating=0.0)
        assert recipe1.quality_rating == 0.0

        recipe2 = PrintRecipe(name="Test", paper_type="Paper", quality_rating=5.0)
        assert recipe2.quality_rating == 5.0

        # Invalid ratings
        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", quality_rating=-1.0)

        with pytest.raises(ValueError):
            PrintRecipe(name="Test", paper_type="Paper", quality_rating=6.0)

    def test_tag_normalization_from_list(self):
        """Test tag normalization from list."""
        recipe = PrintRecipe(
            name="Test",
            paper_type="Paper",
            tags=["Platinum", "ARCHES", "test", "Test"],
        )

        # Tags should be lowercase, sorted, and deduplicated
        assert recipe.tags == ["arches", "platinum", "test"]

    def test_tag_normalization_from_string(self):
        """Test tag normalization from single string."""
        recipe = PrintRecipe(name="Test", paper_type="Paper", tags="SingleTag")

        assert recipe.tags == ["singletag"]

    def test_tag_normalization_with_whitespace(self):
        """Test tag normalization removes whitespace."""
        recipe = PrintRecipe(
            name="Test",
            paper_type="Paper",
            tags=["  test  ", "platinum ", " arches"],
        )

        assert recipe.tags == ["arches", "platinum", "test"]

    def test_clone_recipe_basic(self, sample_recipe):
        """Test basic recipe cloning."""
        cloned = sample_recipe.clone()

        # New ID and timestamps
        assert cloned.recipe_id != sample_recipe.recipe_id
        assert cloned.parent_recipe_id == sample_recipe.recipe_id
        assert cloned.version == sample_recipe.version + 1

        # Same parameters
        assert cloned.name == sample_recipe.name
        assert cloned.paper_type == sample_recipe.paper_type
        assert cloned.pt_pd_ratio == sample_recipe.pt_pd_ratio

    def test_clone_recipe_with_modifications(self, sample_recipe):
        """Test cloning with modifications."""
        modifications = {
            "name": "Modified Recipe",
            "pt_pd_ratio": 0.7,
            "exposure_time_minutes": 15.0,
        }

        cloned = sample_recipe.clone(modifications)

        assert cloned.name == "Modified Recipe"
        assert cloned.pt_pd_ratio == 0.7
        assert cloned.exposure_time_minutes == 15.0
        assert cloned.paper_type == sample_recipe.paper_type  # Unchanged
        assert cloned.parent_recipe_id == sample_recipe.recipe_id

    def test_update_quality_first_print(self, sample_recipe):
        """Test updating quality metrics for first print."""
        sample_recipe.update_quality(rating=4.5, dmin=0.12, dmax=2.3)

        assert sample_recipe.successful_prints == 1
        assert sample_recipe.quality_rating == 4.5
        assert sample_recipe.dmin_achieved == 0.12
        assert sample_recipe.dmax_achieved == 2.3

    def test_update_quality_running_average(self, sample_recipe):
        """Test quality rating running average."""
        sample_recipe.update_quality(rating=4.0, dmin=0.10, dmax=2.2)
        assert sample_recipe.quality_rating == 4.0

        sample_recipe.update_quality(rating=5.0, dmin=0.11, dmax=2.4)
        assert sample_recipe.successful_prints == 2
        assert sample_recipe.quality_rating == 4.5  # (4.0 + 5.0) / 2

        sample_recipe.update_quality(rating=3.0, dmin=0.12, dmax=2.3)
        assert sample_recipe.successful_prints == 3
        assert sample_recipe.quality_rating == 4.0  # (4.0 + 5.0 + 3.0) / 3

    def test_update_quality_updates_modified_at(self, sample_recipe):
        """Test that update_quality updates the modified_at timestamp."""
        original_modified = sample_recipe.modified_at
        sample_recipe.update_quality(rating=4.0, dmin=None, dmax=None)
        assert sample_recipe.modified_at > original_modified

    def test_to_dict_serialization(self, sample_recipe):
        """Test conversion to dictionary."""
        data = sample_recipe.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == sample_recipe.name
        assert data["paper_type"] == sample_recipe.paper_type
        assert isinstance(data["recipe_id"], str)
        assert data["chemistry_type"] == sample_recipe.chemistry_type.value
        assert data["developer"] == sample_recipe.developer.value
        assert data["contrast_agent"] == sample_recipe.contrast_agent.value

    def test_from_dict_deserialization(self, sample_recipe):
        """Test creation from dictionary."""
        data = sample_recipe.to_dict()
        restored = PrintRecipe.from_dict(data)

        assert restored.name == sample_recipe.name
        assert restored.paper_type == sample_recipe.paper_type
        assert restored.pt_pd_ratio == sample_recipe.pt_pd_ratio
        assert restored.chemistry_type == sample_recipe.chemistry_type
        assert restored.developer == sample_recipe.developer

    def test_round_trip_serialization(self, sample_recipe):
        """Test complete round-trip serialization."""
        data = sample_recipe.to_dict()
        restored = PrintRecipe.from_dict(data)
        data2 = restored.to_dict()

        # Should be identical after round trip
        assert data == data2


# ============================================================================
# RECIPEMANAGER TESTS
# ============================================================================


class TestRecipeManager:
    """Tests for RecipeManager CRUD and search operations."""

    def test_create_recipe(self, recipe_manager, sample_recipe_data):
        """Test creating a new recipe."""
        recipe = recipe_manager.create_recipe(**sample_recipe_data)

        assert recipe.name == "Test Recipe"
        assert recipe.paper_type == "Arches Platine"
        assert isinstance(recipe.recipe_id, UUID)

        # Verify it was added to database
        retrieved = recipe_manager.get_recipe_by_id(recipe.recipe_id)
        assert retrieved is not None
        assert retrieved.name == recipe.name

    def test_clone_recipe(self, recipe_manager, sample_recipe):
        """Test cloning an existing recipe."""
        # Add original to database
        recipe_manager.database.add_recipe(sample_recipe)

        # Clone with modifications
        modifications = {"name": "Cloned Recipe", "pt_pd_ratio": 0.8}
        cloned = recipe_manager.clone_recipe(sample_recipe.recipe_id, modifications)

        assert cloned.name == "Cloned Recipe"
        assert cloned.pt_pd_ratio == 0.8
        assert cloned.parent_recipe_id == sample_recipe.recipe_id
        assert cloned.version == sample_recipe.version + 1

        # Verify both exist in database
        assert recipe_manager.get_recipe_by_id(sample_recipe.recipe_id) is not None
        assert recipe_manager.get_recipe_by_id(cloned.recipe_id) is not None

    def test_clone_recipe_not_found(self, recipe_manager):
        """Test cloning non-existent recipe raises error."""
        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            recipe_manager.clone_recipe(fake_id)

    def test_update_recipe(self, recipe_manager, sample_recipe):
        """Test updating an existing recipe."""
        recipe_manager.database.add_recipe(sample_recipe)

        changes = {
            "exposure_time_minutes": 15.0,
            "notes": "Updated recipe",
            "quality_rating": 4.5,
        }

        updated = recipe_manager.update_recipe(sample_recipe.recipe_id, changes)

        assert updated.exposure_time_minutes == 15.0
        assert updated.notes == "Updated recipe"
        assert updated.quality_rating == 4.5
        assert updated.name == sample_recipe.name  # Unchanged

    def test_update_recipe_not_found(self, recipe_manager):
        """Test updating non-existent recipe raises error."""
        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            recipe_manager.update_recipe(fake_id, {"name": "New Name"})

    def test_update_recipe_updates_modified_at(self, recipe_manager, sample_recipe):
        """Test that update_recipe updates the modified_at timestamp."""
        recipe_manager.database.add_recipe(sample_recipe)
        original_modified = sample_recipe.modified_at

        updated = recipe_manager.update_recipe(sample_recipe.recipe_id, {"notes": "Test"})

        assert updated.modified_at > original_modified

    def test_delete_recipe(self, recipe_manager, sample_recipe):
        """Test deleting a recipe."""
        recipe_manager.database.add_recipe(sample_recipe)

        # Verify it exists
        assert recipe_manager.get_recipe_by_id(sample_recipe.recipe_id) is not None

        # Delete it
        result = recipe_manager.delete_recipe(sample_recipe.recipe_id)
        assert result is True

        # Verify it's gone
        assert recipe_manager.get_recipe_by_id(sample_recipe.recipe_id) is None

    def test_delete_recipe_not_found(self, recipe_manager):
        """Test deleting non-existent recipe returns False."""
        fake_id = uuid4()
        result = recipe_manager.delete_recipe(fake_id)
        assert result is False

    def test_list_recipes_no_filters(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test listing all recipes without filters."""
        recipes = recipe_manager.list_recipes()

        assert len(recipes) == 4
        assert all(isinstance(r, PrintRecipe) for r in recipes)

    def test_list_recipes_filter_by_paper_type(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test filtering recipes by paper type."""
        recipes = recipe_manager.list_recipes({"paper_type": "Arches Platine"})

        assert len(recipes) == 2
        assert all(r.paper_type == "Arches Platine" for r in recipes)

    def test_list_recipes_filter_by_tags(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test filtering recipes by tags."""
        recipes = recipe_manager.list_recipes({"tags": ["standard"]})

        assert len(recipes) == 2
        assert all("standard" in r.tags for r in recipes)

    def test_list_recipes_filter_by_chemistry_type(self, recipe_manager):
        """Test filtering by chemistry type."""
        recipe_manager.create_recipe(
            name="Pt Recipe",
            paper_type="Test",
            chemistry_type=ChemistryType.PURE_PLATINUM,
        )
        recipe_manager.create_recipe(
            name="Pd Recipe",
            paper_type="Test",
            chemistry_type=ChemistryType.PURE_PALLADIUM,
        )

        recipes = recipe_manager.list_recipes({"chemistry_type": ChemistryType.PURE_PLATINUM.value})

        assert len(recipes) == 1
        assert recipes[0].chemistry_type == ChemistryType.PURE_PLATINUM

    def test_search_recipes_by_name(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test full-text search by recipe name."""
        results = recipe_manager.search_recipes("Bergger")

        assert len(results) == 1
        assert results[0].name == "Bergger High Contrast"

    def test_search_recipes_by_paper_type(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test search by paper type."""
        results = recipe_manager.search_recipes("arches")

        assert len(results) == 2
        assert all("Arches" in r.paper_type for r in results)

    def test_search_recipes_by_tags(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test search by tags."""
        results = recipe_manager.search_recipes("warm")

        assert len(results) == 1
        assert "warm" in results[0].tags

    def test_search_recipes_case_insensitive(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test that search is case-insensitive."""
        results1 = recipe_manager.search_recipes("ARCHES")
        results2 = recipe_manager.search_recipes("arches")
        results3 = recipe_manager.search_recipes("Arches")

        assert len(results1) == len(results2) == len(results3) == 2

    def test_search_recipes_no_matches(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test search with no matches returns empty list."""
        results = recipe_manager.search_recipes("nonexistent")
        assert results == []

    def test_export_recipe_json(self, recipe_manager, sample_recipe):
        """Test exporting recipe to JSON format."""
        recipe_manager.database.add_recipe(sample_recipe)

        exported = recipe_manager.export_recipe(sample_recipe.recipe_id, RecipeFormat.JSON)

        assert isinstance(exported, str)
        data = json.loads(exported)
        assert data["name"] == sample_recipe.name
        assert data["paper_type"] == sample_recipe.paper_type

    def test_export_recipe_yaml(self, recipe_manager, sample_recipe):
        """Test exporting recipe to YAML format."""
        recipe_manager.database.add_recipe(sample_recipe)

        exported = recipe_manager.export_recipe(sample_recipe.recipe_id, RecipeFormat.YAML)

        assert isinstance(exported, str)
        data = yaml.safe_load(exported)
        assert data["name"] == sample_recipe.name
        assert data["paper_type"] == sample_recipe.paper_type

    def test_export_recipe_not_found(self, recipe_manager):
        """Test exporting non-existent recipe raises error."""
        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            recipe_manager.export_recipe(fake_id)

    def test_import_recipe_json(self, recipe_manager, sample_recipe, tmp_path):
        """Test importing recipe from JSON file."""
        # Export to file
        json_file = tmp_path / "recipe.json"
        data = sample_recipe.to_dict()
        with open(json_file, "w") as f:
            json.dump(data, f)

        # Import
        imported = recipe_manager.import_recipe(json_file)

        assert imported.name == sample_recipe.name
        assert imported.paper_type == sample_recipe.paper_type
        # ID should be different (new recipe)
        assert imported.recipe_id != sample_recipe.recipe_id

    def test_import_recipe_yaml(self, recipe_manager, sample_recipe, tmp_path):
        """Test importing recipe from YAML file."""
        # Export to file
        yaml_file = tmp_path / "recipe.yaml"
        data = sample_recipe.to_dict()
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        # Import
        imported = recipe_manager.import_recipe(yaml_file)

        assert imported.name == sample_recipe.name
        assert imported.paper_type == sample_recipe.paper_type

    def test_import_recipe_file_not_found(self, recipe_manager, tmp_path):
        """Test importing from non-existent file raises error."""
        fake_file = tmp_path / "nonexistent.json"
        with pytest.raises(ValueError, match="not found"):
            recipe_manager.import_recipe(fake_file)

    def test_get_recipe_history_single(self, recipe_manager, sample_recipe):
        """Test getting history for a single recipe."""
        recipe_manager.database.add_recipe(sample_recipe)

        history = recipe_manager.get_recipe_history(sample_recipe.recipe_id)

        assert len(history) == 1
        assert history[0].recipe_id == sample_recipe.recipe_id

    def test_get_recipe_history_with_clones(self, recipe_manager, sample_recipe):
        """Test getting history with multiple versions."""
        recipe_manager.database.add_recipe(sample_recipe)

        # Create two clones
        clone1 = recipe_manager.clone_recipe(sample_recipe.recipe_id, {"name": "Clone 1"})
        clone2 = recipe_manager.clone_recipe(clone1.recipe_id, {"name": "Clone 2"})

        # Get history from the latest
        history = recipe_manager.get_recipe_history(clone2.recipe_id)

        assert len(history) == 3
        assert history[0].recipe_id == sample_recipe.recipe_id
        assert history[1].recipe_id == clone1.recipe_id
        assert history[2].recipe_id == clone2.recipe_id
        assert history[0].version < history[1].version < history[2].version

    def test_get_recipe_history_not_found(self, recipe_manager):
        """Test getting history for non-existent recipe returns empty list."""
        fake_id = uuid4()
        history = recipe_manager.get_recipe_history(fake_id)
        assert history == []

    def test_compare_recipes_two(self, recipe_manager, multiple_recipes):
        """Test comparing two recipes."""
        recipe_ids = [multiple_recipes[0].recipe_id, multiple_recipes[1].recipe_id]
        comparison = recipe_manager.compare_recipes(recipe_ids)

        assert "recipes" in comparison
        assert "differences" in comparison
        assert "similarities" in comparison
        assert len(comparison["recipes"]) == 2

        # They have different paper types
        assert "paper_type" in comparison["differences"]
        # They have different pt_pd_ratios
        assert "pt_pd_ratio" in comparison["differences"]

    def test_compare_recipes_similarities(self, recipe_manager):
        """Test that similar recipes show similarities."""
        # Create two recipes with same paper and developer
        r1 = recipe_manager.create_recipe(
            name="Recipe 1",
            paper_type="Same Paper",
            developer=DeveloperType.POTASSIUM_OXALATE,
            uv_source="UV LED",
        )
        r2 = recipe_manager.create_recipe(
            name="Recipe 2",
            paper_type="Same Paper",
            developer=DeveloperType.POTASSIUM_OXALATE,
            uv_source="UV LED",
        )

        comparison = recipe_manager.compare_recipes([r1.recipe_id, r2.recipe_id])

        assert "paper_type" in comparison["similarities"]
        assert comparison["similarities"]["paper_type"] == "Same Paper"
        assert "developer" in comparison["similarities"]
        assert "uv_source" in comparison["similarities"]

    def test_compare_recipes_not_found(self, recipe_manager):
        """Test comparing non-existent recipe raises error."""
        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            recipe_manager.compare_recipes([fake_id])

    def test_suggest_similar_recipes_by_paper(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test suggesting similar recipes by paper type."""
        params = {"paper_type": "Arches Platine"}
        suggestions = recipe_manager.suggest_similar_recipes(params, limit=10)

        # Should find the 2 Arches recipes
        assert len(suggestions) >= 2
        # Results are (recipe, score) tuples
        assert all(isinstance(s, tuple) for s in suggestions)
        assert all(isinstance(s[0], PrintRecipe) for s in suggestions)
        assert all(isinstance(s[1], float) for s in suggestions)

        # Arches recipes should have high scores
        arches_suggestions = [s for s in suggestions if "Arches" in s[0].paper_type]
        assert len(arches_suggestions) == 2
        assert all(s[1] > 0.5 for s in arches_suggestions)

    def test_suggest_similar_recipes_by_ratio(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test suggesting similar recipes by Pt/Pd ratio."""
        params = {"pt_pd_ratio": 0.5}
        suggestions = recipe_manager.suggest_similar_recipes(params, limit=10)

        # Should find recipes with similar ratios
        assert len(suggestions) > 0
        # Sorted by similarity
        scores = [s[1] for s in suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_suggest_similar_recipes_min_similarity(self, recipe_manager):
        """Test minimum similarity threshold."""
        recipe_manager.create_recipe(
            name="Very Different",
            paper_type="Unique Paper",
            pt_pd_ratio=0.9,
            exposure_time_minutes=100.0,
        )

        params = {"paper_type": "Completely Different"}
        suggestions = recipe_manager.suggest_similar_recipes(params, limit=10, min_similarity=0.9)

        # Very high threshold should yield few or no results
        assert all(s[1] >= 0.9 for s in suggestions)

    def test_suggest_similar_recipes_limit(self, recipe_manager, multiple_recipes):  # noqa: ARG002
        """Test limiting number of suggestions."""
        params = {"paper_type": "Arches Platine"}
        suggestions = recipe_manager.suggest_similar_recipes(params, limit=1)

        assert len(suggestions) <= 1


# ============================================================================
# WORKFLOWAUTOMATION TESTS
# ============================================================================


class TestWorkflowAutomation:
    """Tests for WorkflowAutomation job execution and scheduling."""

    def test_create_batch_job(self, workflow_automation, sample_recipe, tmp_path):
        """Test creating a batch processing job."""
        # Create some fake image paths
        images = [
            tmp_path / "image1.tif",
            tmp_path / "image2.tif",
            tmp_path / "image3.tif",
        ]
        output_dir = tmp_path / "output"

        job = workflow_automation.create_batch_job(images, sample_recipe, output_dir)

        assert isinstance(job, WorkflowJob)
        assert job.recipe_id == sample_recipe.recipe_id
        assert len(job.steps) == 3
        assert job.status == WorkflowStatus.PENDING
        assert all(isinstance(step, WorkflowStep) for step in job.steps)

    def test_batch_job_step_parameters(self, workflow_automation, sample_recipe, tmp_path):
        """Test that batch job steps have correct parameters."""
        images = [tmp_path / "test.tif"]
        output_dir = tmp_path / "output"

        job = workflow_automation.create_batch_job(images, sample_recipe, output_dir)

        step = job.steps[0]
        assert step.action == "process_image"
        assert "image_path" in step.parameters
        assert "recipe" in step.parameters
        assert "output_path" in step.parameters
        assert step.parameters["recipe"] == sample_recipe.to_dict()

    def test_execute_workflow_simple(self, workflow_automation):
        """Test executing a simple workflow."""
        steps = [
            WorkflowStep(name="Step 1", action="test_action", parameters={}),
            WorkflowStep(name="Step 2", action="test_action", parameters={}),
        ]

        job = workflow_automation.execute_workflow(steps)

        assert job.status == WorkflowStatus.COMPLETED
        assert job.progress == 1.0
        assert job.started_at is not None
        assert job.completed_at is not None
        assert all(s.status == WorkflowStatus.COMPLETED for s in job.steps)

    def test_execute_workflow_updates_progress(self, workflow_automation):
        """Test that workflow execution updates progress."""
        steps = [WorkflowStep(name=f"Step {i}", action="test") for i in range(5)]

        job = workflow_automation.execute_workflow(steps)

        # Final progress should be 1.0
        assert job.progress == 1.0
        # All steps should be completed
        assert all(s.status == WorkflowStatus.COMPLETED for s in job.steps)

    def test_execute_workflow_stores_results(self, workflow_automation):
        """Test that step results are stored."""
        steps = [WorkflowStep(name="Step 1", action="test_action")]

        job = workflow_automation.execute_workflow(steps)

        assert job.steps[0].result is not None
        assert "status" in job.steps[0].result
        assert job.steps[0].result["status"] == "simulated"

    def test_execute_workflow_tracks_timing(self, workflow_automation):
        """Test that step and job timing is tracked."""
        steps = [WorkflowStep(name="Step 1", action="test")]

        job = workflow_automation.execute_workflow(steps)

        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.started_at <= job.completed_at

        step = job.steps[0]
        assert step.started_at is not None
        assert step.completed_at is not None
        assert step.started_at <= step.completed_at

    def test_get_workflow_status(self, workflow_automation, sample_recipe, tmp_path):
        """Test getting workflow status."""
        images = [tmp_path / "test.tif"]
        job = workflow_automation.create_batch_job(images, sample_recipe, tmp_path)

        retrieved = workflow_automation.get_workflow_status(job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == job.job_id
        assert retrieved.status == WorkflowStatus.PENDING

    def test_get_workflow_status_not_found(self, workflow_automation):
        """Test getting status for non-existent job returns None."""
        fake_id = uuid4()
        status = workflow_automation.get_workflow_status(fake_id)
        assert status is None

    def test_schedule_workflow(self, workflow_automation):
        """Test scheduling a workflow for later execution."""
        steps = [WorkflowStep(name="Scheduled Step", action="test")]
        schedule_time = datetime.now() + timedelta(hours=1)

        job = workflow_automation.schedule_workflow(steps, schedule_time)

        assert job.scheduled_for == schedule_time
        assert job.status == WorkflowStatus.PENDING
        assert len(job.steps) == 1

    def test_cancel_workflow_pending(self, workflow_automation):
        """Test cancelling a pending workflow."""
        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.schedule_workflow(steps, datetime.now())

        result = workflow_automation.cancel_workflow(job.job_id)

        assert result is True
        assert job.status == WorkflowStatus.CANCELLED
        assert job.completed_at is not None

    def test_cancel_workflow_not_found(self, workflow_automation):
        """Test cancelling non-existent workflow returns False."""
        fake_id = uuid4()
        result = workflow_automation.cancel_workflow(fake_id)
        assert result is False

    def test_cancel_workflow_already_completed(self, workflow_automation):
        """Test that completed workflows cannot be cancelled."""
        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.execute_workflow(steps)

        # Job is now completed
        assert job.status == WorkflowStatus.COMPLETED

        result = workflow_automation.cancel_workflow(job.job_id)
        assert result is False

    def test_log_workflow_result_success(self, workflow_automation):
        """Test logging successful workflow result."""
        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.schedule_workflow(steps, datetime.now())

        result_data = {"output": "success", "files_processed": 3}
        success = workflow_automation.log_workflow_result(job.job_id, result_data, success=True)

        assert success is True
        assert job.status == WorkflowStatus.COMPLETED
        assert job.progress == 1.0
        assert job.completed_at is not None
        assert job.steps[0].result == result_data

    def test_log_workflow_result_failure(self, workflow_automation):
        """Test logging failed workflow result."""
        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.schedule_workflow(steps, datetime.now())

        result_data = {"error": "Processing failed"}
        success = workflow_automation.log_workflow_result(job.job_id, result_data, success=False)

        assert success is True
        assert job.status == WorkflowStatus.FAILED
        assert job.progress == 1.0

    def test_log_workflow_result_not_found(self, workflow_automation):
        """Test logging result for non-existent job returns False."""
        fake_id = uuid4()
        result = workflow_automation.log_workflow_result(fake_id, {})
        assert result is False

    def test_register_callback(self, workflow_automation):
        """Test registering a callback for job completion."""
        callback_called = []

        def callback(job):
            callback_called.append(job.job_id)

        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.schedule_workflow(steps, datetime.now())

        workflow_automation.register_callback(job.job_id, callback)

        # Complete the job
        workflow_automation.log_workflow_result(job.job_id, {}, success=True)

        # Callback should have been called
        assert job.job_id in callback_called

    def test_multiple_callbacks(self, workflow_automation):
        """Test registering multiple callbacks for same job."""
        calls = []

        def callback1(_job):
            calls.append(1)

        def callback2(_job):
            calls.append(2)

        steps = [WorkflowStep(name="Step", action="test")]
        job = workflow_automation.schedule_workflow(steps, datetime.now())

        workflow_automation.register_callback(job.job_id, callback1)
        workflow_automation.register_callback(job.job_id, callback2)

        workflow_automation.log_workflow_result(job.job_id, {}, success=True)

        # Both callbacks should have been called
        assert 1 in calls
        assert 2 in calls

    def test_list_jobs_all(self, workflow_automation):
        """Test listing all jobs."""
        # Create several jobs
        for i in range(3):
            steps = [WorkflowStep(name=f"Step {i}", action="test")]
            workflow_automation.schedule_workflow(steps, datetime.now())

        jobs = workflow_automation.list_jobs()

        assert len(jobs) == 3

    def test_list_jobs_filter_by_status(self, workflow_automation):
        """Test listing jobs filtered by status."""
        # Create pending and completed jobs
        steps1 = [WorkflowStep(name="Pending", action="test")]
        job1 = workflow_automation.schedule_workflow(steps1, datetime.now())

        steps2 = [WorkflowStep(name="Completed", action="test")]
        job2 = workflow_automation.execute_workflow(steps2)

        pending_jobs = workflow_automation.list_jobs(status=WorkflowStatus.PENDING)
        completed_jobs = workflow_automation.list_jobs(status=WorkflowStatus.COMPLETED)

        assert len(pending_jobs) == 1
        assert pending_jobs[0].job_id == job1.job_id
        assert len(completed_jobs) == 1
        assert completed_jobs[0].job_id == job2.job_id

    def test_list_jobs_limit(self, workflow_automation):
        """Test limiting number of jobs returned."""
        # Create 5 jobs
        for i in range(5):
            steps = [WorkflowStep(name=f"Step {i}", action="test")]
            workflow_automation.schedule_workflow(steps, datetime.now())

        jobs = workflow_automation.list_jobs(limit=3)

        assert len(jobs) == 3

    def test_list_jobs_sorted_by_created_at(self, workflow_automation):
        """Test that jobs are sorted by created_at descending."""
        jobs_created = []
        for i in range(3):
            steps = [WorkflowStep(name=f"Step {i}", action="test")]
            job = workflow_automation.schedule_workflow(steps, datetime.now())
            jobs_created.append(job)

        jobs = workflow_automation.list_jobs()

        # Should be in reverse order (most recent first)
        assert jobs[0].job_id == jobs_created[-1].job_id

    def test_job_duration_seconds(self):
        """Test calculating job duration."""
        job = WorkflowJob(name="Test", steps=[])
        job.started_at = datetime(2025, 1, 1, 12, 0, 0)
        job.completed_at = datetime(2025, 1, 1, 12, 5, 30)

        duration = job.duration_seconds

        assert duration == 330.0  # 5 minutes 30 seconds

    def test_job_duration_seconds_not_completed(self):
        """Test duration is None if job not completed."""
        job = WorkflowJob(name="Test", steps=[])
        job.started_at = datetime.now()

        assert job.duration_seconds is None


# ============================================================================
# RECIPEDATABASE TESTS
# ============================================================================


class TestRecipeDatabase:
    """Tests for RecipeDatabase CRUD and persistence."""

    def test_create_database(self, temp_db_path):
        """Test database creation."""
        db = RecipeDatabase(db_path=temp_db_path)
        assert db.db_path.exists()

    def test_add_recipe(self, recipe_db, sample_recipe):
        """Test adding a recipe to database."""
        recipe_db.add_recipe(sample_recipe)

        # Retrieve it
        retrieved = recipe_db.get_recipe(sample_recipe.recipe_id)
        assert retrieved is not None
        assert retrieved.name == sample_recipe.name
        assert retrieved.paper_type == sample_recipe.paper_type

    def test_get_recipe_not_found(self, recipe_db):
        """Test getting non-existent recipe returns None."""
        fake_id = uuid4()
        result = recipe_db.get_recipe(fake_id)
        assert result is None

    def test_get_recipe_uses_cache(self, recipe_db, sample_recipe):
        """Test that get_recipe uses cache."""
        recipe_db.add_recipe(sample_recipe)

        # First retrieval
        recipe_db.get_recipe(sample_recipe.recipe_id)

        # Recipe should be in cache
        assert sample_recipe.recipe_id in recipe_db._recipe_cache

    def test_update_recipe(self, recipe_db, sample_recipe):
        """Test updating a recipe."""
        recipe_db.add_recipe(sample_recipe)

        # Modify and update
        sample_recipe.notes = "Updated notes"
        sample_recipe.quality_rating = 4.5
        recipe_db.update_recipe(sample_recipe)

        # Retrieve and verify
        retrieved = recipe_db.get_recipe(sample_recipe.recipe_id)
        assert retrieved.notes == "Updated notes"
        assert retrieved.quality_rating == 4.5

    def test_delete_recipe(self, recipe_db, sample_recipe):
        """Test deleting a recipe."""
        recipe_db.add_recipe(sample_recipe)

        result = recipe_db.delete_recipe(sample_recipe.recipe_id)
        assert result is True

        # Should not be retrievable
        assert recipe_db.get_recipe(sample_recipe.recipe_id) is None

    def test_delete_recipe_removes_from_cache(self, recipe_db, sample_recipe):
        """Test that delete removes recipe from cache."""
        recipe_db.add_recipe(sample_recipe)
        recipe_db.get_recipe(sample_recipe.recipe_id)  # Add to cache

        recipe_db.delete_recipe(sample_recipe.recipe_id)

        assert sample_recipe.recipe_id not in recipe_db._recipe_cache

    def test_list_all_recipes(self, recipe_db):
        """Test listing all recipes."""
        # Add multiple recipes
        for i in range(3):
            recipe = PrintRecipe(name=f"Recipe {i}", paper_type="Test Paper")
            recipe_db.add_recipe(recipe)

        recipes = recipe_db.list_all_recipes()

        assert len(recipes) == 3
        assert all(isinstance(r, PrintRecipe) for r in recipes)

    def test_query_recipes_by_paper_type(self, recipe_db):
        """Test querying recipes by paper type."""
        recipe_db.add_recipe(PrintRecipe(name="R1", paper_type="Arches Platine"))
        recipe_db.add_recipe(PrintRecipe(name="R2", paper_type="Bergger COT320"))
        recipe_db.add_recipe(PrintRecipe(name="R3", paper_type="Arches Platine"))

        results = recipe_db.query_recipes({"paper_type": "Arches Platine"})

        assert len(results) == 2
        assert all(r.paper_type == "Arches Platine" for r in results)

    def test_query_recipes_by_chemistry_type(self, recipe_db):
        """Test querying by chemistry type."""
        recipe_db.add_recipe(
            PrintRecipe(
                name="R1",
                paper_type="Test",
                chemistry_type=ChemistryType.PURE_PLATINUM,
            )
        )
        recipe_db.add_recipe(
            PrintRecipe(
                name="R2",
                paper_type="Test",
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
            )
        )

        results = recipe_db.query_recipes({"chemistry_type": ChemistryType.PURE_PLATINUM.value})

        assert len(results) == 1
        assert results[0].chemistry_type == ChemistryType.PURE_PLATINUM

    def test_query_recipes_by_min_quality_rating(self, recipe_db):
        """Test querying by minimum quality rating."""
        recipe_db.add_recipe(PrintRecipe(name="R1", paper_type="Test", quality_rating=3.0))
        recipe_db.add_recipe(PrintRecipe(name="R2", paper_type="Test", quality_rating=4.5))
        recipe_db.add_recipe(PrintRecipe(name="R3", paper_type="Test", quality_rating=5.0))

        results = recipe_db.query_recipes({"min_quality_rating": 4.0})

        assert len(results) == 2
        assert all(r.quality_rating >= 4.0 for r in results)

    def test_query_recipes_by_tags(self, recipe_db):
        """Test querying by tags."""
        recipe_db.add_recipe(PrintRecipe(name="R1", paper_type="Test", tags=["platinum", "warm"]))
        recipe_db.add_recipe(PrintRecipe(name="R2", paper_type="Test", tags=["palladium", "cool"]))
        recipe_db.add_recipe(PrintRecipe(name="R3", paper_type="Test", tags=["platinum", "cool"]))

        results = recipe_db.query_recipes({"tags": ["platinum"]})

        assert len(results) == 2
        assert all("platinum" in r.tags for r in results)

    def test_query_recipes_by_uv_source(self, recipe_db):
        """Test querying by UV source."""
        recipe_db.add_recipe(PrintRecipe(name="R1", paper_type="Test", uv_source="UV LED"))
        recipe_db.add_recipe(PrintRecipe(name="R2", paper_type="Test", uv_source="Metal Halide"))

        results = recipe_db.query_recipes({"uv_source": "UV LED"})

        assert len(results) == 1
        assert results[0].uv_source == "UV LED"

    def test_query_recipes_by_developer(self, recipe_db):
        """Test querying by developer type."""
        recipe_db.add_recipe(
            PrintRecipe(
                name="R1",
                paper_type="Test",
                developer=DeveloperType.POTASSIUM_OXALATE,
            )
        )
        recipe_db.add_recipe(
            PrintRecipe(name="R2", paper_type="Test", developer=DeveloperType.AMMONIUM_CITRATE)
        )

        results = recipe_db.query_recipes({"developer": DeveloperType.POTASSIUM_OXALATE.value})

        assert len(results) == 1
        assert results[0].developer == DeveloperType.POTASSIUM_OXALATE

    def test_export_all_json(self, recipe_db, tmp_path):
        """Test exporting all recipes to JSON file."""
        # Add recipes
        for i in range(3):
            recipe_db.add_recipe(PrintRecipe(name=f"Recipe {i}", paper_type="Test"))

        output_file = tmp_path / "export.json"
        recipe_db.export_all(output_file, RecipeFormat.JSON)

        assert output_file.exists()

        # Verify contents
        with open(output_file) as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        assert len(data["recipes"]) == 3

    def test_export_all_yaml(self, recipe_db, tmp_path):
        """Test exporting all recipes to YAML file."""
        recipe_db.add_recipe(PrintRecipe(name="Recipe", paper_type="Test"))

        output_file = tmp_path / "export.yaml"
        recipe_db.export_all(output_file, RecipeFormat.YAML)

        assert output_file.exists()

        with open(output_file) as f:
            data = yaml.safe_load(f)

        assert data["version"] == "1.0"
        assert len(data["recipes"]) == 1

    def test_import_all_json(self, recipe_db, tmp_path):
        """Test importing recipes from JSON file."""
        # Create export file
        recipes_data = {
            "version": "1.0",
            "recipes": [
                PrintRecipe(name="R1", paper_type="Test").to_dict(),
                PrintRecipe(name="R2", paper_type="Test").to_dict(),
            ],
        }

        import_file = tmp_path / "import.json"
        with open(import_file, "w") as f:
            json.dump(recipes_data, f)

        count = recipe_db.import_all(import_file)

        assert count == 2
        assert len(recipe_db.list_all_recipes()) == 2

    def test_import_all_yaml(self, recipe_db, tmp_path):
        """Test importing recipes from YAML file."""
        recipes_data = {
            "version": "1.0",
            "recipes": [PrintRecipe(name="R1", paper_type="Test").to_dict()],
        }

        import_file = tmp_path / "import.yml"
        with open(import_file, "w") as f:
            yaml.dump(recipes_data, f)

        count = recipe_db.import_all(import_file)

        assert count == 1

    def test_get_statistics_empty(self, recipe_db):
        """Test statistics for empty database."""
        stats = recipe_db.get_statistics()

        assert stats["total_recipes"] == 0
        assert stats["unique_paper_types"] == 0
        assert stats["unique_chemistry_types"] == 0
        assert stats["average_quality_rating"] is None

    def test_get_statistics_with_data(self, recipe_db):
        """Test statistics with recipes."""
        recipe_db.add_recipe(
            PrintRecipe(
                name="R1",
                paper_type="Arches Platine",
                chemistry_type=ChemistryType.PURE_PLATINUM,
                quality_rating=4.0,
            )
        )
        recipe_db.add_recipe(
            PrintRecipe(
                name="R2",
                paper_type="Bergger COT320",
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                quality_rating=5.0,
            )
        )
        recipe_db.add_recipe(
            PrintRecipe(
                name="R3",
                paper_type="Arches Platine",
                chemistry_type=ChemistryType.PURE_PLATINUM,
                quality_rating=3.0,
            )
        )

        stats = recipe_db.get_statistics()

        assert stats["total_recipes"] == 3
        assert stats["unique_paper_types"] == 2
        assert stats["unique_chemistry_types"] == 2
        assert stats["average_quality_rating"] == 4.0  # (4.0 + 5.0 + 3.0) / 3

    def test_bulk_operations(self, recipe_db):
        """Test adding and retrieving many recipes (performance test)."""
        # Add 100 recipes
        for i in range(100):
            recipe = PrintRecipe(
                name=f"Recipe {i}",
                paper_type=f"Paper {i % 10}",
                pt_pd_ratio=i / 100,
            )
            recipe_db.add_recipe(recipe)

        # Retrieve all
        all_recipes = recipe_db.list_all_recipes()
        assert len(all_recipes) == 100

        # Query subset
        results = recipe_db.query_recipes({"paper_type": "Paper 5"})
        assert len(results) == 10

    def test_persistence_across_instances(self, temp_db_path, sample_recipe):
        """Test that data persists across database instances."""
        # Create first instance and add recipe
        db1 = RecipeDatabase(db_path=temp_db_path)
        db1.add_recipe(sample_recipe)

        # Close and create new instance
        del db1
        db2 = RecipeDatabase(db_path=temp_db_path)

        # Should still be able to retrieve
        retrieved = db2.get_recipe(sample_recipe.recipe_id)
        assert retrieved is not None
        assert retrieved.name == sample_recipe.name

    def test_concurrent_operations(self, recipe_db):
        """Test multiple operations on same database."""
        recipe1 = PrintRecipe(name="R1", paper_type="Test")
        recipe2 = PrintRecipe(name="R2", paper_type="Test")

        # Add both
        recipe_db.add_recipe(recipe1)
        recipe_db.add_recipe(recipe2)

        # Update first while reading second
        recipe1.notes = "Updated"
        recipe_db.update_recipe(recipe1)
        retrieved2 = recipe_db.get_recipe(recipe2.recipe_id)

        assert retrieved2.name == "R2"
        assert recipe_db.get_recipe(recipe1.recipe_id).notes == "Updated"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_recipe_workflow(self, recipe_manager, sample_recipe_data):
        """Test complete recipe lifecycle."""
        # 1. Create recipe
        recipe = recipe_manager.create_recipe(**sample_recipe_data)
        assert recipe.successful_prints == 0

        # 2. Update quality after a print
        recipe.update_quality(4.5, 0.12, 2.3)
        recipe_manager.database.update_recipe(recipe)

        # 3. Clone with modifications
        cloned = recipe_manager.clone_recipe(
            recipe.recipe_id, {"name": "Cloned Recipe", "exposure_time_minutes": 12.0}
        )

        # 4. Export original
        exported = recipe_manager.export_recipe(recipe.recipe_id, RecipeFormat.JSON)
        assert len(exported) > 0

        # 5. Get history
        history = recipe_manager.get_recipe_history(cloned.recipe_id)
        assert len(history) == 2

        # 6. Search for it
        results = recipe_manager.search_recipes("Cloned")
        assert len(results) == 1

    def test_batch_workflow_with_recipe(self, workflow_automation, recipe_manager, tmp_path):
        """Test creating and executing a batch workflow with a recipe."""
        # Create recipe
        recipe = recipe_manager.create_recipe(name="Batch Recipe", paper_type="Test Paper")

        # Create batch job
        images = [tmp_path / f"image{i}.tif" for i in range(3)]
        output_dir = tmp_path / "output"

        job = workflow_automation.create_batch_job(images, recipe, output_dir)

        # Execute
        workflow_automation.execute_workflow(job.steps)

        # Check status
        status = workflow_automation.get_workflow_status(job.job_id)
        assert status is not None

    def test_import_export_round_trip(self, recipe_manager, sample_recipe, tmp_path):
        """Test complete import/export round trip."""
        # Add original recipe
        recipe_manager.database.add_recipe(sample_recipe)

        # Export to file
        json_file = tmp_path / "export.json"
        exported = recipe_manager.export_recipe(sample_recipe.recipe_id, RecipeFormat.JSON)
        with open(json_file, "w") as f:
            f.write(exported)

        # Import back
        imported = recipe_manager.import_recipe(json_file)

        # Should have same data but different ID
        assert imported.name == sample_recipe.name
        assert imported.paper_type == sample_recipe.paper_type
        assert imported.pt_pd_ratio == sample_recipe.pt_pd_ratio
        assert imported.recipe_id != sample_recipe.recipe_id

    def test_recipe_versioning_chain(self, recipe_manager):
        """Test creating a chain of recipe versions."""
        # Create original
        v1 = recipe_manager.create_recipe(name="Version 1", paper_type="Test", pt_pd_ratio=0.5)

        # Clone to v2
        v2 = recipe_manager.clone_recipe(v1.recipe_id, {"name": "Version 2", "pt_pd_ratio": 0.6})

        # Clone to v3
        v3 = recipe_manager.clone_recipe(v2.recipe_id, {"name": "Version 3", "pt_pd_ratio": 0.7})

        # Get complete history
        history = recipe_manager.get_recipe_history(v3.recipe_id)

        assert len(history) == 3
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[2].version == 3
        assert history[0].pt_pd_ratio == 0.5
        assert history[1].pt_pd_ratio == 0.6
        assert history[2].pt_pd_ratio == 0.7
