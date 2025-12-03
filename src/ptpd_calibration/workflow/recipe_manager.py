"""
Recipe management and workflow automation for repeatable Pt/Pd printing.

This module provides comprehensive recipe management capabilities for storing,
retrieving, and automating platinum/palladium printing workflows. Recipes
capture all parameters needed to reproduce successful prints, including paper,
chemistry, exposure, environmental conditions, and calibration curves.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType


class RecipeFormat(str, Enum):
    """Supported recipe export/import formats."""

    JSON = "json"
    YAML = "yaml"


class WorkflowStatus(str, Enum):
    """Status of a workflow job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PrintRecipe(BaseModel):
    """
    Complete recipe for repeatable platinum/palladium printing.

    Captures all parameters needed to reproduce a successful print,
    including paper, chemistry, exposure, environmental conditions,
    and calibration curve settings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    recipe_id: UUID = Field(default_factory=uuid4, description="Unique recipe identifier")
    name: str = Field(..., min_length=1, max_length=256, description="Recipe name")
    description: Optional[str] = Field(
        default=None, max_length=2048, description="Detailed recipe description"
    )
    tags: list[str] = Field(default_factory=list, description="Searchable tags")

    # Paper settings
    paper_type: str = Field(..., description="Paper type name")
    paper_profile_id: Optional[UUID] = Field(
        default=None, description="Reference to paper profile"
    )

    # Chemistry settings
    chemistry_type: ChemistryType = Field(
        default=ChemistryType.PLATINUM_PALLADIUM, description="Chemistry type"
    )
    pt_pd_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Platinum ratio (0=Pd, 1=Pt)"
    )
    ferric_oxalate_1_drops: float = Field(
        default=24.0, ge=0.0, description="FO #1 drops (base sensitizer)"
    )
    ferric_oxalate_2_drops: float = Field(
        default=0.0, ge=0.0, description="FO #2 drops (contrast agent)"
    )
    metal_drops: float = Field(default=24.0, ge=0.0, description="Total metal drops")
    contrast_agent: ContrastAgent = Field(
        default=ContrastAgent.NONE, description="Contrast agent type"
    )
    contrast_agent_drops: float = Field(
        default=0.0, ge=0.0, description="Contrast agent drops"
    )

    # Developer settings
    developer: DeveloperType = Field(
        default=DeveloperType.POTASSIUM_OXALATE, description="Developer type"
    )
    developer_temperature_f: float = Field(
        default=68.0, ge=40.0, le=90.0, description="Developer temperature (F)"
    )
    development_time_minutes: float = Field(
        default=2.0, ge=0.5, le=10.0, description="Development time (minutes)"
    )

    # Exposure settings
    exposure_time_minutes: float = Field(
        default=10.0, ge=0.1, le=120.0, description="Exposure time (minutes)"
    )
    uv_source: str = Field(default="UV LED", description="UV light source type")
    uv_intensity_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="UV source intensity (%)"
    )

    # Environmental conditions
    humidity_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Relative humidity (%)"
    )
    temperature_f: Optional[float] = Field(
        default=None, ge=40.0, le=90.0, description="Ambient temperature (F)"
    )
    coating_humidity_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Humidity during coating (%)"
    )
    drying_time_hours: Optional[float] = Field(
        default=None, ge=0.0, le=48.0, description="Drying time before exposure (hours)"
    )

    # Curve settings
    curve_id: Optional[UUID] = Field(
        default=None, description="Calibration curve identifier"
    )
    curve_name: Optional[str] = Field(default=None, description="Curve name for reference")

    # Version tracking
    version: int = Field(default=1, ge=1, description="Recipe version number")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    modified_at: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )
    parent_recipe_id: Optional[UUID] = Field(
        default=None, description="Parent recipe for version tracking"
    )

    # Quality metrics
    quality_rating: Optional[float] = Field(
        default=None, ge=0.0, le=5.0, description="Quality rating from past prints (0-5)"
    )
    successful_prints: int = Field(
        default=0, ge=0, description="Number of successful prints with this recipe"
    )
    dmin_achieved: Optional[float] = Field(
        default=None, ge=0.0, description="Achieved minimum density"
    )
    dmax_achieved: Optional[float] = Field(
        default=None, ge=0.0, description="Achieved maximum density"
    )

    # Notes and metadata
    notes: Optional[str] = Field(default=None, description="Additional notes")
    author: Optional[str] = Field(default=None, description="Recipe author")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: Any) -> list[str]:
        """Normalize and deduplicate tags."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return sorted(list(set(tag.lower().strip() for tag in v if tag.strip())))

    def clone(self, modifications: Optional[dict[str, Any]] = None) -> "PrintRecipe":
        """
        Clone this recipe with optional modifications.

        Args:
            modifications: Dictionary of fields to modify in the clone

        Returns:
            New PrintRecipe instance
        """
        data = self.model_dump(exclude={"recipe_id", "created_at", "modified_at", "version"})
        data["parent_recipe_id"] = self.recipe_id
        data["version"] = self.version + 1

        if modifications:
            data.update(modifications)

        return PrintRecipe(**data)

    def update_quality(self, rating: float, dmin: Optional[float], dmax: Optional[float]) -> None:
        """
        Update quality metrics based on a print result.

        Args:
            rating: Quality rating (0-5)
            dmin: Achieved minimum density
            dmax: Achieved maximum density
        """
        self.successful_prints += 1
        if self.quality_rating is None:
            self.quality_rating = rating
        else:
            # Running average
            self.quality_rating = (
                self.quality_rating * (self.successful_prints - 1) + rating
            ) / self.successful_prints

        if dmin is not None:
            self.dmin_achieved = dmin
        if dmax is not None:
            self.dmax_achieved = dmax

        now = datetime.now()
        if self.modified_at and now <= self.modified_at:
            now = self.modified_at + timedelta(microseconds=1)
        self.modified_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert recipe to dictionary with JSON-serializable values."""
        data = self.model_dump(mode="json")
        # Convert UUIDs to strings
        for key in ["recipe_id", "paper_profile_id", "curve_id", "parent_recipe_id"]:
            if data.get(key):
                data[key] = str(data[key])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrintRecipe":
        """Create recipe from dictionary."""
        return cls(**data)


class RecipeManager:
    """
    Manages print recipes for repeatable Pt/Pd printing results.

    Provides CRUD operations, search, filtering, version tracking,
    and comparison capabilities for print recipes.
    """

    def __init__(self, database: Optional["RecipeDatabase"] = None):
        """
        Initialize the recipe manager.

        Args:
            database: Recipe database instance. If None, creates in-memory database.
        """
        self.database = database or RecipeDatabase()

    def create_recipe(self, **params: Any) -> PrintRecipe:
        """
        Create a new print recipe.

        Args:
            **params: Recipe parameters (name, paper_type, etc.)

        Returns:
            Created PrintRecipe instance
        """
        recipe = PrintRecipe(**params)
        self.database.add_recipe(recipe)
        return recipe

    def clone_recipe(
        self, recipe_id: UUID, modifications: Optional[dict[str, Any]] = None
    ) -> PrintRecipe:
        """
        Clone an existing recipe with optional modifications.

        Args:
            recipe_id: ID of recipe to clone
            modifications: Optional dictionary of fields to modify

        Returns:
            Cloned PrintRecipe instance

        Raises:
            ValueError: If recipe not found
        """
        original = self.database.get_recipe(recipe_id)
        if not original:
            raise ValueError(f"Recipe {recipe_id} not found")

        cloned = original.clone(modifications)
        self.database.add_recipe(cloned)
        return cloned

    def update_recipe(self, recipe_id: UUID, changes: dict[str, Any]) -> PrintRecipe:
        """
        Update an existing recipe.

        Args:
            recipe_id: Recipe ID to update
            changes: Dictionary of fields to update

        Returns:
            Updated PrintRecipe instance

        Raises:
            ValueError: If recipe not found
        """
        recipe = self.database.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")

        # Update fields
        for key, value in changes.items():
            if hasattr(recipe, key):
                setattr(recipe, key, value)

        recipe.modified_at = datetime.now()
        self.database.update_recipe(recipe)
        return recipe

    def delete_recipe(self, recipe_id: UUID) -> bool:
        """
        Delete a recipe.

        Args:
            recipe_id: Recipe ID to delete

        Returns:
            True if deleted, False if not found
        """
        return self.database.delete_recipe(recipe_id)

    def list_recipes(self, filters: Optional[dict[str, Any]] = None) -> list[PrintRecipe]:
        """
        List recipes with optional filters.

        Args:
            filters: Optional filters (paper_type, chemistry_type, tags, etc.)

        Returns:
            List of matching recipes
        """
        return self.database.query_recipes(filters or {})

    def search_recipes(self, query: str) -> list[PrintRecipe]:
        """
        Full-text search across recipes.

        Searches in name, description, paper_type, notes, and tags.

        Args:
            query: Search query string

        Returns:
            List of matching recipes
        """
        query_lower = query.lower()
        results = []

        for recipe in self.database.list_all_recipes():
            # Search in multiple fields
            searchable = [
                recipe.name,
                recipe.description or "",
                recipe.paper_type,
                recipe.notes or "",
                " ".join(recipe.tags),
            ]

            if any(query_lower in field.lower() for field in searchable):
                results.append(recipe)

        return results

    def get_recipe_by_id(self, recipe_id: UUID) -> Optional[PrintRecipe]:
        """
        Get a single recipe by ID.

        Args:
            recipe_id: Recipe identifier

        Returns:
            PrintRecipe or None if not found
        """
        return self.database.get_recipe(recipe_id)

    def export_recipe(self, recipe_id: UUID, format: RecipeFormat = RecipeFormat.JSON) -> str:
        """
        Export recipe to JSON or YAML format.

        Args:
            recipe_id: Recipe to export
            format: Export format (JSON or YAML)

        Returns:
            Serialized recipe string

        Raises:
            ValueError: If recipe not found or unsupported format
        """
        recipe = self.database.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe {recipe_id} not found")

        data = recipe.to_dict()

        if format == RecipeFormat.JSON:
            return json.dumps(data, indent=2, default=str)
        elif format == RecipeFormat.YAML:
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_recipe(self, file_path: Path) -> PrintRecipe:
        """
        Import recipe from JSON or YAML file.

        Args:
            file_path: Path to recipe file

        Returns:
            Imported PrintRecipe instance

        Raises:
            ValueError: If file not found or invalid format
        """
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        with open(file_path) as f:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Generate new ID for imported recipe to avoid conflicts
        data["recipe_id"] = str(uuid4())
        recipe = PrintRecipe.from_dict(data)
        self.database.add_recipe(recipe)
        return recipe

    def get_recipe_history(self, recipe_id: UUID) -> list[PrintRecipe]:
        """
        Get version history for a recipe.

        Args:
            recipe_id: Recipe identifier

        Returns:
            List of recipe versions, sorted by version number
        """
        recipe = self.database.get_recipe(recipe_id)
        if not recipe:
            return []

        # Find all recipes in the version chain
        history = [recipe]

        # Walk backwards through parent chain
        current = recipe
        while current.parent_recipe_id:
            parent = self.database.get_recipe(current.parent_recipe_id)
            if parent:
                history.insert(0, parent)
                current = parent
            else:
                break

        # Find all children (recipes derived from this one)
        all_recipes = self.database.list_all_recipes()
        for r in all_recipes:
            if r.parent_recipe_id == recipe_id and r not in history:
                history.append(r)

        return sorted(history, key=lambda r: r.version)

    def compare_recipes(self, recipe_ids: list[UUID]) -> dict[str, Any]:
        """
        Compare multiple recipes side-by-side.

        Args:
            recipe_ids: List of recipe IDs to compare

        Returns:
            Dictionary with comparison data

        Raises:
            ValueError: If any recipe not found
        """
        recipes = []
        for rid in recipe_ids:
            recipe = self.database.get_recipe(rid)
            if not recipe:
                raise ValueError(f"Recipe {rid} not found")
            recipes.append(recipe)

        if not recipes:
            return {}

        # Build comparison structure
        comparison = {
            "recipes": [r.to_dict() for r in recipes],
            "differences": {},
            "similarities": {},
        }

        # Find differences and similarities
        fields_to_compare = [
            "paper_type",
            "pt_pd_ratio",
            "ferric_oxalate_1_drops",
            "ferric_oxalate_2_drops",
            "metal_drops",
            "exposure_time_minutes",
            "uv_source",
            "developer",
            "developer_temperature_f",
        ]

        for field in fields_to_compare:
            values = [getattr(r, field) for r in recipes]
            if len(set(str(v) for v in values)) == 1:
                comparison["similarities"][field] = values[0]
            else:
                comparison["differences"][field] = {
                    str(r.recipe_id): getattr(r, field) for r in recipes
                }

        return comparison

    def suggest_similar_recipes(
        self, params: dict[str, Any], limit: int = 5, min_similarity: float = 0.5
    ) -> list[tuple[PrintRecipe, float]]:
        """
        Find similar recipes based on parameters.

        Args:
            params: Recipe parameters to match against
            limit: Maximum number of suggestions
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of (recipe, similarity_score) tuples, sorted by similarity
        """
        all_recipes = self.database.list_all_recipes()
        similarities = []

        for recipe in all_recipes:
            score = self._calculate_similarity(recipe, params)
            if score >= min_similarity:
                similarities.append((recipe, score))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

    def _calculate_similarity(self, recipe: PrintRecipe, params: dict[str, Any]) -> float:
        """
        Calculate similarity between recipe and parameters.

        Uses weighted scoring of matching fields.

        Args:
            recipe: Recipe to compare
            params: Parameters to match against

        Returns:
            Similarity score (0-1)
        """
        score = 0.0
        weights = 0.0

        # Paper type match (high weight)
        if "paper_type" in params:
            if recipe.paper_type.lower() == params["paper_type"].lower():
                score += 0.25
            weights += 0.25

        # Chemistry type match
        if "chemistry_type" in params:
            if str(recipe.chemistry_type.value) == str(params["chemistry_type"]):
                score += 0.15
            weights += 0.15

        # Metal ratio similarity
        if "pt_pd_ratio" in params:
            ratio_diff = abs(recipe.pt_pd_ratio - float(params["pt_pd_ratio"]))
            score += 0.15 * max(0, 1 - ratio_diff)
            weights += 0.15

        # Exposure time similarity (log scale)
        if "exposure_time_minutes" in params:
            import math

            exp_param = float(params["exposure_time_minutes"])
            if recipe.exposure_time_minutes > 0 and exp_param > 0:
                log_ratio = abs(
                    math.log(recipe.exposure_time_minutes) - math.log(exp_param)
                )
                score += 0.15 * max(0, 1 - log_ratio / 2)
            weights += 0.15

        # Developer match
        if "developer" in params:
            if str(recipe.developer.value) == str(params["developer"]):
                score += 0.10
            weights += 0.10

        # UV source match
        if "uv_source" in params:
            if recipe.uv_source.lower() == str(params["uv_source"]).lower():
                score += 0.10
            weights += 0.10

        # Tag overlap
        if "tags" in params:
            param_tags = set(params["tags"]) if isinstance(params["tags"], list) else set()
            recipe_tags = set(recipe.tags)
            if recipe_tags and param_tags:
                overlap = len(recipe_tags & param_tags) / len(recipe_tags | param_tags)
                score += 0.10 * overlap
            weights += 0.10

        return score / weights if weights > 0 else 0.0


class WorkflowStep(BaseModel):
    """A single step in a workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Step name")
    action: str = Field(..., description="Action to perform")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    depends_on: list[UUID] = Field(default_factory=list, description="Step dependencies")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    result: Optional[Any] = Field(default=None, description="Step result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkflowJob(BaseModel):
    """A workflow automation job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Job name")
    recipe_id: Optional[UUID] = Field(default=None, description="Associated recipe")
    steps: list[WorkflowStep] = Field(default_factory=list)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    scheduled_for: Optional[datetime] = Field(
        default=None, description="Scheduled execution time"
    )
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Job progress (0-1)")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class WorkflowAutomation:
    """
    Automates multi-step workflows for batch processing and scheduled tasks.

    Provides batch job creation, workflow execution, scheduling,
    status monitoring, and result logging.
    """

    def __init__(self):
        """Initialize workflow automation system."""
        self.jobs: dict[UUID, WorkflowJob] = {}
        self._job_callbacks: dict[UUID, list[Callable]] = {}
        self._job_sequence = 0

    def _register_job(self, job: WorkflowJob) -> WorkflowJob:
        """Register a job and ensure strict creation ordering."""
        job.created_at = datetime.now() + timedelta(microseconds=self._job_sequence)
        self._job_sequence += 1
        self.jobs[job.job_id] = job
        return job

    def create_batch_job(
        self, images: list[Path], recipe: PrintRecipe, output_dir: Path
    ) -> WorkflowJob:
        """
        Create a batch processing job for multiple images.

        Args:
            images: List of image file paths
            recipe: Recipe to apply
            output_dir: Output directory for processed images

        Returns:
            Created WorkflowJob instance
        """
        job = WorkflowJob(
            name=f"Batch process {len(images)} images with {recipe.name}",
            recipe_id=recipe.recipe_id,
        )

        # Create workflow steps
        for idx, image_path in enumerate(images):
            step = WorkflowStep(
                name=f"Process {image_path.name}",
                action="process_image",
                parameters={
                    "image_path": str(image_path),
                    "recipe": recipe.to_dict(),
                    "output_path": str(output_dir / f"{image_path.stem}_negative.tif"),
                },
            )
            job.steps.append(step)

        return self._register_job(job)

    def execute_workflow(self, workflow_steps: list[WorkflowStep]) -> WorkflowJob:
        """
        Execute a multi-step workflow.

        Args:
            workflow_steps: List of workflow steps to execute

        Returns:
            WorkflowJob tracking execution
        """
        job = WorkflowJob(name="Custom workflow", steps=workflow_steps)
        self._register_job(job)

        # Start execution
        job.status = WorkflowStatus.RUNNING
        job.started_at = datetime.now()

        try:
            self._execute_job(job)
            job.status = WorkflowStatus.COMPLETED
            job.progress = 1.0
        except Exception as e:
            job.status = WorkflowStatus.FAILED
            # Store error on job for debugging
            if job.steps:
                job.steps[-1].error = str(e)
        finally:
            job.completed_at = datetime.now()
            self._notify_callbacks(job.job_id)

        return job

    def schedule_workflow(
        self, workflow: list[WorkflowStep], schedule: datetime
    ) -> WorkflowJob:
        """
        Schedule a workflow for later execution.

        Args:
            workflow: List of workflow steps
            schedule: Datetime to execute the workflow

        Returns:
            Scheduled WorkflowJob
        """
        job = WorkflowJob(name="Scheduled workflow", steps=workflow, scheduled_for=schedule)
        return self._register_job(job)

    def get_workflow_status(self, job_id: UUID) -> Optional[WorkflowJob]:
        """
        Get the current status of a workflow job.

        Args:
            job_id: Job identifier

        Returns:
            WorkflowJob or None if not found
        """
        return self.jobs.get(job_id)

    def cancel_workflow(self, job_id: UUID) -> bool:
        """
        Cancel a running or pending workflow.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
            job.status = WorkflowStatus.CANCELLED
            job.completed_at = datetime.now()
            self._notify_callbacks(job_id)
            return True

        return False

    def log_workflow_result(
        self, job_id: UUID, result: dict[str, Any], success: bool = True
    ) -> bool:
        """
        Log the completion result of a workflow.

        Args:
            job_id: Job identifier
            result: Result data to log
            success: Whether the workflow succeeded

        Returns:
            True if logged, False if job not found
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        job.status = WorkflowStatus.COMPLETED if success else WorkflowStatus.FAILED
        job.completed_at = datetime.now()
        job.progress = 1.0

        # Store result on the last step or create a result step
        if job.steps:
            job.steps[-1].result = result
        else:
            step = WorkflowStep(
                name="Result",
                action="log_result",
                parameters=result,
                status=WorkflowStatus.COMPLETED,
            )
            job.steps.append(step)

        self._notify_callbacks(job_id)
        return True

    def register_callback(self, job_id: UUID, callback: Callable[[WorkflowJob], None]) -> None:
        """
        Register a callback for job completion.

        Args:
            job_id: Job identifier
            callback: Function to call when job completes
        """
        if job_id not in self._job_callbacks:
            self._job_callbacks[job_id] = []
        self._job_callbacks[job_id].append(callback)

    def list_jobs(
        self, status: Optional[WorkflowStatus] = None, limit: Optional[int] = None
    ) -> list[WorkflowJob]:
        """
        List workflow jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of WorkflowJob instances
        """
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def _execute_job(self, job: WorkflowJob) -> None:
        """
        Execute a workflow job by running all steps.

        Args:
            job: Job to execute
        """
        total_steps = len(job.steps)
        completed_steps = 0

        for step in job.steps:
            if job.status == WorkflowStatus.CANCELLED:
                break

            step.status = WorkflowStatus.RUNNING
            step.started_at = datetime.now()

            try:
                # Execute step based on action
                result = self._execute_step(step)
                step.result = result
                step.status = WorkflowStatus.COMPLETED
            except Exception as e:
                step.status = WorkflowStatus.FAILED
                step.error = str(e)
                raise
            finally:
                step.completed_at = datetime.now()
                completed_steps += 1
                job.progress = completed_steps / total_steps if total_steps > 0 else 1.0

    def _execute_step(self, step: WorkflowStep) -> Any:
        """
        Execute a single workflow step.

        Args:
            step: Step to execute

        Returns:
            Step result

        Note:
            This is a simplified implementation. In production, this would
            integrate with actual image processing, curve application, etc.
        """
        # Placeholder for actual step execution
        # In production, this would dispatch to appropriate handlers
        # based on step.action (process_image, apply_curve, etc.)
        return {"status": "simulated", "action": step.action}

    def _notify_callbacks(self, job_id: UUID) -> None:
        """
        Notify registered callbacks of job completion.

        Args:
            job_id: Job identifier
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        callbacks = self._job_callbacks.get(job_id, [])
        for callback in callbacks:
            try:
                callback(job)
            except Exception:
                # Silently ignore callback errors
                pass


class RecipeDatabase:
    """
    Persistent storage for print recipes.

    Provides SQLite-based storage with CRUD operations, search,
    filtering, and export/import capabilities.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize recipe database.

        Args:
            db_path: Path to SQLite database file.
                    If None, uses ~/.ptpd/recipes.db or in-memory database.
        """
        settings = get_settings()
        if db_path is None:
            db_path = settings.data_dir / "recipes.db"

        self.db_path = db_path
        self._ensure_database()
        self._recipe_cache: dict[UUID, PrintRecipe] = {}

    def _ensure_database(self) -> None:
        """Ensure database and tables exist."""
        if self.db_path != Path(":memory:"):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                recipe_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                paper_type TEXT NOT NULL,
                paper_profile_id TEXT,
                chemistry_type TEXT,
                pt_pd_ratio REAL,
                ferric_oxalate_1_drops REAL,
                ferric_oxalate_2_drops REAL,
                metal_drops REAL,
                contrast_agent TEXT,
                contrast_agent_drops REAL,
                developer TEXT,
                developer_temperature_f REAL,
                development_time_minutes REAL,
                exposure_time_minutes REAL,
                uv_source TEXT,
                uv_intensity_percent REAL,
                humidity_percent REAL,
                temperature_f REAL,
                coating_humidity_percent REAL,
                drying_time_hours REAL,
                curve_id TEXT,
                curve_name TEXT,
                version INTEGER,
                created_at TEXT,
                modified_at TEXT,
                parent_recipe_id TEXT,
                quality_rating REAL,
                successful_prints INTEGER,
                dmin_achieved REAL,
                dmax_achieved REAL,
                notes TEXT,
                author TEXT,
                recipe_json TEXT
            )
        """)

        # Create indices for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_paper_type ON recipes(paper_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chemistry_type ON recipes(chemistry_type)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON recipes(created_at)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_quality_rating ON recipes(quality_rating)"
        )

        conn.commit()
        conn.close()

    def add_recipe(self, recipe: PrintRecipe) -> None:
        """
        Add a recipe to the database.

        Args:
            recipe: Recipe to add
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        recipe_dict = recipe.to_dict()
        tags_json = json.dumps(recipe.tags)
        recipe_json = json.dumps(recipe_dict)

        cursor.execute(
            """
            INSERT OR REPLACE INTO recipes (
                recipe_id, name, description, tags, paper_type, paper_profile_id,
                chemistry_type, pt_pd_ratio, ferric_oxalate_1_drops, ferric_oxalate_2_drops,
                metal_drops, contrast_agent, contrast_agent_drops, developer,
                developer_temperature_f, development_time_minutes, exposure_time_minutes,
                uv_source, uv_intensity_percent, humidity_percent, temperature_f,
                coating_humidity_percent, drying_time_hours, curve_id, curve_name,
                version, created_at, modified_at, parent_recipe_id, quality_rating,
                successful_prints, dmin_achieved, dmax_achieved, notes, author, recipe_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(recipe.recipe_id),
                recipe.name,
                recipe.description,
                tags_json,
                recipe.paper_type,
                str(recipe.paper_profile_id) if recipe.paper_profile_id else None,
                recipe.chemistry_type.value,
                recipe.pt_pd_ratio,
                recipe.ferric_oxalate_1_drops,
                recipe.ferric_oxalate_2_drops,
                recipe.metal_drops,
                recipe.contrast_agent.value,
                recipe.contrast_agent_drops,
                recipe.developer.value,
                recipe.developer_temperature_f,
                recipe.development_time_minutes,
                recipe.exposure_time_minutes,
                recipe.uv_source,
                recipe.uv_intensity_percent,
                recipe.humidity_percent,
                recipe.temperature_f,
                recipe.coating_humidity_percent,
                recipe.drying_time_hours,
                str(recipe.curve_id) if recipe.curve_id else None,
                recipe.curve_name,
                recipe.version,
                recipe.created_at.isoformat(),
                recipe.modified_at.isoformat(),
                str(recipe.parent_recipe_id) if recipe.parent_recipe_id else None,
                recipe.quality_rating,
                recipe.successful_prints,
                recipe.dmin_achieved,
                recipe.dmax_achieved,
                recipe.notes,
                recipe.author,
                recipe_json,
            ),
        )

        conn.commit()
        conn.close()

        # Update cache
        self._recipe_cache[recipe.recipe_id] = recipe

    def get_recipe(self, recipe_id: UUID) -> Optional[PrintRecipe]:
        """
        Get a recipe by ID.

        Args:
            recipe_id: Recipe identifier

        Returns:
            PrintRecipe or None if not found
        """
        # Check cache first
        if recipe_id in self._recipe_cache:
            return self._recipe_cache[recipe_id]

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT recipe_json FROM recipes WHERE recipe_id = ?", (str(recipe_id),))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        recipe_dict = json.loads(row[0])
        recipe = PrintRecipe.from_dict(recipe_dict)
        self._recipe_cache[recipe_id] = recipe
        return recipe

    def update_recipe(self, recipe: PrintRecipe) -> None:
        """
        Update an existing recipe.

        Args:
            recipe: Updated recipe
        """
        recipe.modified_at = datetime.now()
        self.add_recipe(recipe)  # Uses INSERT OR REPLACE

    def delete_recipe(self, recipe_id: UUID) -> bool:
        """
        Delete a recipe.

        Args:
            recipe_id: Recipe identifier

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM recipes WHERE recipe_id = ?", (str(recipe_id),))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        # Remove from cache
        self._recipe_cache.pop(recipe_id, None)

        return deleted

    def list_all_recipes(self) -> list[PrintRecipe]:
        """
        Get all recipes.

        Returns:
            List of all recipes
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT recipe_json FROM recipes ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        recipes = []
        for row in rows:
            recipe_dict = json.loads(row[0])
            recipe = PrintRecipe.from_dict(recipe_dict)
            self._recipe_cache[recipe.recipe_id] = recipe
            recipes.append(recipe)

        return recipes

    def query_recipes(self, filters: dict[str, Any]) -> list[PrintRecipe]:
        """
        Query recipes with filters.

        Args:
            filters: Dictionary of filter criteria

        Returns:
            List of matching recipes
        """
        # Start with all recipes
        recipes = self.list_all_recipes()

        # Apply filters
        if "paper_type" in filters:
            recipes = [
                r for r in recipes if r.paper_type.lower() == filters["paper_type"].lower()
            ]

        if "chemistry_type" in filters:
            recipes = [r for r in recipes if r.chemistry_type.value == filters["chemistry_type"]]

        if "min_quality_rating" in filters:
            min_rating = filters["min_quality_rating"]
            recipes = [
                r for r in recipes if r.quality_rating and r.quality_rating >= min_rating
            ]

        if "tags" in filters:
            filter_tags = set(filters["tags"]) if isinstance(filters["tags"], list) else {filters["tags"]}
            recipes = [r for r in recipes if any(tag in r.tags for tag in filter_tags)]

        if "uv_source" in filters:
            recipes = [
                r for r in recipes if r.uv_source.lower() == filters["uv_source"].lower()
            ]

        if "developer" in filters:
            recipes = [r for r in recipes if r.developer.value == filters["developer"]]

        return recipes

    def export_all(self, output_path: Path, format: RecipeFormat = RecipeFormat.JSON) -> None:
        """
        Export all recipes to a file.

        Args:
            output_path: Output file path
            format: Export format (JSON or YAML)
        """
        recipes = self.list_all_recipes()
        data = {"version": "1.0", "recipes": [r.to_dict() for r in recipes]}

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            if format == RecipeFormat.JSON:
                json.dump(data, f, indent=2, default=str)
            else:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def import_all(self, input_path: Path) -> int:
        """
        Import recipes from a file.

        Args:
            input_path: Input file path

        Returns:
            Number of recipes imported
        """
        with open(input_path) as f:
            if input_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        count = 0
        for recipe_dict in data.get("recipes", []):
            recipe = PrintRecipe.from_dict(recipe_dict)
            self.add_recipe(recipe)
            count += 1

        return count

    def get_statistics(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM recipes")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT paper_type) FROM recipes")
        paper_types = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT chemistry_type) FROM recipes")
        chemistry_types = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(quality_rating) FROM recipes WHERE quality_rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0]

        conn.close()

        return {
            "total_recipes": total,
            "unique_paper_types": paper_types,
            "unique_chemistry_types": chemistry_types,
            "average_quality_rating": avg_rating,
        }
