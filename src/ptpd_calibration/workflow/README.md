# Recipe Management and Workflow Automation

Comprehensive recipe management and workflow automation system for repeatable platinum/palladium printing.

## Overview

The workflow module provides four main components:

1. **PrintRecipe**: Pydantic model for storing complete print recipes
2. **RecipeManager**: High-level interface for recipe management
3. **RecipeDatabase**: SQLite-based persistent storage
4. **WorkflowAutomation**: Batch processing and workflow orchestration

## Features

### Recipe Management

- **Complete Recipe Capture**: Store all parameters needed to reproduce prints
  - Paper type and profile
  - Chemistry settings (Pt/Pd ratio, ferric oxalate, contrast agents)
  - Exposure settings (time, UV source, intensity)
  - Environmental conditions (humidity, temperature, drying time)
  - Developer settings
  - Calibration curve references

- **Version Control**: Clone recipes with modifications and track version history
- **Quality Tracking**: Record print results and maintain quality ratings
- **Search & Discovery**: Full-text search and advanced filtering
- **Similarity Matching**: Find similar recipes based on parameters
- **Export/Import**: JSON and YAML support for sharing recipes

### Workflow Automation

- **Batch Processing**: Process multiple images with the same recipe
- **Multi-step Workflows**: Define and execute complex workflow sequences
- **Scheduling**: Schedule workflows for later execution
- **Progress Tracking**: Monitor job status and progress
- **Callback System**: Register callbacks for job completion events

### Database Features

- **SQLite Storage**: Reliable, file-based persistence
- **Indexed Queries**: Fast searches on common fields
- **Transaction Safety**: ACID compliance for data integrity
- **Caching**: In-memory cache for frequently accessed recipes
- **Statistics**: Query database statistics and usage patterns

## Quick Start

### Basic Recipe Creation

```python
from ptpd_calibration.workflow import RecipeManager
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType

# Create a recipe manager
manager = RecipeManager()

# Create a recipe
recipe = manager.create_recipe(
    name="Classic Palladium Print",
    description="Standard palladium recipe for Arches Platine",
    paper_type="Arches Platine",
    chemistry_type=ChemistryType.PURE_PALLADIUM,
    pt_pd_ratio=0.0,  # Pure palladium
    ferric_oxalate_1_drops=24.0,
    ferric_oxalate_2_drops=6.0,
    metal_drops=24.0,
    contrast_agent=ContrastAgent.NA2,
    contrast_agent_drops=6.0,
    developer=DeveloperType.POTASSIUM_OXALATE,
    exposure_time_minutes=10.0,
    uv_source="UV LED Array",
    humidity_percent=50.0,
    temperature_f=68.0,
    tags=["beginner", "palladium"],
    notes="Excellent starting point for palladium printing"
)

print(f"Created recipe: {recipe.name} (ID: {recipe.recipe_id})")
```

### Recipe Cloning

```python
# Clone with modifications for different paper
modifications = {
    "name": "Classic Palladium - Bergger COT320",
    "paper_type": "Bergger COT320",
    "exposure_time_minutes": 12.0,  # Bergger needs longer exposure
}

cloned = manager.clone_recipe(recipe.recipe_id, modifications)
print(f"Cloned recipe version {cloned.version}")
```

### Searching Recipes

```python
# Full-text search
results = manager.search_recipes("platinum")

# Filter by specific criteria
arches_recipes = manager.list_recipes({"paper_type": "Arches Platine"})

# Filter by tags
beginner_recipes = manager.list_recipes({"tags": ["beginner"]})

# Find similar recipes
similar = manager.suggest_similar_recipes(
    {"paper_type": "Arches Platine", "pt_pd_ratio": 0.0},
    limit=5,
    min_similarity=0.5
)
```

### Recipe Comparison

```python
# Compare multiple recipes
comparison = manager.compare_recipes([recipe_id_1, recipe_id_2])

print("Similarities:", comparison["similarities"])
print("Differences:", comparison["differences"])
```

### Export and Import

```python
from ptpd_calibration.workflow import RecipeFormat

# Export to JSON
json_str = manager.export_recipe(recipe.recipe_id, RecipeFormat.JSON)

# Export to YAML
yaml_str = manager.export_recipe(recipe.recipe_id, RecipeFormat.YAML)

# Import from file
imported = manager.import_recipe(Path("recipe.json"))
```

### Workflow Automation

```python
from ptpd_calibration.workflow import WorkflowAutomation, WorkflowStep
from pathlib import Path

# Create workflow automation system
workflow = WorkflowAutomation()

# Create batch job
images = [Path("image1.tif"), Path("image2.tif"), Path("image3.tif")]
output_dir = Path("output")

job = workflow.create_batch_job(images, recipe, output_dir)
print(f"Created job with {len(job.steps)} steps")

# Check status
status = workflow.get_workflow_status(job.job_id)
print(f"Job status: {status.status.value}, Progress: {status.progress:.0%}")

# Register completion callback
def on_complete(job):
    print(f"Job {job.name} completed!")

workflow.register_callback(job.job_id, on_complete)
```

### Database Operations

```python
from ptpd_calibration.workflow import RecipeDatabase
from pathlib import Path

# Create database
db = RecipeDatabase(Path("~/.ptpd/recipes.db"))

# Add recipe
db.add_recipe(recipe)

# Get recipe
retrieved = db.get_recipe(recipe.recipe_id)

# Query recipes
results = db.query_recipes({
    "paper_type": "Arches Platine",
    "min_quality_rating": 4.0
})

# Get statistics
stats = db.get_statistics()
print(f"Total recipes: {stats['total_recipes']}")
print(f"Average rating: {stats['average_quality_rating']:.2f}")

# Export all recipes
db.export_all(Path("all_recipes.json"), RecipeFormat.JSON)

# Import recipes
count = db.import_all(Path("shared_recipes.json"))
print(f"Imported {count} recipes")
```

### Quality Tracking

```python
# Update quality after a print
recipe.update_quality(
    rating=4.5,  # 0-5 scale
    dmin=0.07,
    dmax=1.72
)

print(f"Quality rating: {recipe.quality_rating:.2f}")
print(f"Successful prints: {recipe.successful_prints}")
print(f"Density range: {recipe.dmin_achieved} - {recipe.dmax_achieved}")
```

## Configuration

The workflow module can be configured via environment variables with the `PTPD_WORKFLOW_` prefix:

```bash
# Database settings
PTPD_WORKFLOW_RECIPE_DB_PATH=/path/to/recipes.db
PTPD_WORKFLOW_AUTO_BACKUP=true
PTPD_WORKFLOW_BACKUP_INTERVAL_HOURS=24

# Batch processing
PTPD_WORKFLOW_DEFAULT_MAX_WORKERS=4
PTPD_WORKFLOW_BATCH_TIMEOUT_MINUTES=60

# Workflow execution
PTPD_WORKFLOW_ENABLE_SCHEDULING=true
PTPD_WORKFLOW_MAX_CONCURRENT_WORKFLOWS=3
```

Or programmatically:

```python
from ptpd_calibration.config import get_settings

settings = get_settings()
settings.workflow.default_max_workers = 8
settings.workflow.batch_timeout_minutes = 120
```

## Data Model

### PrintRecipe Fields

| Field | Type | Description |
|-------|------|-------------|
| recipe_id | UUID | Unique identifier |
| name | str | Recipe name |
| description | str | Detailed description |
| tags | list[str] | Searchable tags |
| paper_type | str | Paper type name |
| paper_profile_id | UUID | Paper profile reference |
| chemistry_type | ChemistryType | Chemistry type enum |
| pt_pd_ratio | float | Platinum ratio (0=Pd, 1=Pt) |
| ferric_oxalate_1_drops | float | FO #1 drops |
| ferric_oxalate_2_drops | float | FO #2 drops (contrast) |
| metal_drops | float | Total metal drops |
| contrast_agent | ContrastAgent | Contrast agent type |
| contrast_agent_drops | float | Contrast agent amount |
| developer | DeveloperType | Developer type |
| developer_temperature_f | float | Developer temp (F) |
| development_time_minutes | float | Development time |
| exposure_time_minutes | float | Exposure time |
| uv_source | str | UV light source |
| uv_intensity_percent | float | UV intensity (%) |
| humidity_percent | float | Relative humidity (%) |
| temperature_f | float | Ambient temperature (F) |
| coating_humidity_percent | float | Coating humidity (%) |
| drying_time_hours | float | Drying time (hours) |
| curve_id | UUID | Calibration curve ID |
| curve_name | str | Curve name reference |
| version | int | Recipe version number |
| created_at | datetime | Creation timestamp |
| modified_at | datetime | Modification timestamp |
| parent_recipe_id | UUID | Parent for version tracking |
| quality_rating | float | Quality rating (0-5) |
| successful_prints | int | Number of successful prints |
| dmin_achieved | float | Achieved minimum density |
| dmax_achieved | float | Achieved maximum density |
| notes | str | Additional notes |
| author | str | Recipe author |

## Examples

See `examples/recipe_management_demo.py` for a comprehensive demonstration of all features.

Run the demo:

```bash
python examples/recipe_management_demo.py
```

## Best Practices

### Recipe Naming

Use descriptive names that include key information:
- Paper type
- Chemistry type
- Key characteristic

Examples:
- "Arches Platine - Pure Palladium - Standard"
- "Bergger COT320 - Pt/Pd Mix - High Contrast"
- "Hahnemuhle - Pure Platinum - Low Humidity"

### Tags

Use consistent tags for better searchability:
- Skill level: `beginner`, `intermediate`, `advanced`
- Chemistry: `platinum`, `palladium`, `mixed`
- Characteristics: `high-contrast`, `warm-tone`, `cold-tone`
- Paper: `smooth`, `textured`, `heavy-sizing`
- Purpose: `testing`, `production`, `exhibition`

### Version Control

When cloning recipes:
- Always provide meaningful modifications
- Document why the changes were made in notes
- Use version tracking to find recipe evolution

### Quality Tracking

- Update quality ratings after each print
- Record actual Dmin/Dmax achieved
- Add notes about print quality and issues
- Use ratings to identify best recipes

### Database Management

- Regular backups (automatic with auto_backup=true)
- Export recipes for sharing with other printers
- Organize with tags and clear naming
- Clean up failed/outdated recipes periodically

## Troubleshooting

### Database Locked Error

If you get "database is locked" errors:
- Ensure only one process accesses the database
- Check file permissions on database file
- Use longer timeout in SQLite connection

### Recipe Not Found

If recipes aren't found after import:
- Check that UUIDs match expected format
- Verify JSON/YAML structure is valid
- Ensure recipe was actually added to database

### Workflow Not Executing

If workflows don't run:
- Check workflow status for errors
- Verify all step dependencies are met
- Review step parameters for correctness
- Check logs for detailed error messages

## API Reference

See inline docstrings for complete API documentation:

```python
help(PrintRecipe)
help(RecipeManager)
help(RecipeDatabase)
help(WorkflowAutomation)
```

## Contributing

When adding features:
1. Follow existing code patterns
2. Add comprehensive docstrings
3. Include type hints
4. Update this README
5. Add tests in `tests/workflow/`
6. Update the demo script with examples

## License

Part of the Platinum-Palladium AI Printing Tool.
See project LICENSE for details.
