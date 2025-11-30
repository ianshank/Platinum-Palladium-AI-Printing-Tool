# Recipe Management Quick Reference

## Import

```python
from ptpd_calibration.workflow import (
    PrintRecipe,
    RecipeManager,
    RecipeDatabase,
    RecipeFormat,
    WorkflowAutomation,
)
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
```

## Create Recipe

```python
manager = RecipeManager()

recipe = manager.create_recipe(
    name="My Recipe",
    paper_type="Arches Platine",
    chemistry_type=ChemistryType.PURE_PALLADIUM,
    pt_pd_ratio=0.0,
    ferric_oxalate_1_drops=24.0,
    metal_drops=24.0,
    exposure_time_minutes=10.0,
    uv_source="UV LED",
    tags=["test", "palladium"],
)
```

## Clone Recipe

```python
cloned = manager.clone_recipe(
    recipe.recipe_id,
    {"paper_type": "Bergger COT320", "exposure_time_minutes": 12.0}
)
```

## Search Recipes

```python
# Full-text search
results = manager.search_recipes("platinum")

# Filter by paper
arches = manager.list_recipes({"paper_type": "Arches Platine"})

# Filter by tags
beginner = manager.list_recipes({"tags": ["beginner"]})

# Filter by quality
high_quality = manager.list_recipes({"min_quality_rating": 4.0})
```

## Find Similar Recipes

```python
similar = manager.suggest_similar_recipes(
    {"paper_type": "Arches Platine", "pt_pd_ratio": 0.0},
    limit=5,
    min_similarity=0.5
)

for recipe, score in similar:
    print(f"{recipe.name}: {score:.0%} similar")
```

## Compare Recipes

```python
comparison = manager.compare_recipes([recipe_id_1, recipe_id_2])

print("Same:", comparison["similarities"])
print("Different:", comparison["differences"])
```

## Update Recipe

```python
updated = manager.update_recipe(
    recipe.recipe_id,
    {"exposure_time_minutes": 11.0, "notes": "Increased exposure"}
)
```

## Export/Import

```python
# Export to JSON
json_str = manager.export_recipe(recipe.recipe_id, RecipeFormat.JSON)

# Export to YAML
yaml_str = manager.export_recipe(recipe.recipe_id, RecipeFormat.YAML)

# Import
imported = manager.import_recipe(Path("recipe.json"))
```

## Quality Tracking

```python
recipe.update_quality(rating=4.5, dmin=0.07, dmax=1.72)

print(f"Quality: {recipe.quality_rating:.1f}/5.0")
print(f"Prints: {recipe.successful_prints}")
print(f"Dmin: {recipe.dmin_achieved}, Dmax: {recipe.dmax_achieved}")
```

## Workflow Automation

```python
workflow = WorkflowAutomation()

# Create batch job
images = [Path("image1.tif"), Path("image2.tif")]
job = workflow.create_batch_job(images, recipe, Path("output"))

# Check status
status = workflow.get_workflow_status(job.job_id)
print(f"Status: {status.status.value}, Progress: {status.progress:.0%}")

# Cancel job
workflow.cancel_workflow(job.job_id)
```

## Database Operations

```python
# Create database
db = RecipeDatabase(Path("~/.ptpd/recipes.db"))

# Add recipe
db.add_recipe(recipe)

# Get recipe
retrieved = db.get_recipe(recipe.recipe_id)

# Query
results = db.query_recipes({"paper_type": "Arches Platine"})

# Statistics
stats = db.get_statistics()
print(f"Total: {stats['total_recipes']}")

# Export all
db.export_all(Path("all_recipes.json"), RecipeFormat.JSON)

# Import all
count = db.import_all(Path("recipes.json"))
```

## Configuration

```python
from ptpd_calibration.config import get_settings

settings = get_settings()

# Database path
settings.workflow.recipe_db_path = Path("/custom/path/recipes.db")

# Workers
settings.workflow.default_max_workers = 8

# Timeouts
settings.workflow.batch_timeout_minutes = 120
```

## Environment Variables

```bash
# Database
export PTPD_WORKFLOW_RECIPE_DB_PATH=/path/to/recipes.db
export PTPD_WORKFLOW_AUTO_BACKUP=true

# Processing
export PTPD_WORKFLOW_DEFAULT_MAX_WORKERS=4
export PTPD_WORKFLOW_BATCH_TIMEOUT_MINUTES=60

# Scheduling
export PTPD_WORKFLOW_ENABLE_SCHEDULING=true
export PTPD_WORKFLOW_MAX_CONCURRENT_WORKFLOWS=3
```

## Common Patterns

### Create Recipe from Existing

```python
# Get existing recipe
original = manager.get_recipe_by_id(recipe_id)

# Clone and modify
new_recipe = manager.clone_recipe(
    original.recipe_id,
    {
        "name": f"{original.name} - Modified",
        "exposure_time_minutes": original.exposure_time_minutes * 1.2,
    }
)
```

### Build Recipe Library

```python
papers = ["Arches Platine", "Bergger COT320", "Hahnemuhle Platinum Rag"]
ratios = [0.0, 0.25, 0.5, 1.0]  # Pd, 25% Pt, 50% Pt, Pt

for paper in papers:
    for ratio in ratios:
        manager.create_recipe(
            name=f"{paper} - {int(ratio*100)}% Pt",
            paper_type=paper,
            pt_pd_ratio=ratio,
            ferric_oxalate_1_drops=24.0,
            metal_drops=24.0,
            exposure_time_minutes=10.0,
            uv_source="UV LED",
            tags=[paper.lower().replace(" ", "_"), f"pt_{int(ratio*100)}"],
        )
```

### Track Print Session

```python
# Create recipe
recipe = manager.create_recipe(name="Test Session", ...)

# After printing
recipe.update_quality(rating=4.0, dmin=0.08, dmax=1.65)
recipe.notes = "Good detail in highlights, shadows a bit flat"
manager.update_recipe(recipe.recipe_id, {"notes": recipe.notes})

# Next print
recipe.update_quality(rating=4.5, dmin=0.07, dmax=1.70)

# Check progress
print(f"Average quality: {recipe.quality_rating:.2f}")
print(f"Total prints: {recipe.successful_prints}")
```

### Batch Process with Workflow

```python
workflow = WorkflowAutomation()

# Create batch jobs for different recipes
for recipe in manager.list_recipes({"tags": ["production"]}):
    images = Path("images").glob(f"*{recipe.paper_type}*.tif")
    job = workflow.create_batch_job(list(images), recipe, Path("output"))

    # Register callback
    def on_complete(j):
        print(f"Completed: {j.name}")

    workflow.register_callback(job.job_id, on_complete)
```

## Common Queries

```python
# Best rated recipes
best = manager.list_recipes({"min_quality_rating": 4.5})

# Beginner-friendly recipes
beginner = manager.list_recipes({"tags": ["beginner"]})

# Pure palladium recipes
palladium = manager.list_recipes({
    "chemistry_type": ChemistryType.PURE_PALLADIUM.value
})

# Quick exposure recipes (< 10 min)
# Note: This requires custom filtering
quick = [r for r in manager.list_recipes() if r.exposure_time_minutes < 10]

# Recipes for specific paper
paper_recipes = manager.list_recipes({"paper_type": "Arches Platine"})
```

## Tips

1. **Use Tags Consistently**: Create a standard set of tags (beginner, advanced, high-contrast, etc.)
2. **Update Quality**: Always update quality metrics after printing
3. **Version Control**: Clone instead of modify when experimenting
4. **Export Regularly**: Export successful recipes for backup
5. **Document Notes**: Add detailed notes about results and observations
6. **Search First**: Before creating a recipe, search for similar ones
7. **Use Filters**: Combine filters to narrow down recipe searches
8. **Track History**: Use version history to see recipe evolution
