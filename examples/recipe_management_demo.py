"""
Demonstration of Recipe Management and Workflow Automation.

This script demonstrates the key features of the recipe management system
for platinum/palladium printing.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptpd_calibration.workflow import (
    PrintRecipe,
    RecipeDatabase,
    RecipeFormat,
    RecipeManager,
    WorkflowAutomation,
    WorkflowStep,
)
from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType


def demo_basic_recipe_creation():
    """Demonstrate creating and managing basic recipes."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Recipe Creation")
    print("=" * 70)

    # Create a recipe manager with in-memory database
    manager = RecipeManager()

    # Create a simple palladium recipe
    recipe = manager.create_recipe(
        name="Classic Palladium Print",
        description="Standard palladium recipe for Arches Platine paper",
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
        tags=["beginner", "palladium", "standard"],
        notes="Excellent starting point for palladium printing",
        author="Demo User",
    )

    print(f"✓ Created recipe: {recipe.name}")
    print(f"  Recipe ID: {recipe.recipe_id}")
    print(f"  Paper: {recipe.paper_type}")
    print(f"  Exposure: {recipe.exposure_time_minutes} minutes")
    print(f"  Tags: {', '.join(recipe.tags)}")

    return manager, recipe


def demo_recipe_cloning(manager, original_recipe):
    """Demonstrate recipe cloning with modifications."""
    print("\n" + "=" * 70)
    print("DEMO 2: Recipe Cloning and Versioning")
    print("=" * 70)

    # Clone recipe with modifications for different paper
    modifications = {
        "name": "Classic Palladium - Bergger COT320",
        "paper_type": "Bergger COT320",
        "exposure_time_minutes": 12.0,  # Bergger needs longer exposure
        "ferric_oxalate_1_drops": 20.0,  # Less absorbent paper
        "notes": "Adapted for Bergger COT320 - longer exposure needed",
    }

    cloned = manager.clone_recipe(original_recipe.recipe_id, modifications)

    print(f"✓ Cloned recipe: {cloned.name}")
    print(f"  Recipe ID: {cloned.recipe_id}")
    print(f"  Version: {cloned.version}")
    print(f"  Parent ID: {cloned.parent_recipe_id}")
    print(f"  Modified exposure: {cloned.exposure_time_minutes} minutes")

    return cloned


def demo_recipe_search(manager):
    """Demonstrate recipe search and filtering."""
    print("\n" + "=" * 70)
    print("DEMO 3: Recipe Search and Filtering")
    print("=" * 70)

    # Create a few more recipes for searching
    manager.create_recipe(
        name="High Contrast Platinum",
        paper_type="Hahnemuhle Platinum Rag",
        chemistry_type=ChemistryType.PURE_PLATINUM,
        pt_pd_ratio=1.0,
        ferric_oxalate_1_drops=24.0,
        ferric_oxalate_2_drops=0.0,
        metal_drops=24.0,
        exposure_time_minutes=15.0,
        uv_source="UV LED Array",
        tags=["platinum", "high-contrast", "advanced"],
    )

    manager.create_recipe(
        name="Warm Tone Pt/Pd Mix",
        paper_type="Arches Platine",
        chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
        pt_pd_ratio=0.25,
        ferric_oxalate_1_drops=24.0,
        ferric_oxalate_2_drops=6.0,
        metal_drops=24.0,
        exposure_time_minutes=11.0,
        uv_source="UV LED Array",
        tags=["mixed", "warm-tone", "beginner"],
    )

    # Search by text
    results = manager.search_recipes("platinum")
    print(f"\n✓ Search for 'platinum': {len(results)} results")
    for r in results:
        print(f"  - {r.name}")

    # Filter by paper type
    arches_recipes = manager.list_recipes({"paper_type": "Arches Platine"})
    print(f"\n✓ Filter by paper 'Arches Platine': {len(arches_recipes)} results")
    for r in arches_recipes:
        print(f"  - {r.name}")

    # Filter by tags
    beginner_recipes = manager.list_recipes({"tags": ["beginner"]})
    print(f"\n✓ Filter by tag 'beginner': {len(beginner_recipes)} results")
    for r in beginner_recipes:
        print(f"  - {r.name}")


def demo_recipe_comparison(manager):
    """Demonstrate recipe comparison."""
    print("\n" + "=" * 70)
    print("DEMO 4: Recipe Comparison")
    print("=" * 70)

    # Get all recipes
    recipes = manager.list_recipes()
    if len(recipes) >= 2:
        recipe_ids = [r.recipe_id for r in recipes[:2]]

        comparison = manager.compare_recipes(recipe_ids)

        print(f"\n✓ Comparing {len(recipe_ids)} recipes:")
        for r in recipes[:2]:
            print(f"  - {r.name}")

        print("\nSimilarities:")
        for field, value in comparison.get("similarities", {}).items():
            print(f"  {field}: {value}")

        print("\nDifferences:")
        for field, values in comparison.get("differences", {}).items():
            print(f"  {field}:")
            for recipe_id, value in values.items():
                print(f"    {recipe_id}: {value}")


def demo_similar_recipes(manager):
    """Demonstrate finding similar recipes."""
    print("\n" + "=" * 70)
    print("DEMO 5: Similar Recipe Suggestions")
    print("=" * 70)

    # Define search parameters
    params = {
        "paper_type": "Arches Platine",
        "pt_pd_ratio": 0.0,
        "exposure_time_minutes": 10.0,
        "tags": ["beginner"],
    }

    print("\nSearching for recipes similar to:")
    print(f"  Paper: {params['paper_type']}")
    print(f"  Pt/Pd Ratio: {params['pt_pd_ratio']}")
    print(f"  Exposure: {params['exposure_time_minutes']} min")

    similar = manager.suggest_similar_recipes(params, limit=3)

    print(f"\n✓ Found {len(similar)} similar recipes:")
    for recipe, score in similar:
        print(f"  - {recipe.name} (similarity: {score:.2%})")


def demo_recipe_export_import(manager):
    """Demonstrate recipe export and import."""
    print("\n" + "=" * 70)
    print("DEMO 6: Recipe Export and Import")
    print("=" * 70)

    # Get a recipe to export
    recipes = manager.list_recipes()
    if recipes:
        recipe = recipes[0]

        # Export to JSON
        json_export = manager.export_recipe(recipe.recipe_id, RecipeFormat.JSON)
        print(f"\n✓ Exported '{recipe.name}' to JSON")
        print(f"  Length: {len(json_export)} characters")
        print(f"  Preview: {json_export[:200]}...")

        # Export to YAML
        yaml_export = manager.export_recipe(recipe.recipe_id, RecipeFormat.YAML)
        print(f"\n✓ Exported '{recipe.name}' to YAML")
        print(f"  Length: {len(yaml_export)} characters")
        print(f"  Preview: {yaml_export[:200]}...")


def demo_workflow_automation():
    """Demonstrate workflow automation."""
    print("\n" + "=" * 70)
    print("DEMO 7: Workflow Automation")
    print("=" * 70)

    workflow = WorkflowAutomation()

    # Create a simple recipe for batch processing
    recipe = PrintRecipe(
        name="Batch Processing Recipe",
        paper_type="Arches Platine",
        pt_pd_ratio=0.0,
        ferric_oxalate_1_drops=24.0,
        metal_drops=24.0,
        exposure_time_minutes=10.0,
        uv_source="UV LED",
    )

    # Simulate batch processing of images
    image_paths = [
        Path("/tmp/image1.tif"),
        Path("/tmp/image2.tif"),
        Path("/tmp/image3.tif"),
    ]
    output_dir = Path("/tmp/output")

    job = workflow.create_batch_job(image_paths, recipe, output_dir)

    print(f"✓ Created batch job: {job.name}")
    print(f"  Job ID: {job.job_id}")
    print(f"  Number of steps: {len(job.steps)}")
    print(f"  Status: {job.status.value}")

    print("\nWorkflow steps:")
    for idx, step in enumerate(job.steps, 1):
        print(f"  {idx}. {step.name}")

    return workflow, job


def demo_workflow_status(workflow, job):
    """Demonstrate workflow status tracking."""
    print("\n" + "=" * 70)
    print("DEMO 8: Workflow Status and Monitoring")
    print("=" * 70)

    # Get status
    status = workflow.get_workflow_status(job.job_id)
    if status:
        print(f"✓ Job status: {status.status.value}")
        print(f"  Progress: {status.progress:.0%}")
        print(f"  Created: {status.created_at}")

        # List all jobs
        all_jobs = workflow.list_jobs(limit=5)
        print(f"\n✓ Total jobs in system: {len(all_jobs)}")


def demo_database_persistence():
    """Demonstrate database persistence."""
    print("\n" + "=" * 70)
    print("DEMO 9: Database Persistence")
    print("=" * 70)

    # Create database with temporary file
    db_path = Path("/tmp/ptpd_demo_recipes.db")
    db = RecipeDatabase(db_path)

    # Add some recipes
    for i in range(3):
        recipe = PrintRecipe(
            name=f"Test Recipe {i + 1}",
            paper_type="Test Paper",
            pt_pd_ratio=i * 0.3,
            ferric_oxalate_1_drops=24.0,
            metal_drops=24.0,
            exposure_time_minutes=10.0 + i,
            uv_source="UV LED",
        )
        db.add_recipe(recipe)

    print(f"✓ Created database at: {db_path}")
    print(f"  Database size: {db_path.stat().st_size} bytes")

    # Get statistics
    stats = db.get_statistics()
    print("\n✓ Database statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clean up
    db_path.unlink()
    print("\n✓ Cleaned up demo database")


def demo_quality_tracking(manager):
    """Demonstrate quality tracking and updates."""
    print("\n" + "=" * 70)
    print("DEMO 10: Quality Tracking")
    print("=" * 70)

    # Create a recipe
    recipe = manager.create_recipe(
        name="Quality Tracking Demo",
        paper_type="Arches Platine",
        pt_pd_ratio=0.0,
        ferric_oxalate_1_drops=24.0,
        metal_drops=24.0,
        exposure_time_minutes=10.0,
        uv_source="UV LED",
    )

    print(f"✓ Created recipe: {recipe.name}")
    print(f"  Initial quality rating: {recipe.quality_rating}")
    print(f"  Successful prints: {recipe.successful_prints}")

    # Simulate printing and updating quality
    print("\n✓ Simulating prints and quality updates...")

    # First print
    recipe.update_quality(rating=4.0, dmin=0.08, dmax=1.65)
    print(f"  Print 1: Rating 4.0 → Current rating: {recipe.quality_rating:.2f}")

    # Second print
    recipe.update_quality(rating=4.5, dmin=0.07, dmax=1.70)
    print(f"  Print 2: Rating 4.5 → Current rating: {recipe.quality_rating:.2f}")

    # Third print
    recipe.update_quality(rating=5.0, dmin=0.07, dmax=1.72)
    print(f"  Print 3: Rating 5.0 → Current rating: {recipe.quality_rating:.2f}")

    print(f"\n✓ Final statistics:")
    print(f"  Average quality: {recipe.quality_rating:.2f}/5.0")
    print(f"  Successful prints: {recipe.successful_prints}")
    print(f"  Dmin achieved: {recipe.dmin_achieved}")
    print(f"  Dmax achieved: {recipe.dmax_achieved}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PLATINUM/PALLADIUM RECIPE MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 70)

    try:
        # Basic operations
        manager, recipe = demo_basic_recipe_creation()
        cloned = demo_recipe_cloning(manager, recipe)

        # Search and discovery
        demo_recipe_search(manager)
        demo_recipe_comparison(manager)
        demo_similar_recipes(manager)

        # Export/Import
        demo_recipe_export_import(manager)

        # Workflow automation
        workflow, job = demo_workflow_automation()
        demo_workflow_status(workflow, job)

        # Database and persistence
        demo_database_persistence()

        # Quality tracking
        demo_quality_tracking(manager)

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
