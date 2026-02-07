"""
Interactive tutorial system for Pt/Pd printing.

Provides structured, step-by-step tutorials for learning the Pt/Pd printing workflow,
from complete beginner guides to advanced techniques.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TutorialDifficulty(str, Enum):
    """Tutorial difficulty levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ActionType(str, Enum):
    """Types of actions required in tutorial steps."""

    READ = "read"
    PRACTICE = "practice"
    CALCULATE = "calculate"
    MIX = "mix"
    COAT = "coat"
    EXPOSE = "expose"
    DEVELOP = "develop"
    VERIFY = "verify"


class TutorialStep(BaseModel):
    """A single step in a tutorial."""

    step_number: int = Field(..., ge=1, description="Step number in sequence")
    title: str = Field(..., min_length=1, max_length=200, description="Step title")
    content: str = Field(..., min_length=1, description="Detailed step instructions")
    action: ActionType = Field(..., description="Type of action required")
    validation: str | None = Field(
        default=None, description="Validation criteria or expected outcome"
    )
    tips: list[str] = Field(default_factory=list, description="Additional tips for this step")
    warnings: list[str] = Field(
        default_factory=list, description="Important warnings or safety notes"
    )
    estimated_time_minutes: int | None = Field(
        default=None, ge=1, description="Estimated time to complete step"
    )


class Tutorial(BaseModel):
    """A complete tutorial with multiple steps."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=100, description="Tutorial identifier")
    display_name: str = Field(..., min_length=1, max_length=200, description="Display title")
    description: str = Field(..., min_length=1, description="Tutorial overview")
    difficulty: TutorialDifficulty = Field(..., description="Difficulty level")
    estimated_time: int = Field(..., ge=1, description="Total estimated time in minutes")
    prerequisites: list[str] = Field(
        default_factory=list, description="Required prior tutorials or knowledge"
    )
    steps: list[TutorialStep] = Field(..., min_length=1, description="Tutorial steps")
    learning_objectives: list[str] = Field(
        default_factory=list, description="What students will learn"
    )
    materials_needed: list[str] = Field(
        default_factory=list, description="Required materials and tools"
    )
    tags: list[str] = Field(default_factory=list, description="Searchable tags")


class UserProgress(BaseModel):
    """Tracks user progress through a tutorial."""

    tutorial_name: str = Field(..., description="Tutorial identifier")
    current_step: int = Field(default=0, ge=0, description="Current step number (0 = not started)")
    completed_steps: list[int] = Field(default_factory=list, description="Completed step numbers")
    started_at: datetime | None = Field(default=None, description="When tutorial was started")
    completed_at: datetime | None = Field(default=None, description="When tutorial was completed")
    notes: str = Field(default="", description="User notes")


# Tutorial content database
TUTORIALS_DATA = {
    "first_print": {
        "display_name": "Your First Platinum/Palladium Print",
        "description": (
            "A comprehensive beginner's guide to making your first Pt/Pd print. "
            "This tutorial covers all basics from chemistry preparation to final print."
        ),
        "difficulty": TutorialDifficulty.BEGINNER,
        "estimated_time": 180,
        "prerequisites": [],
        "learning_objectives": [
            "Understand the basic Pt/Pd printing process",
            "Safely prepare sensitizer solutions",
            "Coat paper evenly",
            "Make proper UV exposure",
            "Develop and clear prints correctly",
        ],
        "materials_needed": [
            "Ferric oxalate (24% solution)",
            "Palladium chloride (7% solution)",
            "Platinum chloride (15% solution)",
            "Potassium oxalate developer",
            "EDTA clearing agent",
            "Sizing solution (arrowroot or gelatin)",
            "100% cotton paper (Arches Platine recommended)",
            "Glass coating rod",
            "UV light source",
            "Digital negative or transparency",
            "Contact printing frame",
            "Trays for development and clearing",
            "Gloves and safety equipment",
        ],
        "tags": ["beginner", "first-print", "basics", "pt-pd"],
        "steps": [
            {
                "step_number": 1,
                "title": "Safety and Workspace Setup",
                "content": (
                    "Before beginning, ensure your workspace is properly set up:\n\n"
                    "1. Work in a well-ventilated area\n"
                    "2. Wear nitrile gloves throughout the process\n"
                    "3. Have clean, dedicated trays for each step\n"
                    "4. Prepare fresh distilled water in spray bottle\n"
                    "5. Set up drying rack away from UV light\n"
                    "6. Ensure UV light source is ready and calibrated"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Workspace is organized with all materials accessible",
                "warnings": [
                    "Platinum and palladium salts are light-sensitive - keep in dark storage",
                    "Always wear gloves when handling chemistry",
                    "Keep chemistry away from food preparation areas",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 2,
                "title": "Size Your Paper",
                "content": (
                    "Paper sizing is critical for even coating:\n\n"
                    "1. Cut paper to desired size (8x10 recommended for first print)\n"
                    "2. Prepare sizing solution (2% arrowroot or gelatin)\n"
                    "3. Heat solution to 40°C (104°F)\n"
                    "4. Brush or spray coating evenly on paper\n"
                    "5. Hang to dry completely (at least 1 hour)\n"
                    "6. Paper should feel smooth and slightly stiff when ready"
                ),
                "action": ActionType.COAT,
                "validation": "Paper is evenly sized and completely dry",
                "tips": [
                    "Multiple thin coats are better than one thick coat",
                    "Sized paper can be stored for months in dry conditions",
                    "Test sizing by placing water drop - it should bead up, not absorb",
                ],
                "estimated_time_minutes": 90,
            },
            {
                "step_number": 3,
                "title": "Prepare Sensitizer Solution",
                "content": (
                    "Mix your sensitizer in subdued lighting:\n\n"
                    "1. Work under safe light (tungsten only, no UV)\n"
                    "2. For 50/50 Pt/Pd ratio, mix equal parts:\n"
                    "   - 1ml ferric oxalate (24%)\n"
                    "   - 0.5ml palladium chloride (7%)\n"
                    "   - 0.5ml platinum chloride (15%)\n"
                    "3. Mix gently but thoroughly\n"
                    "4. This makes enough for approximately 8x10 print\n"
                    "5. Use immediately for best results"
                ),
                "action": ActionType.MIX,
                "validation": "Solution is well-mixed with no separation",
                "warnings": [
                    "Sensitizer is extremely light-sensitive once mixed",
                    "Work quickly but carefully",
                    "Any UV exposure will fog the print",
                ],
                "tips": [
                    "Keep stock solutions refrigerated",
                    "Label all chemistry clearly with dates",
                    "Room humidity should be 50-65% for optimal results",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 4,
                "title": "Coat the Paper",
                "content": (
                    "Apply sensitizer evenly to sized paper:\n\n"
                    "1. Place paper on clean, flat surface\n"
                    "2. Pour sensitizer in line across top of paper\n"
                    "3. Use glass rod to spread evenly downward\n"
                    "4. Use smooth, consistent strokes\n"
                    "5. Ensure complete, even coverage\n"
                    "6. Check for streaks or dry spots\n"
                    "7. Dry in dark (use fan on low if needed)"
                ),
                "action": ActionType.COAT,
                "validation": "Paper is evenly coated with no visible streaks",
                "tips": [
                    "Practice coating motion on plain paper first",
                    "Single smooth stroke is better than multiple passes",
                    "Coating should appear slightly glossy when wet",
                    "Typical dry time is 30-60 minutes depending on humidity",
                ],
                "warnings": [
                    "Do not over-work the coating - it can cause streaking",
                    "Keep in complete darkness while drying",
                ],
                "estimated_time_minutes": 45,
            },
            {
                "step_number": 5,
                "title": "Prepare for Exposure",
                "content": (
                    "Set up your contact printing frame:\n\n"
                    "1. Ensure negative is clean and dust-free\n"
                    "2. Place coated paper in frame (coated side up)\n"
                    "3. Place negative on top (emulsion side down)\n"
                    "4. Close frame ensuring perfect contact\n"
                    "5. Check for any gaps or air bubbles\n"
                    "6. Prepare timer for exposure"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Frame is properly loaded with good contact",
                "tips": [
                    "Perfect contact is essential for sharp prints",
                    "Use frame with sprung back for even pressure",
                    "Test exposure time will vary by UV source (typically 3-15 minutes)",
                ],
                "estimated_time_minutes": 5,
            },
            {
                "step_number": 6,
                "title": "Make the Exposure",
                "content": (
                    "Expose your print to UV light:\n\n"
                    "1. For first print, start with recommended time for your UV source:\n"
                    "   - Sunlight: 5-10 minutes (depends on conditions)\n"
                    "   - UV unit: 3-8 minutes (check unit specs)\n"
                    "2. Place frame under UV source\n"
                    "3. Start timer\n"
                    "4. Ensure even illumination across frame\n"
                    "5. Do not disturb during exposure\n"
                    "6. Remove from UV when timer completes"
                ),
                "action": ActionType.EXPOSE,
                "validation": "Exposure completed for recommended duration",
                "tips": [
                    "Image should be faintly visible after proper exposure",
                    "Under-exposure = low density print",
                    "Over-exposure = loss of highlight detail",
                    "Keep exposure log for future reference",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 7,
                "title": "Develop the Print",
                "content": (
                    "Develop your exposed print:\n\n"
                    "1. Prepare 20% potassium oxalate developer (68-75°F)\n"
                    "2. Fill tray with enough developer to cover print\n"
                    "3. Quickly immerse print face-down\n"
                    "4. Agitate gently and continuously\n"
                    "5. Image will appear and deepen (1-3 minutes)\n"
                    "6. Development is complete when no further change occurs\n"
                    "7. Remove and rinse briefly in water"
                ),
                "action": ActionType.DEVELOP,
                "validation": "Image is fully developed with good density",
                "tips": [
                    "Fresh developer gives best results",
                    "Gentle agitation prevents staining",
                    "Development typically completes in 2-3 minutes",
                    "Image will clear and intensify during development",
                ],
                "warnings": [
                    "Do not over-develop - it won't increase density",
                    "Maintain consistent developer temperature",
                ],
                "estimated_time_minutes": 5,
            },
            {
                "step_number": 8,
                "title": "Clear the Print",
                "content": (
                    "Remove unexposed sensitizer:\n\n"
                    "1. First rinse: plain water (2 minutes)\n"
                    "2. EDTA clearing bath (5 minutes):\n"
                    "   - Use 1% EDTA solution\n"
                    "   - Gentle agitation\n"
                    "3. Second rinse: running water (10 minutes)\n"
                    "4. Final rinse: distilled water (2 minutes)\n"
                    "5. Carefully remove from water"
                ),
                "action": ActionType.DEVELOP,
                "validation": "Whites are clean with no yellow staining",
                "tips": [
                    "Proper clearing is essential for archival permanence",
                    "Image may lighten slightly during clearing - this is normal",
                    "EDTA is preferred over traditional citric acid clearing",
                ],
                "estimated_time_minutes": 20,
            },
            {
                "step_number": 9,
                "title": "Dry and Evaluate",
                "content": (
                    "Final drying and assessment:\n\n"
                    "1. Blot print gently with clean towel\n"
                    "2. Hang to dry or place on blotters\n"
                    "3. Keep flat while drying\n"
                    "4. Allow to dry completely (several hours)\n"
                    "5. Evaluate final result:\n"
                    "   - Check highlight detail\n"
                    "   - Assess shadow density\n"
                    "   - Look for overall tone and color\n"
                    "   - Note any areas for improvement"
                ),
                "action": ActionType.VERIFY,
                "validation": "Print is dry and ready for evaluation",
                "tips": [
                    "Image will be slightly darker when dry",
                    "Color will warm slightly as print ages",
                    "First print is for learning - perfection comes with practice",
                ],
                "estimated_time_minutes": 5,
            },
            {
                "step_number": 10,
                "title": "Document and Learn",
                "content": (
                    "Record your results for future reference:\n\n"
                    "1. Note all process parameters:\n"
                    "   - Paper type and sizing\n"
                    "   - Chemistry ratios\n"
                    "   - Exposure time and source\n"
                    "   - Developer type and temperature\n"
                    "2. Identify what worked well\n"
                    "3. Note areas for improvement\n"
                    "4. Plan adjustments for next print\n"
                    "5. Congratulations on your first Pt/Pd print!"
                ),
                "action": ActionType.VERIFY,
                "validation": "Process notes are complete and organized",
                "tips": [
                    "Keep detailed notes - they're invaluable for consistency",
                    "Compare results with exposure targets",
                    "Join online communities to share and learn",
                ],
                "estimated_time_minutes": 10,
            },
        ],
    },
    "calibration_workflow": {
        "display_name": "Calibration and Linearization Workflow",
        "description": (
            "Learn how to create calibrated prints using step tablets and "
            "linearization curves for consistent, repeatable results."
        ),
        "difficulty": TutorialDifficulty.INTERMEDIATE,
        "estimated_time": 120,
        "prerequisites": ["first_print"],
        "learning_objectives": [
            "Understand the need for calibration in Pt/Pd printing",
            "Create and expose step tablet test prints",
            "Measure and analyze density readings",
            "Generate linearization curves",
            "Apply curves to final prints",
        ],
        "materials_needed": [
            "All materials from first print tutorial",
            "21-step tablet digital file",
            "Scanner or densitometer",
            "Calibration software (this tool)",
        ],
        "tags": ["intermediate", "calibration", "curves", "linearization"],
        "steps": [
            {
                "step_number": 1,
                "title": "Understanding Linearization",
                "content": (
                    "Why we need calibration:\n\n"
                    "Pt/Pd printing is non-linear - a 50% gray in your digital file "
                    "won't print as 50% density. Calibration creates a curve that "
                    "corrects this, ensuring:\n\n"
                    "- Accurate tonal reproduction\n"
                    "- Predictable results\n"
                    "- Consistent output across papers and chemistry\n"
                    "- Full tonal range utilization\n\n"
                    "The process:\n"
                    "1. Print step tablet with known values\n"
                    "2. Measure actual output densities\n"
                    "3. Generate correction curve\n"
                    "4. Apply curve to future images"
                ),
                "action": ActionType.READ,
                "validation": "Understand basic calibration principles",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 2,
                "title": "Print Your Step Tablet",
                "content": (
                    "Create a calibration step tablet:\n\n"
                    "1. Load 21-step tablet file (0-100% in 5% increments)\n"
                    "2. Print to transparency or create digital negative\n"
                    "3. Prepare paper exactly as you will for final prints:\n"
                    "   - Same sizing method\n"
                    "   - Same paper type\n"
                    "4. Mix sensitizer with your target chemistry ratio\n"
                    "5. Coat tablet at same size as test print\n"
                    "6. Use your standard exposure settings\n"
                    "7. Develop and clear normally"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Step tablet is properly printed and processed",
                "tips": [
                    "Consistency is key - use exact process for finals",
                    "Print multiple tablets if testing variables",
                    "Label each tablet with date and parameters",
                ],
                "estimated_time_minutes": 45,
            },
            {
                "step_number": 3,
                "title": "Scan and Measure the Tablet",
                "content": (
                    "Capture accurate density readings:\n\n"
                    "1. Let tablet dry completely (24 hours is best)\n"
                    "2. Scan at high resolution (600 DPI minimum):\n"
                    "   - Use flatbed scanner\n"
                    "   - Scan in color mode\n"
                    "   - Disable all auto-corrections\n"
                    "3. Save as uncompressed TIFF\n"
                    "4. Load scan into calibration software\n"
                    "5. Software will:\n"
                    "   - Detect step patches\n"
                    "   - Measure density values\n"
                    "   - Calculate actual response curve"
                ),
                "action": ActionType.VERIFY,
                "validation": "All 21 steps are accurately measured",
                "tips": [
                    "Clean scanner glass thoroughly",
                    "Use same scanner for all measurements",
                    "Consistent lighting in scanner is critical",
                ],
                "warnings": [
                    "Scanner profiles can affect readings - use raw mode",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 4,
                "title": "Generate Linearization Curve",
                "content": (
                    "Create your correction curve:\n\n"
                    "1. Software analyzes measured vs. expected values\n"
                    "2. Review curve characteristics:\n"
                    "   - Dmin (minimum density - paper white)\n"
                    "   - Dmax (maximum density - deepest black)\n"
                    "   - Overall shape and smoothness\n"
                    "3. Choose linearization target:\n"
                    "   - Linear: straight density response\n"
                    "   - Paper white: preserves natural paper tone\n"
                    "   - Aesthetic: enhanced for visual preference\n"
                    "4. Generate and save curve file\n"
                    "5. Name descriptively (paper-chemistry-date)"
                ),
                "action": ActionType.CALCULATE,
                "validation": "Curve is generated and saved",
                "tips": [
                    "Most users prefer 'paper white' linearization",
                    "Save curves with detailed metadata",
                    "Keep curve library organized by paper/chemistry",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 5,
                "title": "Apply Curve to Images",
                "content": (
                    "Use your calibration curve:\n\n"
                    "1. Open image in Photoshop or editing software\n"
                    "2. Apply curve via:\n"
                    "   - Curves adjustment layer (manual)\n"
                    "   - Import .acv curve file\n"
                    "   - Use calibration software's curve applicator\n"
                    "3. Verify curve is applied correctly\n"
                    "4. Make creative adjustments if desired\n"
                    "5. Output corrected image for printing\n"
                    "6. Print using exact same process as calibration tablet"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Curve is properly applied to test image",
                "tips": [
                    "Always apply curve before creative adjustments",
                    "Keep uncurved original as backup",
                    "Test on simple image first",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 6,
                "title": "Validate and Iterate",
                "content": (
                    "Verify calibration accuracy:\n\n"
                    "1. Print test image with applied curve\n"
                    "2. Compare to digital preview\n"
                    "3. Check key areas:\n"
                    "   - Highlight detail and separation\n"
                    "   - Midtone accuracy\n"
                    "   - Shadow depth and detail\n"
                    "4. If results are off:\n"
                    "   - Ensure exact process match\n"
                    "   - Re-check measurements\n"
                    "   - Consider re-calibration\n"
                    "5. Document final working process"
                ),
                "action": ActionType.VERIFY,
                "validation": "Prints match expected tonal values",
                "tips": [
                    "Re-calibrate when changing any variable",
                    "Environmental factors affect results",
                    "Build curve library for different conditions",
                ],
                "estimated_time_minutes": 25,
            },
        ],
    },
    "chemistry_mixing": {
        "display_name": "Chemistry Mixing and Ratios",
        "description": (
            "Master the art of mixing Pt/Pd sensitizer solutions, understanding "
            "metal ratios, and adjusting chemistry for different aesthetic effects."
        ),
        "difficulty": TutorialDifficulty.INTERMEDIATE,
        "estimated_time": 60,
        "prerequisites": ["first_print"],
        "learning_objectives": [
            "Understand Pt/Pd ratio effects on print characteristics",
            "Mix accurate chemistry solutions",
            "Adjust formulas for different papers",
            "Use contrast agents effectively",
            "Calculate chemistry costs and yields",
        ],
        "materials_needed": [
            "Ferric oxalate",
            "Palladium chloride",
            "Platinum chloride",
            "Contrast agents (optional)",
            "Graduated cylinders or pipettes",
            "Mixing containers",
            "Scale (0.01g precision)",
        ],
        "tags": ["chemistry", "mixing", "ratios", "formulas"],
        "steps": [
            {
                "step_number": 1,
                "title": "Understanding Metal Ratios",
                "content": (
                    "Platinum vs. Palladium characteristics:\n\n"
                    "Pure Palladium (0% Pt / 100% Pd):\n"
                    "- Warmer, brown tones\n"
                    "- Lower contrast\n"
                    "- Less expensive\n"
                    "- Faster exposure\n\n"
                    "50/50 Mix (50% Pt / 50% Pd):\n"
                    "- Neutral to warm tones\n"
                    "- Moderate contrast\n"
                    "- Balanced cost\n"
                    "- Most popular ratio\n\n"
                    "Pure Platinum (100% Pt / 0% Pd):\n"
                    "- Cooler, neutral tones\n"
                    "- Higher contrast\n"
                    "- More expensive\n"
                    "- Slower exposure"
                ),
                "action": ActionType.READ,
                "validation": "Understand how ratios affect print character",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 2,
                "title": "Standard Formula Mixing",
                "content": (
                    "Mix the classic Pt/Pd formula:\n\n"
                    "For 8x10 print (approximately 4ml total):\n\n"
                    "Solution A (sensitizer):\n"
                    "- 2ml ferric oxalate (24%)\n"
                    "- Mix with:\n\n"
                    "Solution B (metals):\n"
                    "For 50/50 ratio:\n"
                    "- 1ml palladium chloride (7% or 15%)\n"
                    "- 1ml platinum chloride (15%)\n\n"
                    "Mixing procedure:\n"
                    "1. Measure Solution A accurately\n"
                    "2. Add Solution B components\n"
                    "3. Mix gently but thoroughly\n"
                    "4. Use immediately\n\n"
                    "Yields: 1ml covers approximately 20 sq inches"
                ),
                "action": ActionType.MIX,
                "validation": "Formula is accurately mixed",
                "tips": [
                    "Measure carefully - small variations matter",
                    "Keep stock solutions refrigerated",
                    "Label everything with concentration and date",
                ],
                "warnings": [
                    "Work in subdued lighting",
                    "Mixed solution is light-sensitive",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 3,
                "title": "Adjusting for Contrast",
                "content": (
                    "Add contrast agents for enhanced density:\n\n"
                    "Common contrast agents and typical amounts:\n\n"
                    "1. Potassium Chlorate:\n"
                    "   - Start with 2-4 drops per 4ml\n"
                    "   - Increases contrast moderately\n"
                    "   - Coolest tone\n\n"
                    "2. NA2 (Sodium Chloroplatinate):\n"
                    "   - Very powerful contrast boost\n"
                    "   - Start with 1 drop per 4ml\n"
                    "   - Can increase Dmax significantly\n\n"
                    "3. Dichromate:\n"
                    "   - Moderate contrast increase\n"
                    "   - Warmest tone\n"
                    "   - Start with 2-3 drops per 4ml\n\n"
                    "Always test contrast agents with calibration prints first!"
                ),
                "action": ActionType.MIX,
                "validation": "Understand contrast agent usage",
                "tips": [
                    "Start with minimal amounts and increase gradually",
                    "Keep detailed notes on amounts used",
                    "Contrast agents affect exposure time",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 4,
                "title": "Paper-Specific Adjustments",
                "content": (
                    "Optimize chemistry for different papers:\n\n"
                    "Highly Sized Papers (Arches Platine, COT320):\n"
                    "- Standard formula works well\n"
                    "- May reduce ferric oxalate slightly for cooler tones\n\n"
                    "Less Sized Papers:\n"
                    "- May need increased metal salts\n"
                    "- Can add glycerin (1-2 drops) to slow absorption\n\n"
                    "Unsized Papers:\n"
                    "- Require more sensitizer\n"
                    "- Pre-size with arrowroot or gelatin\n"
                    "- May need multiple coats\n\n"
                    "Always test new papers with calibration tablets!"
                ),
                "action": ActionType.READ,
                "validation": "Understand paper-specific considerations",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 5,
                "title": "Cost Calculation and Efficiency",
                "content": (
                    "Calculate chemistry costs:\n\n"
                    "Typical costs (2024 prices, approximate):\n"
                    "- Palladium chloride: $60-80 per 30ml\n"
                    "- Platinum chloride: $40-60 per 30ml\n"
                    "- Ferric oxalate: $20-30 per 100ml\n\n"
                    "Cost per 8x10 print (50/50 ratio):\n"
                    "- Palladium: $2-3\n"
                    "- Platinum: $1-2\n"
                    "- Ferric oxalate: $0.50\n"
                    "- Total: $3.50-5.50 per print\n\n"
                    "Efficiency tips:\n"
                    "- Mix only what you need\n"
                    "- Use coating rod efficiently\n"
                    "- Consider smaller prints for testing\n"
                    "- Pure Pd is more economical for practice"
                ),
                "action": ActionType.CALCULATE,
                "validation": "Can calculate chemistry costs",
                "tips": [
                    "Buy chemistry in larger quantities for savings",
                    "Share costs with local printing community",
                    "Track usage to optimize efficiency",
                ],
                "estimated_time_minutes": 15,
            },
        ],
    },
    "coating_techniques": {
        "display_name": "Paper Coating Techniques",
        "description": (
            "Learn professional paper coating methods for even, streak-free "
            "sensitizer application and consistent results."
        ),
        "difficulty": TutorialDifficulty.INTERMEDIATE,
        "estimated_time": 45,
        "prerequisites": ["first_print"],
        "learning_objectives": [
            "Master glass rod coating technique",
            "Achieve even, streak-free coatings",
            "Troubleshoot common coating problems",
            "Learn alternative coating methods",
            "Optimize coating for different paper sizes",
        ],
        "materials_needed": [
            "Glass coating rod",
            "Practice paper (sized)",
            "Sensitizer (or water for practice)",
            "Flat coating surface",
            "Clips or tape",
        ],
        "tags": ["coating", "technique", "application"],
        "steps": [
            {
                "step_number": 1,
                "title": "Glass Rod Technique Fundamentals",
                "content": (
                    "Master the basic glass rod stroke:\n\n"
                    "1. Secure paper to flat surface (tape edges lightly)\n"
                    "2. Pour sensitizer in line across top edge\n"
                    "3. Hold rod at 30-45° angle\n"
                    "4. Draw rod down in ONE smooth motion\n"
                    "5. Maintain even pressure and speed\n"
                    "6. Lift rod at bottom in quick motion\n\n"
                    "Key principles:\n"
                    "- Single stroke is ideal\n"
                    "- Consistent speed prevents streaking\n"
                    "- Even pressure ensures uniform coating\n"
                    "- Practice with water first!"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Can execute smooth, even rod stroke",
                "tips": [
                    "Practice on scrap paper until confident",
                    "Watch coating flow behind rod",
                    "Adjust angle if pooling or skipping occurs",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 2,
                "title": "Troubleshooting Coating Issues",
                "content": (
                    "Fix common coating problems:\n\n"
                    "Streaking:\n"
                    "- Cause: Multiple passes or uneven pressure\n"
                    "- Fix: Single smooth stroke, practice consistency\n\n"
                    "Pooling at edges:\n"
                    "- Cause: Too much sensitizer or slow speed\n"
                    "- Fix: Reduce volume, increase speed\n\n"
                    "Dry spots:\n"
                    "- Cause: Insufficient sensitizer or poor sizing\n"
                    "- Fix: Check sizing quality, increase volume slightly\n\n"
                    "Bubbles:\n"
                    "- Cause: Vigorous mixing or too fast application\n"
                    "- Fix: Mix gently, let settle before coating\n\n"
                    "Uneven density:\n"
                    "- Cause: Variable coating thickness\n"
                    "- Fix: Maintain consistent rod angle and pressure"
                ),
                "action": ActionType.READ,
                "validation": "Understand common issues and solutions",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 3,
                "title": "Alternative Coating Methods",
                "content": (
                    "Other coating techniques:\n\n"
                    "Brush Coating:\n"
                    "- Use wide, soft brush (hake brush ideal)\n"
                    "- Apply in criss-cross pattern\n"
                    "- Good for small prints or practice\n"
                    "- Can show brush marks\n\n"
                    "Spray Coating:\n"
                    "- Use airbrush or spray bottle\n"
                    "- Multiple thin coats\n"
                    "- Very even coverage possible\n"
                    "- Requires more chemistry\n\n"
                    "Puddle Pusher:\n"
                    "- Acrylic or glass applicator\n"
                    "- Similar to rod technique\n"
                    "- Good control over coating thickness\n\n"
                    "Each method has advantages - experiment to find your preference!"
                ),
                "action": ActionType.READ,
                "validation": "Aware of alternative coating methods",
                "tips": [
                    "Glass rod is most popular for good reason",
                    "Brush good for learning and small prints",
                    "Spray excellent for very large prints",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 4,
                "title": "Coating Different Sizes",
                "content": (
                    "Adapt technique for various print sizes:\n\n"
                    "Small prints (5x7 and under):\n"
                    "- Use less sensitizer (1-2ml)\n"
                    "- Shorter rod or brush may be easier\n"
                    "- Quick drying time\n\n"
                    "Standard prints (8x10 to 11x14):\n"
                    "- Standard rod technique ideal\n"
                    "- 3-5ml sensitizer typical\n"
                    "- Moderate drying time\n\n"
                    "Large prints (16x20 and larger):\n"
                    "- May need longer rod or spray method\n"
                    "- Coat quickly to prevent drying during application\n"
                    "- Consider coating in sections\n"
                    "- Plan for extended drying time\n\n"
                    "Practice scaling technique before expensive large prints!"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Can adapt coating for different sizes",
                "tips": [
                    "Start with 8x10 to learn technique",
                    "Scale up gradually as confidence builds",
                    "Large prints require more workspace and planning",
                ],
                "estimated_time_minutes": 10,
            },
        ],
    },
    "exposure_control": {
        "display_name": "UV Exposure Techniques and Control",
        "description": (
            "Master UV exposure for optimal print quality, learn to work with "
            "different UV sources, and understand exposure testing methods."
        ),
        "difficulty": TutorialDifficulty.INTERMEDIATE,
        "estimated_time": 60,
        "prerequisites": ["first_print", "calibration_workflow"],
        "learning_objectives": [
            "Understand UV exposure principles",
            "Work with different UV sources effectively",
            "Perform exposure tests accurately",
            "Compensate for environmental variables",
            "Achieve consistent exposure results",
        ],
        "materials_needed": [
            "UV light source",
            "Contact printing frame",
            "Timer",
            "UV meter (optional but recommended)",
            "Test negatives",
            "Coated paper",
        ],
        "tags": ["exposure", "UV", "timing", "testing"],
        "steps": [
            {
                "step_number": 1,
                "title": "Understanding UV Sources",
                "content": (
                    "Different UV light sources and characteristics:\n\n"
                    "Natural Sunlight:\n"
                    "- Free and readily available\n"
                    "- Variable intensity (weather, season, time)\n"
                    "- Broad UV spectrum\n"
                    "- Requires exposure testing each session\n\n"
                    "UV LED Units:\n"
                    "- Consistent output\n"
                    "- Long lifespan\n"
                    "- Narrow UV spectrum (typically 365-405nm)\n"
                    "- Energy efficient\n\n"
                    "Fluorescent UV Tubes:\n"
                    "- Good UV output\n"
                    "- Tubes degrade over time\n"
                    "- Broader spectrum than LED\n"
                    "- Need replacement every 500-1000 hours\n\n"
                    "Metal Halide:\n"
                    "- Very powerful\n"
                    "- Fast exposures\n"
                    "- Generates heat\n"
                    "- Professional/commercial use"
                ),
                "action": ActionType.READ,
                "validation": "Understand UV source characteristics",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 2,
                "title": "Exposure Testing Methods",
                "content": (
                    "Determine optimal exposure time:\n\n"
                    "Method 1: Strip Test\n"
                    "1. Coat paper strip\n"
                    "2. Place negative over strip\n"
                    "3. Cover strip with opaque card\n"
                    "4. Expose, revealing strip every 1-2 minutes\n"
                    "5. Develop and evaluate density progression\n"
                    "6. Choose time with best Dmax and highlight detail\n\n"
                    "Method 2: Multiple Print Test\n"
                    "1. Coat several identical pieces\n"
                    "2. Expose at different times (e.g., 3, 5, 7, 9 minutes)\n"
                    "3. Process identically\n"
                    "4. Compare results when dry\n"
                    "5. Select optimal exposure\n\n"
                    "Method 3: Digital Step Tablet\n"
                    "1. Use this tool's exposure calculator\n"
                    "2. Print step tablet\n"
                    "3. Measure and analyze\n"
                    "4. Software recommends optimal time"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Can perform exposure test",
                "tips": [
                    "Test with actual image negative, not just step tablet",
                    "Process test prints identically",
                    "Evaluate when completely dry",
                    "Keep detailed notes of all variables",
                ],
                "estimated_time_minutes": 20,
            },
            {
                "step_number": 3,
                "title": "Environmental Compensation",
                "content": (
                    "Adjust for environmental variables:\n\n"
                    "Humidity Effects:\n"
                    "- High humidity (>65%): Longer exposure needed\n"
                    "- Low humidity (<45%): Shorter exposure needed\n"
                    "- Ideal range: 50-60%\n"
                    "- Track humidity with each session\n\n"
                    "Temperature Effects:\n"
                    "- Warmer temps: Faster exposure\n"
                    "- Cooler temps: Slower exposure\n"
                    "- Keep workspace temperature stable\n\n"
                    "UV Source Variations:\n"
                    "- Sunlight: Test each session\n"
                    "- Artificial: Test when bulbs changed\n"
                    "- Use UV meter for consistency\n"
                    "- Log source intensity if possible\n\n"
                    "Chemistry Age:\n"
                    "- Fresh chemistry: Standard exposure\n"
                    "- Aged stock solutions: May need more exposure\n"
                    "- Mixed sensitizer: Use immediately"
                ),
                "action": ActionType.READ,
                "validation": "Understand environmental factors",
                "tips": [
                    "Keep environmental log with prints",
                    "Use dehumidifier/humidifier to control humidity",
                    "Refrigerate chemistry to extend life",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 4,
                "title": "Achieving Consistency",
                "content": (
                    "Strategies for repeatable exposures:\n\n"
                    "1. Standardize Your Process:\n"
                    "   - Use same UV source when possible\n"
                    "   - Maintain consistent chemistry ratios\n"
                    "   - Control environmental conditions\n"
                    "   - Document everything\n\n"
                    "2. Use Measurement Tools:\n"
                    "   - UV meter for light intensity\n"
                    "   - Hygrometer for humidity\n"
                    "   - Thermometer for temperature\n"
                    "   - Timer (accurate to seconds)\n\n"
                    "3. Create Exposure Charts:\n"
                    "   - Record baseline exposures\n"
                    "   - Note adjustment factors\n"
                    "   - Update with experience\n"
                    "   - Include environmental data\n\n"
                    "4. Perform Regular Testing:\n"
                    "   - Test when changing variables\n"
                    "   - Verify with step tablets\n"
                    "   - Recalibrate periodically"
                ),
                "action": ActionType.VERIFY,
                "validation": "Have system for consistent exposures",
                "tips": [
                    "Consistency beats perfection - establish routine",
                    "Small variations are normal - don't overthink",
                    "Experience builds intuition over time",
                ],
                "estimated_time_minutes": 15,
            },
        ],
    },
    "troubleshooting": {
        "display_name": "Troubleshooting Common Problems",
        "description": (
            "Identify and fix common issues in Pt/Pd printing, from coating "
            "problems to development issues and print defects."
        ),
        "difficulty": TutorialDifficulty.INTERMEDIATE,
        "estimated_time": 90,
        "prerequisites": ["first_print"],
        "learning_objectives": [
            "Diagnose common printing problems",
            "Understand root causes of defects",
            "Apply appropriate corrections",
            "Prevent future issues",
            "Salvage problematic prints when possible",
        ],
        "materials_needed": [
            "Problem prints for analysis",
            "Fresh chemistry",
            "Test materials",
        ],
        "tags": ["troubleshooting", "problems", "defects", "fixes"],
        "steps": [
            {
                "step_number": 1,
                "title": "Low Density / Weak Shadows",
                "content": (
                    "Problem: Print lacks depth, shadows are weak or gray\n\n"
                    "Possible Causes:\n"
                    "1. Underexposure\n"
                    "   - Increase exposure time by 25-50%\n"
                    "   - Check UV source strength\n"
                    "   - Verify negative density\n\n"
                    "2. Weak Sensitizer\n"
                    "   - Use fresh chemistry\n"
                    "   - Increase metal salt concentrations\n"
                    "   - Check ferric oxalate strength\n\n"
                    "3. Poor Coating\n"
                    "   - Apply more sensitizer\n"
                    "   - Ensure even coating\n"
                    "   - Check paper sizing\n\n"
                    "4. Development Issues\n"
                    "   - Use fresh developer\n"
                    "   - Ensure proper temperature (68-75°F)\n"
                    "   - Develop until no further change\n\n"
                    "5. High Humidity\n"
                    "   - Increase exposure time\n"
                    "   - Dry coated paper in low humidity\n"
                    "   - Use dehumidifier"
                ),
                "action": ActionType.VERIFY,
                "validation": "Can diagnose density problems",
                "tips": [
                    "Test one variable at a time",
                    "Keep detailed notes to track changes",
                    "Underexposure is most common cause",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 2,
                "title": "Highlight Blocking / Loss of Detail",
                "content": (
                    "Problem: Highlights are too dense, losing delicate detail\n\n"
                    "Possible Causes:\n"
                    "1. Overexposure\n"
                    "   - Reduce exposure time by 20-30%\n"
                    "   - Check UV source (may be too strong)\n"
                    "   - Ensure proper contact in frame\n\n"
                    "2. Negative Too Dense\n"
                    "   - Check negative density range\n"
                    "   - May need curve adjustment\n"
                    "   - Consider new negative with less density\n\n"
                    "3. Too Much Contrast Agent\n"
                    "   - Reduce or eliminate contrast agents\n"
                    "   - Test with plain formula first\n\n"
                    "4. Poor Clearing\n"
                    "   - Ensure adequate EDTA clearing\n"
                    "   - Extend clearing time\n"
                    "   - Use fresh clearing solutions\n\n"
                    "Prevention:\n"
                    "- Always test exposure with step tablet\n"
                    "- Use negatives with proper density range\n"
                    "- Avoid excess contrast agents initially"
                ),
                "action": ActionType.VERIFY,
                "validation": "Understand highlight problems",
                "tips": [
                    "Perfect contact in frame is critical",
                    "Highlights are harder to fix than shadows",
                    "Conservative exposure preserves detail",
                ],
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 3,
                "title": "Streaking and Uneven Coating",
                "content": (
                    "Problem: Visible streaks or uneven density in coated areas\n\n"
                    "Causes and Solutions:\n\n"
                    "1. Coating Technique:\n"
                    "   - Use single, smooth glass rod stroke\n"
                    "   - Don't over-work the coating\n"
                    "   - Maintain consistent pressure and speed\n"
                    "   - Practice with water on scrap paper\n\n"
                    "2. Paper Sizing Issues:\n"
                    "   - Ensure even sizing coverage\n"
                    "   - Size may be too thin or uneven\n"
                    "   - Test sizing with water drop test\n"
                    "   - Re-size if necessary\n\n"
                    "3. Sensitizer Precipitation:\n"
                    "   - Mix sensitizer thoroughly\n"
                    "   - Use fresh chemistry\n"
                    "   - Check ferric oxalate for crystals\n"
                    "   - Filter if necessary\n\n"
                    "4. Surface Contamination:\n"
                    "   - Clean coating rod thoroughly\n"
                    "   - Ensure paper is clean and dry\n"
                    "   - Work in clean environment\n"
                    "   - Use clean water for sizing"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Can identify streaking causes",
                "tips": [
                    "Glass rod technique takes practice",
                    "Sizing quality is foundation of good coating",
                    "Keep all tools scrupulously clean",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 4,
                "title": "Yellow Staining / Poor Clearing",
                "content": (
                    "Problem: Print shows yellow staining in highlights or overall\n\n"
                    "Causes and Solutions:\n\n"
                    "1. Insufficient Clearing:\n"
                    "   - Extend EDTA clearing time (10-15 minutes)\n"
                    "   - Use fresh EDTA solution (1-2%)\n"
                    "   - Ensure continuous agitation\n"
                    "   - Follow with extended water rinse\n\n"
                    "2. Inadequate Rinsing:\n"
                    "   - Initial rinse: 5 minutes running water\n"
                    "   - Post-EDTA rinse: 15-20 minutes\n"
                    "   - Final distilled water rinse\n\n"
                    "3. Old or Exhausted Clearing Bath:\n"
                    "   - Mix fresh EDTA solution\n"
                    "   - Don't reuse clearing baths\n"
                    "   - One solution per 4-6 prints maximum\n\n"
                    "4. Overexposure:\n"
                    "   - Reduces clearing efficiency\n"
                    "   - More sensitizer to clear\n"
                    "   - Reduce exposure time\n\n"
                    "Salvaging Yellow Prints:\n"
                    "- Re-clear in fresh EDTA (up to 30 minutes)\n"
                    "- Some yellowing may be permanent\n"
                    "- Prevention is easier than cure"
                ),
                "action": ActionType.VERIFY,
                "validation": "Understand clearing issues",
                "warnings": [
                    "Insufficient clearing affects archival stability",
                    "Yellow stains can worsen over time",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 5,
                "title": "Edge Fogging and Light Leaks",
                "content": (
                    "Problem: Unwanted density at edges or streaks of fog\n\n"
                    "Causes and Solutions:\n\n"
                    "1. Light Leaks During Coating/Drying:\n"
                    "   - Check darkroom for light leaks\n"
                    "   - Use amber or red safelight only\n"
                    "   - Cover coated paper while drying\n"
                    "   - Dry in complete darkness if possible\n\n"
                    "2. Frame Contact Issues:\n"
                    "   - Ensure perfect contact at edges\n"
                    "   - Check frame springs/pressure\n"
                    "   - Use proper size negative\n"
                    "   - Add padding if needed\n\n"
                    "3. Handling Coated Paper:\n"
                    "   - Touch only edges\n"
                    "   - Use clean, dry hands or gloves\n"
                    "   - Don't expose coated paper to light\n\n"
                    "4. Storage Issues:\n"
                    "   - Store unexposed coated paper in darkness\n"
                    "   - Use black plastic or foil wrapping\n"
                    "   - Coat and use within hours for best results\n\n"
                    "Prevention:\n"
                    "- Test darkroom with sensitive material\n"
                    "- Minimize time between coating and exposure\n"
                    "- Handle coated paper minimally"
                ),
                "action": ActionType.VERIFY,
                "validation": "Can prevent fogging issues",
                "tips": [
                    "Even small light leaks can cause problems",
                    "Pt/Pd is very light-sensitive when coated",
                    "Paranoia about light exposure is justified",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 6,
                "title": "Mottle and Uneven Development",
                "content": (
                    "Problem: Blotchy or uneven appearance after development\n\n"
                    "Causes and Solutions:\n\n"
                    "1. Inadequate Agitation:\n"
                    "   - Agitate continuously during development\n"
                    "   - Use gentle rocking motion\n"
                    "   - Ensure developer covers print completely\n"
                    "   - Avoid aggressive agitation\n\n"
                    "2. Development Temperature:\n"
                    "   - Maintain 68-75°F (20-24°C)\n"
                    "   - Use thermometer\n"
                    "   - Warmer developer can cause mottle\n"
                    "   - Pre-warm chemicals to room temp\n\n"
                    "3. Developer Strength:\n"
                    "   - Use fresh developer (20% potassium oxalate)\n"
                    "   - Don't reuse exhausted developer\n"
                    "   - Mix fresh for important prints\n\n"
                    "4. Coating Issues:\n"
                    "   - Ensure even sensitizer coating\n"
                    "   - Check for precipitation in sensitizer\n"
                    "   - Verify paper sizing uniformity\n\n"
                    "5. Hard Water:\n"
                    "   - Use distilled or filtered water\n"
                    "   - Minerals can cause uneven development\n"
                    "   - Especially important for final rinses"
                ),
                "action": ActionType.VERIFY,
                "validation": "Understand mottle causes",
                "tips": [
                    "Fresh developer is cheap insurance",
                    "Gentle, continuous agitation is key",
                    "Water quality matters more than you think",
                ],
                "estimated_time_minutes": 20,
            },
        ],
    },
    "advanced_techniques": {
        "display_name": "Advanced Printing Techniques",
        "description": (
            "Explore advanced methods including split-grade printing, contrast "
            "masking, combination printing, and creative variations."
        ),
        "difficulty": TutorialDifficulty.ADVANCED,
        "estimated_time": 120,
        "prerequisites": ["first_print", "calibration_workflow", "chemistry_mixing"],
        "learning_objectives": [
            "Master split-grade printing techniques",
            "Create and use contrast masks",
            "Perform combination printing",
            "Experiment with creative chemistry variations",
            "Achieve maximum print quality and control",
        ],
        "materials_needed": [
            "All standard Pt/Pd materials",
            "Multiple chemistry formulations",
            "Masking materials",
            "Registration system",
            "Test papers",
        ],
        "tags": ["advanced", "masking", "split-grade", "creative"],
        "steps": [
            {
                "step_number": 1,
                "title": "Split-Grade Printing Overview",
                "content": (
                    "Split-grade printing uses multiple exposures with different "
                    "chemistry formulations to achieve extended tonal range and control.\n\n"
                    "Basic Concept:\n"
                    "1. First exposure: Low contrast chemistry (highlights)\n"
                    "2. Second exposure: High contrast chemistry (shadows)\n"
                    "3. Combines best of both in single print\n\n"
                    "Benefits:\n"
                    "- Extended density range beyond single chemistry\n"
                    "- Independent control of highlight/shadow contrast\n"
                    "- Can compensate for difficult negatives\n"
                    "- Creates unique tonal characteristics\n\n"
                    "Challenges:\n"
                    "- Requires precise registration\n"
                    "- More complex workflow\n"
                    "- Doubled chemistry and time costs\n"
                    "- Needs careful testing\n\n"
                    "Best suited for:\n"
                    "- Negatives with extreme density range\n"
                    "- Maximum tonal control requirements\n"
                    "- Fine art printing where time investment justified"
                ),
                "action": ActionType.READ,
                "validation": "Understand split-grade principles",
                "estimated_time_minutes": 10,
            },
            {
                "step_number": 2,
                "title": "Split-Grade Chemistry Formulation",
                "content": (
                    "Prepare two contrast-differentiated chemistries:\n\n"
                    "Low Contrast Formula (for highlights):\n"
                    "- Pure palladium or Pd-heavy ratio (70% Pd / 30% Pt)\n"
                    "- Standard ferric oxalate\n"
                    "- NO contrast agents\n"
                    "- Gentle, lower contrast\n\n"
                    "High Contrast Formula (for shadows):\n"
                    "- Pure platinum or Pt-heavy ratio (70% Pt / 30% Pd)\n"
                    "- Can increase ferric oxalate slightly\n"
                    "- Add contrast agent (NA2 or potassium chlorate)\n"
                    "- Aggressive, higher contrast\n\n"
                    "Testing Process:\n"
                    "1. Test each formula separately first\n"
                    "2. Create calibration curves for both\n"
                    "3. Determine individual exposure times\n"
                    "4. Test various exposure ratios (50/50, 60/40, etc.)\n"
                    "5. Evaluate combined results"
                ),
                "action": ActionType.MIX,
                "validation": "Can formulate split-grade chemistries",
                "tips": [
                    "Start with 50/50 exposure ratio",
                    "Keep detailed notes on ratios and results",
                    "Test on same paper you'll use for finals",
                ],
                "estimated_time_minutes": 20,
            },
            {
                "step_number": 3,
                "title": "Registration and Multiple Coating",
                "content": (
                    "Achieve precise registration for multiple exposures:\n\n"
                    "Method 1: Sequential Coating\n"
                    "1. Coat with first chemistry\n"
                    "2. Dry completely\n"
                    "3. Make first exposure\n"
                    "4. Coat with second chemistry over first\n"
                    "5. Dry and make second exposure\n"
                    "6. Develop both layers together\n\n"
                    "Pros: Simple registration\n"
                    "Cons: Second coat can disturb first\n\n"
                    "Method 2: Separate Prints Combined\n"
                    "1. Make two separate prints\n"
                    "2. Register and sandwich when dry\n"
                    "3. Scan/photograph combined result\n"
                    "4. Make final print from combined image\n\n"
                    "Pros: Independent optimization\n"
                    "Cons: Extra steps, potential for artifacts\n\n"
                    "Registration Tips:\n"
                    "- Use pin registration system\n"
                    "- Mark paper and negative positions\n"
                    "- Maintain exact frame position\n"
                    "- Test registration with practice runs"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Can achieve accurate registration",
                "warnings": [
                    "Registration errors will show as doubled edges",
                    "Practice thoroughly before final prints",
                ],
                "estimated_time_minutes": 30,
            },
            {
                "step_number": 4,
                "title": "Contrast Masking Techniques",
                "content": (
                    "Create and use masks for selective contrast control:\n\n"
                    "Unsharp Masking:\n"
                    "1. Create positive mask from negative\n"
                    "2. Make mask slightly out of focus (unsharp)\n"
                    "3. Register mask with negative\n"
                    "4. Print through both\n"
                    "5. Effect: Compressed density range, enhanced local contrast\n\n"
                    "Highlight/Shadow Masks:\n"
                    "1. Create mask protecting specific areas\n"
                    "2. Use for dodging/burning during exposure\n"
                    "3. Can be hand-cut or digitally generated\n"
                    "4. Allows selective contrast control\n\n"
                    "Digital Masking:\n"
                    "1. Create masks in Photoshop/imaging software\n"
                    "2. Generate separate negatives for different areas\n"
                    "3. Make multiple exposures through different negatives\n"
                    "4. Most precise control possible\n\n"
                    "Applications:\n"
                    "- Rescue difficult negatives\n"
                    "- Achieve impossible density ranges\n"
                    "- Creative tonal effects\n"
                    "- Fine art control"
                ),
                "action": ActionType.PRACTICE,
                "validation": "Understand masking principles",
                "tips": [
                    "Start with simple unsharp masks",
                    "Digital masking offers most control",
                    "Test thoroughly before complex masks",
                ],
                "estimated_time_minutes": 25,
            },
            {
                "step_number": 5,
                "title": "Creative Chemistry Variations",
                "content": (
                    "Experiment with alternative formulations:\n\n"
                    "Ziatype Process:\n"
                    "- Uses different sensitizer (ammonium ferric citrate)\n"
                    "- Broader tonal scale\n"
                    "- Different color characteristics\n"
                    "- Requires specific chemistry\n\n"
                    "NA2 Process:\n"
                    "- Extremely high contrast capability\n"
                    "- Very deep blacks possible\n"
                    "- Requires careful control\n"
                    "- Can be combined with standard Pt/Pd\n\n"
                    "Kallitype Combination:\n"
                    "- Mix Pt/Pd with iron-based process\n"
                    "- Unique brown-to-neutral tones\n"
                    "- Experimental territory\n\n"
                    "Toning Variations:\n"
                    "- Selenium toning after clearing\n"
                    "- Gold toning for color shifts\n"
                    "- Tea toning for warmth\n\n"
                    "Developer Variations:\n"
                    "- Ammonium citrate (warmer tones)\n"
                    "- Sodium citrate (similar to ammonium)\n"
                    "- EDTA developer (unique characteristics)\n"
                    "- Temperature variations for color control"
                ),
                "action": ActionType.READ,
                "validation": "Aware of creative variations",
                "tips": [
                    "Research thoroughly before trying exotic processes",
                    "Keep detailed notes on experiments",
                    "Master basics before advanced variations",
                ],
                "estimated_time_minutes": 15,
            },
            {
                "step_number": 6,
                "title": "Maximum Quality Optimization",
                "content": (
                    "Fine-tuning for absolute best quality:\n\n"
                    "Paper Selection and Preparation:\n"
                    "- Use highest quality 100% cotton paper\n"
                    "- Double sizing for optimal surface\n"
                    "- Consider paper texture for image content\n\n"
                    "Chemistry Optimization:\n"
                    "- Use freshest possible stock solutions\n"
                    "- Precise measurements (calibrated tools)\n"
                    "- Mix immediately before use\n"
                    "- Control solution temperature\n\n"
                    "Negative Quality:\n"
                    "- Maximum negative quality from digital source\n"
                    "- Proper density range for process\n"
                    "- Precise calibration curves\n"
                    "- Dust-free preparation\n\n"
                    "Environmental Control:\n"
                    "- Optimal humidity (55-60%)\n"
                    "- Stable temperature (68-72°F)\n"
                    "- Dust-free workspace\n"
                    "- Consistent UV source\n\n"
                    "Development Refinement:\n"
                    "- Fresh developer for each session\n"
                    "- Precise temperature control\n"
                    "- Optimal agitation technique\n"
                    "- Extended clearing for archival quality\n\n"
                    "The difference between good and exceptional prints:\n"
                    "- Attention to every detail\n"
                    "- Consistent process\n"
                    "- Quality materials\n"
                    "- Patience and care"
                ),
                "action": ActionType.VERIFY,
                "validation": "Understand quality optimization",
                "tips": [
                    "Small improvements accumulate",
                    "Consistency is foundation of quality",
                    "Invest in quality materials and tools",
                    "Practice and experience matter most",
                ],
                "estimated_time_minutes": 20,
            },
        ],
    },
}


class TutorialManager:
    """Manages tutorial access and user progress."""

    def __init__(self):
        """Initialize tutorial manager."""
        self.tutorials: dict[str, Tutorial] = {}
        self._load_tutorials()

    def _load_tutorials(self) -> None:
        """Load tutorial data into Tutorial objects."""
        for name, data in TUTORIALS_DATA.items():
            steps = [TutorialStep(**step_data) for step_data in data["steps"]]
            tutorial = Tutorial(
                name=name,
                display_name=data["display_name"],
                description=data["description"],
                difficulty=data["difficulty"],
                estimated_time=data["estimated_time"],
                prerequisites=data.get("prerequisites", []),
                steps=steps,
                learning_objectives=data.get("learning_objectives", []),
                materials_needed=data.get("materials_needed", []),
                tags=data.get("tags", []),
            )
            self.tutorials[name] = tutorial

    def get_available_tutorials(
        self, difficulty: TutorialDifficulty | None = None
    ) -> list[Tutorial]:
        """
        Get list of available tutorials.

        Args:
            difficulty: Optional filter by difficulty level

        Returns:
            List of Tutorial objects
        """
        tutorials = list(self.tutorials.values())
        if difficulty:
            tutorials = [t for t in tutorials if t.difficulty == difficulty]
        return sorted(tutorials, key=lambda t: (t.difficulty.value, t.name))

    def get_tutorial(self, name: str) -> Tutorial | None:
        """
        Get specific tutorial by name.

        Args:
            name: Tutorial identifier

        Returns:
            Tutorial object or None if not found
        """
        return self.tutorials.get(name)

    def start_tutorial(self, name: str, user_progress: UserProgress | None = None) -> UserProgress:
        """
        Start a tutorial, creating or updating progress.

        Args:
            name: Tutorial identifier
            user_progress: Existing progress or None to create new

        Returns:
            UserProgress object
        """
        if name not in self.tutorials:
            raise ValueError(f"Tutorial '{name}' not found")

        if user_progress is None:
            user_progress = UserProgress(
                tutorial_name=name, started_at=datetime.now(), current_step=1
            )
        elif user_progress.current_step == 0:
            user_progress.current_step = 1
            user_progress.started_at = datetime.now()

        return user_progress

    def validate_step(
        self, tutorial_name: str, step_index: int, user_action: Any
    ) -> tuple[bool, str]:
        """
        Validate if a step is completed correctly.

        Args:
            tutorial_name: Tutorial identifier
            step_index: Step number (1-based)
            user_action: User's action data

        Returns:
            Tuple of (is_valid, message)
        """
        tutorial = self.get_tutorial(tutorial_name)
        if not tutorial:
            return False, f"Tutorial '{tutorial_name}' not found"

        if step_index < 1 or step_index > len(tutorial.steps):
            return False, f"Invalid step number: {step_index}"

        step = tutorial.steps[step_index - 1]

        # Basic validation - can be extended with specific validation logic
        if step.validation:
            return True, f"Step {step_index} complete: {step.validation}"
        else:
            return True, f"Step {step_index} marked as complete"

    def get_progress(self, tutorial_name: str, user_progress: UserProgress) -> dict[str, Any]:
        """
        Get progress information for a tutorial.

        Args:
            tutorial_name: Tutorial identifier
            user_progress: User's progress data

        Returns:
            Dict with progress details
        """
        tutorial = self.get_tutorial(tutorial_name)
        if not tutorial:
            return {"error": f"Tutorial '{tutorial_name}' not found"}

        total_steps = len(tutorial.steps)
        completed = len(user_progress.completed_steps)
        current = user_progress.current_step

        return {
            "tutorial_name": tutorial.display_name,
            "total_steps": total_steps,
            "completed_steps": completed,
            "current_step": current,
            "progress_percent": (completed / total_steps * 100) if total_steps > 0 else 0,
            "is_complete": user_progress.completed_at is not None,
            "started_at": user_progress.started_at,
            "completed_at": user_progress.completed_at,
        }

    def mark_complete(self, tutorial_name: str, user_progress: UserProgress) -> UserProgress:
        """
        Mark tutorial as completed.

        Args:
            tutorial_name: Tutorial identifier
            user_progress: User's progress data

        Returns:
            Updated UserProgress object
        """
        tutorial = self.get_tutorial(tutorial_name)
        if not tutorial:
            raise ValueError(f"Tutorial '{tutorial_name}' not found")

        user_progress.completed_at = datetime.now()
        user_progress.current_step = len(tutorial.steps)
        user_progress.completed_steps = list(range(1, len(tutorial.steps) + 1))

        return user_progress

    def get_next_tutorial(self, completed_tutorial: str) -> Tutorial | None:
        """
        Get recommended next tutorial based on prerequisites.

        Args:
            completed_tutorial: Name of completed tutorial

        Returns:
            Recommended next Tutorial or None
        """
        # Find tutorials that list this one as a prerequisite
        candidates = [
            t
            for t in self.tutorials.values()
            if completed_tutorial in t.prerequisites
        ]

        if candidates:
            # Return the easiest matching tutorial
            return sorted(candidates, key=lambda t: t.difficulty.value)[0]

        return None
