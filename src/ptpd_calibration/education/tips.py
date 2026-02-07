"""
Tips and best practices system for Pt/Pd printing.

Provides contextual tips, random helpful advice, and category-based
recommendations to improve printing workflow and results.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TipCategory(str, Enum):
    """Categories for organizing tips."""

    SAFETY = "safety"
    CHEMISTRY = "chemistry"
    COATING = "coating"
    EXPOSURE = "exposure"
    DEVELOPMENT = "development"
    CLEARING = "clearing"
    TROUBLESHOOTING = "troubleshooting"
    WORKFLOW = "workflow"
    COST_SAVING = "cost_saving"
    QUALITY = "quality"
    BEGINNER = "beginner"
    ADVANCED = "advanced"


class TipDifficulty(str, Enum):
    """Difficulty levels for tips."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ALL = "all"


class Tip(BaseModel):
    """A single tip or best practice."""

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=1, description="The tip content")
    category: TipCategory = Field(..., description="Tip category")
    difficulty: TipDifficulty = Field(default=TipDifficulty.ALL, description="Skill level")
    conditions: list[str] = Field(
        default_factory=list, description="Conditions when this tip is relevant"
    )
    priority: int = Field(default=1, ge=1, le=5, description="Importance (1=low, 5=critical)")
    related_terms: list[str] = Field(
        default_factory=list, description="Related glossary terms"
    )


# Comprehensive tips database
TIPS_DATA = [
    # Safety Tips
    {
        "content": "Always wear nitrile gloves when handling platinum and palladium salts. These chemicals can cause skin sensitization with repeated exposure.",
        "category": TipCategory.SAFETY,
        "difficulty": TipDifficulty.ALL,
        "priority": 5,
        "conditions": ["handling_chemistry", "mixing", "coating"],
        "related_terms": ["chemistry", "sensitizer"],
    },
    {
        "content": "Work in a well-ventilated area. While Pt/Pd chemistry is relatively safe, good ventilation prevents buildup of any chemical vapors.",
        "category": TipCategory.SAFETY,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["workspace_setup", "coating", "development"],
        "related_terms": ["workspace"],
    },
    {
        "content": "Keep food and drink completely separate from your printing workspace. Use dedicated equipment that never contacts food.",
        "category": TipCategory.SAFETY,
        "difficulty": TipDifficulty.ALL,
        "priority": 5,
        "conditions": ["workspace_setup"],
        "related_terms": ["safety"],
    },
    {
        "content": "Label all chemistry bottles clearly with contents and date. Store in dark, cool location away from light and children.",
        "category": TipCategory.SAFETY,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["storage", "chemistry_mixing"],
        "related_terms": ["chemistry", "storage"],
    },
    # Chemistry Tips
    {
        "content": "Store platinum and palladium stock solutions in refrigerator. This extends shelf life from months to over a year.",
        "category": TipCategory.CHEMISTRY,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["storage"],
        "related_terms": ["platinum_chloride", "palladium_chloride", "storage"],
    },
    {
        "content": "Mix sensitizer immediately before coating. Mixed sensitizer degrades quickly and loses sensitivity within hours.",
        "category": TipCategory.CHEMISTRY,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["coating", "chemistry_mixing"],
        "related_terms": ["sensitizer", "coating"],
    },
    {
        "content": "Start with 50/50 Pt/Pd ratio for balanced characteristics. You can adjust toward more Pt (cooler, higher contrast) or more Pd (warmer, gentler) once familiar with the process.",
        "category": TipCategory.CHEMISTRY,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["first_print", "chemistry_mixing"],
        "related_terms": ["metal_ratio", "platinum_chloride", "palladium_chloride"],
    },
    {
        "content": "Pure palladium is more economical for practice prints. Save expensive platinum-heavy ratios for final work.",
        "category": TipCategory.COST_SAVING,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["testing", "practice"],
        "related_terms": ["palladium_chloride", "cost"],
    },
    {
        "content": "If ferric oxalate shows yellow crystals, it may be degraded. Fresh ferric oxalate should be green and crystal-free in solution.",
        "category": TipCategory.CHEMISTRY,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 4,
        "conditions": ["chemistry_mixing", "troubleshooting"],
        "related_terms": ["ferric_oxalate", "sensitizer"],
    },
    {
        "content": "Add contrast agents one drop at a time and test. It's much easier to add more than to deal with excessive contrast.",
        "category": TipCategory.CHEMISTRY,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 4,
        "conditions": ["chemistry_mixing", "contrast_adjustment"],
        "related_terms": ["contrast_agent", "na2"],
    },
    # Coating Tips
    {
        "content": "Practice glass rod coating with water on scrap paper until you can do it smoothly and evenly. This is free practice that saves expensive chemistry.",
        "category": TipCategory.COATING,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 4,
        "conditions": ["learning", "coating"],
        "related_terms": ["glass_rod", "coating"],
    },
    {
        "content": "A single smooth stroke produces better results than multiple passes. Once you've coated, resist the urge to 'fix' areas.",
        "category": TipCategory.COATING,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["coating"],
        "related_terms": ["glass_rod", "coating", "streaking"],
    },
    {
        "content": "Clean your glass rod thoroughly between coatings. Even small dried sensitizer particles can cause streaks.",
        "category": TipCategory.COATING,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["coating"],
        "related_terms": ["glass_rod", "coating"],
    },
    {
        "content": "Coat paper slightly larger than your image area. This gives you clean edges to handle and room for trimming.",
        "category": TipCategory.COATING,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["coating"],
        "related_terms": ["coating", "paper"],
    },
    {
        "content": "If coating shows streaks, check your sizing. Uneven or poor sizing is often the culprit, not coating technique.",
        "category": TipCategory.TROUBLESHOOTING,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 3,
        "conditions": ["coating", "troubleshooting"],
        "related_terms": ["sizing", "coating", "streaking"],
    },
    {
        "content": "Use a fan on low speed to help dry coated paper faster, but keep in complete darkness. Typical drying time: 30-60 minutes.",
        "category": TipCategory.COATING,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["coating", "drying"],
        "related_terms": ["coating", "drying"],
    },
    # Exposure Tips
    {
        "content": "When in doubt about exposure time, bracket: make 3 prints at -20%, normal, and +20%. This helps dial in perfect exposure quickly.",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["exposure", "testing"],
        "related_terms": ["exposure", "testing"],
    },
    {
        "content": "Perfect contact in your printing frame is essential. Even small air gaps will cause unsharp areas. Check before each exposure.",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["exposure"],
        "related_terms": ["contact_printing", "printing_frame"],
    },
    {
        "content": "If using sunlight, test exposure each session. UV intensity varies dramatically with weather, time of day, and season.",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["exposure", "sunlight"],
        "related_terms": ["exposure", "uv_light"],
    },
    {
        "content": "Track humidity with each printing session. You may need to adjust exposure time by 20-30% between low and high humidity days.",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 3,
        "conditions": ["exposure", "environmental"],
        "related_terms": ["humidity", "exposure"],
    },
    {
        "content": "After proper exposure, you should see a faint image on the dried, coated paper. If you see nothing, exposure is likely too short.",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["exposure", "troubleshooting"],
        "related_terms": ["exposure"],
    },
    {
        "content": "UV fluorescent tubes degrade over time. If exposures are getting longer, tubes may need replacement (typically every 500-1000 hours).",
        "category": TipCategory.EXPOSURE,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 3,
        "conditions": ["exposure", "equipment"],
        "related_terms": ["uv_light", "exposure"],
    },
    # Development Tips
    {
        "content": "Use fresh developer for important prints. Developer is inexpensive compared to printing materials and fresh gives best results.",
        "category": TipCategory.DEVELOPMENT,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["development"],
        "related_terms": ["developer", "development"],
    },
    {
        "content": "Agitate continuously during development with gentle rocking motion. This prevents mottle and ensures even development.",
        "category": TipCategory.DEVELOPMENT,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["development"],
        "related_terms": ["development", "agitation", "mottle"],
    },
    {
        "content": "Development temperature affects print tone. Warmer developer (toward 75°F) gives warmer tones; cooler (toward 68°F) gives cooler tones.",
        "category": TipCategory.DEVELOPMENT,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 2,
        "conditions": ["development", "tone_control"],
        "related_terms": ["development", "tone"],
    },
    {
        "content": "Development is complete when no further change occurs, typically 1-3 minutes. Longer development won't increase density.",
        "category": TipCategory.DEVELOPMENT,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["development"],
        "related_terms": ["development"],
    },
    {
        "content": "If you're seeing mottle, check your water quality. Hard water can cause uneven development. Use distilled water if needed.",
        "category": TipCategory.TROUBLESHOOTING,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 3,
        "conditions": ["development", "troubleshooting"],
        "related_terms": ["mottle", "development"],
    },
    # Clearing Tips
    {
        "content": "Don't skimp on clearing time. Proper clearing is essential for archival permanence and clean highlights. 10-15 minutes in EDTA is ideal.",
        "category": TipCategory.CLEARING,
        "difficulty": TipDifficulty.ALL,
        "priority": 5,
        "conditions": ["clearing"],
        "related_terms": ["clearing", "edta", "archival"],
    },
    {
        "content": "If highlights show yellow staining, you can try re-clearing in fresh EDTA. Catch it early before it sets permanently.",
        "category": TipCategory.TROUBLESHOOTING,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 3,
        "conditions": ["clearing", "troubleshooting"],
        "related_terms": ["yellow_staining", "clearing", "edta"],
    },
    {
        "content": "Three baths are essential: initial water rinse (2-5 min), EDTA clearing (10-15 min), final wash (15-20 min running water).",
        "category": TipCategory.CLEARING,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["clearing"],
        "related_terms": ["clearing", "edta"],
    },
    {
        "content": "Use distilled water for final rinse. This prevents water spots and ensures cleanest possible result.",
        "category": TipCategory.CLEARING,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 2,
        "conditions": ["clearing", "quality"],
        "related_terms": ["clearing"],
    },
    {
        "content": "Don't reuse EDTA clearing baths. Mix fresh for each session - it's inexpensive and ensures proper clearing.",
        "category": TipCategory.CLEARING,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["clearing"],
        "related_terms": ["edta", "clearing"],
    },
    # Workflow Tips
    {
        "content": "Keep a detailed printing log: paper, chemistry ratios, exposure time, humidity, developer, and results. This data becomes invaluable.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.ALL,
        "priority": 4,
        "conditions": ["documentation"],
        "related_terms": ["workflow", "calibration"],
    },
    {
        "content": "Organize your workspace with clear zones: dry (chemistry storage), coating (subdued light), exposure, and wet (development/clearing).",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["workspace_setup"],
        "related_terms": ["workflow"],
    },
    {
        "content": "Prepare all chemistry and materials before coating. Once you start coating, you want to work continuously without searching for items.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["workflow", "coating"],
        "related_terms": ["workflow", "coating"],
    },
    {
        "content": "Print multiple test strips when testing variables. This saves time and uses chemistry efficiently.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["testing"],
        "related_terms": ["workflow", "testing"],
    },
    {
        "content": "Let prints dry completely (24 hours) before evaluating density and making printing decisions. Wet prints appear darker.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["evaluation"],
        "related_terms": ["workflow"],
    },
    # Quality Tips
    {
        "content": "Use 100% cotton paper for archival quality. Cheaper papers are fine for testing but won't last for fine art work.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["paper_selection"],
        "related_terms": ["paper", "archival"],
    },
    {
        "content": "A good digital negative is half the battle. Invest time in creating quality negatives with proper density range and calibration curves.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 4,
        "conditions": ["negative_making"],
        "related_terms": ["digital_negative", "negative", "calibration"],
    },
    {
        "content": "Calibrate for each paper/chemistry combination. A curve made for one paper won't work properly on another.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 4,
        "conditions": ["calibration"],
        "related_terms": ["calibration", "linearization", "curve"],
    },
    {
        "content": "Small improvements in each step compound into dramatically better final prints. Pursue excellence at every stage.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 3,
        "conditions": ["quality"],
        "related_terms": ["workflow", "quality"],
    },
    # Cost Saving Tips
    {
        "content": "Size your own paper rather than buying pre-sized. It's cheaper and you can optimize sizing for your process.",
        "category": TipCategory.COST_SAVING,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 2,
        "conditions": ["cost_saving"],
        "related_terms": ["sizing", "paper"],
    },
    {
        "content": "Practice on smaller prints (5x7) before committing to large sizes. This saves chemistry and paper while learning.",
        "category": TipCategory.COST_SAVING,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["learning", "cost_saving"],
        "related_terms": ["workflow"],
    },
    {
        "content": "Join a local or online Pt/Pd printing community. Share chemistry costs and learn from others' experience.",
        "category": TipCategory.COST_SAVING,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["community"],
        "related_terms": ["workflow"],
    },
    {
        "content": "Measure chemistry carefully to avoid waste. A digital scale with 0.01g precision pays for itself quickly.",
        "category": TipCategory.COST_SAVING,
        "difficulty": TipDifficulty.ALL,
        "priority": 3,
        "conditions": ["chemistry_mixing"],
        "related_terms": ["chemistry"],
    },
    # Beginner Tips
    {
        "content": "Your first prints won't be perfect - that's expected! Each print teaches you something. Keep notes and iterate.",
        "category": TipCategory.BEGINNER,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["first_print", "learning"],
        "related_terms": ["workflow"],
    },
    {
        "content": "Start with a proven formula (50/50 Pt/Pd, standard ferric oxalate) rather than experimenting. Learn the basics first.",
        "category": TipCategory.BEGINNER,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 4,
        "conditions": ["first_print"],
        "related_terms": ["chemistry", "metal_ratio"],
    },
    {
        "content": "Watch videos of experienced printers coating. Seeing the technique in motion helps more than still photos or text.",
        "category": TipCategory.BEGINNER,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["learning", "coating"],
        "related_terms": ["coating"],
    },
    {
        "content": "Don't get discouraged by initial failures. Even experienced printers make mistakes. The process has many variables to master.",
        "category": TipCategory.BEGINNER,
        "difficulty": TipDifficulty.BEGINNER,
        "priority": 3,
        "conditions": ["learning"],
        "related_terms": ["workflow"],
    },
    # Advanced Tips
    {
        "content": "Split-grade printing (two exposures with different chemistry) can extend tonal range beyond what single chemistry achieves.",
        "category": TipCategory.ADVANCED,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 2,
        "conditions": ["advanced"],
        "related_terms": ["advanced"],
    },
    {
        "content": "Temperature of developer affects print tone more than most realize. Experiment with 65-80°F range for tonal control.",
        "category": TipCategory.ADVANCED,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 2,
        "conditions": ["tone_control", "advanced"],
        "related_terms": ["developer", "tone"],
    },
    {
        "content": "Creating digital unsharp masks can help print difficult negatives with extreme density ranges. This is advanced but powerful.",
        "category": TipCategory.ADVANCED,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 2,
        "conditions": ["advanced", "negative_making"],
        "related_terms": ["negative", "advanced"],
    },
    {
        "content": "Different papers have different 'personalities' with Pt/Pd. Print same image on several papers to find your aesthetic preference.",
        "category": TipCategory.ADVANCED,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 2,
        "conditions": ["paper_selection", "advanced"],
        "related_terms": ["paper"],
    },
    # Additional practical tips
    {
        "content": "Store sized paper flat in dry location. It can last months if kept clean and dry. Date your sized paper batches.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["storage"],
        "related_terms": ["sizing", "paper", "storage"],
    },
    {
        "content": "A clip test (cutting small piece from coated paper to test) can save full prints if you're uncertain about exposure.",
        "category": TipCategory.WORKFLOW,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 2,
        "conditions": ["testing", "exposure"],
        "related_terms": ["exposure", "testing"],
    },
    {
        "content": "The image will continue to subtly change (usually warming and slightly darkening) over first few weeks after printing.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.INTERMEDIATE,
        "priority": 2,
        "conditions": ["evaluation"],
        "related_terms": ["archival", "tone"],
    },
    {
        "content": "Clean workspace prevents dust in coating. A microfiber cloth wipe-down before coating session helps.",
        "category": TipCategory.QUALITY,
        "difficulty": TipDifficulty.ALL,
        "priority": 2,
        "conditions": ["coating", "workspace_setup"],
        "related_terms": ["coating", "quality"],
    },
    {
        "content": "Consider UV exposure units with integrators (measure actual UV delivered) for ultimate consistency across sessions.",
        "category": TipCategory.ADVANCED,
        "difficulty": TipDifficulty.ADVANCED,
        "priority": 1,
        "conditions": ["equipment", "exposure"],
        "related_terms": ["uv_light", "exposure"],
    },
]


class TipsManager:
    """Manages tips and best practices for Pt/Pd printing."""

    def __init__(self):
        """Initialize tips manager."""
        self.tips: list[Tip] = []
        self.seen_tips: set[UUID] = set()
        self._load_tips()

    def _load_tips(self) -> None:
        """Load tips data into Tip objects."""
        self.tips = [Tip(**tip_data) for tip_data in TIPS_DATA]

    def get_contextual_tips(
        self,
        context: str,
        difficulty: TipDifficulty | None = None,
        limit: int = 5,
    ) -> list[Tip]:
        """
        Get tips relevant to current operation context.

        Args:
            context: Current context/operation (e.g., 'coating', 'exposure')
            difficulty: Optional filter by difficulty level
            limit: Maximum number of tips to return

        Returns:
            List of relevant Tip objects, sorted by priority
        """
        context_lower = context.lower()
        relevant_tips = []

        for tip in self.tips:
            # Check if context matches any conditions
            if any(context_lower in cond.lower() for cond in tip.conditions):
                # Apply difficulty filter if specified
                if difficulty and tip.difficulty not in [difficulty, TipDifficulty.ALL]:
                    continue
                relevant_tips.append(tip)

        # Sort by priority (higher first) and return top N
        relevant_tips.sort(key=lambda t: t.priority, reverse=True)
        return relevant_tips[:limit]

    def get_random_tip(
        self,
        category: TipCategory | None = None,
        difficulty: TipDifficulty | None = None,
        unseen_only: bool = False,
    ) -> Tip | None:
        """
        Get a random tip.

        Args:
            category: Optional filter by category
            difficulty: Optional filter by difficulty
            unseen_only: If True, only return tips not previously seen

        Returns:
            Random Tip object or None
        """
        import random

        candidates = self.tips.copy()

        # Apply filters
        if category:
            candidates = [t for t in candidates if t.category == category]

        if difficulty:
            candidates = [
                t for t in candidates if t.difficulty in [difficulty, TipDifficulty.ALL]
            ]

        if unseen_only:
            candidates = [t for t in candidates if t.id not in self.seen_tips]

        if not candidates:
            return None

        tip = random.choice(candidates)
        self.seen_tips.add(tip.id)
        return tip

    def get_tips_by_category(
        self, category: TipCategory, difficulty: TipDifficulty | None = None
    ) -> list[Tip]:
        """
        Get all tips in a specific category.

        Args:
            category: Tip category
            difficulty: Optional filter by difficulty

        Returns:
            List of Tip objects in category
        """
        tips = [t for t in self.tips if t.category == category]

        if difficulty:
            tips = [t for t in tips if t.difficulty in [difficulty, TipDifficulty.ALL]]

        # Sort by priority
        return sorted(tips, key=lambda t: t.priority, reverse=True)

    def mark_tip_seen(self, tip_id: UUID) -> None:
        """
        Mark a tip as seen.

        Args:
            tip_id: UUID of tip to mark
        """
        self.seen_tips.add(tip_id)

    def get_unseen_tips(
        self,
        category: TipCategory | None = None,
        difficulty: TipDifficulty | None = None,
    ) -> list[Tip]:
        """
        Get all unseen tips.

        Args:
            category: Optional filter by category
            difficulty: Optional filter by difficulty

        Returns:
            List of unseen Tip objects
        """
        unseen = [t for t in self.tips if t.id not in self.seen_tips]

        if category:
            unseen = [t for t in unseen if t.category == category]

        if difficulty:
            unseen = [t for t in unseen if t.difficulty in [difficulty, TipDifficulty.ALL]]

        return sorted(unseen, key=lambda t: t.priority, reverse=True)

    def reset_seen_tips(self) -> None:
        """Clear the seen tips tracking."""
        self.seen_tips.clear()

    def get_high_priority_tips(
        self, min_priority: int = 4, difficulty: TipDifficulty | None = None
    ) -> list[Tip]:
        """
        Get high-priority tips (critical information).

        Args:
            min_priority: Minimum priority level (1-5)
            difficulty: Optional filter by difficulty

        Returns:
            List of high-priority Tip objects
        """
        high_priority = [t for t in self.tips if t.priority >= min_priority]

        if difficulty:
            high_priority = [
                t for t in high_priority if t.difficulty in [difficulty, TipDifficulty.ALL]
            ]

        return sorted(high_priority, key=lambda t: t.priority, reverse=True)

    def search_tips(self, query: str) -> list[Tip]:
        """
        Search tips by content.

        Args:
            query: Search string (case-insensitive)

        Returns:
            List of matching Tip objects
        """
        query_lower = query.lower()
        results = [t for t in self.tips if query_lower in t.content.lower()]

        return sorted(results, key=lambda t: t.priority, reverse=True)

    def get_all_categories(self) -> list[TipCategory]:
        """
        Get list of all tip categories.

        Returns:
            List of TipCategory values
        """
        categories = {tip.category for tip in self.tips}
        return sorted(categories, key=lambda c: c.value)

    def get_tips_for_related_term(self, term: str) -> list[Tip]:
        """
        Get tips related to a specific glossary term.

        Args:
            term: Glossary term

        Returns:
            List of related Tip objects
        """
        term_lower = term.lower()
        related = [
            t
            for t in self.tips
            if any(term_lower in rt.lower() for rt in t.related_terms)
        ]

        return sorted(related, key=lambda t: t.priority, reverse=True)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about tips collection.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_tips": len(self.tips),
            "seen_tips": len(self.seen_tips),
            "unseen_tips": len(self.tips) - len(self.seen_tips),
            "tips_by_category": {
                cat.value: len([t for t in self.tips if t.category == cat])
                for cat in TipCategory
            },
            "tips_by_difficulty": {
                diff.value: len([t for t in self.tips if t.difficulty == diff])
                for diff in TipDifficulty
            },
            "high_priority_count": len([t for t in self.tips if t.priority >= 4]),
        }
