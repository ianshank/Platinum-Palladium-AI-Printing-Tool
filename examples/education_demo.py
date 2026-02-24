#!/usr/bin/env python
"""
Demonstration of educational components for Pt/Pd printing tool.

This script demonstrates how to use the tutorials, glossary, and tips systems.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptpd_calibration.education import (
    Glossary,
    TipsManager,
    TutorialManager,
)


def demo_tutorials():
    """Demonstrate tutorial system."""
    print("=" * 70)
    print("TUTORIAL SYSTEM DEMONSTRATION")
    print("=" * 70)

    tm = TutorialManager()

    # List all tutorials
    print("\n📚 Available Tutorials:")
    print("-" * 70)
    for tutorial in tm.get_available_tutorials():
        print(f"\n{tutorial.display_name}")
        print(f"  Difficulty: {tutorial.difficulty.value.upper()}")
        print(f"  Time: {tutorial.estimated_time} minutes")
        print(f"  Steps: {len(tutorial.steps)}")
        print(f"  Description: {tutorial.description}")

    # Get a specific tutorial
    print("\n" + "=" * 70)
    print("TUTORIAL DETAILS: First Print")
    print("=" * 70)

    first_print = tm.get_tutorial("first_print")
    if first_print:
        print(f"\n{first_print.display_name}")
        print(f"\n{first_print.description}")
        print(f"\nPrerequisites: {first_print.prerequisites or 'None'}")
        print("\nLearning Objectives:")
        for obj in first_print.learning_objectives:
            print(f"  • {obj}")

        print("\nMaterials Needed:")
        for material in first_print.materials_needed[:5]:  # Show first 5
            print(f"  • {material}")
        print(f"  ... and {len(first_print.materials_needed) - 5} more items")

        print("\nFirst 3 Steps:")
        for step in first_print.steps[:3]:
            print(f"\n  Step {step.step_number}: {step.title}")
            print(f"  Action: {step.action.value}")
            # Show first 100 chars of content
            content_preview = step.content[:100] + "..." if len(step.content) > 100 else step.content
            print(f"  {content_preview}")

    # Simulate user progress
    print("\n" + "=" * 70)
    print("USER PROGRESS TRACKING")
    print("=" * 70)

    progress = tm.start_tutorial("first_print")
    print(f"\nStarted tutorial: {progress.tutorial_name}")
    print(f"Current step: {progress.current_step}")

    # Get progress information
    progress_info = tm.get_progress("first_print", progress)
    print(f"\nProgress: {progress_info['completed_steps']}/{progress_info['total_steps']} steps")
    print(f"Completion: {progress_info['progress_percent']:.0f}%")


def demo_glossary():
    """Demonstrate glossary system."""
    print("\n\n" + "=" * 70)
    print("GLOSSARY SYSTEM DEMONSTRATION")
    print("=" * 70)

    glossary = Glossary()

    # Lookup specific terms
    print("\n📖 Term Lookup Examples:")
    print("-" * 70)

    terms_to_demo = ["ferric_oxalate", "dmax", "platinum_chloride", "coating"]
    for term_name in terms_to_demo:
        term = glossary.lookup(term_name)
        if term:
            print(f"\n{term.term.upper()}")
            print(f"Category: {term.category.value}")
            # Show first 150 chars of definition
            definition_preview = term.definition[:150] + "..." if len(term.definition) > 150 else term.definition
            print(f"Definition: {definition_preview}")
            if term.synonyms:
                print(f"Also known as: {', '.join(term.synonyms)}")

    # Search functionality
    print("\n\n🔍 Search Results for 'exposure':")
    print("-" * 70)
    results = glossary.search("exposure")
    for term in results[:5]:  # Show first 5 results
        print(f"  • {term.term} ({term.category.value})")

    # Category browsing
    print("\n\n📂 Chemistry Terms:")
    print("-" * 70)
    from ptpd_calibration.education.glossary import TermCategory
    chemistry_terms = glossary.get_by_category(TermCategory.CHEMISTRY)
    for term in chemistry_terms[:8]:  # Show first 8
        print(f"  • {term.term}")
    print(f"  ... and {len(chemistry_terms) - 8} more chemistry terms")


def demo_tips():
    """Demonstrate tips system."""
    print("\n\n" + "=" * 70)
    print("TIPS SYSTEM DEMONSTRATION")
    print("=" * 70)

    tips_mgr = TipsManager()

    # Get contextual tips
    print("\n💡 Tips for Coating:")
    print("-" * 70)
    coating_tips = tips_mgr.get_contextual_tips("coating", limit=5)
    for i, tip in enumerate(coating_tips, 1):
        print(f"\n{i}. [{tip.category.value.upper()}] Priority: {'⭐' * tip.priority}")
        print(f"   {tip.content}")

    # Get tips by category
    print("\n\n⚠️  Safety Tips:")
    print("-" * 70)
    from ptpd_calibration.education.tips import TipCategory
    safety_tips = tips_mgr.get_tips_by_category(TipCategory.SAFETY)
    for tip in safety_tips[:3]:
        print(f"\n• Priority: {'⭐' * tip.priority}")
        print(f"  {tip.content}")

    # Random tip
    print("\n\n🎲 Random Tip:")
    print("-" * 70)
    random_tip = tips_mgr.get_random_tip()
    if random_tip:
        print(f"[{random_tip.category.value.upper()}] {random_tip.content}")

    # High priority tips
    print("\n\n🔥 Critical Tips (Priority 5):")
    print("-" * 70)
    critical_tips = tips_mgr.get_high_priority_tips(min_priority=5)
    for i, tip in enumerate(critical_tips[:3], 1):
        print(f"\n{i}. {tip.content}")

    # Statistics
    print("\n\n📊 Tips Statistics:")
    print("-" * 70)
    stats = tips_mgr.get_statistics()
    print(f"Total tips: {stats['total_tips']}")
    print(f"High priority tips: {stats['high_priority_count']}")
    print("\nTips by category:")
    for cat, count in sorted(stats['tips_by_category'].items()):
        if count > 0:
            print(f"  • {cat}: {count}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Pt/Pd Printing Educational Components Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    demo_tutorials()
    demo_glossary()
    demo_tips()

    print("\n\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nThe educational system includes:")
    print("  • 7 comprehensive tutorials covering beginner to advanced topics")
    print("  • 43 detailed glossary terms with definitions and cross-references")
    print("  • 58 practical tips organized by category and context")
    print("\nAll components are fully integrated and ready to use!")
    print()


if __name__ == "__main__":
    main()
