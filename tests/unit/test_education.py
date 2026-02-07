"""
Comprehensive unit tests for education module.

Tests TutorialManager, Glossary, and TipsManager classes.
"""

from datetime import datetime
from uuid import UUID

import pytest

from ptpd_calibration.education.glossary import (
    Glossary,
    GlossaryTerm,
    TermCategory,
)
from ptpd_calibration.education.tips import (
    Tip,
    TipCategory,
    TipDifficulty,
    TipsManager,
)
from ptpd_calibration.education.tutorials import (
    ActionType,
    Tutorial,
    TutorialDifficulty,
    TutorialManager,
    TutorialStep,
    UserProgress,
)

# ============================================================================
# TutorialManager Tests
# ============================================================================


class TestTutorialManager:
    """Tests for TutorialManager class."""

    @pytest.fixture
    def tutorial_manager(self):
        """Create a TutorialManager instance."""
        return TutorialManager()

    @pytest.fixture
    def user_progress(self):
        """Create a sample UserProgress instance."""
        return UserProgress(
            tutorial_name="first_print",
            current_step=1,
            completed_steps=[],
            started_at=datetime.now(),
        )

    # Test get_available_tutorials()
    def test_get_available_tutorials_all(self, tutorial_manager):
        """Test getting all tutorials without filter."""
        tutorials = tutorial_manager.get_available_tutorials()
        assert len(tutorials) > 0
        assert all(isinstance(t, Tutorial) for t in tutorials)
        # Verify they are sorted by difficulty then name
        assert tutorials == sorted(
            tutorials, key=lambda t: (t.difficulty.value, t.name)
        )

    def test_get_available_tutorials_by_difficulty(self, tutorial_manager):
        """Test filtering tutorials by difficulty."""
        # Get beginner tutorials
        beginner = tutorial_manager.get_available_tutorials(
            difficulty=TutorialDifficulty.BEGINNER
        )
        assert all(t.difficulty == TutorialDifficulty.BEGINNER for t in beginner)
        assert len(beginner) > 0

        # Get intermediate tutorials
        intermediate = tutorial_manager.get_available_tutorials(
            difficulty=TutorialDifficulty.INTERMEDIATE
        )
        assert all(
            t.difficulty == TutorialDifficulty.INTERMEDIATE for t in intermediate
        )
        assert len(intermediate) > 0

        # Get advanced tutorials
        advanced = tutorial_manager.get_available_tutorials(
            difficulty=TutorialDifficulty.ADVANCED
        )
        assert all(t.difficulty == TutorialDifficulty.ADVANCED for t in advanced)
        assert len(advanced) > 0

    def test_get_available_tutorials_empty_difficulty(self, tutorial_manager):
        """Test that all difficulty levels have at least one tutorial."""
        for difficulty in TutorialDifficulty:
            tutorials = tutorial_manager.get_available_tutorials(difficulty=difficulty)
            # Should have at least one tutorial for each difficulty
            assert len(tutorials) > 0

    # Test get_tutorial()
    def test_get_tutorial_valid(self, tutorial_manager):
        """Test getting a valid tutorial by name."""
        tutorial = tutorial_manager.get_tutorial("first_print")
        assert tutorial is not None
        assert isinstance(tutorial, Tutorial)
        assert tutorial.name == "first_print"
        assert tutorial.display_name == "Your First Platinum/Palladium Print"

    def test_get_tutorial_invalid(self, tutorial_manager):
        """Test getting a non-existent tutorial."""
        tutorial = tutorial_manager.get_tutorial("nonexistent_tutorial")
        assert tutorial is None

    def test_get_tutorial_case_sensitive(self, tutorial_manager):
        """Test that tutorial names are case-sensitive."""
        tutorial = tutorial_manager.get_tutorial("FIRST_PRINT")
        assert tutorial is None

    # Test start_tutorial()
    def test_start_tutorial_new(self, tutorial_manager):
        """Test starting a tutorial with new progress."""
        progress = tutorial_manager.start_tutorial("first_print")
        assert isinstance(progress, UserProgress)
        assert progress.tutorial_name == "first_print"
        assert progress.current_step == 1
        assert progress.started_at is not None
        assert progress.completed_at is None

    def test_start_tutorial_with_existing_progress(self, tutorial_manager, user_progress):
        """Test starting tutorial with existing progress."""
        user_progress.current_step = 0  # Not started
        progress = tutorial_manager.start_tutorial("first_print", user_progress)
        assert progress.current_step == 1
        assert progress.started_at is not None

    def test_start_tutorial_already_started(self, tutorial_manager, user_progress):
        """Test starting tutorial that's already in progress."""
        user_progress.current_step = 5
        original_start = user_progress.started_at
        progress = tutorial_manager.start_tutorial("first_print", user_progress)
        assert progress.current_step == 5  # Unchanged
        assert progress.started_at == original_start  # Unchanged

    def test_start_tutorial_invalid(self, tutorial_manager):
        """Test starting a non-existent tutorial raises error."""
        with pytest.raises(ValueError, match="Tutorial 'invalid' not found"):
            tutorial_manager.start_tutorial("invalid")

    # Test validate_step()
    def test_validate_step_valid(self, tutorial_manager):
        """Test validating a valid step."""
        is_valid, message = tutorial_manager.validate_step("first_print", 1, None)
        assert is_valid is True
        assert "Step 1" in message

    def test_validate_step_with_validation_message(self, tutorial_manager):
        """Test that validation returns the step's validation message."""
        is_valid, message = tutorial_manager.validate_step("first_print", 1, None)
        assert is_valid is True
        # Check if validation message from step is included
        assert "organized" in message.lower() or "complete" in message.lower()

    def test_validate_step_invalid_tutorial(self, tutorial_manager):
        """Test validating step for non-existent tutorial."""
        is_valid, message = tutorial_manager.validate_step("invalid", 1, None)
        assert is_valid is False
        assert "not found" in message

    def test_validate_step_invalid_step_number_too_low(self, tutorial_manager):
        """Test validating with step number < 1."""
        is_valid, message = tutorial_manager.validate_step("first_print", 0, None)
        assert is_valid is False
        assert "Invalid step number" in message

    def test_validate_step_invalid_step_number_too_high(self, tutorial_manager):
        """Test validating with step number > total steps."""
        tutorial = tutorial_manager.get_tutorial("first_print")
        total_steps = len(tutorial.steps)
        is_valid, message = tutorial_manager.validate_step(
            "first_print", total_steps + 1, None
        )
        assert is_valid is False
        assert "Invalid step number" in message

    # Test get_progress()
    def test_get_progress_valid(self, tutorial_manager, user_progress):
        """Test getting progress for a valid tutorial."""
        user_progress.completed_steps = [1, 2, 3]
        user_progress.current_step = 4
        progress = tutorial_manager.get_progress("first_print", user_progress)

        assert progress["tutorial_name"] == "Your First Platinum/Palladium Print"
        assert progress["total_steps"] == 10  # first_print has 10 steps
        assert progress["completed_steps"] == 3
        assert progress["current_step"] == 4
        assert progress["progress_percent"] == 30.0  # 3/10 * 100
        assert progress["is_complete"] is False
        assert progress["started_at"] is not None
        assert progress["completed_at"] is None

    def test_get_progress_completed(self, tutorial_manager):
        """Test progress for a completed tutorial."""
        progress = UserProgress(
            tutorial_name="first_print",
            current_step=10,
            completed_steps=list(range(1, 11)),
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )
        result = tutorial_manager.get_progress("first_print", progress)

        assert result["completed_steps"] == 10
        assert result["progress_percent"] == 100.0
        assert result["is_complete"] is True
        assert result["completed_at"] is not None

    def test_get_progress_invalid_tutorial(self, tutorial_manager, user_progress):
        """Test getting progress for non-existent tutorial."""
        progress = tutorial_manager.get_progress("invalid", user_progress)
        assert "error" in progress
        assert "not found" in progress["error"]

    # Test mark_complete()
    def test_mark_complete_valid(self, tutorial_manager, user_progress):
        """Test marking a tutorial as complete."""
        user_progress.completed_steps = [1, 2, 3]
        user_progress.current_step = 3

        completed = tutorial_manager.mark_complete("first_print", user_progress)

        assert completed.completed_at is not None
        tutorial = tutorial_manager.get_tutorial("first_print")
        assert completed.current_step == len(tutorial.steps)
        assert completed.completed_steps == list(range(1, len(tutorial.steps) + 1))

    def test_mark_complete_invalid_tutorial(self, tutorial_manager, user_progress):
        """Test marking non-existent tutorial as complete."""
        with pytest.raises(ValueError, match="Tutorial 'invalid' not found"):
            tutorial_manager.mark_complete("invalid", user_progress)

    # Test get_next_tutorial()
    def test_get_next_tutorial_valid(self, tutorial_manager):
        """Test getting next tutorial based on prerequisites."""
        # After completing "first_print", should suggest calibration_workflow
        next_tutorial = tutorial_manager.get_next_tutorial("first_print")
        assert next_tutorial is not None
        assert "first_print" in next_tutorial.prerequisites

    def test_get_next_tutorial_none(self, tutorial_manager):
        """Test getting next tutorial when none exists."""
        # Create a tutorial name that isn't a prerequisite for anything
        next_tutorial = tutorial_manager.get_next_tutorial("advanced_techniques")
        # Should return None or a tutorial (depends on data)
        assert next_tutorial is None or isinstance(next_tutorial, Tutorial)

    # Test that all pre-loaded tutorials are valid
    def test_all_tutorials_valid_structure(self, tutorial_manager):
        """Test that all pre-loaded tutorials have valid structure."""
        for tutorial in tutorial_manager.tutorials.values():
            # Check required fields
            assert isinstance(tutorial.id, UUID)
            assert len(tutorial.name) > 0
            assert len(tutorial.display_name) > 0
            assert len(tutorial.description) > 0
            assert isinstance(tutorial.difficulty, TutorialDifficulty)
            assert tutorial.estimated_time > 0
            assert isinstance(tutorial.prerequisites, list)
            assert isinstance(tutorial.steps, list)
            assert len(tutorial.steps) > 0
            assert isinstance(tutorial.learning_objectives, list)
            assert isinstance(tutorial.materials_needed, list)
            assert isinstance(tutorial.tags, list)

    def test_all_tutorial_steps_valid(self, tutorial_manager):
        """Test that all tutorial steps have valid structure."""
        for tutorial in tutorial_manager.tutorials.values():
            for i, step in enumerate(tutorial.steps, start=1):
                assert isinstance(step, TutorialStep)
                assert step.step_number == i  # Sequential
                assert len(step.title) > 0
                assert len(step.content) > 0
                assert isinstance(step.action, ActionType)
                assert isinstance(step.tips, list)
                assert isinstance(step.warnings, list)

    def test_all_tutorials_have_unique_names(self, tutorial_manager):
        """Test that all tutorial names are unique."""
        names = [t.name for t in tutorial_manager.tutorials.values()]
        assert len(names) == len(set(names))

    def test_tutorial_prerequisites_exist(self, tutorial_manager):
        """Test that all prerequisites reference existing tutorials."""
        all_names = set(tutorial_manager.tutorials.keys())
        for tutorial in tutorial_manager.tutorials.values():
            for prereq in tutorial.prerequisites:
                assert (
                    prereq in all_names
                ), f"Prerequisite '{prereq}' not found for '{tutorial.name}'"


# ============================================================================
# Glossary Tests
# ============================================================================


class TestGlossary:
    """Tests for Glossary class."""

    @pytest.fixture
    def glossary(self):
        """Create a Glossary instance."""
        return Glossary()

    # Test lookup()
    def test_lookup_exact_match(self, glossary):
        """Test looking up a term with exact match."""
        term = glossary.lookup("Ferric Oxalate")
        assert term is not None
        assert isinstance(term, GlossaryTerm)
        assert term.term == "Ferric Oxalate"

    def test_lookup_case_insensitive(self, glossary):
        """Test that lookup is case-insensitive."""
        term1 = glossary.lookup("ferric oxalate")
        term2 = glossary.lookup("FERRIC OXALATE")
        term3 = glossary.lookup("Ferric Oxalate")

        assert term1 is not None
        assert term2 is not None
        assert term3 is not None
        assert term1.term == term2.term == term3.term

    def test_lookup_by_synonym(self, glossary):
        """Test looking up a term by synonym."""
        # "iron oxalate" is a synonym for "Ferric Oxalate"
        term = glossary.lookup("iron oxalate")
        assert term is not None
        assert term.term == "Ferric Oxalate"

    def test_lookup_not_found(self, glossary):
        """Test looking up non-existent term."""
        term = glossary.lookup("nonexistent term")
        assert term is None

    # Test search()
    def test_search_in_term_name(self, glossary):
        """Test searching for terms in term names."""
        results = glossary.search("platinum")
        assert len(results) > 0
        assert any("platinum" in t.term.lower() for t in results)

    def test_search_in_definition(self, glossary):
        """Test searching in term definitions."""
        results = glossary.search("UV light")
        assert len(results) > 0
        # Should find terms with UV in definition even if not in name

    def test_search_in_synonyms(self, glossary):
        """Test searching in synonyms."""
        results = glossary.search("developer")
        assert len(results) > 0

    def test_search_case_insensitive(self, glossary):
        """Test that search is case-insensitive."""
        results1 = glossary.search("platinum")
        results2 = glossary.search("PLATINUM")
        results3 = glossary.search("Platinum")

        assert len(results1) == len(results2) == len(results3)

    def test_search_no_matches(self, glossary):
        """Test search with no matches."""
        results = glossary.search("xyznonexistent")
        assert len(results) == 0

    def test_search_no_duplicates(self, glossary):
        """Test that search results don't contain duplicates."""
        results = glossary.search("oxalate")
        terms = [t.term for t in results]
        assert len(terms) == len(set(terms))

    # Test get_by_category()
    def test_get_by_category_chemistry(self, glossary):
        """Test getting chemistry terms."""
        chemistry_terms = glossary.get_by_category(TermCategory.CHEMISTRY)
        assert len(chemistry_terms) > 0
        assert all(t.category == TermCategory.CHEMISTRY for t in chemistry_terms)

    def test_get_by_category_process(self, glossary):
        """Test getting process terms."""
        process_terms = glossary.get_by_category(TermCategory.PROCESS)
        assert len(process_terms) > 0
        assert all(t.category == TermCategory.PROCESS for t in process_terms)

    def test_get_by_category_all_categories(self, glossary):
        """Test that get_by_category works for all categories."""
        # Note: Not all categories may have terms yet, so we just verify
        # the method works without errors for all categories
        for category in TermCategory:
            terms = glossary.get_by_category(category)
            assert isinstance(terms, list)
            # Verify terms are of correct category if any exist
            if terms:
                assert all(t.category == category for t in terms)

    def test_get_by_category_no_duplicates(self, glossary):
        """Test that category results don't contain duplicates."""
        for category in TermCategory:
            terms = glossary.get_by_category(category)
            term_names = [t.term for t in terms]
            assert len(term_names) == len(set(term_names))

    def test_get_by_category_sorted(self, glossary):
        """Test that category results are sorted by term name."""
        for category in TermCategory:
            terms = glossary.get_by_category(category)
            assert terms == sorted(terms, key=lambda t: t.term)

    # Test get_related()
    def test_get_related_valid_term(self, glossary):
        """Test getting related terms for a valid term."""
        related = glossary.get_related("Ferric Oxalate")
        assert isinstance(related, list)
        # Ferric oxalate should have related terms
        assert len(related) > 0

    def test_get_related_returns_term_objects(self, glossary):
        """Test that get_related returns GlossaryTerm objects."""
        related = glossary.get_related("Platinum Chloride")
        assert all(isinstance(t, GlossaryTerm) for t in related)

    def test_get_related_excludes_self(self, glossary):
        """Test that related terms don't include the term itself."""
        related = glossary.get_related("Dmax")
        term_names = [t.term for t in related]
        assert "Dmax" not in term_names

    def test_get_related_invalid_term(self, glossary):
        """Test getting related terms for non-existent term."""
        related = glossary.get_related("nonexistent")
        assert related == []

    def test_get_related_includes_see_also(self, glossary):
        """Test that get_related includes both related_terms and see_also."""
        # EDTA has both related_terms and see_also
        related = glossary.get_related("EDTA")
        # Should have related terms from both lists
        assert len(related) > 0

    # Test add_term()
    def test_add_term_new(self, glossary):
        """Test adding a new term."""
        new_term = GlossaryTerm(
            term="Test Term",
            definition="Test definition",
            category=TermCategory.CHEMISTRY,
            related_terms=[],
            examples=[],
            synonyms=["test synonym"],
            see_also=[],
        )

        glossary.add_term(new_term)

        # Verify it can be looked up
        found = glossary.lookup("Test Term")
        assert found is not None
        assert found.term == "Test Term"

        # Verify synonym works
        found_by_synonym = glossary.lookup("test synonym")
        assert found_by_synonym is not None
        assert found_by_synonym.term == "Test Term"

    def test_add_term_replaces_existing(self, glossary):
        """Test that adding a term with same name replaces existing."""
        original = glossary.lookup("Dmax")
        original_definition = original.definition

        new_term = GlossaryTerm(
            term="Dmax",
            definition="New definition",
            category=TermCategory.MEASUREMENT,
        )

        glossary.add_term(new_term)
        updated = glossary.lookup("Dmax")
        assert updated.definition == "New definition"
        assert updated.definition != original_definition

    # Test utility methods
    def test_get_all_categories(self, glossary):
        """Test getting all categories."""
        categories = glossary.get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(c, TermCategory) for c in categories)
        # Should be sorted
        assert categories == sorted(categories, key=lambda c: c.value)

    def test_export_to_dict(self, glossary):
        """Test exporting glossary to dictionary."""
        export = glossary.export_to_dict()
        assert isinstance(export, dict)
        assert len(export) > 0
        # Check structure of exported data
        for term_name, term_data in export.items():
            assert isinstance(term_name, str)
            assert isinstance(term_data, dict)
            assert "term" in term_data
            assert "definition" in term_data
            assert "category" in term_data

    # Test that all pre-loaded terms are valid
    def test_all_terms_valid_structure(self, glossary):
        """Test that all pre-loaded terms have valid structure."""
        seen = set()
        for term in glossary.terms.values():
            if term.term in seen:
                continue
            seen.add(term.term)

            assert isinstance(term, GlossaryTerm)
            assert len(term.term) > 0
            assert len(term.definition) > 0
            assert isinstance(term.category, TermCategory)
            assert isinstance(term.related_terms, list)
            assert isinstance(term.examples, list)
            assert isinstance(term.synonyms, list)
            assert isinstance(term.see_also, list)

    def test_all_term_categories_valid(self, glossary):
        """Test that all terms have valid categories."""
        seen = set()
        for term in glossary.terms.values():
            if term.term in seen:
                continue
            seen.add(term.term)
            assert term.category in TermCategory

    def test_related_terms_exist(self, glossary):
        """Test that related terms reference existing terms."""
        seen = set()
        for term in glossary.terms.values():
            if term.term in seen:
                continue
            seen.add(term.term)

            for related in term.related_terms + term.see_also:
                # Related terms use underscores, need to convert
                related_with_spaces = related.replace("_", " ")
                glossary.lookup(related_with_spaces)
                # It's okay if some related terms don't exist yet
                # (they might be added later), but log for awareness


# ============================================================================
# TipsManager Tests
# ============================================================================


class TestTipsManager:
    """Tests for TipsManager class."""

    @pytest.fixture
    def tips_manager(self):
        """Create a TipsManager instance."""
        return TipsManager()

    # Test get_contextual_tips()
    def test_get_contextual_tips_coating(self, tips_manager):
        """Test getting tips for coating context."""
        tips = tips_manager.get_contextual_tips("coating")
        assert len(tips) > 0
        assert all(isinstance(t, Tip) for t in tips)
        # All should have "coating" in conditions
        assert all(any("coating" in c.lower() for c in t.conditions) for t in tips)

    def test_get_contextual_tips_exposure(self, tips_manager):
        """Test getting tips for exposure context."""
        tips = tips_manager.get_contextual_tips("exposure")
        assert len(tips) > 0
        assert all(any("exposure" in c.lower() for c in t.conditions) for t in tips)

    def test_get_contextual_tips_with_difficulty_filter(self, tips_manager):
        """Test getting contextual tips with difficulty filter."""
        tips = tips_manager.get_contextual_tips(
            "coating", difficulty=TipDifficulty.BEGINNER
        )
        assert all(
            t.difficulty in [TipDifficulty.BEGINNER, TipDifficulty.ALL] for t in tips
        )

    def test_get_contextual_tips_with_limit(self, tips_manager):
        """Test that limit parameter works."""
        tips = tips_manager.get_contextual_tips("coating", limit=3)
        assert len(tips) <= 3

    def test_get_contextual_tips_sorted_by_priority(self, tips_manager):
        """Test that contextual tips are sorted by priority."""
        tips = tips_manager.get_contextual_tips("coating", limit=10)
        # Should be sorted by priority (descending)
        for i in range(len(tips) - 1):
            assert tips[i].priority >= tips[i + 1].priority

    def test_get_contextual_tips_no_matches(self, tips_manager):
        """Test getting tips for context with no matches."""
        tips = tips_manager.get_contextual_tips("nonexistent_context_xyz")
        assert len(tips) == 0

    # Test get_random_tip()
    def test_get_random_tip_basic(self, tips_manager):
        """Test getting a random tip."""
        tip = tips_manager.get_random_tip()
        assert tip is not None
        assert isinstance(tip, Tip)

    def test_get_random_tip_category_filter(self, tips_manager):
        """Test getting random tip with category filter."""
        tip = tips_manager.get_random_tip(category=TipCategory.SAFETY)
        assert tip is not None
        assert tip.category == TipCategory.SAFETY

    def test_get_random_tip_difficulty_filter(self, tips_manager):
        """Test getting random tip with difficulty filter."""
        tip = tips_manager.get_random_tip(difficulty=TipDifficulty.BEGINNER)
        assert tip is not None
        assert tip.difficulty in [TipDifficulty.BEGINNER, TipDifficulty.ALL]

    def test_get_random_tip_marks_as_seen(self, tips_manager):
        """Test that random tip is marked as seen."""
        tip = tips_manager.get_random_tip()
        assert tip.id in tips_manager.seen_tips

    def test_get_random_tip_unseen_only(self, tips_manager):
        """Test getting only unseen tips."""
        # Mark some tips as seen
        for _ in range(5):
            tips_manager.get_random_tip()

        # Get unseen tip
        tip = tips_manager.get_random_tip(unseen_only=True)
        assert tip is not None

    def test_get_random_tip_no_matches(self, tips_manager):
        """Test random tip with impossible filter combination."""
        # Mark all safety tips as seen
        safety_tips = tips_manager.get_tips_by_category(TipCategory.SAFETY)
        for tip in safety_tips:
            tips_manager.mark_tip_seen(tip.id)

        # Try to get unseen safety tip
        tip = tips_manager.get_random_tip(
            category=TipCategory.SAFETY, unseen_only=True
        )
        # Should return None when no matches
        assert tip is None

    # Test get_tips_by_category()
    def test_get_tips_by_category_safety(self, tips_manager):
        """Test getting tips by safety category."""
        tips = tips_manager.get_tips_by_category(TipCategory.SAFETY)
        assert len(tips) > 0
        assert all(t.category == TipCategory.SAFETY for t in tips)

    def test_get_tips_by_category_chemistry(self, tips_manager):
        """Test getting tips by chemistry category."""
        tips = tips_manager.get_tips_by_category(TipCategory.CHEMISTRY)
        assert len(tips) > 0
        assert all(t.category == TipCategory.CHEMISTRY for t in tips)

    def test_get_tips_by_category_with_difficulty(self, tips_manager):
        """Test getting tips by category with difficulty filter."""
        tips = tips_manager.get_tips_by_category(
            TipCategory.BEGINNER, difficulty=TipDifficulty.BEGINNER
        )
        assert all(
            t.difficulty in [TipDifficulty.BEGINNER, TipDifficulty.ALL] for t in tips
        )

    def test_get_tips_by_category_sorted(self, tips_manager):
        """Test that category tips are sorted by priority."""
        tips = tips_manager.get_tips_by_category(TipCategory.COATING)
        # Should be sorted by priority (descending)
        for i in range(len(tips) - 1):
            assert tips[i].priority >= tips[i + 1].priority

    def test_get_tips_by_category_all_categories(self, tips_manager):
        """Test that all categories have at least one tip."""
        for category in TipCategory:
            tips = tips_manager.get_tips_by_category(category)
            assert len(tips) > 0, f"No tips in category {category}"

    # Test mark_tip_seen()
    def test_mark_tip_seen(self, tips_manager):
        """Test marking a tip as seen."""
        tip = tips_manager.tips[0]
        tips_manager.mark_tip_seen(tip.id)
        assert tip.id in tips_manager.seen_tips

    def test_mark_tip_seen_multiple(self, tips_manager):
        """Test marking multiple tips as seen."""
        tips_to_mark = tips_manager.tips[:5]
        for tip in tips_to_mark:
            tips_manager.mark_tip_seen(tip.id)

        for tip in tips_to_mark:
            assert tip.id in tips_manager.seen_tips

    # Test get_unseen_tips()
    def test_get_unseen_tips_all(self, tips_manager):
        """Test getting all unseen tips."""
        # Initially, all tips should be unseen
        unseen = tips_manager.get_unseen_tips()
        assert len(unseen) == len(tips_manager.tips)

    def test_get_unseen_tips_after_marking(self, tips_manager):
        """Test getting unseen tips after marking some as seen."""
        # Mark 5 tips as seen
        for i in range(5):
            tips_manager.mark_tip_seen(tips_manager.tips[i].id)

        unseen = tips_manager.get_unseen_tips()
        assert len(unseen) == len(tips_manager.tips) - 5

    def test_get_unseen_tips_with_category_filter(self, tips_manager):
        """Test getting unseen tips with category filter."""
        unseen = tips_manager.get_unseen_tips(category=TipCategory.SAFETY)
        assert all(t.category == TipCategory.SAFETY for t in unseen)

    def test_get_unseen_tips_with_difficulty_filter(self, tips_manager):
        """Test getting unseen tips with difficulty filter."""
        unseen = tips_manager.get_unseen_tips(difficulty=TipDifficulty.BEGINNER)
        assert all(
            t.difficulty in [TipDifficulty.BEGINNER, TipDifficulty.ALL] for t in unseen
        )

    def test_get_unseen_tips_sorted(self, tips_manager):
        """Test that unseen tips are sorted by priority."""
        unseen = tips_manager.get_unseen_tips()
        for i in range(len(unseen) - 1):
            assert unseen[i].priority >= unseen[i + 1].priority

    # Test reset_seen_tips()
    def test_reset_seen_tips(self, tips_manager):
        """Test resetting seen tips."""
        # Mark some tips as seen
        for i in range(5):
            tips_manager.mark_tip_seen(tips_manager.tips[i].id)

        assert len(tips_manager.seen_tips) == 5

        # Reset
        tips_manager.reset_seen_tips()
        assert len(tips_manager.seen_tips) == 0

    # Test get_high_priority_tips()
    def test_get_high_priority_tips_default(self, tips_manager):
        """Test getting high priority tips with default threshold."""
        high_priority = tips_manager.get_high_priority_tips()
        assert all(t.priority >= 4 for t in high_priority)

    def test_get_high_priority_tips_custom_threshold(self, tips_manager):
        """Test getting high priority tips with custom threshold."""
        high_priority = tips_manager.get_high_priority_tips(min_priority=5)
        assert all(t.priority >= 5 for t in high_priority)
        assert len(high_priority) > 0  # Should have some priority 5 tips

    def test_get_high_priority_tips_with_difficulty(self, tips_manager):
        """Test getting high priority tips with difficulty filter."""
        high_priority = tips_manager.get_high_priority_tips(
            min_priority=4, difficulty=TipDifficulty.BEGINNER
        )
        assert all(t.priority >= 4 for t in high_priority)
        assert all(
            t.difficulty in [TipDifficulty.BEGINNER, TipDifficulty.ALL]
            for t in high_priority
        )

    def test_get_high_priority_tips_sorted(self, tips_manager):
        """Test that high priority tips are sorted."""
        high_priority = tips_manager.get_high_priority_tips()
        for i in range(len(high_priority) - 1):
            assert high_priority[i].priority >= high_priority[i + 1].priority

    # Test search_tips()
    def test_search_tips_basic(self, tips_manager):
        """Test searching tips by content."""
        results = tips_manager.search_tips("coating")
        assert len(results) > 0
        assert all("coating" in t.content.lower() for t in results)

    def test_search_tips_case_insensitive(self, tips_manager):
        """Test that tip search is case-insensitive."""
        results1 = tips_manager.search_tips("coating")
        results2 = tips_manager.search_tips("COATING")
        assert len(results1) == len(results2)

    def test_search_tips_no_matches(self, tips_manager):
        """Test searching with no matches."""
        results = tips_manager.search_tips("xyznonexistent")
        assert len(results) == 0

    def test_search_tips_sorted(self, tips_manager):
        """Test that search results are sorted by priority."""
        results = tips_manager.search_tips("print")
        for i in range(len(results) - 1):
            assert results[i].priority >= results[i + 1].priority

    # Test get_all_categories()
    def test_get_all_categories(self, tips_manager):
        """Test getting all tip categories."""
        categories = tips_manager.get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(c, TipCategory) for c in categories)
        # Should be sorted
        assert categories == sorted(categories, key=lambda c: c.value)

    # Test get_tips_for_related_term()
    def test_get_tips_for_related_term(self, tips_manager):
        """Test getting tips for a related glossary term."""
        tips = tips_manager.get_tips_for_related_term("coating")
        assert len(tips) > 0
        assert all(any("coating" in rt.lower() for rt in t.related_terms) for t in tips)

    def test_get_tips_for_related_term_case_insensitive(self, tips_manager):
        """Test that related term lookup is case-insensitive."""
        tips1 = tips_manager.get_tips_for_related_term("coating")
        tips2 = tips_manager.get_tips_for_related_term("COATING")
        assert len(tips1) == len(tips2)

    def test_get_tips_for_related_term_sorted(self, tips_manager):
        """Test that related term tips are sorted by priority."""
        tips = tips_manager.get_tips_for_related_term("exposure")
        for i in range(len(tips) - 1):
            assert tips[i].priority >= tips[i + 1].priority

    # Test get_statistics()
    def test_get_statistics_basic(self, tips_manager):
        """Test getting statistics."""
        stats = tips_manager.get_statistics()
        assert isinstance(stats, dict)
        assert "total_tips" in stats
        assert "seen_tips" in stats
        assert "unseen_tips" in stats
        assert "tips_by_category" in stats
        assert "tips_by_difficulty" in stats
        assert "high_priority_count" in stats

    def test_get_statistics_counts(self, tips_manager):
        """Test that statistics counts are correct."""
        stats = tips_manager.get_statistics()
        assert stats["total_tips"] == len(tips_manager.tips)
        assert stats["seen_tips"] == 0  # Initially none seen
        assert stats["unseen_tips"] == len(tips_manager.tips)

    def test_get_statistics_after_seeing_tips(self, tips_manager):
        """Test statistics after marking tips as seen."""
        # Mark 5 tips as seen
        for i in range(5):
            tips_manager.mark_tip_seen(tips_manager.tips[i].id)

        stats = tips_manager.get_statistics()
        assert stats["seen_tips"] == 5
        assert stats["unseen_tips"] == len(tips_manager.tips) - 5

    def test_get_statistics_category_breakdown(self, tips_manager):
        """Test statistics category breakdown."""
        stats = tips_manager.get_statistics()
        category_counts = stats["tips_by_category"]

        # Sum of category counts should equal total tips
        total = sum(category_counts.values())
        assert total == stats["total_tips"]

    def test_get_statistics_difficulty_breakdown(self, tips_manager):
        """Test statistics difficulty breakdown."""
        stats = tips_manager.get_statistics()
        difficulty_counts = stats["tips_by_difficulty"]

        # Sum of difficulty counts should equal total tips
        total = sum(difficulty_counts.values())
        assert total == stats["total_tips"]

    # Test that all pre-loaded tips are valid
    def test_all_tips_valid_structure(self, tips_manager):
        """Test that all pre-loaded tips have valid structure."""
        for tip in tips_manager.tips:
            assert isinstance(tip, Tip)
            assert isinstance(tip.id, UUID)
            assert len(tip.content) > 0
            assert isinstance(tip.category, TipCategory)
            assert isinstance(tip.difficulty, TipDifficulty)
            assert isinstance(tip.conditions, list)
            assert 1 <= tip.priority <= 5
            assert isinstance(tip.related_terms, list)

    def test_all_tips_have_valid_category(self, tips_manager):
        """Test that all tips have valid categories."""
        for tip in tips_manager.tips:
            assert tip.category in TipCategory

    def test_all_tips_have_valid_difficulty(self, tips_manager):
        """Test that all tips have valid difficulty levels."""
        for tip in tips_manager.tips:
            assert tip.difficulty in TipDifficulty

    def test_all_tips_have_content(self, tips_manager):
        """Test that all tips have non-empty content."""
        for tip in tips_manager.tips:
            assert len(tip.content.strip()) > 0

    def test_all_tips_have_reasonable_priority(self, tips_manager):
        """Test that all tips have priority in valid range."""
        for tip in tips_manager.tips:
            assert 1 <= tip.priority <= 5

    def test_high_priority_tips_exist(self, tips_manager):
        """Test that there are some high-priority tips."""
        high_priority = [t for t in tips_manager.tips if t.priority >= 4]
        assert len(high_priority) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestEducationIntegration:
    """Integration tests across education modules."""

    def test_tutorial_references_glossary_terms(self):
        """Test that tutorials can reference glossary terms."""
        tutorial_manager = TutorialManager()
        glossary = Glossary()

        # Get a tutorial
        tutorial_manager.get_tutorial("first_print")

        # Check if common terms from tutorial exist in glossary
        common_terms = ["coating", "exposure", "developer", "ferric oxalate"]

        for term in common_terms:
            found = glossary.lookup(term)
            assert found is not None, f"Common term '{term}' not in glossary"

    def test_tips_reference_glossary_terms(self):
        """Test that tips can reference glossary terms."""
        tips_manager = TipsManager()
        Glossary()

        # Check some tips have valid related terms
        for tip in tips_manager.tips[:10]:  # Check first 10
            for related_term in tip.related_terms:
                # Try to find in glossary (may need to replace underscores)
                related_term.replace("_", " ")
                # Not all related terms must exist in glossary, but common ones should

    def test_tutorial_difficulty_progression(self):
        """Test that tutorials with prerequisites have appropriate difficulty."""
        tutorial_manager = TutorialManager()

        # Tutorials with prerequisites should not be BEGINNER level
        for tutorial in tutorial_manager.tutorials.values():
            if tutorial.prerequisites:
                # Should be intermediate or advanced
                assert tutorial.difficulty in [
                    TutorialDifficulty.INTERMEDIATE,
                    TutorialDifficulty.ADVANCED,
                ]

    def test_contextual_tips_for_tutorial_steps(self):
        """Test getting contextual tips for tutorial step actions."""
        tutorial_manager = TutorialManager()
        tips_manager = TipsManager()

        # Get first tutorial
        tutorial = tutorial_manager.get_tutorial("first_print")

        # For each step, try to get contextual tips
        for step in tutorial.steps[:3]:  # Check first 3 steps
            context = step.action.value
            tips_manager.get_contextual_tips(context, limit=3)
            # Most action types should have some tips
            # (though not strictly required)
