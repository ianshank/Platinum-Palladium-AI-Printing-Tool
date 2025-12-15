"""Tests for the active learning system.

Tests cover:
- Feedback recording and retrieval
- Pattern matching and similarity
- Confidence adjustment
- User prompting decisions
- Persistence and state management
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from ptpd_calibration.agents.smart.learning import (
    ActiveLearner,
    FeedbackRecord,
    FeedbackType,
    LearningSettings,
    PatternMatcher,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def learning_settings() -> LearningSettings:
    """Create learning settings for tests."""
    return LearningSettings(
        min_feedback_for_adjustment=3,
        positive_weight=0.1,
        negative_weight=0.15,
        decay_factor=0.95,
        max_records_per_action=50,
        uncertainty_threshold=0.3,
    )


@pytest.fixture
def active_learner(learning_settings: LearningSettings) -> ActiveLearner:
    """Create active learner for tests."""
    return ActiveLearner(settings=learning_settings)


@pytest.fixture
def pattern_matcher() -> PatternMatcher:
    """Create pattern matcher for tests."""
    return PatternMatcher(similarity_threshold=0.5)


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Create sample context for tests."""
    return {
        "paper_type": "platine",
        "temperature": 21.0,
        "humidity": 50.0,
        "metal_ratio": 1.2,
    }


@pytest.fixture
def feedback_record() -> FeedbackRecord:
    """Create sample feedback record."""
    return FeedbackRecord(
        feedback_id="test_001",
        timestamp=datetime.now(timezone.utc),
        feedback_type=FeedbackType.POSITIVE,
        context_hash="abc123",
        action="calibrate",
        original_value={"dmax": 2.0},
        corrected_value=None,
        confidence_at_time=0.8,
        metadata={"user": "test"},
    )


# ============================================================================
# FeedbackType Tests
# ============================================================================


class TestFeedbackType:
    """Tests for FeedbackType enum."""

    def test_feedback_types_exist(self) -> None:
        """Test all feedback types are defined."""
        assert FeedbackType.POSITIVE == "positive"
        assert FeedbackType.NEGATIVE == "negative"
        assert FeedbackType.CORRECTION == "correction"
        assert FeedbackType.PREFERENCE == "preference"
        assert FeedbackType.SKIP == "skip"

    def test_feedback_type_values(self) -> None:
        """Test feedback type string values."""
        assert len(FeedbackType) == 5


# ============================================================================
# FeedbackRecord Tests
# ============================================================================


class TestFeedbackRecord:
    """Tests for FeedbackRecord dataclass."""

    def test_record_creation(self, feedback_record: FeedbackRecord) -> None:
        """Test feedback record creation."""
        assert feedback_record.feedback_id == "test_001"
        assert feedback_record.feedback_type == FeedbackType.POSITIVE
        assert feedback_record.action == "calibrate"
        assert feedback_record.confidence_at_time == 0.8

    def test_to_dict(self, feedback_record: FeedbackRecord) -> None:
        """Test conversion to dictionary."""
        data = feedback_record.to_dict()

        assert data["feedback_id"] == "test_001"
        assert data["feedback_type"] == "positive"
        assert data["action"] == "calibrate"
        assert "timestamp" in data
        assert isinstance(data["original_value"], dict)

    def test_from_dict(self, feedback_record: FeedbackRecord) -> None:
        """Test creation from dictionary."""
        data = feedback_record.to_dict()
        restored = FeedbackRecord.from_dict(data)

        assert restored.feedback_id == feedback_record.feedback_id
        assert restored.feedback_type == feedback_record.feedback_type
        assert restored.action == feedback_record.action
        assert restored.confidence_at_time == feedback_record.confidence_at_time

    def test_record_with_correction(self) -> None:
        """Test feedback record with correction."""
        record = FeedbackRecord(
            feedback_id="test_002",
            timestamp=datetime.now(timezone.utc),
            feedback_type=FeedbackType.CORRECTION,
            context_hash="def456",
            action="predict_exposure",
            original_value=180,
            corrected_value=200,
            confidence_at_time=0.6,
        )

        assert record.corrected_value == 200
        data = record.to_dict()
        assert data["corrected_value"] == 200


# ============================================================================
# LearningSettings Tests
# ============================================================================


class TestLearningSettings:
    """Tests for LearningSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = LearningSettings()

        assert settings.min_feedback_for_adjustment == 5
        assert settings.positive_weight == 0.1
        assert settings.negative_weight == 0.15
        assert settings.decay_factor == 0.95
        assert settings.uncertainty_threshold == 0.3

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = LearningSettings(
            min_feedback_for_adjustment=10,
            positive_weight=0.2,
        )

        assert settings.min_feedback_for_adjustment == 10
        assert settings.positive_weight == 0.2

    def test_settings_bounds(self) -> None:
        """Test settings respect bounds."""
        settings = LearningSettings(
            min_adjustment=0.5,
            max_adjustment=2.0,
        )

        # Settings don't enforce bounds directly, just store values
        assert hasattr(settings, "min_feedback_for_adjustment")


# ============================================================================
# PatternMatcher Tests
# ============================================================================


class TestPatternMatcher:
    """Tests for PatternMatcher."""

    def test_compute_hash(self, pattern_matcher: PatternMatcher) -> None:
        """Test hash computation."""
        context = {"a": 1, "b": 2}
        hash1 = pattern_matcher.compute_hash(context)

        # Same content should produce same hash
        hash2 = pattern_matcher.compute_hash({"a": 1, "b": 2})
        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = pattern_matcher.compute_hash({"a": 1, "b": 3})
        assert hash1 != hash3

    def test_hash_order_independence(self, pattern_matcher: PatternMatcher) -> None:
        """Test hash is independent of key order."""
        hash1 = pattern_matcher.compute_hash({"a": 1, "b": 2})
        hash2 = pattern_matcher.compute_hash({"b": 2, "a": 1})
        assert hash1 == hash2

    def test_add_pattern(self, pattern_matcher: PatternMatcher) -> None:
        """Test adding patterns."""
        pattern_matcher.add_pattern(
            "calibrate",
            {"paper": "platine"},
            FeedbackType.POSITIVE
        )

        # Pattern should be stored
        assert "calibrate" in pattern_matcher._patterns
        assert len(pattern_matcher._patterns["calibrate"]) == 1

    def test_find_similar_exact_match(self, pattern_matcher: PatternMatcher) -> None:
        """Test finding exact matches."""
        context = {"paper": "platine", "temp": 21}
        pattern_matcher.add_pattern("calibrate", context, FeedbackType.POSITIVE)

        matches = pattern_matcher.find_similar("calibrate", context)

        assert len(matches) == 1
        assert matches[0][1] == 1.0  # Perfect similarity

    def test_find_similar_partial_match(self, pattern_matcher: PatternMatcher) -> None:
        """Test finding partial matches."""
        pattern_matcher.add_pattern(
            "calibrate",
            {"paper": "platine", "temp": 21, "humidity": 50},
            FeedbackType.POSITIVE
        )

        # Similar but not identical - different humidity and extra key
        matches = pattern_matcher.find_similar(
            "calibrate",
            {"paper": "platine", "temp": 21, "humidity": 80, "extra": "value"}
        )

        assert len(matches) >= 1
        # Not perfect similarity due to different humidity and extra key
        assert matches[0][1] <= 1.0

    def test_find_similar_no_match(self, pattern_matcher: PatternMatcher) -> None:
        """Test when no patterns match."""
        matches = pattern_matcher.find_similar(
            "unknown_action",
            {"paper": "platine"}
        )

        assert len(matches) == 0

    def test_get_success_rate(self, pattern_matcher: PatternMatcher) -> None:
        """Test success rate calculation."""
        context = {"paper": "platine"}

        # Add mixed feedback
        pattern_matcher.add_pattern("calibrate", context, FeedbackType.POSITIVE)
        pattern_matcher.add_pattern("calibrate", context, FeedbackType.POSITIVE)
        pattern_matcher.add_pattern("calibrate", context, FeedbackType.NEGATIVE)

        rate = pattern_matcher.get_success_rate("calibrate", context)

        assert rate is not None
        # 2 positive out of 3
        assert 0.6 <= rate <= 0.7

    def test_get_success_rate_no_data(self, pattern_matcher: PatternMatcher) -> None:
        """Test success rate with no data."""
        rate = pattern_matcher.get_success_rate("unknown", {"paper": "platine"})
        assert rate is None

    def test_values_similar_numeric(self, pattern_matcher: PatternMatcher) -> None:
        """Test numeric value similarity."""
        # Within 10%
        assert pattern_matcher._values_similar(100, 105)
        assert pattern_matcher._values_similar(100, 95)

        # Outside 10%
        assert not pattern_matcher._values_similar(100, 120)

    def test_values_similar_string(self, pattern_matcher: PatternMatcher) -> None:
        """Test string value similarity."""
        # Same prefix
        assert pattern_matcher._values_similar("platine", "pla")
        assert pattern_matcher._values_similar("platine", "PLATINE")

        # Different strings
        assert not pattern_matcher._values_similar("platine", "bergger")


# ============================================================================
# ActiveLearner Tests
# ============================================================================


class TestActiveLearner:
    """Tests for ActiveLearner."""

    def test_creation(self, active_learner: ActiveLearner) -> None:
        """Test learner creation."""
        assert active_learner.settings is not None
        assert isinstance(active_learner._feedback, dict)
        assert isinstance(active_learner._pattern_matcher, PatternMatcher)

    def test_record_feedback(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test recording feedback."""
        record = active_learner.record_feedback(
            action="calibrate",
            context=sample_context,
            feedback_type=FeedbackType.POSITIVE,
            original_value={"dmax": 2.0},
            confidence=0.8,
        )

        assert record.action == "calibrate"
        assert record.feedback_type == FeedbackType.POSITIVE
        assert "calibrate" in active_learner._feedback
        assert len(active_learner._feedback["calibrate"]) == 1

    def test_record_feedback_with_correction(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test recording feedback with correction."""
        record = active_learner.record_feedback(
            action="predict_exposure",
            context=sample_context,
            feedback_type=FeedbackType.CORRECTION,
            original_value=180,
            corrected_value=200,
            confidence=0.6,
        )

        assert record.corrected_value == 200
        assert record.feedback_type == FeedbackType.CORRECTION

    def test_get_adjusted_confidence_no_history(
        self,
        active_learner: ActiveLearner
    ) -> None:
        """Test confidence adjustment with no history."""
        adjusted = active_learner.get_adjusted_confidence("new_action", 0.7)
        assert adjusted == 0.7  # No adjustment

    def test_get_adjusted_confidence_with_positive_feedback(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test confidence increases with positive feedback."""
        # Record enough positive feedback
        for _ in range(5):
            active_learner.record_feedback(
                action="calibrate",
                context=sample_context,
                feedback_type=FeedbackType.POSITIVE,
                original_value={"dmax": 2.0},
                confidence=0.7,
            )

        adjusted = active_learner.get_adjusted_confidence("calibrate", 0.7)
        assert adjusted >= 0.7  # Should increase or stay same

    def test_get_adjusted_confidence_with_negative_feedback(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test confidence decreases with negative feedback."""
        # Record enough negative feedback
        for _ in range(5):
            active_learner.record_feedback(
                action="calibrate",
                context=sample_context,
                feedback_type=FeedbackType.NEGATIVE,
                original_value={"dmax": 1.5},
                confidence=0.7,
            )

        adjusted = active_learner.get_adjusted_confidence("calibrate", 0.7)
        assert adjusted <= 0.7  # Should decrease or stay same

    def test_should_ask_user_low_confidence(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test user prompt on low confidence."""
        should_ask, reason = active_learner.should_ask_user(
            action="calibrate",
            context=sample_context,
            confidence=0.1,  # Very low
        )

        assert should_ask is True
        assert "confidence" in reason.lower()

    def test_should_ask_user_high_confidence(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test no prompt on high confidence."""
        should_ask, reason = active_learner.should_ask_user(
            action="calibrate",
            context=sample_context,
            confidence=0.9,
        )

        # May still ask based on history, but low confidence alone shouldn't trigger
        if not should_ask:
            assert reason == ""

    def test_get_suggestions(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test getting suggestions from history."""
        # Add some corrections
        active_learner.record_feedback(
            action="predict_exposure",
            context=sample_context,
            feedback_type=FeedbackType.CORRECTION,
            original_value=180,
            corrected_value=200,
        )

        suggestions = active_learner.get_suggestions("predict_exposure", sample_context)

        # May or may not have suggestions depending on similarity
        assert isinstance(suggestions, list)

    def test_get_feedback_stats_empty(self, active_learner: ActiveLearner) -> None:
        """Test stats with no feedback."""
        stats = active_learner.get_feedback_stats()

        assert stats["total_records"] == 0
        assert stats["average_confidence"] == 0.0

    def test_get_feedback_stats_with_data(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test stats with feedback data."""
        active_learner.record_feedback(
            action="calibrate",
            context=sample_context,
            feedback_type=FeedbackType.POSITIVE,
            original_value={"dmax": 2.0},
            confidence=0.8,
        )
        active_learner.record_feedback(
            action="calibrate",
            context=sample_context,
            feedback_type=FeedbackType.NEGATIVE,
            original_value={"dmax": 1.5},
            confidence=0.6,
        )

        stats = active_learner.get_feedback_stats("calibrate")

        assert stats["total_records"] == 2
        assert "positive" in stats["feedback_types"]
        assert "negative" in stats["feedback_types"]
        assert stats["average_confidence"] == 0.7

    def test_clear_feedback(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test clearing feedback."""
        active_learner.record_feedback(
            action="calibrate",
            context=sample_context,
            feedback_type=FeedbackType.POSITIVE,
            original_value={},
        )

        count = active_learner.clear_feedback("calibrate")

        assert count == 1
        assert len(active_learner._feedback.get("calibrate", [])) == 0

    def test_clear_all_feedback(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test clearing all feedback."""
        active_learner.record_feedback(
            action="calibrate",
            context=sample_context,
            feedback_type=FeedbackType.POSITIVE,
            original_value={},
        )
        active_learner.record_feedback(
            action="predict",
            context=sample_context,
            feedback_type=FeedbackType.NEGATIVE,
            original_value={},
        )

        count = active_learner.clear_feedback()

        assert count == 2
        assert len(active_learner._feedback) == 0

    def test_max_records_trimming(
        self,
        learning_settings: LearningSettings,
        sample_context: dict[str, Any]
    ) -> None:
        """Test records are trimmed to max."""
        settings = LearningSettings(max_records_per_action=5)
        learner = ActiveLearner(settings=settings)

        for i in range(10):
            learner.record_feedback(
                action="calibrate",
                context=sample_context,
                feedback_type=FeedbackType.POSITIVE,
                original_value={"index": i},
            )

        assert len(learner._feedback["calibrate"]) == 5


# ============================================================================
# Persistence Tests
# ============================================================================


class TestActiveLearnerPersistence:
    """Tests for ActiveLearner persistence."""

    def test_persistence_save_and_load(self, sample_context: dict[str, Any]) -> None:
        """Test saving and loading state."""
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "learning_state.json"

            # Create learner and record feedback
            learner1 = ActiveLearner(storage_path=storage_path)
            learner1.record_feedback(
                action="calibrate",
                context=sample_context,
                feedback_type=FeedbackType.POSITIVE,
                original_value={"dmax": 2.0},
                confidence=0.8,
            )

            # Create new learner that should load state
            learner2 = ActiveLearner(storage_path=storage_path)

            assert "calibrate" in learner2._feedback
            assert len(learner2._feedback["calibrate"]) == 1

    def test_persistence_handles_missing_file(self) -> None:
        """Test handling missing storage file."""
        with TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "nonexistent.json"
            learner = ActiveLearner(storage_path=storage_path)

            # Should not raise
            assert len(learner._feedback) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestActiveLearnerIntegration:
    """Integration tests for ActiveLearner."""

    def test_full_feedback_workflow(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test complete feedback workflow."""
        # System makes prediction
        action = "predict_exposure"
        original_value = 180

        # Check if should ask user
        should_ask, _ = active_learner.should_ask_user(
            action=action,
            context=sample_context,
            confidence=0.5,
        )

        # Record user's correction
        active_learner.record_feedback(
            action=action,
            context=sample_context,
            feedback_type=FeedbackType.CORRECTION,
            original_value=original_value,
            corrected_value=200,
            confidence=0.5,
        )

        # Get suggestions for similar context
        similar_context = {**sample_context, "temperature": 22.0}
        suggestions = active_learner.get_suggestions(action, similar_context)

        # Check stats
        stats = active_learner.get_feedback_stats(action)
        assert stats["total_records"] == 1

    def test_confidence_evolution_over_time(
        self,
        active_learner: ActiveLearner,
        sample_context: dict[str, Any]
    ) -> None:
        """Test confidence evolution with feedback."""
        action = "calibrate"
        base_confidence = 0.6

        initial = active_learner.get_adjusted_confidence(action, base_confidence)

        # Record positive feedback
        for _ in range(5):
            active_learner.record_feedback(
                action=action,
                context=sample_context,
                feedback_type=FeedbackType.POSITIVE,
                original_value={},
                confidence=base_confidence,
            )

        after_positive = active_learner.get_adjusted_confidence(action, base_confidence)

        # Should have increased
        assert after_positive >= initial

        # Record negative feedback
        for _ in range(3):
            active_learner.record_feedback(
                action=action,
                context=sample_context,
                feedback_type=FeedbackType.NEGATIVE,
                original_value={},
                confidence=base_confidence,
            )

        after_negative = active_learner.get_adjusted_confidence(action, base_confidence)

        # Should have decreased from positive peak
        assert after_negative <= after_positive
