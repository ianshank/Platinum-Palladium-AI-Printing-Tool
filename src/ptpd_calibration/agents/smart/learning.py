"""Active learning system for continuous improvement from user feedback.

This module implements an active learning system that:
- Records user feedback on calibrations and predictions
- Identifies patterns in successful and failed attempts
- Adjusts confidence and recommendations based on history
- Suggests when user input would be most valuable
"""

from __future__ import annotations

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedbackType(str, Enum):
    """Types of user feedback."""

    POSITIVE = "positive"  # User confirmed result was good
    NEGATIVE = "negative"  # User indicated result was wrong
    CORRECTION = "correction"  # User provided corrected value
    PREFERENCE = "preference"  # User indicated a preference
    SKIP = "skip"  # User skipped/ignored the suggestion


@dataclass
class FeedbackRecord:
    """Record of user feedback on a system action."""

    feedback_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    context_hash: str  # Hash of input context for pattern matching
    action: str  # What the system did
    original_value: Any  # System's original output
    corrected_value: Optional[Any] = None  # User's correction if applicable
    confidence_at_time: float = 0.0  # System's confidence when action was taken
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feedback_id": self.feedback_id,
            "timestamp": self.timestamp.isoformat(),
            "feedback_type": self.feedback_type.value,
            "context_hash": self.context_hash,
            "action": self.action,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "confidence_at_time": self.confidence_at_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        """Create from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            feedback_type=FeedbackType(data["feedback_type"]),
            context_hash=data["context_hash"],
            action=data["action"],
            original_value=data["original_value"],
            corrected_value=data.get("corrected_value"),
            confidence_at_time=data.get("confidence_at_time", 0.0),
            metadata=data.get("metadata", {}),
        )


class LearningSettings(BaseSettings):
    """Settings for the active learning system."""

    model_config = SettingsConfigDict(
        env_prefix="PTPD_LEARNING_",
        env_file=".env",
        extra="ignore",
    )

    # Minimum feedback records before adjusting confidence
    min_feedback_for_adjustment: int = Field(
        default=5,
        description="Minimum feedback records before adjusting confidence"
    )

    # Weight for positive feedback
    positive_weight: float = Field(
        default=0.1,
        description="How much positive feedback increases confidence"
    )

    # Weight for negative feedback
    negative_weight: float = Field(
        default=0.15,
        description="How much negative feedback decreases confidence"
    )

    # Decay factor for old feedback
    decay_factor: float = Field(
        default=0.95,
        description="Decay factor per week for old feedback importance"
    )

    # Maximum records to keep per action
    max_records_per_action: int = Field(
        default=100,
        description="Maximum feedback records to keep per action type"
    )

    # Uncertainty threshold for asking user
    uncertainty_threshold: float = Field(
        default=0.3,
        description="Ask user when confidence is below this threshold"
    )

    # Storage path for feedback data
    storage_path: Optional[str] = Field(
        default=None,
        description="Path to store feedback data"
    )


T = TypeVar("T")


class LearningModel(ABC, Generic[T]):
    """Abstract base class for learning models."""

    @abstractmethod
    def update(self, feedback: FeedbackRecord) -> None:
        """Update model with new feedback."""
        pass

    @abstractmethod
    def predict(self, context: dict[str, Any]) -> tuple[T, float]:
        """Make prediction with confidence."""
        pass

    @abstractmethod
    def get_uncertainty(self, context: dict[str, Any]) -> float:
        """Get uncertainty for a given context."""
        pass


class PatternMatcher:
    """Matches contexts to find similar historical patterns."""

    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize pattern matcher.

        Args:
            similarity_threshold: Minimum similarity to consider a match
        """
        self.similarity_threshold = similarity_threshold
        self._patterns: dict[str, list[dict[str, Any]]] = {}

    def compute_hash(self, context: dict[str, Any]) -> str:
        """Compute a hash for a context dictionary.

        Args:
            context: Context dictionary to hash

        Returns:
            Hash string for the context
        """
        # Sort keys for consistent hashing
        normalized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def add_pattern(self, action: str, context: dict[str, Any], outcome: Any) -> None:
        """Add a pattern to the matcher.

        Args:
            action: The action taken
            context: The context when action was taken
            outcome: The outcome (positive/negative/value)
        """
        if action not in self._patterns:
            self._patterns[action] = []

        self._patterns[action].append({
            "context": context,
            "context_hash": self.compute_hash(context),
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc),
        })

    def find_similar(
        self,
        action: str,
        context: dict[str, Any],
        limit: int = 10
    ) -> list[tuple[dict[str, Any], float]]:
        """Find similar patterns for a given context.

        Args:
            action: The action to find patterns for
            context: Current context to match against
            limit: Maximum number of matches to return

        Returns:
            List of (pattern, similarity_score) tuples
        """
        if action not in self._patterns:
            return []

        matches = []
        context_keys = set(context.keys())

        for pattern in self._patterns[action]:
            pattern_context = pattern["context"]
            pattern_keys = set(pattern_context.keys())

            # Calculate Jaccard similarity for keys
            key_intersection = len(context_keys & pattern_keys)
            key_union = len(context_keys | pattern_keys)
            key_similarity = key_intersection / key_union if key_union > 0 else 0

            # Calculate value similarity for common keys
            common_keys = context_keys & pattern_keys
            if common_keys:
                value_matches = sum(
                    1 for k in common_keys
                    if self._values_similar(context.get(k), pattern_context.get(k))
                )
                value_similarity = value_matches / len(common_keys)
            else:
                value_similarity = 0

            # Combined similarity
            similarity = (key_similarity + value_similarity) / 2

            if similarity >= self.similarity_threshold:
                matches.append((pattern, similarity))

        # Sort by similarity (descending) and return top matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]

    def _values_similar(self, v1: Any, v2: Any) -> bool:
        """Check if two values are similar.

        Args:
            v1: First value
            v2: Second value

        Returns:
            True if values are similar
        """
        if v1 == v2:
            return True

        # Numeric similarity (within 10%)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 == 0 and v2 == 0:
                return True
            avg = (abs(v1) + abs(v2)) / 2
            return abs(v1 - v2) / avg < 0.1 if avg > 0 else False

        # String similarity (case-insensitive prefix)
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower().startswith(v2.lower()[:3]) or v2.lower().startswith(v1.lower()[:3])

        return False

    def get_success_rate(self, action: str, context: dict[str, Any]) -> Optional[float]:
        """Get historical success rate for similar contexts.

        Args:
            action: The action to check
            context: Current context

        Returns:
            Success rate (0-1) or None if no matches
        """
        similar = self.find_similar(action, context)
        if not similar:
            return None

        positive = sum(
            1 for pattern, _ in similar
            if pattern["outcome"] in [FeedbackType.POSITIVE, FeedbackType.POSITIVE.value, True]
        )
        return positive / len(similar)


class ActiveLearner:
    """Main active learning system that coordinates feedback and learning."""

    def __init__(
        self,
        settings: Optional[LearningSettings] = None,
        storage_path: Optional[Path] = None
    ):
        """Initialize active learner.

        Args:
            settings: Learning settings
            storage_path: Optional path for persistent storage
        """
        self.settings = settings or LearningSettings()
        self._storage_path = storage_path or (
            Path(self.settings.storage_path) if self.settings.storage_path else None
        )

        # Feedback storage by action
        self._feedback: dict[str, list[FeedbackRecord]] = {}

        # Pattern matcher for finding similar contexts
        self._pattern_matcher = PatternMatcher()

        # Confidence adjustments by action
        self._confidence_adjustments: dict[str, float] = {}

        # Load persisted data if available
        if self._storage_path and self._storage_path.exists():
            self._load_state()

    def record_feedback(
        self,
        action: str,
        context: dict[str, Any],
        feedback_type: FeedbackType,
        original_value: Any,
        corrected_value: Optional[Any] = None,
        confidence: float = 0.0,
        metadata: Optional[dict[str, Any]] = None
    ) -> FeedbackRecord:
        """Record user feedback on a system action.

        Args:
            action: The action/operation that was performed
            context: Context when the action was taken
            feedback_type: Type of feedback
            original_value: The system's original output
            corrected_value: User's correction if applicable
            confidence: System's confidence at time of action
            metadata: Additional metadata

        Returns:
            The created feedback record
        """
        context_hash = self._pattern_matcher.compute_hash(context)

        record = FeedbackRecord(
            feedback_id=f"{action}_{context_hash}_{datetime.now(timezone.utc).timestamp()}",
            timestamp=datetime.now(timezone.utc),
            feedback_type=feedback_type,
            context_hash=context_hash,
            action=action,
            original_value=original_value,
            corrected_value=corrected_value,
            confidence_at_time=confidence,
            metadata=metadata or {},
        )

        # Store feedback
        if action not in self._feedback:
            self._feedback[action] = []
        self._feedback[action].append(record)

        # Trim to max records
        if len(self._feedback[action]) > self.settings.max_records_per_action:
            self._feedback[action] = self._feedback[action][-self.settings.max_records_per_action:]

        # Update pattern matcher
        outcome = feedback_type if feedback_type != FeedbackType.CORRECTION else corrected_value
        self._pattern_matcher.add_pattern(action, context, outcome)

        # Update confidence adjustments
        self._update_confidence_adjustment(action)

        # Persist state
        if self._storage_path:
            self._save_state()

        return record

    def get_adjusted_confidence(self, action: str, base_confidence: float) -> float:
        """Get confidence adjusted based on historical feedback.

        Args:
            action: The action being performed
            base_confidence: The system's base confidence

        Returns:
            Adjusted confidence (0-1)
        """
        adjustment = self._confidence_adjustments.get(action, 0.0)
        adjusted = base_confidence + adjustment
        return max(0.0, min(1.0, adjusted))

    def should_ask_user(
        self,
        action: str,
        context: dict[str, Any],
        confidence: float
    ) -> tuple[bool, str]:
        """Determine if system should ask user for input.

        Args:
            action: The action being considered
            context: Current context
            confidence: Current confidence level

        Returns:
            Tuple of (should_ask, reason)
        """
        adjusted_confidence = self.get_adjusted_confidence(action, confidence)

        # Check confidence threshold
        if adjusted_confidence < self.settings.uncertainty_threshold:
            return True, f"Low confidence ({adjusted_confidence:.2f})"

        # Check historical success rate
        success_rate = self._pattern_matcher.get_success_rate(action, context)
        if success_rate is not None and success_rate < 0.5:
            return True, f"Low historical success rate ({success_rate:.2f})"

        # Check for conflicting historical feedback
        similar = self._pattern_matcher.find_similar(action, context)
        if len(similar) >= 3:
            outcomes = [p["outcome"] for p, _ in similar]
            unique_outcomes = len(set(str(o) for o in outcomes))
            if unique_outcomes > 2:
                return True, "Conflicting historical outcomes"

        return False, ""

    def get_suggestions(
        self,
        action: str,
        context: dict[str, Any]
    ) -> list[tuple[Any, float]]:
        """Get suggestions based on historical patterns.

        Args:
            action: The action being performed
            context: Current context

        Returns:
            List of (suggested_value, confidence) tuples
        """
        similar = self._pattern_matcher.find_similar(action, context)
        if not similar:
            return []

        # Collect corrections from similar patterns
        suggestions: dict[str, tuple[Any, float]] = {}

        for pattern, similarity in similar:
            outcome = pattern["outcome"]
            if isinstance(outcome, FeedbackType):
                continue  # Skip non-value outcomes

            key = str(outcome)
            if key in suggestions:
                # Average the confidence
                existing_value, existing_conf = suggestions[key]
                suggestions[key] = (outcome, (existing_conf + similarity) / 2)
            else:
                suggestions[key] = (outcome, similarity)

        # Sort by confidence
        result = list(suggestions.values())
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_feedback_stats(self, action: Optional[str] = None) -> dict[str, Any]:
        """Get statistics about collected feedback.

        Args:
            action: Optional action to filter by

        Returns:
            Dictionary of feedback statistics
        """
        if action:
            records = self._feedback.get(action, [])
            actions = [action]
        else:
            records = []
            for action_records in self._feedback.values():
                records.extend(action_records)
            actions = list(self._feedback.keys())

        if not records:
            return {
                "total_records": 0,
                "actions": actions,
                "feedback_types": {},
                "average_confidence": 0.0,
            }

        # Count by feedback type
        type_counts = {}
        for record in records:
            type_name = record.feedback_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Calculate average confidence
        avg_confidence = sum(r.confidence_at_time for r in records) / len(records)

        return {
            "total_records": len(records),
            "actions": actions,
            "feedback_types": type_counts,
            "average_confidence": avg_confidence,
            "confidence_adjustments": {
                a: self._confidence_adjustments.get(a, 0.0)
                for a in actions
            },
        }

    def _update_confidence_adjustment(self, action: str) -> None:
        """Update confidence adjustment for an action based on feedback.

        Args:
            action: The action to update
        """
        records = self._feedback.get(action, [])
        if len(records) < self.settings.min_feedback_for_adjustment:
            return

        # Calculate weighted adjustment based on feedback
        adjustment = 0.0
        total_weight = 0.0
        now = datetime.now(timezone.utc)

        for record in records:
            # Calculate time-based decay
            age_days = (now - record.timestamp).days
            age_weeks = age_days / 7
            weight = self.settings.decay_factor ** age_weeks

            if record.feedback_type == FeedbackType.POSITIVE:
                adjustment += self.settings.positive_weight * weight
            elif record.feedback_type == FeedbackType.NEGATIVE:
                adjustment -= self.settings.negative_weight * weight
            elif record.feedback_type == FeedbackType.CORRECTION:
                # Corrections are treated as negative feedback
                adjustment -= self.settings.negative_weight * weight * 0.5

            total_weight += weight

        if total_weight > 0:
            self._confidence_adjustments[action] = adjustment / total_weight

    def _save_state(self) -> None:
        """Save state to persistent storage."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "feedback": {
                action: [r.to_dict() for r in records]
                for action, records in self._feedback.items()
            },
            "confidence_adjustments": self._confidence_adjustments,
        }

        self._storage_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            state = json.loads(self._storage_path.read_text())

            # Load feedback
            for action, records in state.get("feedback", {}).items():
                self._feedback[action] = [
                    FeedbackRecord.from_dict(r) for r in records
                ]
                # Update pattern matcher
                for record in self._feedback[action]:
                    # Reconstruct context from hash (we can't fully restore)
                    # Just add the outcomes to pattern matcher
                    pass

            # Load confidence adjustments
            self._confidence_adjustments = state.get("confidence_adjustments", {})
        except (json.JSONDecodeError, KeyError):
            # Start fresh on error
            pass

    def clear_feedback(self, action: Optional[str] = None) -> int:
        """Clear stored feedback.

        Args:
            action: Optional action to clear (clears all if None)

        Returns:
            Number of records cleared
        """
        if action:
            count = len(self._feedback.get(action, []))
            self._feedback.pop(action, None)
            self._confidence_adjustments.pop(action, None)
        else:
            count = sum(len(records) for records in self._feedback.values())
            self._feedback.clear()
            self._confidence_adjustments.clear()

        if self._storage_path:
            self._save_state()

        return count
