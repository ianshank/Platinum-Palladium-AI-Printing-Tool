"""End-to-end and user journey tests for the agent system.

These tests simulate complete user workflows from start to finish,
verifying the entire system works together as users would experience it.

Test scenarios cover:
- New user setup and first calibration
- Experienced user optimizing workflow
- Troubleshooting session
- Quality improvement journey
- Multi-session learning
"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from ptpd_calibration.agents.router import (
    TaskRouter,
    TaskCategory,
    TaskComplexity,
    RouterSettings,
)
from ptpd_calibration.agents.skills import (
    CalibrationSkill,
    ChemistrySkill,
    QualitySkill,
    TroubleshootingSkill,
    create_default_skills,
    SkillContext,
    SkillRegistry,
)
from ptpd_calibration.agents.smart.learning import (
    ActiveLearner,
    FeedbackType,
    LearningSettings,
)
from ptpd_calibration.agents.smart.environment import (
    EnvironmentAdapter,
    EnvironmentConditions,
    EnvironmentSettings,
)
from ptpd_calibration.agents.smart.prediction import (
    QualityPredictor,
    QualityFeatures,
    PredictionSettings,
)


# ============================================================================
# Agent System Facade for User Journey Tests
# ============================================================================


class AgentSystem:
    """Facade class that integrates all agent components for user journeys.

    This provides a unified interface similar to what a real application would use.
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        learning_settings: LearningSettings | None = None,
        env_settings: EnvironmentSettings | None = None,
        prediction_settings: PredictionSettings | None = None,
    ):
        """Initialize the agent system.

        Args:
            storage_path: Path for persistent storage
            learning_settings: Settings for active learning
            env_settings: Settings for environment adaptation
            prediction_settings: Settings for quality prediction
        """
        self.router = TaskRouter()
        self.skills = create_default_skills()
        self.learner = ActiveLearner(
            settings=learning_settings or LearningSettings(min_feedback_for_adjustment=3),
            storage_path=storage_path / "learning.json" if storage_path else None,
        )
        self.env_adapter = EnvironmentAdapter(
            settings=env_settings or EnvironmentSettings()
        )
        self.predictor = QualityPredictor(
            settings=prediction_settings or PredictionSettings(min_data_points=3)
        )

        # Session state
        self._current_conditions: EnvironmentConditions | None = None
        self._session_tasks: list[str] = []

    def set_environment(
        self,
        temperature: float,
        humidity: float,
        **kwargs: Any
    ) -> tuple[bool, list[str]]:
        """Set current environmental conditions.

        Args:
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            **kwargs: Additional environment parameters

        Returns:
            Tuple of (is_optimal, warnings)
        """
        self._current_conditions = EnvironmentConditions(
            temperature_celsius=temperature,
            humidity_percent=humidity,
            **kwargs
        )
        return self._current_conditions.is_optimal()

    def process_task(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Process a user task through the agent system.

        Args:
            task: User's task description
            context: Additional context
            **kwargs: Task-specific parameters

        Returns:
            Result dictionary with status, data, and suggestions
        """
        self._session_tasks.append(task)

        # Route the task
        routing = self.router.route(task, context)

        # Find and execute skill
        skill_result = self.skills.find_best_skill(task)

        if skill_result is None:
            return {
                "success": False,
                "message": "No skill found to handle this task",
                "suggestions": ["Try rephrasing your request"],
            }

        skill, confidence = skill_result

        # Build skill context
        skill_context = None
        if context or self._current_conditions:
            skill_context = SkillContext(
                paper_type=context.get("paper_type") if context else None,
                temperature=(
                    self._current_conditions.temperature_celsius
                    if self._current_conditions else None
                ),
                humidity=(
                    self._current_conditions.humidity_percent
                    if self._current_conditions else None
                ),
            )

        # Execute skill
        result = skill.execute(task, skill_context, **kwargs)

        return {
            "success": result.success,
            "category": routing.category.value,
            "complexity": routing.complexity.value,
            "data": result.data,
            "message": result.message,
            "confidence": result.confidence,
            "suggestions": result.suggestions,
            "next_actions": result.next_actions,
            "skill_used": skill.name,
        }

    def provide_feedback(
        self,
        task: str,
        feedback_type: FeedbackType,
        context: dict[str, Any] | None = None,
        original_value: Any = None,
        corrected_value: Any = None,
        confidence: float = 0.5,
    ) -> None:
        """Provide feedback on a task result.

        Args:
            task: The task that was performed
            feedback_type: Type of feedback
            context: Task context
            original_value: System's original output
            corrected_value: User's correction if applicable
            confidence: System's confidence at time of task
        """
        self.learner.record_feedback(
            action=task,
            context=context or {},
            feedback_type=feedback_type,
            original_value=original_value,
            corrected_value=corrected_value,
            confidence=confidence,
        )

    def predict_quality(
        self,
        features: QualityFeatures
    ) -> dict[str, Any]:
        """Predict print quality.

        Args:
            features: Quality features

        Returns:
            Prediction result dictionary
        """
        # Add environmental data if available
        if self._current_conditions:
            features = QualityFeatures(
                dmin=features.dmin,
                dmax=features.dmax,
                contrast=features.contrast,
                curve_smoothness=features.curve_smoothness,
                metal_ratio=features.metal_ratio,
                coating_amount=features.coating_amount,
                sensitizer_ratio=features.sensitizer_ratio,
                temperature=self._current_conditions.temperature_celsius,
                humidity=self._current_conditions.humidity_percent,
                exposure_time=features.exposure_time,
                paper_type=features.paper_type,
                coating_age_hours=features.coating_age_hours,
            )

        prediction = self.predictor.predict(features)

        return {
            "predicted_quality": prediction.predicted_quality,
            "grade": prediction.quality_grade.value,
            "confidence": prediction.confidence,
            "confidence_interval": prediction.confidence_interval,
            "risks": prediction.risk_factors,
            "recommendations": prediction.recommendations,
            "acceptable": prediction.is_acceptable(),
        }

    def adapt_exposure(self, base_exposure: float) -> dict[str, Any]:
        """Adapt exposure time for current conditions.

        Args:
            base_exposure: Base exposure time in seconds

        Returns:
            Adaptation result dictionary
        """
        if not self._current_conditions:
            return {
                "adjusted_exposure": base_exposure,
                "factor": 1.0,
                "message": "No environmental data available",
            }

        result = self.env_adapter.adapt_exposure(base_exposure, self._current_conditions)

        return {
            "original_exposure": result.original_exposure,
            "adjusted_exposure": result.adjusted_exposure,
            "factor": result.adjustment_factor,
            "adjustments": result.adjustments_applied,
            "recommendations": result.recommendations,
            "warnings": result.warnings,
        }

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current session.

        Returns:
            Session summary dictionary
        """
        return {
            "tasks_processed": len(self._session_tasks),
            "task_history": self._session_tasks.copy(),
            "has_environment": self._current_conditions is not None,
            "learning_stats": self.learner.get_feedback_stats(),
            "prediction_stats": self.predictor.get_statistics(),
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def agent_system() -> AgentSystem:
    """Create agent system for tests."""
    return AgentSystem()


@pytest.fixture
def persistent_system(tmp_path: Path) -> AgentSystem:
    """Create agent system with persistent storage."""
    return AgentSystem(storage_path=tmp_path)


# ============================================================================
# User Journey: New User First Calibration
# ============================================================================


class TestNewUserFirstCalibration:
    """Journey: A new user sets up their first calibration."""

    def test_new_user_complete_journey(self, agent_system: AgentSystem) -> None:
        """Test complete new user journey from setup to first print."""
        # Step 1: User sets their environment
        is_optimal, warnings = agent_system.set_environment(
            temperature=21.0,
            humidity=50.0
        )
        assert is_optimal, "Optimal conditions should be detected"

        # Step 2: User asks about coating calculation
        result = agent_system.process_task(
            "calculate coating for 8x10 print",
            print_size=(8, 10)
        )
        assert result["success"]
        assert result["category"] == "chemistry"

        # Step 3: User provides positive feedback
        agent_system.provide_feedback(
            task="coating_calculation",
            feedback_type=FeedbackType.POSITIVE,
            context={"paper_type": "platine", "size": "8x10"},
            original_value=result["data"],
        )

        # Step 4: User asks about exposure time
        exposure_result = agent_system.adapt_exposure(180.0)
        assert exposure_result["factor"] == pytest.approx(1.0, rel=0.1)

        # Step 5: User checks predicted quality
        features = QualityFeatures(
            dmax=2.0,
            dmin=0.1,
            contrast=1.8,
            exposure_time=180.0,
            coating_age_hours=0.5,
        )
        prediction = agent_system.predict_quality(features)
        assert prediction["acceptable"]

        # Step 6: Verify session summary
        summary = agent_system.get_session_summary()
        assert summary["tasks_processed"] >= 1
        assert summary["has_environment"]

    def test_new_user_with_suboptimal_conditions(
        self,
        agent_system: AgentSystem
    ) -> None:
        """Test new user journey with suboptimal environmental conditions."""
        # User's workspace is too warm and humid
        is_optimal, warnings = agent_system.set_environment(
            temperature=28.0,
            humidity=70.0
        )
        assert not is_optimal
        assert len(warnings) >= 2

        # User asks about coating - should still work
        result = agent_system.process_task(
            "calculate coating for 5x7 print",
            print_size=(5, 7)
        )
        assert result["success"]

        # Exposure should be adjusted
        exposure_result = agent_system.adapt_exposure(180.0)
        assert exposure_result["factor"] != 1.0
        assert len(exposure_result["warnings"]) > 0

        # Quality prediction should flag risks
        features = QualityFeatures(
            dmax=2.0,
            exposure_time=180.0,
        )
        prediction = agent_system.predict_quality(features)
        assert len(prediction["risks"]) > 0


# ============================================================================
# User Journey: Experienced User Workflow Optimization
# ============================================================================


class TestExperiencedUserOptimization:
    """Journey: An experienced user optimizes their workflow."""

    def test_optimization_workflow(self, agent_system: AgentSystem) -> None:
        """Test experienced user optimizing their printing workflow."""
        # Setup environment
        agent_system.set_environment(temperature=22.0, humidity=48.0)

        # Step 1: Ask for optimal metal ratio
        ratio_result = agent_system.process_task(
            "calculate chemistry ratio for warm tone",
            target_tone="warm"
        )
        assert ratio_result["success"]
        # Category can be chemistry or prediction depending on routing
        assert ratio_result["category"] in ["chemistry", "prediction"]

        # Step 2: Calculate optimized coating
        coating_result = agent_system.process_task(
            "calculate coating for 11x14 print",
            print_size=(11, 14)
        )
        assert coating_result["success"]

        # Step 3: Check quality with multiple scenarios
        scenarios = [
            {"exposure_time": 150.0, "coating_age_hours": 0.5},
            {"exposure_time": 180.0, "coating_age_hours": 1.0},
            {"exposure_time": 200.0, "coating_age_hours": 2.0},
        ]

        best_quality = 0
        best_scenario = None

        for scenario in scenarios:
            features = QualityFeatures(
                dmax=2.0,
                dmin=0.1,
                contrast=1.8,
                exposure_time=scenario["exposure_time"],
                coating_age_hours=scenario["coating_age_hours"],
            )
            prediction = agent_system.predict_quality(features)
            if prediction["predicted_quality"] > best_quality:
                best_quality = prediction["predicted_quality"]
                best_scenario = scenario

        assert best_scenario is not None
        assert best_quality > 0

    def test_learning_from_repeated_sessions(
        self,
        agent_system: AgentSystem
    ) -> None:
        """Test system learns from repeated user sessions."""
        context = {"paper_type": "platine", "print_size": "8x10"}

        # Simulate multiple sessions with feedback
        for session in range(5):
            agent_system.set_environment(
                temperature=20.0 + session * 0.5,
                humidity=50.0
            )

            # Process task
            result = agent_system.process_task(
                "calculate coating",
                context=context,
                print_size=(8, 10)
            )

            # Provide positive feedback
            agent_system.provide_feedback(
                task="coating",
                feedback_type=FeedbackType.POSITIVE,
                context=context,
                original_value=result["data"],
                confidence=0.8,
            )

        # Check learning stats
        summary = agent_system.get_session_summary()
        assert summary["learning_stats"]["total_records"] >= 5


# ============================================================================
# User Journey: Troubleshooting Session
# ============================================================================


class TestTroubleshootingJourney:
    """Journey: User troubleshoots print problems."""

    def test_troubleshooting_session(self, agent_system: AgentSystem) -> None:
        """Test complete troubleshooting session."""
        # User has a problem
        result = agent_system.process_task(
            "troubleshoot and diagnose issues",
            symptoms=["uneven coating", "low contrast"]
        )

        assert result["success"]
        assert result["category"] == "troubleshooting"
        assert "diagnoses" in result["data"]

        # User tries fix and it works
        agent_system.provide_feedback(
            task="troubleshoot",
            feedback_type=FeedbackType.POSITIVE,
            context={"symptoms": ["uneven coating"]},
            original_value=result["data"],
        )

        # User has another problem
        result2 = agent_system.process_task(
            "troubleshoot my print problems",
            symptoms=["fogging"]
        )
        assert result2["success"]

    def test_iterative_troubleshooting(self, agent_system: AgentSystem) -> None:
        """Test iterative troubleshooting with multiple attempts."""
        symptoms = ["low dmax", "flat midtones"]

        # First attempt
        result1 = agent_system.process_task(
            "diagnose print problems",
            symptoms=symptoms
        )
        assert result1["success"]

        # Feedback: didn't work
        agent_system.provide_feedback(
            task="troubleshoot",
            feedback_type=FeedbackType.NEGATIVE,
            context={"symptoms": symptoms},
            original_value=result1["data"],
        )

        # User asks about quality with problematic conditions
        features = QualityFeatures(
            dmax=1.2,  # Low Dmax as reported
            contrast=1.0,  # Low contrast
            exposure_time=80.0,  # Short exposure
            coating_age_hours=20.0,  # Old coating
        )
        prediction = agent_system.predict_quality(features)

        # Should identify issues (low dmax, short exposure, or old coating)
        # If no specific risks, should at least have recommendations
        assert len(prediction["risks"]) > 0 or len(prediction["recommendations"]) > 0


# ============================================================================
# User Journey: Quality Improvement Over Time
# ============================================================================


class TestQualityImprovementJourney:
    """Journey: User improves print quality over multiple sessions."""

    def test_quality_improvement_tracking(
        self,
        agent_system: AgentSystem
    ) -> None:
        """Test tracking quality improvement over time."""
        agent_system.set_environment(temperature=21.0, humidity=50.0)

        # Record multiple print sessions with improving quality
        quality_history = []

        for session in range(5):
            # Each session gets slightly better
            quality = 60 + session * 8

            features = QualityFeatures(
                dmin=0.15 - session * 0.01,  # Improving dmin
                dmax=1.7 + session * 0.06,   # Improving dmax
                contrast=1.5 + session * 0.06,  # Improving contrast
                exposure_time=180.0,
                coating_age_hours=1.0,
            )

            # Record historical data
            agent_system.predictor.add_historical_data(
                features=features,
                actual_quality=float(quality),
                notes=f"Session {session + 1}",
            )

            quality_history.append(quality)

        # Verify improvement is tracked
        stats = agent_system.predictor.get_statistics()
        assert stats["total_prints"] == 5
        assert stats["quality"]["max"] > stats["quality"]["min"]

        # Latest prediction should reflect learning
        current_features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            exposure_time=180.0,
        )
        prediction = agent_system.predict_quality(current_features)
        # Confidence should be higher with more data
        assert prediction["confidence"] > 0.3


# ============================================================================
# User Journey: Multi-Session Learning
# ============================================================================


class TestMultiSessionLearning:
    """Journey: System learns across multiple sessions with persistence."""

    def test_persistent_learning(self, tmp_path: Path) -> None:
        """Test learning persists across sessions."""
        # Session 1: Initial use
        system1 = AgentSystem(storage_path=tmp_path)
        system1.set_environment(temperature=21.0, humidity=50.0)

        result1 = system1.process_task(
            "calculate coating",
            print_size=(8, 10)
        )

        system1.provide_feedback(
            task="coating",
            feedback_type=FeedbackType.POSITIVE,
            context={"size": "8x10"},
            original_value=result1["data"],
        )

        stats1 = system1.learner.get_feedback_stats()
        assert stats1["total_records"] == 1

        # Session 2: New instance (simulating restart)
        system2 = AgentSystem(storage_path=tmp_path)

        # Should have loaded previous feedback
        stats2 = system2.learner.get_feedback_stats()
        assert stats2["total_records"] >= 1

        # Add more feedback
        system2.provide_feedback(
            task="coating",
            feedback_type=FeedbackType.POSITIVE,
            context={"size": "8x10"},
            original_value={"test": True},
        )

        stats3 = system2.learner.get_feedback_stats()
        assert stats3["total_records"] >= 2


# ============================================================================
# User Journey: Complex Multi-Step Workflow
# ============================================================================


class TestComplexWorkflow:
    """Journey: User executes a complex multi-step printing workflow."""

    def test_complete_printing_workflow(
        self,
        agent_system: AgentSystem
    ) -> None:
        """Test a complete printing workflow from planning to execution."""
        # Step 1: Set up environment
        is_optimal, warnings = agent_system.set_environment(
            temperature=21.0,
            humidity=50.0,
            altitude_meters=300.0
        )
        assert is_optimal

        # Step 2: Plan coating
        coating_result = agent_system.process_task(
            "calculate coating for 16x20 large format print",
            print_size=(16, 20)
        )
        assert coating_result["success"]

        # Step 3: Get metal ratio recommendation
        ratio_result = agent_system.process_task(
            "recommend metal ratio for neutral tones",
            target_tone="neutral"
        )
        assert ratio_result["success"]

        # Step 4: Calculate exposure
        exposure_data = agent_system.adapt_exposure(240.0)  # Longer base for large print
        assert "adjusted_exposure" in exposure_data

        # Step 5: Pre-print quality prediction
        features = QualityFeatures(
            dmax=2.0,
            dmin=0.1,
            contrast=1.8,
            metal_ratio=1.0,  # Neutral
            exposure_time=exposure_data["adjusted_exposure"],
            coating_age_hours=0.5,
        )
        prediction = agent_system.predict_quality(features)
        assert prediction["acceptable"]

        # Step 6: Verify session captured all steps
        summary = agent_system.get_session_summary()
        assert summary["tasks_processed"] >= 2

    def test_workflow_with_corrections(
        self,
        agent_system: AgentSystem
    ) -> None:
        """Test workflow where user corrects system suggestions."""
        agent_system.set_environment(temperature=22.0, humidity=52.0)

        # System suggests exposure
        exposure_data = agent_system.adapt_exposure(180.0)
        original_exposure = exposure_data["adjusted_exposure"]

        # User finds actual exposure should be different
        actual_exposure = 200.0

        # Record correction
        agent_system.provide_feedback(
            task="exposure_prediction",
            feedback_type=FeedbackType.CORRECTION,
            context={
                "temperature": 22.0,
                "humidity": 52.0,
            },
            original_value=original_exposure,
            corrected_value=actual_exposure,
        )

        # System should consider this for future
        stats = agent_system.get_session_summary()
        feedback = stats["learning_stats"]
        assert "correction" in feedback.get("feedback_types", {})


# ============================================================================
# Edge Cases in User Journeys
# ============================================================================


class TestJourneyEdgeCases:
    """Test edge cases in user journeys."""

    def test_empty_workflow(self, agent_system: AgentSystem) -> None:
        """Test system handles empty workflow gracefully."""
        summary = agent_system.get_session_summary()
        assert summary["tasks_processed"] == 0
        assert not summary["has_environment"]

    def test_unknown_task_handling(self, agent_system: AgentSystem) -> None:
        """Test system handles unknown tasks gracefully."""
        result = agent_system.process_task(
            "do something completely unrelated to printing"
        )
        # Should either handle gracefully or indicate no skill found
        assert "success" in result
        if not result["success"]:
            assert "suggestions" in result

    def test_rapid_task_sequence(self, agent_system: AgentSystem) -> None:
        """Test system handles rapid sequence of tasks."""
        agent_system.set_environment(temperature=21.0, humidity=50.0)

        tasks = [
            ("calculate coating", {"print_size": (8, 10)}),
            ("recommend metal ratio", {"target_tone": "warm"}),
            ("troubleshoot issues", {"symptoms": ["low contrast"]}),
            ("check quality", {}),
        ]

        results = []
        for task, kwargs in tasks:
            result = agent_system.process_task(task, **kwargs)
            results.append(result)

        # All should complete
        assert len(results) == 4
        # Most should succeed
        successful = sum(1 for r in results if r["success"])
        assert successful >= 3

    def test_contradictory_feedback(self, agent_system: AgentSystem) -> None:
        """Test system handles contradictory feedback."""
        context = {"paper_type": "platine"}

        # Positive feedback
        agent_system.provide_feedback(
            task="coating",
            feedback_type=FeedbackType.POSITIVE,
            context=context,
            original_value={"amount": 2.0},
        )

        # Negative feedback for same context
        agent_system.provide_feedback(
            task="coating",
            feedback_type=FeedbackType.NEGATIVE,
            context=context,
            original_value={"amount": 2.0},
        )

        # System should still function
        stats = agent_system.get_session_summary()
        assert stats["learning_stats"]["total_records"] == 2
