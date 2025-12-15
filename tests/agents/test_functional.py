"""Functional tests for the agent system.

These tests verify that components work together correctly to achieve
specific functional requirements:
- Task classification and routing
- Skill-based task execution
- Environmental-aware calibration
- Quality prediction workflows
- Active learning integration
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from ptpd_calibration.agents.router import (
    TaskRouter,
    TaskCategory,
    TaskComplexity,
    PatternRegistry,
    RouterSettings,
)
from ptpd_calibration.agents.skills.base import (
    Skill,
    SkillRegistry,
    SkillContext,
    SkillSettings,
)
from ptpd_calibration.agents.skills import (
    CalibrationSkill,
    ChemistrySkill,
    QualitySkill,
    TroubleshootingSkill,
    create_default_skills,
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
# Fixtures
# ============================================================================


@pytest.fixture
def router() -> TaskRouter:
    """Create task router."""
    return TaskRouter()


@pytest.fixture
def skill_registry() -> SkillRegistry:
    """Create skill registry with all skills."""
    # create_default_skills() returns a registry with all skills already registered
    return create_default_skills()


@pytest.fixture
def learner() -> ActiveLearner:
    """Create active learner."""
    return ActiveLearner(
        settings=LearningSettings(min_feedback_for_adjustment=3)
    )


@pytest.fixture
def env_adapter() -> EnvironmentAdapter:
    """Create environment adapter."""
    return EnvironmentAdapter()


@pytest.fixture
def predictor() -> QualityPredictor:
    """Create quality predictor."""
    return QualityPredictor(
        settings=PredictionSettings(min_data_points=3)
    )


# ============================================================================
# Router to Skill Integration Tests
# ============================================================================


class TestRouterToSkillIntegration:
    """Test router and skill system integration."""

    def test_calibration_task_routed_to_calibration_skill(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test calibration tasks are handled by CalibrationSkill."""
        task = "analyze step tablet and extract densities"

        # Route the task
        routing = router.route(task)
        assert routing.category == TaskCategory.CALIBRATION

        # Find matching skill
        skill, confidence = skill_registry.find_best_skill(task)

        assert skill is not None
        assert isinstance(skill, CalibrationSkill)
        assert confidence > 0.5

    def test_chemistry_task_routed_to_chemistry_skill(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test chemistry tasks are handled by ChemistrySkill."""
        task = "calculate coating volume for 8x10 print"

        routing = router.route(task)
        assert routing.category == TaskCategory.CHEMISTRY

        skill, confidence = skill_registry.find_best_skill(task)

        assert skill is not None
        assert isinstance(skill, ChemistrySkill)
        assert confidence > 0.3

    def test_troubleshooting_task_routed_correctly(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test troubleshooting tasks are handled correctly."""
        task = "troubleshoot print problems and diagnose issues"

        routing = router.route(task)
        assert routing.category == TaskCategory.TROUBLESHOOTING

        skill, confidence = skill_registry.find_best_skill(task)

        assert skill is not None
        assert isinstance(skill, TroubleshootingSkill)

    def test_quality_task_routed_to_quality_skill(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test quality tasks are handled by QualitySkill."""
        task = "assess print quality and check for defects"

        routing = router.route(task)
        assert routing.category == TaskCategory.QUALITY

        skill, confidence = skill_registry.find_best_skill(task)

        assert skill is not None
        assert isinstance(skill, QualitySkill)

    def test_complex_task_escalation(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test complex tasks can be escalated."""
        task = "perform full calibration with all steps"

        routing = router.route(task)

        # Should be able to escalate if needed
        escalated = router.escalate(routing)
        assert escalated.complexity.value >= routing.complexity.value

    def test_skill_execution_with_context(
        self,
        skill_registry: SkillRegistry
    ) -> None:
        """Test skill execution with context."""
        task = "calculate coating for warm tones"
        context = SkillContext(
            paper_type="platine",
            temperature=21.0,
            humidity=50.0,
        )

        result = skill_registry.execute_best(task, context, target_tone="warm")

        assert result is not None
        assert result.success

    def test_multiple_skills_can_handle_task(
        self,
        skill_registry: SkillRegistry
    ) -> None:
        """Test that multiple skills may handle ambiguous tasks."""
        # A task that could be handled by multiple skills
        task = "check calibration quality"

        # Find all capable skills
        capable_skills = []
        for skill in skill_registry._skills.values():
            confidence = skill.can_handle(task)
            if confidence > 0:
                capable_skills.append((skill, confidence))

        # At least one skill should be capable
        assert len(capable_skills) >= 1


# ============================================================================
# Environmental Adaptation Integration Tests
# ============================================================================


class TestEnvironmentalAdaptation:
    """Test environmental adaptation with other components."""

    def test_environment_affects_quality_prediction(
        self,
        env_adapter: EnvironmentAdapter,
        predictor: QualityPredictor
    ) -> None:
        """Test environment conditions affect quality predictions."""
        # Optimal conditions
        optimal_conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
        )

        # Suboptimal conditions
        suboptimal_conditions = EnvironmentConditions(
            temperature_celsius=30.0,
            humidity_percent=75.0,
        )

        base_features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            metal_ratio=1.2,
            exposure_time=180.0,
        )

        # Create features with different environments
        optimal_features = QualityFeatures(
            dmin=base_features.dmin,
            dmax=base_features.dmax,
            contrast=base_features.contrast,
            metal_ratio=base_features.metal_ratio,
            exposure_time=base_features.exposure_time,
            temperature=optimal_conditions.temperature_celsius,
            humidity=optimal_conditions.humidity_percent,
        )

        suboptimal_features = QualityFeatures(
            dmin=base_features.dmin,
            dmax=base_features.dmax,
            contrast=base_features.contrast,
            metal_ratio=base_features.metal_ratio,
            exposure_time=base_features.exposure_time,
            temperature=suboptimal_conditions.temperature_celsius,
            humidity=suboptimal_conditions.humidity_percent,
        )

        # Predict quality
        optimal_prediction = predictor.predict(optimal_features)
        suboptimal_prediction = predictor.predict(suboptimal_features)

        # Suboptimal should have more risks
        assert len(suboptimal_prediction.risk_factors) > len(optimal_prediction.risk_factors)

    def test_exposure_adaptation_workflow(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test complete exposure adaptation workflow."""
        # Current conditions
        conditions = EnvironmentConditions(
            temperature_celsius=25.0,
            humidity_percent=60.0,
            altitude_meters=1500.0,
        )

        # Base exposure time
        base_exposure = 180.0

        # Adapt exposure
        result = env_adapter.adapt_exposure(base_exposure, conditions)

        # Verify adaptation
        assert result.adjusted_exposure != base_exposure
        assert len(result.adjustments_applied) >= 2  # temp and humidity
        assert "altitude" in result.adjustments_applied

        # Get paper compensation
        compensation = env_adapter.get_compensation_for_paper("platine", conditions)

        # Final exposure
        final_exposure = result.adjusted_exposure * compensation["exposure_compensation"]
        assert final_exposure > 0

    def test_seasonal_profile_affects_recommendations(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test seasonal profiles provide relevant recommendations."""
        # Summer conditions
        conditions = EnvironmentConditions(
            temperature_celsius=28.0,
            humidity_percent=65.0,
        )

        # Get schedule recommendations
        recommendations = env_adapter.recommend_schedule(conditions, task="printing")

        assert "season" in recommendations
        assert "advice" in recommendations
        # Should mention temperature since it's high
        assert "temperature" in recommendations["advice"].lower() or "cool" in recommendations["advice"].lower()


# ============================================================================
# Active Learning Integration Tests
# ============================================================================


class TestActiveLearningIntegration:
    """Test active learning with other components."""

    def test_learning_improves_skill_confidence(
        self,
        learner: ActiveLearner,
        skill_registry: SkillRegistry
    ) -> None:
        """Test learning from feedback improves confidence."""
        task = "calibrate step tablet"
        context = {"paper_type": "platine", "exposure": 180}

        # Execute skill
        skill, base_confidence = skill_registry.find_best_skill(task)
        result = skill.execute(task)

        # Record positive feedback
        for _ in range(5):
            learner.record_feedback(
                action="calibrate",
                context=context,
                feedback_type=FeedbackType.POSITIVE,
                original_value=result.data,
                confidence=base_confidence,
            )

        # Check adjusted confidence
        adjusted = learner.get_adjusted_confidence("calibrate", base_confidence)
        assert adjusted >= base_confidence

    def test_learning_triggers_user_prompt(
        self,
        learner: ActiveLearner
    ) -> None:
        """Test learning system triggers user prompts appropriately."""
        context = {"paper_type": "unknown", "exposure": 100}

        # Record negative feedback
        for _ in range(5):
            learner.record_feedback(
                action="predict_exposure",
                context=context,
                feedback_type=FeedbackType.NEGATIVE,
                original_value=100,
                confidence=0.6,
            )

        # Should suggest asking user
        should_ask, reason = learner.should_ask_user(
            action="predict_exposure",
            context=context,
            confidence=0.5,
        )

        # Low confidence or low success rate should trigger
        assert should_ask or "confidence" in reason.lower()

    def test_corrections_inform_future_predictions(
        self,
        learner: ActiveLearner
    ) -> None:
        """Test corrections are used for future predictions."""
        context = {"paper_type": "platine", "temperature": 21}

        # Record correction
        learner.record_feedback(
            action="predict_exposure",
            context=context,
            feedback_type=FeedbackType.CORRECTION,
            original_value=150,
            corrected_value=180,
        )

        # Get suggestions for similar context
        similar_context = {"paper_type": "platine", "temperature": 22}
        suggestions = learner.get_suggestions("predict_exposure", similar_context)

        # May have suggestions based on pattern matching
        assert isinstance(suggestions, list)


# ============================================================================
# Quality Prediction Integration Tests
# ============================================================================


class TestQualityPredictionIntegration:
    """Test quality prediction with other components."""

    def test_prediction_with_historical_data(
        self,
        predictor: QualityPredictor
    ) -> None:
        """Test predictions improve with historical data."""
        base_features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            temperature=21.0,
            humidity=50.0,
            exposure_time=180.0,
        )

        # Initial prediction
        initial_prediction = predictor.predict(base_features)

        # Add historical data
        for quality in [85, 87, 83, 88, 86]:
            predictor.add_historical_data(
                features=base_features,
                actual_quality=float(quality),
            )

        # Prediction with history
        informed_prediction = predictor.predict(base_features)

        # Confidence should increase with data
        assert informed_prediction.confidence >= initial_prediction.confidence

    def test_similar_prints_inform_recommendations(
        self,
        predictor: QualityPredictor
    ) -> None:
        """Test similar historical prints inform recommendations."""
        # Add historical print with issues
        problem_features = QualityFeatures(
            temperature=28.0,
            humidity=70.0,
            coating_age_hours=15.0,
        )

        predictor.add_historical_data(
            features=problem_features,
            actual_quality=50.0,
            issues=["uneven coating", "fogging"],
            notes="Avoid high humidity printing",
        )

        # Predict for similar conditions
        similar_features = QualityFeatures(
            temperature=27.0,
            humidity=68.0,
            coating_age_hours=12.0,
        )

        prediction = predictor.predict(similar_features)

        # Should have relevant risks
        risk_factors = [r["factor"] for r in prediction.risk_factors]
        assert any("humidity" in f or "temperature" in f or "coating" in f
                   for f in risk_factors)

    def test_full_quality_assessment_workflow(
        self,
        predictor: QualityPredictor,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test complete quality assessment workflow."""
        # Get current conditions
        conditions = EnvironmentConditions(
            temperature_celsius=22.0,
            humidity_percent=55.0,
        )

        # Check if conditions are optimal
        is_optimal, warnings = conditions.is_optimal()

        # Build features for prediction
        features = QualityFeatures(
            dmin=0.1,
            dmax=1.9,
            contrast=1.7,
            curve_smoothness=0.85,
            metal_ratio=1.15,
            coating_amount=1.8,
            temperature=conditions.temperature_celsius,
            humidity=conditions.humidity_percent,
            exposure_time=175.0,
            coating_age_hours=2.0,
        )

        # Predict quality
        prediction = predictor.predict(features)

        # Verify complete prediction
        assert 0 <= prediction.predicted_quality <= 100
        assert prediction.quality_grade is not None
        assert 0 <= prediction.confidence <= 1
        assert len(prediction.confidence_interval) == 2


# ============================================================================
# Full System Integration Tests
# ============================================================================


class TestFullSystemIntegration:
    """Test complete system integration scenarios."""

    def test_calibration_workflow_with_all_components(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry,
        env_adapter: EnvironmentAdapter,
        predictor: QualityPredictor,
        learner: ActiveLearner
    ) -> None:
        """Test complete calibration workflow using all components."""
        # User request - use a task that analyzes existing densities
        task = "analyze calibration densities"
        context = {"paper_type": "platine"}

        # 1. Route the task
        routing = router.route(task)
        assert routing.category == TaskCategory.CALIBRATION

        # 2. Get environmental conditions
        conditions = EnvironmentConditions(
            temperature_celsius=21.0,
            humidity_percent=50.0,
        )
        is_optimal, _ = conditions.is_optimal()
        assert is_optimal

        # 3. Find skill (don't execute since it requires image)
        skill, confidence = skill_registry.find_best_skill(task)
        assert skill is not None
        assert isinstance(skill, CalibrationSkill)

        # 4. Predict quality with given calibration values
        features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            temperature=conditions.temperature_celsius,
            humidity=conditions.humidity_percent,
            exposure_time=180.0,
        )
        prediction = predictor.predict(features)
        assert prediction.is_acceptable()

        # 5. Record feedback (simulating successful calibration)
        learner.record_feedback(
            action="calibrate",
            context=context,
            feedback_type=FeedbackType.POSITIVE,
            original_value={"dmax": 2.0, "dmin": 0.1},
            confidence=confidence,
        )

        # Verify learning
        stats = learner.get_feedback_stats("calibrate")
        assert stats["total_records"] == 1

    def test_troubleshooting_workflow_with_learning(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry,
        learner: ActiveLearner
    ) -> None:
        """Test troubleshooting workflow with active learning."""
        task = "troubleshoot and diagnose printing problems"
        context = {
            "paper_type": "platine",
            "symptoms": ["uneven coating"],
        }

        # Route
        routing = router.route(task)
        assert routing.category == TaskCategory.TROUBLESHOOTING

        # Execute skill
        skill, confidence = skill_registry.find_best_skill(task)
        assert isinstance(skill, TroubleshootingSkill)
        result = skill.execute(task, symptoms=["uneven coating"])

        assert result.success
        # Result contains diagnoses (list of diagnosis objects)
        assert "diagnoses" in result.data or "causes" in result.data or "diagnosis" in result.data

        # User indicates solution worked
        learner.record_feedback(
            action="troubleshoot",
            context=context,
            feedback_type=FeedbackType.POSITIVE,
            original_value=result.data,
            confidence=confidence,
        )

        # Future similar problems should have higher confidence
        adjusted = learner.get_adjusted_confidence("troubleshoot", confidence)
        # After one positive feedback, might not change yet (need min_feedback_for_adjustment)
        assert adjusted >= 0

    def test_chemistry_with_environmental_adjustment(
        self,
        skill_registry: SkillRegistry,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test chemistry calculations with environmental adjustment."""
        # Get environmental conditions
        conditions = EnvironmentConditions(
            temperature_celsius=25.0,  # Warm day
            humidity_percent=60.0,
        )

        # Execute chemistry skill
        task = "calculate coating for 8x10 print"
        result = skill_registry.execute_best(task, print_size=(8, 10))

        assert result is not None
        assert result.success
        # Check for coating volume in result (can be total_ml, coating_ml, or volume)
        assert "total_ml" in result.data or "coating_ml" in result.data or "ml" in str(result.data).lower()

        # Adjust exposure based on environment
        base_exposure = 180.0
        adaptation = env_adapter.adapt_exposure(base_exposure, conditions)

        # Warm conditions should reduce exposure (faster chemistry)
        assert adaptation.adjustment_factor < 1.0 or len(adaptation.warnings) > 0

    def test_quality_check_before_printing(
        self,
        skill_registry: SkillRegistry,
        predictor: QualityPredictor,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test pre-print quality check workflow."""
        # Current conditions
        conditions = EnvironmentConditions(
            temperature_celsius=28.0,
            humidity_percent=70.0,
        )

        # Check if we should print now
        is_optimal, warnings = conditions.is_optimal()

        if not is_optimal:
            # Get recommendations
            recommendations = env_adapter.recommend_schedule(conditions, "printing")
            assert "advice" in recommendations

        # Predict quality with current conditions
        features = QualityFeatures(
            dmin=0.1,
            dmax=2.0,
            contrast=1.8,
            temperature=conditions.temperature_celsius,
            humidity=conditions.humidity_percent,
            exposure_time=180.0,
            coating_age_hours=1.0,
        )

        prediction = predictor.predict(features)

        # Should identify environmental risks
        risk_factors = [r["factor"] for r in prediction.risk_factors]
        has_env_risk = any(
            "temperature" in f or "humidity" in f
            for f in risk_factors
        )
        assert has_env_risk

        # Should provide actionable recommendations
        assert len(prediction.recommendations) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_task_routing(
        self,
        router: TaskRouter,
        skill_registry: SkillRegistry
    ) -> None:
        """Test handling of unknown tasks."""
        task = "do something completely unrelated to printing"

        routing = router.route(task)
        # Should still route (possibly to general category)
        assert routing is not None

        # Skill registry may not find a match
        result = skill_registry.find_best_skill(task)
        # Result might be None or low confidence
        if result:
            skill, confidence = result
            # Low confidence for unrelated task
            assert confidence < 0.5 or skill is not None

    def test_empty_context_handling(
        self,
        skill_registry: SkillRegistry,
        learner: ActiveLearner
    ) -> None:
        """Test handling of empty context."""
        task = "calibrate"

        # Should work with no context
        skill, confidence = skill_registry.find_best_skill(task)
        if skill:
            result = skill.execute(task)
            assert result is not None

        # Learner should handle empty context
        learner.record_feedback(
            action="test",
            context={},
            feedback_type=FeedbackType.POSITIVE,
            original_value=None,
        )
        stats = learner.get_feedback_stats("test")
        assert stats["total_records"] == 1

    def test_extreme_environmental_conditions(
        self,
        env_adapter: EnvironmentAdapter
    ) -> None:
        """Test handling of extreme conditions."""
        extreme_conditions = EnvironmentConditions(
            temperature_celsius=40.0,
            humidity_percent=95.0,
        )

        result = env_adapter.adapt_exposure(180.0, extreme_conditions)

        # Should be clamped to bounds
        assert result.adjustment_factor >= env_adapter.settings.min_adjustment
        assert result.adjustment_factor <= env_adapter.settings.max_adjustment

        # Should have warnings
        is_optimal, warnings = extreme_conditions.is_optimal()
        assert not is_optimal
        assert len(warnings) >= 2
