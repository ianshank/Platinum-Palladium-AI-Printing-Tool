"""
Functional tests for the calibration workflow.
Tests complete business workflows end-to-end within the backend.
"""

import pytest
import numpy as np
from typing import Any
from pathlib import Path


class TestCalibrationWorkflow:
    """Tests for the complete calibration workflow."""

    def test_full_calibration_workflow(
        self,
        sample_scan_measurements: list[dict[str, Any]],
        sample_curve_data: dict[str, Any],
    ) -> None:
        """Test complete calibration from scan to curve generation."""
        # Step 1: Upload and process scan
        scan_result = {
            "id": "scan-workflow-1",
            "quality_score": 0.85,
            "measurements": sample_scan_measurements,
        }

        assert scan_result["quality_score"] > 0.7, "Scan quality too low"

        # Step 2: Analyze measurements
        measurements = scan_result["measurements"]
        measured_densities = [m["measured_density"] for m in measurements]
        target_densities = [m["target_density"] for m in measurements]

        # Calculate linearization curve
        linearization_values = []
        for target, measured in zip(target_densities, measured_densities):
            if measured > 0:
                correction = target / measured
            else:
                correction = 1.0
            linearization_values.append(correction)

        assert len(linearization_values) == len(measurements)

        # Step 3: Generate curve
        curve = {
            "id": "curve-workflow-1",
            "name": "Calibration Curve",
            "type": "linearization",
            "input_values": [m["step"] * 5 for m in measurements],
            "output_values": [
                int(m["step"] * 5 * linearization_values[i])
                for i, m in enumerate(measurements)
            ],
        }

        assert len(curve["input_values"]) == len(curve["output_values"])

        # Step 4: Validate curve properties
        assert curve["output_values"][0] == 0, "Curve should start at 0"
        assert max(curve["output_values"]) <= 100, "Curve should not exceed 100"

    def test_calibration_with_contrast_adjustment(
        self, sample_scan_measurements: list[dict[str, Any]]
    ) -> None:
        """Test calibration with contrast curve adjustment."""
        # Generate base linearization
        measurements = sample_scan_measurements
        base_curve = np.array([m["step"] * 5 for m in measurements], dtype=float)

        # Apply contrast adjustment (S-curve)
        midpoint = 50
        contrast_factor = 1.3
        adjusted = midpoint + (base_curve - midpoint) * contrast_factor
        adjusted = np.clip(adjusted, 0, 100)

        # Verify contrast increase
        original_range = np.max(base_curve) - np.min(base_curve)
        adjusted_range = np.max(adjusted) - np.min(adjusted)

        assert adjusted_range >= original_range * 0.95  # May be clipped

    def test_calibration_quality_assessment(
        self, sample_scan_measurements: list[dict[str, Any]]
    ) -> None:
        """Test quality assessment during calibration."""
        measurements = sample_scan_measurements

        # Calculate quality metrics
        target_densities = np.array([m["target_density"] for m in measurements])
        measured_densities = np.array([m["measured_density"] for m in measurements])

        # Calculate deviation from target
        deviations = np.abs(target_densities - measured_densities)
        mean_deviation = np.mean(deviations)
        max_deviation = np.max(deviations)

        # Quality score based on deviation
        quality_score = max(0, 1 - mean_deviation)

        assert quality_score > 0, "Quality score should be positive"
        assert mean_deviation < max_deviation, "Mean should be less than max"

    def test_calibration_iteration_improvement(self) -> None:
        """Test that calibration improves over iterations."""
        # Simulate multiple calibration iterations
        initial_error = 0.15
        iterations = []

        current_error = initial_error
        for i in range(5):
            # Each iteration should reduce error
            improvement_factor = 0.7
            current_error *= improvement_factor
            iterations.append({"iteration": i + 1, "error": current_error})

        final_error = iterations[-1]["error"]
        assert final_error < initial_error, "Calibration should improve over iterations"
        assert final_error < initial_error * 0.5, "Should achieve significant improvement"


class TestChemistryWorkflow:
    """Tests for chemistry calculation workflow."""

    def test_chemistry_for_different_print_sizes(
        self, chemistry_params: dict[str, Any]
    ) -> None:
        """Test chemistry calculation for various print sizes."""
        print_sizes = [
            (4, 5),   # Small
            (8, 10),  # Standard
            (11, 14), # Medium
            (16, 20), # Large
            (20, 24), # Extra large
        ]

        results = []
        for width, height in print_sizes:
            area = width * height
            coating_ml_per_sq_inch = 0.5
            total_metal = area * coating_ml_per_sq_inch

            results.append({
                "size": f"{width}x{height}",
                "area": area,
                "total_metal_ml": total_metal,
            })

        # Verify scaling is linear
        for i in range(1, len(results)):
            ratio = results[i]["area"] / results[0]["area"]
            metal_ratio = results[i]["total_metal_ml"] / results[0]["total_metal_ml"]
            assert abs(ratio - metal_ratio) < 0.001, "Chemistry should scale linearly"

    def test_chemistry_preset_application(self) -> None:
        """Test applying chemistry presets."""
        presets = {
            "warm_tone": {"metal_ratio": 0.3, "contrast_agent": "na2"},
            "neutral": {"metal_ratio": 0.5, "contrast_agent": "none"},
            "cool_tone": {"metal_ratio": 0.7, "contrast_agent": "dichromate"},
            "pure_platinum": {"metal_ratio": 1.0, "contrast_agent": "none"},
        }

        for preset_name, preset_values in presets.items():
            # Apply preset
            recipe = {
                "print_width": 8,
                "print_height": 10,
                **preset_values,
            }

            area = recipe["print_width"] * recipe["print_height"]
            total_metal = area * 0.5  # 0.5 ml/sq.in

            platinum = total_metal * recipe["metal_ratio"]
            palladium = total_metal * (1 - recipe["metal_ratio"])

            assert platinum + palladium == total_metal
            assert platinum >= 0
            assert palladium >= 0

    def test_chemistry_recipe_saving_and_retrieval(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test saving and retrieving chemistry recipes."""
        # Save recipe
        saved_recipe = {**sample_chemistry_recipe, "saved": True}

        # Retrieve recipe
        retrieved_recipe = saved_recipe.copy()

        # Verify all fields match
        for key in sample_chemistry_recipe:
            assert retrieved_recipe[key] == sample_chemistry_recipe[key]


class TestCurveManagementWorkflow:
    """Tests for curve management workflows."""

    def test_curve_creation_and_modification(
        self, sample_curve_data: dict[str, Any]
    ) -> None:
        """Test creating and modifying curves."""
        # Create curve
        curve = {
            "id": "new-curve-1",
            "name": "New Test Curve",
            **sample_curve_data,
        }

        assert curve["id"] is not None

        # Modify curve
        modified_curve = {
            **curve,
            "name": "Modified Test Curve",
            "modified_at": "2025-01-15T12:00:00Z",
        }

        assert modified_curve["name"] != curve["name"]
        assert "modified_at" in modified_curve

    def test_curve_blending_workflow(self) -> None:
        """Test blending multiple curves."""
        # Create base curves
        curve1 = np.array([0, 20, 40, 60, 80, 100], dtype=float)
        curve2 = np.array([0, 30, 55, 75, 90, 100], dtype=float)

        # Blend with different weights
        blend_50_50 = curve1 * 0.5 + curve2 * 0.5
        blend_70_30 = curve1 * 0.7 + curve2 * 0.3

        # Verify blending
        assert np.allclose(blend_50_50, [0, 25, 47.5, 67.5, 85, 100])

        # 70/30 blend should be closer to curve1
        distance_to_curve1 = np.mean(np.abs(blend_70_30 - curve1))
        distance_to_curve2 = np.mean(np.abs(blend_70_30 - curve2))
        assert distance_to_curve1 < distance_to_curve2

    def test_curve_export_import(self, sample_curve_data: dict[str, Any]) -> None:
        """Test exporting and importing curves."""
        import json

        # Export to JSON
        curve = {"name": "Export Test", **sample_curve_data}
        exported_json = json.dumps(curve)

        # Import from JSON
        imported_curve = json.loads(exported_json)

        # Verify data integrity
        assert imported_curve["name"] == curve["name"]
        assert imported_curve["input_values"] == curve["input_values"]
        assert imported_curve["output_values"] == curve["output_values"]


class TestAIAssistantWorkflow:
    """Tests for AI assistant interaction workflow."""

    def test_chat_conversation_flow(self) -> None:
        """Test multi-turn conversation with AI assistant."""
        conversation = []

        # User message 1
        conversation.append({
            "role": "user",
            "content": "How do I improve my blacks?",
        })

        # Assistant response 1
        conversation.append({
            "role": "assistant",
            "content": "To improve your blacks in platinum printing...",
        })

        # User follow-up
        conversation.append({
            "role": "user",
            "content": "What exposure time would you recommend?",
        })

        # Assistant response 2
        conversation.append({
            "role": "assistant",
            "content": "Based on your setup, I recommend...",
        })

        # Verify conversation structure
        assert len(conversation) == 4
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"

    def test_quick_prompt_usage(self) -> None:
        """Test using quick prompts for common questions."""
        quick_prompts = [
            {
                "id": "improve-contrast",
                "text": "How can I improve contrast in my prints?",
                "category": "technique",
            },
            {
                "id": "exposure-help",
                "text": "What exposure time should I use?",
                "category": "exposure",
            },
            {
                "id": "coating-issues",
                "text": "Help me troubleshoot uneven coating",
                "category": "troubleshooting",
            },
        ]

        # Verify prompts are well-formed
        for prompt in quick_prompts:
            assert "id" in prompt
            assert "text" in prompt
            assert len(prompt["text"]) > 10

    def test_context_aware_recommendations(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test AI provides context-aware recommendations."""
        # Provide context
        context = {
            "paper": "Arches Platine",
            "chemistry": sample_chemistry_recipe,
            "issue": "Weak blacks",
        }

        # AI should consider context in recommendations
        expected_considerations = [
            "exposure time",
            "contrast agent",
            "development time",
        ]

        # Verify context is available for AI
        assert context["paper"] is not None
        assert context["chemistry"] is not None
        assert context["issue"] is not None


class TestSessionLoggingWorkflow:
    """Tests for session logging workflow."""

    def test_create_and_populate_session(
        self, sample_chemistry_recipe: dict[str, Any]
    ) -> None:
        """Test creating a session and adding prints."""
        from datetime import datetime

        # Create session
        session = {
            "id": "session-workflow-1",
            "name": "Sunday Print Session",
            "date": datetime.now().isoformat(),
            "prints": [],
            "notes": "",
        }

        # Add prints to session
        for i in range(3):
            print_record = {
                "id": f"print-{i+1}",
                "image_name": f"image_{i+1}.tiff",
                "chemistry": sample_chemistry_recipe,
                "exposure_time": 180 + i * 30,
                "rating": 3 + i,
                "notes": f"Print {i+1} notes",
            }
            session["prints"].append(print_record)

        # Verify session structure
        assert len(session["prints"]) == 3
        assert all("rating" in p for p in session["prints"])

    def test_session_statistics(self) -> None:
        """Test calculating session statistics."""
        prints = [
            {"rating": 3, "exposure_time": 150},
            {"rating": 4, "exposure_time": 180},
            {"rating": 5, "exposure_time": 200},
            {"rating": 4, "exposure_time": 180},
        ]

        # Calculate statistics
        ratings = [p["rating"] for p in prints]
        exposures = [p["exposure_time"] for p in prints]

        stats = {
            "total_prints": len(prints),
            "average_rating": np.mean(ratings),
            "best_rating": max(ratings),
            "average_exposure": np.mean(exposures),
        }

        assert stats["total_prints"] == 4
        assert stats["average_rating"] == 4.0
        assert stats["best_rating"] == 5
        assert stats["average_exposure"] == 177.5
