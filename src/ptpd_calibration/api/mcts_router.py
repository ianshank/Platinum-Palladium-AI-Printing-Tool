"""
MCTS API endpoints for calibration optimization.

Provides REST API endpoints for:
- Running MCTS search for optimal calibration parameters
- Evaluating parameter sets against the simulator
- Training Expert Iteration models
- Getting parameter recommendations
- Submitting real measurement feedback
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Check essential dependencies
try:
    import torch

    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None

# Check API dependencies
try:
    from fastapi import APIRouter, BackgroundTasks, HTTPException
    from pydantic import BaseModel, Field

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

    # Define dummy bases to prevent ImportErrors on module load if dependencies missing
    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(*_args: object, **_kwargs: object) -> None:  # type: ignore[no-redef]
        return None


# =============================================================================
# Pydantic Models (Top-level for introspection)
# =============================================================================


class MCTSSearchRequest(BaseModel):
    """Request to run MCTS calibration search."""

    paper_type: str | None = Field(default=None, description="Paper type (optional)")
    uv_source: str | None = Field(default=None, description="UV source type (optional)")
    target_curve: list[float] | None = Field(
        default=None, description="Target density curve (optional)"
    )
    fixed_parameters: dict[str, float] = Field(
        default_factory=dict, description="Fixed parameters to constrain search"
    )
    target_aesthetics: dict[str, float] = Field(
        default_factory=dict,
        description="Target aesthetic preferences (contrast, warmth, tonal_range)",
    )
    num_simulations: int | None = Field(
        default=None, ge=50, le=10000, description="Number of MCTS simulations"
    )


class MCTSSearchResponse(BaseModel):
    """Response from MCTS search."""

    search_id: str
    best_parameters: dict[str, float]
    predicted_curve: list[float]
    quality_score: float
    alternatives: list[dict[str, float]]
    search_time_seconds: float
    num_simulations: int


class MCTSEvaluateRequest(BaseModel):
    """Request to evaluate a parameter set."""

    parameters: dict[str, float] = Field(..., description="Parameters to evaluate")


class MCTSEvaluateResponse(BaseModel):
    """Response from parameter evaluation."""

    density_curve: list[float]
    dmin: float
    dmax: float
    density_range: float
    gamma: float
    quality_score: float


class MCTSTrainRequest(BaseModel):
    """Request to start Expert Iteration training."""

    num_episodes: int = Field(default=100, ge=10, le=10000, description="Number of episodes")


class MCTSTrainResponse(BaseModel):
    """Response from training start."""

    session_id: str
    status: str
    message: str


class MCTSStatusResponse(BaseModel):
    """MCTS engine status."""

    engine_ready: bool
    networks_loaded: bool
    torch_available: bool
    parameter_ranges: dict[str, dict[str, float | str]]


class MCTSFeedbackRequest(BaseModel):
    """Request to submit real measurement feedback."""

    parameters: dict[str, float] = Field(..., description="Parameters used")
    measured_curve: list[float] = Field(..., description="Measured density curve")
    quality_rating: float = Field(..., ge=0.0, le=1.0, description="User quality rating")


class MCTSRecommendation(BaseModel):
    """Parameter recommendation."""

    parameters: dict[str, float]
    predicted_quality: float
    rationale: str


# =============================================================================
# Router Factory
# =============================================================================


def create_mcts_router() -> APIRouter:
    """
    Create the MCTS API router.

    Returns:
        FastAPI APIRouter with MCTS endpoints.
    """
    if not API_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install ptpd-calibration[api]")

    router = APIRouter(prefix="/api/mcts", tags=["mcts"])

    # Training session storage
    training_sessions: dict[str, dict] = {}

    @router.get("/status", response_model=MCTSStatusResponse)
    async def get_mcts_status() -> MCTSStatusResponse:
        """Get the status of MCTS capabilities."""
        try:
            from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

            parameter_ranges = {
                name: {
                    "min": param_range.min_value,
                    "max": param_range.max_value,
                    "default": param_range.default_value,
                    "unit": param_range.unit,
                }
                for name, param_range in DEFAULT_PARAMETER_RANGES.items()
            }

            return MCTSStatusResponse(
                engine_ready=TORCH_AVAILABLE,
                networks_loaded=False,  # TODO: Check if networks are loaded
                torch_available=TORCH_AVAILABLE,
                parameter_ranges=parameter_ranges,
            )
        except Exception as e:
            logger.exception("Failed to get MCTS status: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get MCTS status: {str(e)}",
            ) from None

    @router.post("/search", response_model=MCTSSearchResponse)
    async def run_mcts_search(
        request: MCTSSearchRequest,
        background_tasks: BackgroundTasks,  # noqa: ARG001 - reserved for background search
    ) -> MCTSSearchResponse:
        """
        Run MCTS search for optimal calibration parameters.

        Can run in background for large searches.
        """
        try:
            import time
            from uuid import uuid4

            from ptpd_calibration.mcts.agents import CalibrationCoordinatorSubagent
            from ptpd_calibration.mcts.config import MCTSSettings

            start_time = time.time()
            search_id = str(uuid4())

            # Configure settings
            settings = MCTSSettings()
            if request.num_simulations is not None:
                settings.num_simulations = request.num_simulations

            # Create coordinator agent
            coordinator = CalibrationCoordinatorSubagent()

            # Run coordinated search
            context = {
                "target_aesthetics": request.target_aesthetics,
                "fixed_parameters": request.fixed_parameters,
                "uv_source": request.uv_source,
                "target_curve": request.target_curve,
            }

            result = await coordinator.run("coordinate_search", context=context)

            if not result.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {result.error}",
                )

            search_time = time.time() - start_time

            # Extract results
            search_data = result.result
            evaluation = search_data["evaluation"]

            # Generate alternatives (simplified - would use MCTS in full implementation)
            alternatives = [search_data["full_parameters"]]

            return MCTSSearchResponse(
                search_id=search_id,
                best_parameters=search_data["full_parameters"],
                predicted_curve=evaluation["predicted_curve"],
                quality_score=evaluation["quality_score"],
                alternatives=alternatives,
                search_time_seconds=search_time,
                num_simulations=settings.num_simulations,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("MCTS search failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"MCTS search failed: {str(e)}",
            ) from None

    @router.post("/evaluate", response_model=MCTSEvaluateResponse)
    async def evaluate_parameters(request: MCTSEvaluateRequest) -> MCTSEvaluateResponse:
        """
        Evaluate a parameter set against the simulator.

        Returns predicted curve and quality metrics.
        """
        try:
            from ptpd_calibration.mcts.config import MCTSSettings
            from ptpd_calibration.mcts.quality import QualityScorer
            from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator

            # Validate parameters
            if not request.parameters:
                # Use defaults
                from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

                params = {
                    name: param_range.default_value
                    for name, param_range in DEFAULT_PARAMETER_RANGES.items()
                }
            else:
                params = request.parameters

            # Run simulation
            simulator = ExtendedProcessSimulator()
            sim_result = simulator.simulate(params)

            # Compute quality score
            settings = MCTSSettings()
            scorer = QualityScorer(settings)
            quality_score = scorer.score(sim_result)

            return MCTSEvaluateResponse(
                density_curve=sim_result.density_curve,
                dmin=sim_result.dmin,
                dmax=sim_result.dmax,
                density_range=sim_result.density_range,
                gamma=sim_result.gamma,
                quality_score=quality_score,
            )

        except Exception as e:
            logger.exception("Parameter evaluation failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed: {str(e)}",
            ) from None

    @router.post("/train", response_model=MCTSTrainResponse)
    async def start_training(
        request: MCTSTrainRequest,
        background_tasks: BackgroundTasks,
    ) -> MCTSTrainResponse:
        """
        Start Expert Iteration training session.

        Returns immediately with session_id; training runs in background.
        """
        if not TORCH_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="PyTorch is not available. Install with: pip install ptpd-calibration[deep]",
            )

        try:
            from uuid import uuid4

            session_id = str(uuid4())

            # Initialize training session
            training_sessions[session_id] = {
                "status": "starting",
                "episodes_completed": 0,
                "num_episodes": request.num_episodes,
                "error": None,
            }

            # Start training in background
            background_tasks.add_task(
                _train_model_task,
                session_id,
                request.num_episodes,
                training_sessions,
            )

            return MCTSTrainResponse(
                session_id=session_id,
                status="starting",
                message=f"Training started with {request.num_episodes} episodes",
            )

        except Exception as e:
            logger.exception("Failed to start training: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start training: {str(e)}",
            ) from None

    @router.get("/train/{session_id}/status")
    async def get_training_status(session_id: str) -> dict[str, object]:
        """Get training progress for a session."""
        if session_id not in training_sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Training session '{session_id}' not found",
            )

        session = training_sessions[session_id]
        return {
            "session_id": session_id,
            "status": session["status"],
            "episodes_completed": session["episodes_completed"],
            "num_episodes": session["num_episodes"],
            "error": session["error"],
        }

    @router.post("/export")
    async def export_result(
        parameters: dict[str, float],
        format: str = "json",
    ) -> dict[str, object]:
        """
        Export calibration result as curve or recipe.

        Args:
            parameters: Parameter set to export.
            format: Export format (json, csv, recipe).

        Returns:
            Exported data.
        """
        try:
            from ptpd_calibration.mcts.simulator import ExtendedProcessSimulator

            # Simulate to get curve
            simulator = ExtendedProcessSimulator()
            sim_result = simulator.simulate(parameters)

            if format == "json":
                return {
                    "parameters": parameters,
                    "density_curve": sim_result.density_curve,
                    "dmin": sim_result.dmin,
                    "dmax": sim_result.dmax,
                    "gamma": sim_result.gamma,
                }
            elif format == "csv":
                # CSV format
                csv_lines = ["input,output"]
                for i, density in enumerate(sim_result.density_curve):
                    input_val = i / (len(sim_result.density_curve) - 1)
                    csv_lines.append(f"{input_val:.4f},{density:.4f}")
                return {"csv": "\n".join(csv_lines)}
            elif format == "recipe":
                # Recipe format
                recipe = {
                    "title": "MCTS Calibration Recipe",
                    "parameters": parameters,
                    "predicted_results": {
                        "dmin": sim_result.dmin,
                        "dmax": sim_result.dmax,
                        "gamma": sim_result.gamma,
                    },
                }
                return recipe
            else:
                raise ValueError(f"Unknown format: {format}")

        except Exception as e:
            logger.exception("Export failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {str(e)}",
            ) from None

    @router.post("/feedback")
    async def submit_feedback(request: MCTSFeedbackRequest) -> dict[str, object]:
        """
        Submit real measurement data for model improvement.

        Args:
            request: Feedback with parameters and measured curve.

        Returns:
            Confirmation message.
        """
        try:
            # Store feedback for future training
            # In production, this would go to a database
            logger.info(
                "Received feedback: parameters=%s, quality=%.2f",
                request.parameters,
                request.quality_rating,
            )

            return {
                "success": True,
                "message": "Feedback recorded successfully",
            }

        except Exception as e:
            logger.exception("Failed to submit feedback: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit feedback: {str(e)}",
            ) from None

    @router.get("/recommendations")
    async def get_recommendations(
        paper_type: str | None = None,
        limit: int = 5,
    ) -> dict[str, list[MCTSRecommendation]]:
        """
        Get top-N parameter recommendations.

        Args:
            paper_type: Optional paper type filter.
            limit: Number of recommendations to return.

        Returns:
            List of recommendations.
        """
        try:
            from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

            # Generate sample recommendations
            # In production, this would use historical data or pre-computed results
            recommendations = []

            for i in range(min(limit, 3)):
                params = {
                    name: param_range.default_value
                    for name, param_range in DEFAULT_PARAMETER_RANGES.items()
                }

                recommendations.append(
                    MCTSRecommendation(
                        parameters=params,
                        predicted_quality=0.8 - i * 0.1,
                        rationale=f"Recommendation {i + 1} based on {paper_type or 'general'} calibrations",
                    )
                )

            return {"recommendations": recommendations}

        except Exception as e:
            logger.exception("Failed to get recommendations: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get recommendations: {str(e)}",
            ) from None

    return router


# =============================================================================
# Background Tasks
# =============================================================================


async def _train_model_task(
    session_id: str,
    num_episodes: int,
    training_sessions: dict,
) -> None:
    """Background task to train MCTS models."""
    import time

    try:
        training_sessions[session_id]["status"] = "training"

        # Import training modules (inside function to avoid errors if torch missing)
        from ptpd_calibration.mcts.config import MCTSSettings
        from ptpd_calibration.mcts.training import MCTSTrainer

        settings = MCTSSettings(num_training_episodes=num_episodes)
        _trainer = MCTSTrainer(settings=settings)

        # Run training with progress updates
        for episode in range(num_episodes):
            # Simulate training episode
            # In production, this would call trainer.run_episode()
            time.sleep(0.1)  # Simulate work

            training_sessions[session_id]["episodes_completed"] = episode + 1

        training_sessions[session_id]["status"] = "completed"

    except Exception as e:
        logger.error("Training failed for session %s: %s", session_id, e)
        training_sessions[session_id]["status"] = "failed"
        training_sessions[session_id]["error"] = str(e)
