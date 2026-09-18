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
from typing import TYPE_CHECKING

from ptpd_calibration.config import get_settings
from ptpd_calibration.core.logging import sanitize_log_text

if TYPE_CHECKING:
    from ptpd_calibration.mcts.feedback import FeedbackStore

logger = logging.getLogger(__name__)

# Request bounds (SEC-03), resolved once from APISettings so they can be used
# as literals in the module-level pydantic models below.
_API_LIMITS = get_settings().api
MAX_LIST_LENGTH: int = _API_LIMITS.max_list_length
MAX_STRING_LENGTH: int = _API_LIMITS.max_string_length

# Values reported in MCTSSearchResponse.search_backend
SEARCH_BACKEND_ENGINE = "mcts_engine"
SEARCH_BACKEND_HEURISTIC = "coordinator_heuristic"

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
    from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
    from fastapi.concurrency import run_in_threadpool
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

    paper_type: str | None = Field(
        default=None, max_length=MAX_STRING_LENGTH, description="Paper type (optional)"
    )
    uv_source: str | None = Field(
        default=None, max_length=MAX_STRING_LENGTH, description="UV source type (optional)"
    )
    target_curve: list[float] | None = Field(
        default=None,
        max_length=MAX_LIST_LENGTH,
        description="Target density curve (optional); scored against when its length "
        "matches the simulated curve",
    )
    fixed_parameters: dict[str, float] = Field(
        default_factory=dict,
        max_length=MAX_LIST_LENGTH,
        description="Fixed parameters to constrain search",
    )
    target_aesthetics: dict[str, float] = Field(
        default_factory=dict,
        max_length=MAX_LIST_LENGTH,
        description="Target aesthetic preferences (contrast, warmth, tonal_range)",
    )
    num_simulations: int | None = Field(
        default=None,
        ge=50,
        le=10000,
        description="Number of MCTS simulations (clamped to MCTSSettings.max_simulations_per_request)",
    )


class MCTSSearchResponse(BaseModel):
    """Response from MCTS search."""

    search_id: str
    best_parameters: dict[str, float]
    predicted_curve: list[float]
    quality_score: float
    alternatives: list[dict[str, float]]
    search_time_seconds: float
    num_simulations: int = Field(description="Simulations actually performed")
    engine_used: bool = Field(
        default=False, description="True when MCTSEngine.search produced this result"
    )
    search_backend: str = Field(
        default=SEARCH_BACKEND_HEURISTIC,
        description=f"'{SEARCH_BACKEND_ENGINE}' or '{SEARCH_BACKEND_HEURISTIC}'",
    )
    target_curve_used: bool = Field(
        default=False,
        description="True when quality_score was computed against the supplied target_curve",
    )


class MCTSEvaluateRequest(BaseModel):
    """Request to evaluate a parameter set."""

    parameters: dict[str, float] = Field(
        ..., max_length=MAX_LIST_LENGTH, description="Parameters to evaluate"
    )


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

    parameters: dict[str, float] = Field(
        ..., max_length=MAX_LIST_LENGTH, description="Parameters used"
    )
    measured_curve: list[float] = Field(
        ...,
        min_length=1,
        max_length=MAX_LIST_LENGTH,
        description="Measured density curve",
    )
    quality_rating: float = Field(..., ge=0.0, le=1.0, description="User quality rating")
    notes: str | None = Field(
        default=None, max_length=MAX_STRING_LENGTH, description="Free-text notes about the print"
    )


class MCTSRecommendation(BaseModel):
    """Parameter recommendation."""

    parameters: dict[str, float]
    predicted_quality: float
    rationale: str


# =============================================================================
# Router Factory
# =============================================================================


def _engine_fixed_parameters(
    request: MCTSSearchRequest, search_data: dict[str, object]
) -> dict[str, float]:
    """Parameters the engine must not search over.

    The caller's ``fixed_parameters`` are always pinned. When the caller also
    expressed ``target_aesthetics``, the chemistry the coordinator derived from
    them (metal ratio, ferric oxalate, coating weight) is pinned as well so the
    engine optimises the remaining exposure/development dimensions for that look.
    """
    from ptpd_calibration.mcts.config import DEFAULT_PARAMETER_RANGES

    fixed: dict[str, float] = {}
    if request.target_aesthetics:
        suggestion = search_data.get("chemistry_suggestion") or {}
        if isinstance(suggestion, dict):
            fixed.update(
                {
                    name: float(value)
                    for name, value in suggestion.items()
                    if name in DEFAULT_PARAMETER_RANGES
                }
            )
    fixed.update(request.fixed_parameters)
    logger.debug("Engine fixed parameters: %s", sorted(fixed))
    return fixed


def create_mcts_router(feedback_store: FeedbackStore | None = None) -> APIRouter:
    """
    Create the MCTS API router.

    Args:
        feedback_store: Store for measured feedback. Defaults to a
            :class:`~ptpd_calibration.mcts.feedback.FeedbackStore` at the path
            configured in ``MCTSSettings``.

    Returns:
        FastAPI APIRouter with MCTS endpoints.
    """
    if not API_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install ptpd-calibration[api]")

    from ptpd_calibration.mcts.feedback import FeedbackRecord
    from ptpd_calibration.mcts.feedback import FeedbackStore as _FeedbackStore

    router = APIRouter(prefix="/api/mcts", tags=["mcts"])

    # Training session storage
    training_sessions: dict[str, dict] = {}

    # Measured feedback persistence (SCI-06). Explicit None check: an injected
    # store must never be replaced just because it happens to be empty.
    store = feedback_store if feedback_store is not None else _FeedbackStore()
    logger.debug("MCTS feedback store: %s", store.path)

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

        The coordinator subagent first resolves chemistry from the requested
        aesthetics, then the real ``MCTSEngine`` searches the remaining
        dimensions against ``target_curve`` in a worker thread so the event
        loop stays responsive. If the engine fails for any reason the
        coordinator's single heuristic evaluation is returned instead and the
        response says so via ``engine_used`` / ``search_backend`` /
        ``num_simulations`` (SCI-06).
        """
        try:
            import time
            from uuid import uuid4

            from ptpd_calibration.mcts.agents import CalibrationCoordinatorSubagent
            from ptpd_calibration.mcts.config import MCTSSettings
            from ptpd_calibration.mcts.engine import MCTSEngine

            start_time = time.time()
            search_id = str(uuid4())

            # Honour the requested budget within the configured per-request cap
            base_settings = MCTSSettings()
            requested = request.num_simulations or base_settings.num_simulations
            num_simulations = min(requested, base_settings.max_simulations_per_request)
            if num_simulations != requested:
                logger.debug(
                    "Clamped num_simulations %d -> %d (max_simulations_per_request)",
                    requested,
                    num_simulations,
                )
            settings = MCTSSettings(num_simulations=num_simulations)

            # Step 1: coordinator resolves chemistry from aesthetics and scores
            # one heuristic parameter set against the target curve.
            coordinator = CalibrationCoordinatorSubagent()
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

            search_data = result.result
            evaluation = search_data["evaluation"]
            heuristic_params = search_data["full_parameters"]

            # Coordinator-only outcome: exactly one simulate + score was run.
            payload: dict[str, object] = {
                "best_parameters": heuristic_params,
                "predicted_curve": evaluation["predicted_curve"],
                "quality_score": evaluation["quality_score"],
                "alternatives": [heuristic_params],
                "num_simulations": 1,
                "engine_used": False,
                "search_backend": SEARCH_BACKEND_HEURISTIC,
                "target_curve_used": bool(evaluation.get("target_curve_used", False)),
            }

            # Step 2: run the real tree search off the event loop.
            engine_fixed = _engine_fixed_parameters(request, search_data)
            try:
                engine = MCTSEngine(settings=settings)
                search_result = await run_in_threadpool(
                    engine.search,
                    target_curve=request.target_curve,
                    fixed_parameters=engine_fixed,
                    paper_type=request.paper_type,
                    uv_source=request.uv_source,
                )
            except Exception as exc:
                logger.exception(
                    "MCTSEngine.search failed for %s; returning coordinator heuristic: %s",
                    search_id,
                    exc,
                )
            else:
                payload = {
                    "best_parameters": search_result.best_parameters,
                    "predicted_curve": search_result.predicted_curve,
                    "quality_score": search_result.quality_score,
                    "alternatives": search_result.alternatives,
                    "num_simulations": search_result.num_simulations,
                    "engine_used": True,
                    "search_backend": SEARCH_BACKEND_ENGINE,
                    "target_curve_used": request.target_curve is not None
                    and len(request.target_curve) == len(search_result.predicted_curve),
                }

            search_time = time.time() - start_time
            logger.info(
                "MCTS search %s: backend=%s simulations=%s quality=%.4f time=%.2fs",
                search_id,
                payload["search_backend"],
                payload["num_simulations"],
                float(payload["quality_score"]),  # type: ignore[arg-type]
                search_time,
            )

            return MCTSSearchResponse(
                search_id=search_id,
                search_time_seconds=search_time,
                **payload,  # type: ignore[arg-type]
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

            # Clamp to the deployment's own ceiling, as /search does with
            # num_simulations: the request field's bound is the widest value the
            # schema allows, not the most a given deployment wants to run.
            from ptpd_calibration.mcts.config import MCTSSettings

            num_episodes = min(request.num_episodes, MCTSSettings().max_episodes_per_request)
            if num_episodes != request.num_episodes:
                logger.debug(
                    "Clamped requested episodes %d to %d", request.num_episodes, num_episodes
                )

            # Initialize training session
            training_sessions[session_id] = {
                "status": "starting",
                "episodes_completed": 0,
                "num_episodes": num_episodes,
                "error": None,
            }

            # Start training in background
            background_tasks.add_task(
                _train_model_task,
                session_id,
                num_episodes,
                training_sessions,
            )

            return MCTSTrainResponse(
                session_id=session_id,
                status="starting",
                message=f"Training started with {num_episodes} episodes",
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
        parameters: dict[str, float] = Body(..., max_length=MAX_LIST_LENGTH),
        format: str = Query("json", max_length=MAX_STRING_LENGTH),
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
                # Never interpolate the raw value into the message: it is a
                # query parameter, and a newline in it forged a second log
                # record at ERROR, which every default level emits.
                raise ValueError(f"Unknown format: {sanitize_log_text(format)}")

        except Exception as e:
            logger.exception("Export failed: %s", sanitize_log_text(e))
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {str(e)}",
            ) from None

    @router.post("/feedback")
    async def submit_feedback(request: MCTSFeedbackRequest) -> dict[str, object]:
        """
        Persist real measurement data for model improvement.

        Records are appended as JSON lines with ``provenance="measured"`` to the
        file configured by ``MCTSSettings.feedback_path`` (SCI-06).

        Args:
            request: Feedback with parameters and measured curve.

        Returns:
            Confirmation message with the stored record id.
        """
        try:
            record = store.append(
                FeedbackRecord(
                    parameters=request.parameters,
                    measured_curve=request.measured_curve,
                    quality_rating=request.quality_rating,
                    notes=request.notes,
                )
            )
            logger.info(
                "Stored measured feedback %s: %d curve points, quality=%.2f",
                record.id,
                len(record.measured_curve),
                record.quality_rating,
            )

            return {
                "success": True,
                "message": "Feedback recorded successfully",
                "record_id": record.id,
                "provenance": record.provenance,
                "timestamp": record.timestamp.isoformat(),
            }

        except Exception as e:
            logger.exception("Failed to submit feedback: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to submit feedback: {str(e)}",
            ) from None

    @router.get("/recommendations")
    async def get_recommendations(
        paper_type: str | None = Query(None, max_length=MAX_STRING_LENGTH),
        limit: int = Query(5, ge=0, le=MAX_LIST_LENGTH),
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


def _train_model_task(
    session_id: str,
    num_episodes: int,
    training_sessions: dict,
) -> None:
    """Background task to train MCTS models.

    Deliberately ``def`` and not ``async def``. Starlette awaits an async
    background task directly on the event loop (``BackgroundTask.__call__``
    branches on ``is_async``), so the per-episode pause below -- and the real
    training that will replace it -- ran *on* the loop: one request to this
    endpoint stopped the whole API answering anything, health checks included,
    for the length of the run. A synchronous task is handed to the threadpool
    instead, which is also the right shape for the CPU-bound trainer that the
    stub stands in for.
    """
    import time

    try:
        training_sessions[session_id]["status"] = "training"

        # Import training modules (inside function to avoid errors if torch missing)
        from ptpd_calibration.mcts.config import MCTSSettings
        from ptpd_calibration.mcts.training import MCTSTrainer

        settings = MCTSSettings(num_training_episodes=num_episodes)
        _trainer = MCTSTrainer(settings=settings)
        delay = settings.training_episode_delay_seconds
        logger.debug(
            "Training session %s starting: %d episodes, %.3fs pause per episode",
            session_id,
            num_episodes,
            delay,
        )

        # Run training with progress updates
        for episode in range(num_episodes):
            # Simulate training episode
            # In production, this would call trainer.run_episode()
            if delay:
                time.sleep(delay)

            training_sessions[session_id]["episodes_completed"] = episode + 1

        logger.debug("Training session %s completed %d episodes", session_id, num_episodes)
        training_sessions[session_id]["status"] = "completed"

    except Exception as e:
        logger.error("Training failed for session %s: %s", session_id, e)
        training_sessions[session_id]["status"] = "failed"
        training_sessions[session_id]["error"] = str(e)
