"""
FastAPI server for PTPD Calibration System.
"""

import tempfile
from pathlib import Path
from uuid import UUID

from ptpd_calibration.config import get_settings


def create_app():
    """Create the FastAPI application."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse  # noqa: F401
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI is required. Install with: pip install ptpd-calibration[api]")

    from ptpd_calibration.config import TabletType
    from ptpd_calibration.core.models import CalibrationRecord, CurveData
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, CurveType, DeveloperType
    from ptpd_calibration.curves import (
        BlendMode,
        CurveAIEnhancer,
        CurveGenerator,
        CurveModifier,
        EnhancementGoal,
        SmoothingMethod,
        load_quad_file,
        load_quad_string,
        save_curve,
    )
    from ptpd_calibration.detection import StepTabletReader
    from ptpd_calibration.gcp.storage import get_storage_backend
    from ptpd_calibration.ml import CalibrationDatabase

    # Initialize app
    settings = get_settings()
    app = FastAPI(
        title="PTPD Calibration API",
        description="AI-powered calibration system for platinum/palladium printing",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # State
    storage_backend = get_storage_backend(settings.gcp)
    database = CalibrationDatabase(storage_backend=storage_backend)
    database.load_from_storage()

    upload_dir = settings.api.upload_dir or Path(tempfile.mkdtemp())

    def save_curve_to_storage(curve: CurveData) -> None:
        """Save curve to storage backend."""
        path = f"curves/{str(curve.id)}.json"
        storage_backend.save(path, curve.model_dump_json())

    def get_curve_from_storage(curve_id: str) -> CurveData | None:
        """Get curve from storage backend."""
        path = f"curves/{curve_id}.json"
        try:
            data = storage_backend.load(path)
            return CurveData.model_validate_json(data)
        except Exception:
            return None

    # Pydantic models
    class AnalyzeRequest(BaseModel):
        densities: list[float]

    class CurveRequest(BaseModel):
        densities: list[float]
        name: str = "Calibration Curve"
        curve_type: str = "linear"
        paper_type: str | None = None
        chemistry: str | None = None

    class CalibrationRequest(BaseModel):
        paper_type: str
        exposure_time: float
        metal_ratio: float = 0.5
        contrast_agent: str = "none"
        contrast_amount: float = 0.0
        developer: str = "potassium_oxalate"
        chemistry_type: str = "platinum_palladium"
        densities: list[float] = []
        notes: str | None = None

    class ChatRequest(BaseModel):
        message: str
        include_history: bool = True

    class RecipeRequest(BaseModel):
        paper_type: str
        characteristics: str

    class TroubleshootRequest(BaseModel):
        problem: str

    class CurveModifyRequest(BaseModel):
        input_values: list[float]
        output_values: list[float]
        name: str = "Modified Curve"
        adjustment_type: str = (
            "brightness"  # brightness, contrast, gamma, levels, highlights, shadows, midtones
        )
        amount: float = 0.0
        # Additional parameters for specific adjustments
        pivot: float = 0.5  # For contrast
        black_point: float = 0.0  # For levels
        white_point: float = 1.0  # For levels

    class CurveSmoothRequest(BaseModel):
        input_values: list[float]
        output_values: list[float]
        name: str = "Smoothed Curve"
        method: str = "gaussian"  # gaussian, savgol, moving_average, spline
        strength: float = 0.5
        preserve_endpoints: bool = True

    class CurveBlendRequest(BaseModel):
        curve1_inputs: list[float]
        curve1_outputs: list[float]
        curve2_inputs: list[float]
        curve2_outputs: list[float]
        name: str = "Blended Curve"
        mode: str = "weighted"  # average, weighted, multiply, screen, overlay, min, max
        weight: float = 0.5

    class CurveEnhanceRequest(BaseModel):
        input_values: list[float]
        output_values: list[float]
        name: str = "Enhanced Curve"
        goal: str = "linearization"  # linearization, maximize_range, smooth_gradation, highlight_detail, shadow_detail, neutral_midtones, print_stability
        paper_type: str | None = None
        additional_context: str | None = None

    # Curve storage for session (replaced by persistent storage backend)
    # curve_storage: dict[str, CurveData] = {}

    # Routes
    @app.get("/")
    async def root():
        return {"message": "PTPD Calibration API", "version": "1.0.0"}

    @app.get("/api/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/api/analyze")
    async def analyze_densities(request: AnalyzeRequest):
        """Analyze density measurements."""
        from ptpd_calibration.curves.analysis import CurveAnalyzer

        analysis = CurveAnalyzer.analyze_linearity(request.densities)
        suggestions = CurveAnalyzer.suggest_adjustments(request.densities)

        return {
            "dmin": min(request.densities),
            "dmax": max(request.densities),
            "range": max(request.densities) - min(request.densities),
            "is_monotonic": analysis.is_monotonic,
            "max_error": analysis.max_error,
            "rms_error": analysis.rms_error,
            "suggestions": suggestions,
        }

    # Allowlisted scan file extensions (case-insensitive)
    _ALLOWED_SCAN_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"})

    @app.post("/api/scan/upload")
    async def upload_scan(
        file: UploadFile = File(...),
        tablet_type: str = Form("stouffer_21"),
    ):
        """Upload and process a step tablet scan."""
        import logging
        from uuid import uuid4

        logger = logging.getLogger(__name__)

        # ── Sanitise client-supplied filename ──────────────────────────
        original_filename = file.filename or "unknown"
        # Extract extension safely (only basename, no path separators)
        safe_basename = Path(original_filename).name  # strips ../ segments
        suffix = Path(safe_basename).suffix.lower()

        if suffix not in _ALLOWED_SCAN_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_SCAN_EXTENSIONS))}",
            )

        # Server-generated unique key — never trust client filename for paths
        scan_id = uuid4().hex
        safe_name = f"{scan_id}{suffix}"
        file_path = upload_dir / safe_name

        logger.debug("Scan upload: original=%s safe=%s", original_filename, safe_name)

        # ── Stream upload to disk with size enforcement ─────────────
        max_bytes = settings.api.max_upload_size_mb * 1024 * 1024
        bytes_written = 0
        _CHUNK_SIZE = 64 * 1024  # 64 KB chunks

        try:
            with open(file_path, "wb") as f:
                while True:
                    chunk = await file.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        # Clean up partial file before rejecting
                        f.close()
                        if file_path.exists():
                            file_path.unlink()
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds maximum size of "
                            f"{settings.api.max_upload_size_mb} MB",
                        )
                    f.write(chunk)
        except HTTPException:
            raise  # Re-raise 413 without catching it below
        except OSError as exc:
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

        try:
            # Process scan
            reader = StepTabletReader(tablet_type=TabletType(tablet_type))
            result = reader.read(file_path)

            # Persist raw scan to storage with server-generated key
            with open(file_path, "rb") as f:
                storage_backend.save(f"scans/{safe_name}", f.read())

            return {
                "success": True,
                "extraction_id": str(result.extraction.id),
                "original_filename": original_filename,
                "num_patches": result.extraction.num_patches,
                "densities": result.extraction.get_densities(),
                "dmin": result.extraction.dmin,
                "dmax": result.extraction.dmax,
                "range": result.extraction.density_range,
                "quality": result.extraction.overall_quality,
                "warnings": result.extraction.warnings,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            # Cleanup temp file
            if file_path.exists():
                file_path.unlink()

    @app.post("/api/curves/generate")
    async def generate_curve(request: CurveRequest):
        """Generate a calibration curve."""
        generator = CurveGenerator()

        try:
            curve = generator.generate(
                request.densities,
                curve_type=CurveType(request.curve_type),
                name=request.name,
                paper_type=request.paper_type,
                chemistry=request.chemistry,
            )

            # Store the generated curve
            save_curve_to_storage(curve)

            return {
                "success": True,
                "curve_id": str(curve.id),
                "name": curve.name,
                "num_points": len(curve.input_values),
                "input_values": curve.input_values[:10],  # Sample
                "output_values": curve.output_values[:10],
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/curves/export")
    async def export_curve(
        densities: list[float] = Form(...),
        name: str = Form("curve"),
        format: str = Form("qtr"),
    ):
        """Export a curve to file."""
        generator = CurveGenerator()
        curve = generator.generate(densities, name=name)

        # Create temp file
        ext_map = {"qtr": ".txt", "piezography": ".ppt", "csv": ".csv", "json": ".json"}
        ext = ext_map.get(format, ".txt")
        output_path = upload_dir / f"{name}{ext}"

        save_curve(curve, output_path, format=format)

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"{name}{ext}",
        )

    @app.post("/api/curves/upload-quad")
    async def upload_quad_file(
        file: UploadFile = File(...),
        channel: str = Form("K"),
    ):
        """
        Upload and parse a QTR .quad file.

        Returns the parsed profile with all channels and metadata.
        """
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # Parse the .quad file
            profile = load_quad_file(file_path)

            # Convert requested channel to CurveData and store
            if channel.upper() in profile.channels:
                curve_data = profile.to_curve_data(channel.upper())
                save_curve_to_storage(curve_data)
            else:
                curve_data = None

            return {
                "success": True,
                "profile_name": profile.profile_name,
                "resolution": profile.resolution,
                "ink_limit": profile.ink_limit,
                "media_type": profile.media_type,
                "all_channels": profile.all_channel_names,
                "active_channels": profile.active_channels,
                "curve_id": str(curve_data.id) if curve_data else None,
                "curve_data": {
                    "input_values": curve_data.input_values[:20] if curve_data else [],
                    "output_values": curve_data.output_values[:20] if curve_data else [],
                }
                if curve_data
                else None,
                "summary": profile.summary(),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            # Cleanup
            if file_path.exists():
                file_path.unlink()

    @app.post("/api/curves/parse-quad")
    async def parse_quad_content(
        content: str = Form(...),
        name: str = Form("Uploaded Profile"),
        channel: str = Form("K"),
    ):
        """
        Parse .quad content from a string (for pasting quad data directly).
        """
        try:
            profile = load_quad_string(content, name)

            # Convert requested channel to CurveData and store
            if channel.upper() in profile.channels:
                curve_data = profile.to_curve_data(channel.upper())
                save_curve_to_storage(curve_data)
            else:
                curve_data = None

            return {
                "success": True,
                "profile_name": profile.profile_name,
                "active_channels": profile.active_channels,
                "curve_id": str(curve_data.id) if curve_data else None,
                "curve_data": {
                    "input_values": curve_data.input_values if curve_data else [],
                    "output_values": curve_data.output_values if curve_data else [],
                }
                if curve_data
                else None,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/curves/modify")
    async def modify_curve(request: CurveModifyRequest):
        """
        Apply modifications to a curve.

        Supports: brightness, contrast, gamma, levels, highlights, shadows, midtones
        """
        try:
            # Create CurveData from request
            curve = CurveData(
                name=request.name,
                input_values=request.input_values,
                output_values=request.output_values,
            )

            modifier = CurveModifier()
            adjustment_type = request.adjustment_type.lower()

            if adjustment_type == "brightness":
                modified = modifier.adjust_brightness(curve, request.amount)
            elif adjustment_type == "contrast":
                modified = modifier.adjust_contrast(curve, request.amount, pivot=request.pivot)
            elif adjustment_type == "gamma":
                # For gamma, amount should be > 0; default to 1.0 + amount
                gamma_value = max(0.1, 1.0 + request.amount) if request.amount != 0 else 1.0
                modified = modifier.adjust_gamma(curve, gamma_value)
            elif adjustment_type == "levels":
                modified = modifier.adjust_levels(
                    curve,
                    black_point=request.black_point,
                    white_point=request.white_point,
                )
            elif adjustment_type == "highlights":
                modified = modifier.adjust_highlights(curve, request.amount)
            elif adjustment_type == "shadows":
                modified = modifier.adjust_shadows(curve, request.amount)
            elif adjustment_type == "midtones":
                modified = modifier.adjust_midtones(curve, request.amount)
            else:
                raise ValueError(f"Unknown adjustment type: {adjustment_type}")

            # Store the modified curve
            save_curve_to_storage(modified)

            return {
                "success": True,
                "curve_id": str(modified.id),
                "name": modified.name,
                "adjustment_applied": adjustment_type,
                "input_values": modified.input_values,
                "output_values": modified.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/curves/smooth")
    async def smooth_curve(request: CurveSmoothRequest):
        """
        Apply smoothing to a curve.

        Supports: gaussian, savgol, moving_average, spline
        """
        try:
            # Create CurveData from request
            curve = CurveData(
                name=request.name,
                input_values=request.input_values,
                output_values=request.output_values,
            )

            modifier = CurveModifier(preserve_endpoints=request.preserve_endpoints)
            method = SmoothingMethod(request.method.lower())
            smoothed = modifier.smooth(
                curve,
                method=method,
                strength=request.strength,
            )

            # Store the smoothed curve
            save_curve_to_storage(smoothed)

            return {
                "success": True,
                "curve_id": str(smoothed.id),
                "name": smoothed.name,
                "method_applied": request.method,
                "input_values": smoothed.input_values,
                "output_values": smoothed.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/curves/blend")
    async def blend_curves(request: CurveBlendRequest):
        """
        Blend two curves together.

        Supports: average, weighted, multiply, screen, overlay, min, max
        """
        try:
            # Create CurveData from request
            curve1 = CurveData(
                name="Curve 1",
                input_values=request.curve1_inputs,
                output_values=request.curve1_outputs,
            )
            curve2 = CurveData(
                name="Curve 2",
                input_values=request.curve2_inputs,
                output_values=request.curve2_outputs,
            )

            modifier = CurveModifier()
            mode = BlendMode(request.mode.lower())
            blended = modifier.blend(
                curve1,
                curve2,
                mode=mode,
                weight=request.weight,
            )
            blended.name = request.name

            # Store the blended curve
            save_curve_to_storage(blended)

            return {
                "success": True,
                "curve_id": str(blended.id),
                "name": blended.name,
                "mode_applied": request.mode,
                "input_values": blended.input_values,
                "output_values": blended.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/curves/enhance")
    async def enhance_curve_ai(request: CurveEnhanceRequest):
        """
        Apply AI-powered enhancement to a curve.

        Uses LLM analysis to suggest and apply improvements based on the specified goal.
        """
        try:
            # Create CurveData from request
            curve = CurveData(
                name=request.name,
                input_values=request.input_values,
                output_values=request.output_values,
                paper_type=request.paper_type,
            )

            enhancer = CurveAIEnhancer()
            goal = EnhancementGoal(request.goal.lower())

            # Try LLM enhancement first, fall back to algorithmic
            try:
                result = await enhancer.enhance_with_llm(
                    curve,
                    goal=goal,
                    additional_context=request.additional_context,
                )
            except Exception:
                # Fall back to algorithmic enhancement
                result = await enhancer.analyze_and_enhance(
                    curve,
                    goal=goal,
                )

            # Store the enhanced curve
            save_curve_to_storage(result.enhanced_curve)

            return {
                "success": True,
                "curve_id": str(result.enhanced_curve.id),
                "name": result.enhanced_curve.name,
                "goal": result.goal.value,
                "confidence": result.confidence,
                "analysis": result.analysis,
                "changes_made": result.changes_made,
                "input_values": result.enhanced_curve.input_values,
                "output_values": result.enhanced_curve.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/curves/{curve_id}")
    async def get_stored_curve(curve_id: str):
        """Get a stored curve by ID."""
        curve = get_curve_from_storage(curve_id)
        if not curve:
            raise HTTPException(status_code=404, detail="Curve not found")

        return {
            "curve_id": str(curve.id),
            "name": curve.name,
            "curve_type": curve.curve_type.value if curve.curve_type else None,
            "paper_type": curve.paper_type,
            "input_values": curve.input_values,
            "output_values": curve.output_values,
            "notes": curve.notes,
        }

    @app.post("/api/curves/{curve_id}/enforce-monotonicity")
    async def enforce_monotonicity(
        curve_id: str,
        direction: str = "increasing",
    ):
        """Enforce monotonicity on a stored curve."""
        curve = get_curve_from_storage(curve_id)
        if not curve:
            raise HTTPException(status_code=404, detail="Curve not found")

        try:
            modifier = CurveModifier()
            modified = modifier.enforce_monotonicity(curve, direction=direction)
            save_curve_to_storage(modified)

            return {
                "success": True,
                "curve_id": str(modified.id),
                "name": modified.name,
                "input_values": modified.input_values,
                "output_values": modified.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/calibrations")
    async def list_calibrations(
        paper_type: str | None = None,
        limit: int = 50,
    ):
        """List calibration records."""
        records = database.query(paper_type=paper_type)

        return {
            "count": len(records),
            "records": [
                {
                    "id": str(r.id),
                    "paper_type": r.paper_type,
                    "exposure_time": r.exposure_time,
                    "metal_ratio": r.metal_ratio,
                    "timestamp": r.timestamp.isoformat(),
                    "dmax": max(r.measured_densities) if r.measured_densities else 0,
                }
                for r in records[:limit]
            ],
        }

    @app.post("/api/calibrations")
    async def create_calibration(request: CalibrationRequest):
        """Create a new calibration record."""
        record = CalibrationRecord(
            paper_type=request.paper_type,
            exposure_time=request.exposure_time,
            metal_ratio=request.metal_ratio,
            contrast_agent=ContrastAgent(request.contrast_agent),
            contrast_amount=request.contrast_amount,
            developer=DeveloperType(request.developer),
            chemistry_type=ChemistryType(request.chemistry_type),
            measured_densities=request.densities,
            notes=request.notes,
        )

        database.add_record(record)

        return {
            "success": True,
            "id": str(record.id),
            "message": "Calibration saved",
        }

    @app.get("/api/calibrations/{calibration_id}")
    async def get_calibration(calibration_id: str):
        """Get a specific calibration record."""
        record = database.get_record(UUID(calibration_id))
        if not record:
            raise HTTPException(status_code=404, detail="Calibration not found")

        return record.model_dump(mode="json")

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Chat with the AI assistant."""
        try:
            from ptpd_calibration.llm import create_assistant

            assistant = create_assistant(database=database)
            response = await assistant.chat(
                request.message,
                include_history=request.include_history,
            )

            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/recipe")
    async def suggest_recipe(request: RecipeRequest):
        """Get recipe suggestion."""
        try:
            from ptpd_calibration.llm import create_assistant

            assistant = create_assistant(database=database)
            response = await assistant.suggest_recipe(
                request.paper_type,
                request.characteristics,
            )

            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/chat/troubleshoot")
    async def troubleshoot(request: TroubleshootRequest):
        """Get troubleshooting help."""
        try:
            from ptpd_calibration.llm import create_assistant

            assistant = create_assistant(database=database)
            response = await assistant.troubleshoot(request.problem)

            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/statistics")
    async def get_statistics():
        """Get database statistics."""
        return database.get_statistics()

    return app


def main():
    """Run the API server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required. Install with: pip install ptpd-calibration[api]")

    settings = get_settings()
    app = create_app()

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
