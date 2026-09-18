"""
FastAPI server for PTPD Calibration System.
"""

import logging
import re
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

from ptpd_calibration.api.observability import RequestContextMiddleware
from ptpd_calibration.config import Settings, get_settings
from ptpd_calibration.core.logging import setup_logging

_log = logging.getLogger(__name__)

# Extension of the JSON records written for stored curves.
_CURVE_SUFFIX = ".json"

# Stored curve ids are UUIDs (CurveData.id), so the id taken from the URL is
# matched against that shape before it is ever joined to a path.
_CURVE_ID_PATTERN = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def _api_version() -> str:
    """Return the installed package version, or a marker when run from a tree.

    The API reported a hard-coded "1.0.0" whatever was deployed, so an operator
    could not tell which build answered a request.
    """
    try:
        return version("ptpd-calibration")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "0.0.0+unknown"


def create_app(settings: Settings | None = None):
    """Create the FastAPI application.

    Args:
        settings: Optional settings override. Defaults to the process-wide
            ``get_settings()`` instance; tests pass an explicit ``Settings``
            to exercise limits without touching global state.
    """
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field
        from starlette.background import BackgroundTask
    except ImportError as err:
        raise ImportError(
            "FastAPI is required. Install with: pip install ptpd-calibration[api]"
        ) from err

    from ptpd_calibration.api.security import (
        RequestBodyLimitMiddleware,
        kb_to_bytes,
        mb_to_bytes,
        safe_export_name,
        safe_suffix,
        server_upload_path,
        stored_record_path,
        stream_upload_to_path,
        unlink_quietly,
    )
    from ptpd_calibration.config import ExportFormat, TabletType
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
    from ptpd_calibration.curves.parser import QuadParserLimits
    from ptpd_calibration.detection import StepTabletReader
    from ptpd_calibration.imaging import (
        ColorMode,
        ExportSettings,
        ImageFormat,
        ImageProcessor,
    )
    from ptpd_calibration.imaging.safe_image import (
        ImageDecodeError,
        ImageDecodeSettings,
        ImageTooLargeError,
    )
    from ptpd_calibration.ml import CalibrationDatabase

    # Initialize app
    settings = settings or get_settings()

    # Configure logging deliberately, here, rather than leaving it to whichever
    # module happens to call get_logger() first: that made the level, format
    # and destination depend on import order, so a deployment could not choose
    # them and the debug logging on guarded paths was invisible.
    setup_logging(
        level=settings.log_level,
        log_file=settings.log_file,
        json_format=settings.log_json,
    )
    _log.info(
        "Starting %s version %s (log level %s, json=%s)",
        settings.app_name,
        _api_version(),
        settings.log_level,
        settings.log_json,
    )

    app = FastAPI(
        title="PTPD Calibration API",
        description="AI-powered calibration system for platinum/palladium printing",
        version=_api_version(),
    )

    # Request bounds shared by every endpoint (SEC-03). Values come from
    # APISettings so deployments can tune them via PTPD_API_* variables.
    max_list = settings.api.max_list_length
    max_str = settings.api.max_string_length
    max_upload_bytes = mb_to_bytes(settings.api.max_upload_size_mb)
    upload_chunk_bytes = kb_to_bytes(settings.api.upload_chunk_size_kb)
    # A pasted .quad profile is legitimately far longer than the general string
    # cap: 256 values per channel across eight channels runs to thousands of
    # lines, so max_str would reject valid content. The parser's own limit is
    # the right bound, and it measures a string the same way (PTPD_QUAD_*), so
    # an oversize paste is refused by the framework instead of being buffered
    # and parsed before the parser rejects it.
    max_quad_content = QuadParserLimits().max_bytes

    # Body-size cap is added before CORS so that CORS wraps it and a 413 still
    # carries the CORS headers a browser needs to surface the error.
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=mb_to_bytes(settings.api.max_request_body_mb),
    )

    # Added last so it runs first: every request, including one rejected by the
    # body cap above, is logged with its identifier.
    app.add_middleware(RequestContextMiddleware)

    # CORS (SEC-07): allow_credentials is read from settings and defaults to
    # False; APISettings refuses the wildcard-origin + credentials combination.
    cors_origins_set = set(settings.api.cors_origins)
    # Allow localhost:3000 only in reload (development) mode
    if settings.api.reload:
        cors_origins_set.add("http://localhost:3000")
    cors_origins = list(cors_origins_set)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _log.debug(
        "API limits: body=%dMB upload=%dMB list=%d str=%d cors_credentials=%s",
        settings.api.max_request_body_mb,
        settings.api.max_upload_size_mb,
        max_list,
        max_str,
        settings.api.cors_allow_credentials,
    )

    # State
    database = CalibrationDatabase()
    upload_dir = settings.api.upload_dir or Path(tempfile.mkdtemp())
    upload_dir.mkdir(parents=True, exist_ok=True)
    deep_learning_model_storage: dict = {}  # Storage for trained DL models

    # Optional routers. Which of these mounted is reported by /api/health, so a
    # caller can tell a missing extra from a broken deployment.
    optional_routers: dict[str, bool] = {}

    try:
        from ptpd_calibration.api.deep_learning import create_deep_learning_router

        deep_router = create_deep_learning_router(database, deep_learning_model_storage)
        app.include_router(deep_router)
        optional_routers["deep_learning"] = True
    except ImportError as exc:
        _log.info("Deep-learning routes unavailable: %s", exc)
        optional_routers["deep_learning"] = False

    try:
        from ptpd_calibration.api.mcts_router import create_mcts_router

        mcts_router = create_mcts_router()
        app.include_router(mcts_router)
        optional_routers["mcts"] = True
    except ImportError as exc:
        _log.info("Search routes unavailable: %s", exc)
        optional_routers["mcts"] = False

    # Pydantic models. Every list and string field is bounded (SEC-03) using
    # the limits above so an oversized payload fails validation with 422
    # before any processing happens.
    class AnalyzeRequest(BaseModel):
        densities: list[float] = Field(max_length=max_list)

    class CurveRequest(BaseModel):
        densities: list[float] = Field(max_length=max_list)
        name: str = Field(default="Calibration Curve", max_length=max_str)
        curve_type: str = Field(default="linear", max_length=max_str)
        paper_type: str | None = Field(default=None, max_length=max_str)
        chemistry: str | None = Field(default=None, max_length=max_str)

    class CalibrationRequest(BaseModel):
        paper_type: str = Field(max_length=max_str)
        exposure_time: float
        metal_ratio: float = 0.5
        contrast_agent: str = Field(default="none", max_length=max_str)
        contrast_amount: float = 0.0
        developer: str = Field(default="potassium_oxalate", max_length=max_str)
        chemistry_type: str = Field(default="platinum_palladium", max_length=max_str)
        densities: list[float] = Field(default_factory=list, max_length=max_list)
        notes: str | None = Field(default=None, max_length=max_str)

    class ChatRequest(BaseModel):
        message: str = Field(max_length=max_str)
        include_history: bool = True

    class RecipeRequest(BaseModel):
        paper_type: str = Field(max_length=max_str)
        characteristics: str = Field(max_length=max_str)

    class TroubleshootRequest(BaseModel):
        problem: str = Field(max_length=max_str)

    class CurveModifyRequest(BaseModel):
        input_values: list[float] = Field(max_length=max_list)
        output_values: list[float] = Field(max_length=max_list)
        name: str = Field(default="Modified Curve", max_length=max_str)
        adjustment_type: str = Field(
            default="brightness",  # brightness, contrast, gamma, levels, highlights, shadows, midtones
            max_length=max_str,
        )
        amount: float = 0.0
        # Additional parameters for specific adjustments
        pivot: float = 0.5  # For contrast
        black_point: float = 0.0  # For levels
        white_point: float = 1.0  # For levels

    class CurveSmoothRequest(BaseModel):
        input_values: list[float] = Field(max_length=max_list)
        output_values: list[float] = Field(max_length=max_list)
        name: str = Field(default="Smoothed Curve", max_length=max_str)
        method: str = Field(
            default="gaussian", max_length=max_str
        )  # gaussian, savgol, moving_average, spline
        strength: float = 0.5
        preserve_endpoints: bool = True

    class CurveBlendRequest(BaseModel):
        curve1_inputs: list[float] = Field(max_length=max_list)
        curve1_outputs: list[float] = Field(max_length=max_list)
        curve2_inputs: list[float] = Field(max_length=max_list)
        curve2_outputs: list[float] = Field(max_length=max_list)
        name: str = Field(default="Blended Curve", max_length=max_str)
        mode: str = Field(
            default="weighted", max_length=max_str
        )  # average, weighted, multiply, screen, overlay, min, max
        weight: float = 0.5

    class CurveEnhanceRequest(BaseModel):
        input_values: list[float] = Field(max_length=max_list)
        output_values: list[float] = Field(max_length=max_list)
        name: str = Field(default="Enhanced Curve", max_length=max_str)
        goal: str = Field(
            default="linearization", max_length=max_str
        )  # linearization, maximize_range, smooth_gradation, highlight_detail, shadow_detail, neutral_midtones, print_stability
        paper_type: str | None = Field(default=None, max_length=max_str)
        additional_context: str | None = Field(default=None, max_length=max_str)

    # Curve storage — write-through cache backed by JSON files on disk
    curves_dir = upload_dir.parent / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curve_storage: dict[str, CurveData] = {}

    def _curve_path(curve_id: str) -> Path | None:
        """Path of a stored curve, or None when the id is not a safe component.

        The id reaches this from a URL path parameter, so it goes through
        ``stored_record_path``, which rejects separators and traversal and
        re-checks that the resolved path is still under ``curves_dir``.
        """
        return stored_record_path(curves_dir, curve_id, _CURVE_SUFFIX, pattern=_CURVE_ID_PATTERN)

    def _store_curve(curve: CurveData) -> None:
        """Cache curve in memory and persist to disk."""
        curve_storage[str(curve.id)] = curve
        path = _curve_path(str(curve.id))
        if path is None:  # pragma: no cover - ids are server-generated UUIDs
            _log.error("Refusing to persist curve with unsafe id %r", curve.id)
            return
        try:
            path.write_text(curve.model_dump_json(), encoding="utf-8")
        except Exception:
            _log.warning("Failed to persist curve %s to disk", curve.id, exc_info=True)

    def _get_curve(curve_id: str) -> CurveData | None:
        """Return curve from memory cache, falling back to disk."""
        if curve_id in curve_storage:
            return curve_storage[curve_id]
        path = _curve_path(curve_id)
        if path is None:
            return None
        if path.exists():
            try:
                loaded = CurveData.model_validate_json(path.read_text(encoding="utf-8"))
                curve_storage[curve_id] = loaded
                return loaded
            except Exception:
                _log.warning("Failed to load curve %s from disk", curve_id, exc_info=True)
                return None
        return None

    # Routes
    @app.get("/")
    async def root():
        return {"message": "PTPD Calibration API", "version": "1.0.0"}

    @app.get("/api/health")
    async def health():
        """Report what is actually running and which optional parts are usable.

        A static "healthy" cannot distinguish a working deployment from one
        whose language-model provider is unconfigured or whose optional
        machine-learning extra is missing, which are the two states an operator
        most often needs to tell apart.
        """
        return {
            "status": "healthy",
            "version": _api_version(),
            "log_level": settings.log_level,
            "llm_provider_configured": bool(settings.llm.get_active_api_key()),
            "features": dict(optional_routers),
        }

    @app.post("/api/analyze")
    async def analyze_densities(request: AnalyzeRequest):
        """Analyze density measurements."""
        if not request.densities:
            raise HTTPException(status_code=422, detail="Densities list cannot be empty")

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

    @app.post("/api/scan/upload")
    async def upload_scan(
        file: UploadFile = File(...),
        tablet_type: str = Form("stouffer_21", max_length=max_str),
    ):
        """Upload and process a step tablet scan.

        The client filename is only used to pick an allowlisted extension; the
        file is streamed to a server-generated path under a size cap (SEC-01/02).
        """
        original_filename = file.filename or "unknown"
        suffix = safe_suffix(file.filename, settings.api.allowed_scan_extensions)
        file_path = server_upload_path(upload_dir, suffix)
        _log.debug("Scan upload: original=%r server_path=%s", original_filename, file_path.name)

        await stream_upload_to_path(file, file_path, max_upload_bytes, upload_chunk_bytes)

        try:
            # Process scan
            reader = StepTabletReader(tablet_type=TabletType(tablet_type))
            result = reader.read(file_path)

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
        except ImageTooLargeError as e:
            # Decode guard tripped on the header (pixel or frame cap, SEC-04):
            # the request is well-formed but exceeds configured limits.
            _log.warning("Scan upload rejected before decode: %s", e)
            raise HTTPException(status_code=413, detail=str(e)) from None
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        finally:
            # Only the server-generated path is ever removed
            unlink_quietly(file_path)

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

            # Every other curve route stores its result, and this one returned
            # a curve_id regardless, so fetching or exporting a generated curve
            # answered 404 for the one endpoint a calibration actually starts
            # from.
            _store_curve(curve)

            return {
                "success": True,
                "curve_id": str(curve.id),
                "name": curve.name,
                "num_points": len(curve.input_values),
                "input_values": curve.input_values,
                "output_values": curve.output_values,
            }
        except ValueError as e:
            # The generator refuses input it cannot invert (a reversed wedge, a
            # non-finite patch, a series that rises and falls) and names the
            # offending patch; that message is the useful part of the response.
            _log.debug("Curve generation refused: %s", e)
            raise HTTPException(status_code=422, detail=str(e)) from None
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    # Download extension per supported export format (the exporter's declared set)
    _export_extensions: dict[str, str] = {
        ExportFormat.QTR.value: ".txt",
        ExportFormat.PIEZOGRAPHY.value: ".ppt",
        ExportFormat.CSV.value: ".csv",
        ExportFormat.JSON.value: ".json",
    }

    def _export_curve_response(curve: CurveData, name: str, format: str) -> FileResponse:
        """Write ``curve`` to a server-named temp file and return it as a download.

        The client-supplied ``name`` only reaches the ``Content-Disposition``
        header after sanitisation; the on-disk path is a uuid so no request can
        choose where the server writes. The temp file is removed once the
        response has been sent.
        """
        normalized = format.lower()
        ext = _export_extensions.get(normalized)
        if ext is None:
            _log.debug("Rejected export format %r", format)
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported export format '{format}'. "
                f"Supported: {', '.join(sorted(_export_extensions))}",
            )
        download_stem = safe_export_name(name, max_length=settings.api.max_export_name_length)
        output_path = server_upload_path(upload_dir, ext)
        save_curve(curve, output_path, format=normalized)
        _log.debug(
            "Exported curve %s as %s -> %s (download name %s%s)",
            curve.id,
            normalized,
            output_path.name,
            download_stem,
            ext,
        )
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"{download_stem}{ext}",
            background=BackgroundTask(unlink_quietly, output_path),
        )

    @app.post("/api/curves/export")
    async def export_curve(
        densities: list[float] = Form(..., max_length=max_list),
        name: str = Form("curve", max_length=max_str),
        format: str = Form("qtr", max_length=max_str),
    ):
        """Export a curve to file."""
        generator = CurveGenerator()
        curve = generator.generate(densities, name=name)
        return _export_curve_response(curve, name, format)

    @app.post("/api/curves/{curve_id}/export")
    async def export_stored_curve(
        curve_id: str,
        format: str = Query("qtr", max_length=max_str),
    ):
        """Export a previously stored curve by ID."""
        curve = _get_curve(curve_id)
        if not curve:
            raise HTTPException(status_code=404, detail="Curve not found")
        return _export_curve_response(curve, curve.name, format)

    # Download extension and media type per negative export format. ORIGINAL is
    # absent on purpose: a negative is a new artefact, so the caller states the
    # format it wants rather than inheriting the scan's.
    _negative_formats: dict[str, tuple[str, str]] = {
        ImageFormat.TIFF.value: (".tiff", "image/tiff"),
        ImageFormat.TIFF_16BIT.value: (".tiff", "image/tiff"),
        ImageFormat.PNG.value: (".png", "image/png"),
        ImageFormat.PNG_16BIT.value: (".png", "image/png"),
        ImageFormat.JPEG.value: (".jpg", "image/jpeg"),
        ImageFormat.JPEG_HIGH.value: (".jpg", "image/jpeg"),
    }

    # A negative is printed at full size, so this path does not inherit the
    # shared decode limits, which shrink an image to bound analysis work.
    _negative_decode_settings = ImageDecodeSettings(
        max_pixels=settings.api.negative_export_max_pixels,
        downsample_max_side=settings.api.negative_export_max_side,
    )

    def _negative_curve(curve_id: str | None, densities: list[float] | None) -> CurveData | None:
        """Resolve the curve to apply, by stored id or from measured densities.

        Neither is required: inverting an already linearised file is a real
        request, and refusing it would make the endpoint less useful than the
        Gradio tab it replaces.
        """
        if curve_id:
            stored = _get_curve(curve_id)
            if stored is None:
                raise HTTPException(status_code=404, detail="Curve not found")
            return stored
        if densities:
            try:
                return CurveGenerator().generate(densities, name="negative")
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from None
        return None

    @app.post("/api/export/negative")
    async def export_negative(
        file: UploadFile = File(...),
        curve_id: str | None = Form(None, max_length=max_str),
        densities: list[float] | None = Form(None, max_length=max_list),
        name: str = Form("negative", max_length=max_str),
        format: str = Form(ImageFormat.TIFF_16BIT.value, max_length=max_str),
        invert: bool = Form(True),
        color_mode: str = Form(ColorMode.GRAYSCALE.value, max_length=max_str),
    ):
        """Turn an uploaded image into a digital negative and return the file.

        The curve comes from ``curve_id`` (a previously stored curve) or from
        ``densities`` (generated on the spot); with neither, the image is only
        inverted. The upload is streamed to a server-generated path under the
        same size cap as every other upload and removed as soon as it is
        decoded; the rendered negative is removed once the response is sent.
        """
        target = format.lower()
        if target not in _negative_formats:
            _log.debug("Rejected negative export format %r", format)
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported negative format '{format}'. "
                f"Supported: {', '.join(sorted(_negative_formats))}",
            )
        extension, media_type = _negative_formats[target]

        try:
            mode = ColorMode(color_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported color mode '{color_mode}'. "
                f"Supported: {', '.join(sorted(m.value for m in ColorMode))}",
            ) from None

        curve = _negative_curve(curve_id, densities)
        download_stem = safe_export_name(
            name, default="negative", max_length=settings.api.max_export_name_length
        )

        suffix = safe_suffix(file.filename, settings.api.allowed_scan_extensions)
        source_path = server_upload_path(upload_dir, suffix)
        await stream_upload_to_path(file, source_path, max_upload_bytes, upload_chunk_bytes)

        processor = ImageProcessor(decode_settings=_negative_decode_settings)
        try:
            negative = processor.create_digital_negative(
                source_path, curve=curve, invert=invert, color_mode=mode
            )
        except ImageTooLargeError as exc:
            # Decode guard tripped on the header, before any pixel was read.
            _log.warning("Negative source rejected before decode: %s", exc)
            raise HTTPException(status_code=413, detail=str(exc)) from None
        except ImageDecodeError as exc:
            _log.debug("Negative source refused: %s", exc)
            raise HTTPException(status_code=415, detail=str(exc)) from None
        except Exception as exc:
            _log.warning("Negative rendering failed", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc)) from None
        finally:
            # The source is fully decoded by now; only the server path is removed.
            unlink_quietly(source_path)

        output_path = server_upload_path(upload_dir, extension)
        try:
            processor.export(negative, output_path, ExportSettings(format=ImageFormat(target)))
        except ValueError as exc:
            # A combination the writers refuse, such as 16-bit colour as PNG.
            # Left unhandled this escaped as a 500 and stranded the file.
            unlink_quietly(output_path)
            _log.debug("Negative export refused: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except Exception as exc:
            unlink_quietly(output_path)
            _log.warning("Negative export failed", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc)) from None
        _log.debug(
            "Exported negative: curve=%s format=%s mode=%s inverted=%s -> %s",
            curve.id if curve else None,
            target,
            negative.image.mode,
            invert,
            output_path.name,
        )
        return FileResponse(
            output_path,
            media_type=media_type,
            filename=f"{download_stem}{extension}",
            background=BackgroundTask(unlink_quietly, output_path),
        )

    @app.post("/api/curves/upload-quad")
    async def upload_quad_file(
        file: UploadFile = File(...),
        channel: str = Form("K", max_length=max_str),
    ):
        """
        Upload and parse a QTR .quad file.

        Returns the parsed profile with all channels and metadata. The upload is
        streamed to a server-generated path under the configured size cap; the
        client filename only selects an allowlisted extension (SEC-01/02).
        """
        suffix = safe_suffix(file.filename, settings.api.allowed_quad_extensions)
        file_path = server_upload_path(upload_dir, suffix)
        _log.debug("Quad upload: original=%r server_path=%s", file.filename, file_path.name)

        await stream_upload_to_path(file, file_path, max_upload_bytes, upload_chunk_bytes)

        try:
            # Parse the .quad file
            profile = load_quad_file(file_path)

            # Convert requested channel to CurveData and store
            if channel.upper() in profile.channels:
                curve_data = profile.to_curve_data(channel.upper())
                _store_curve(curve_data)
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
            raise HTTPException(status_code=400, detail=str(e)) from None
        finally:
            # Only the server-generated path is ever removed
            unlink_quietly(file_path)

    @app.post("/api/curves/parse-quad")
    async def parse_quad_content(
        content: str = Form(..., max_length=max_quad_content),
        name: str = Form("Uploaded Profile", max_length=max_str),
        channel: str = Form("K", max_length=max_str),
    ):
        """
        Parse .quad content from a string (for pasting quad data directly).
        """
        try:
            profile = load_quad_string(content, name)

            # Validate the parsed profile has recognizable .quad content.
            # _post_process() always adds default disabled channels, so we check
            # raw_sections (INI-style sections found) OR active_channels (channels
            # with non-zero curve data) to distinguish real .quad from random text.
            if not profile.raw_sections and not profile.active_channels:
                raise ValueError(
                    "Content does not appear to be valid .quad format: "
                    "no sections or channel curves found"
                )

            # Convert requested channel to CurveData and store
            if channel.upper() in profile.channels:
                curve_data = profile.to_curve_data(channel.upper())
                _store_curve(curve_data)
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
            raise HTTPException(status_code=400, detail=str(e)) from None

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
            _store_curve(modified)

            return {
                "success": True,
                "curve_id": str(modified.id),
                "name": modified.name,
                "adjustment_applied": adjustment_type,
                "input_values": modified.input_values,
                "output_values": modified.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

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
            _store_curve(smoothed)

            return {
                "success": True,
                "curve_id": str(smoothed.id),
                "name": smoothed.name,
                "method_applied": request.method,
                "input_values": smoothed.input_values,
                "output_values": smoothed.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

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
            _store_curve(blended)

            return {
                "success": True,
                "curve_id": str(blended.id),
                "name": blended.name,
                "mode_applied": request.mode,
                "input_values": blended.input_values,
                "output_values": blended.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

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

            # Try LLM enhancement first, fall back to algorithmic. The keyword
            # was `additional_context`, which is not this method's parameter, so
            # every call raised TypeError and the fallback below swallowed it:
            # the LLM path was unreachable and nothing said so. Log the fallback
            # at warning so a future mismatch is visible rather than silent.
            try:
                result = await enhancer.enhance_with_llm(
                    curve,
                    goal=goal,
                    user_requirements=request.additional_context,
                )
            except Exception:
                _log.warning(
                    "LLM enhancement unavailable, falling back to algorithmic", exc_info=True
                )
                result = await enhancer.analyze_and_enhance(
                    curve,
                    goal=goal,
                )

            # Store the enhanced curve
            _store_curve(result.enhanced_curve)

            # `goal` and `changes_made` are not fields of EnhancementResult, so
            # reading them raised AttributeError and the handler answered 400 to
            # every well-formed request. The goal is the validated request
            # value; the adjustments are `adjustments_applied`.
            return {
                "success": True,
                "curve_id": str(result.enhanced_curve.id),
                "name": result.enhanced_curve.name,
                "goal": goal.value,
                "confidence": result.confidence,
                "analysis": result.analysis,
                "changes_made": result.adjustments_applied,
                "suggestions": result.suggestions,
                "input_values": result.enhanced_curve.input_values,
                "output_values": result.enhanced_curve.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    @app.get("/api/curves/{curve_id}")
    async def get_stored_curve(curve_id: str):
        """Get a stored curve by ID."""
        curve = _get_curve(curve_id)
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
        curve = _get_curve(curve_id)
        if not curve:
            raise HTTPException(status_code=404, detail="Curve not found")

        try:
            modifier = CurveModifier()
            modified = modifier.enforce_monotonicity(curve, direction=direction)
            _store_curve(modified)

            return {
                "success": True,
                "curve_id": str(modified.id),
                "name": modified.name,
                "input_values": modified.input_values,
                "output_values": modified.output_values,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

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
        from pydantic import ValidationError

        try:
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
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None

        database.add_record(record)

        return {
            "success": True,
            "id": str(record.id),
            "message": "Calibration saved",
        }

    @app.get("/api/calibrations/{calibration_id}")
    async def get_calibration(calibration_id: str):
        """Get a specific calibration record."""
        try:
            uid = UUID(calibration_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid calibration ID format") from None
        record = database.get_record(uid)
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
            raise HTTPException(status_code=500, detail=str(e)) from None

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
            raise HTTPException(status_code=500, detail=str(e)) from None

    @app.post("/api/chat/troubleshoot")
    async def troubleshoot(request: TroubleshootRequest):
        """Get troubleshooting help."""
        try:
            from ptpd_calibration.llm import create_assistant

            assistant = create_assistant(database=database)
            response = await assistant.troubleshoot(request.problem)

            return {"response": response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from None

    @app.get("/api/statistics")
    async def get_statistics():
        """Get database statistics."""
        return database.get_statistics()

    return app


def main():
    """Run the API server."""
    try:
        import uvicorn
    except ImportError as err:
        raise ImportError(
            "uvicorn is required. Install with: pip install ptpd-calibration[api]"
        ) from err

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
