"""
FastAPI server for PTPD Calibration System.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from uuid import UUID

from ptpd_calibration.config import get_settings


def create_app():
    """Create the FastAPI application."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required. Install with: pip install ptpd-calibration[api]"
        )

    from ptpd_calibration.config import ExportFormat, TabletType
    from ptpd_calibration.core.models import CalibrationRecord
    from ptpd_calibration.core.types import ChemistryType, ContrastAgent, CurveType, DeveloperType
    from ptpd_calibration.curves import CurveGenerator, save_curve
    from ptpd_calibration.detection import StepTabletReader
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
    database = CalibrationDatabase()
    upload_dir = settings.api.upload_dir or Path(tempfile.mkdtemp())

    # Pydantic models
    class AnalyzeRequest(BaseModel):
        densities: list[float]

    class CurveRequest(BaseModel):
        densities: list[float]
        name: str = "Calibration Curve"
        curve_type: str = "linear"
        paper_type: Optional[str] = None
        chemistry: Optional[str] = None

    class CalibrationRequest(BaseModel):
        paper_type: str
        exposure_time: float
        metal_ratio: float = 0.5
        contrast_agent: str = "none"
        contrast_amount: float = 0.0
        developer: str = "potassium_oxalate"
        chemistry_type: str = "platinum_palladium"
        densities: list[float] = []
        notes: Optional[str] = None

    class ChatRequest(BaseModel):
        message: str
        include_history: bool = True

    class RecipeRequest(BaseModel):
        paper_type: str
        characteristics: str

    class TroubleshootRequest(BaseModel):
        problem: str

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

    @app.post("/api/scan/upload")
    async def upload_scan(
        file: UploadFile = File(...),
        tablet_type: str = Form("stouffer_21"),
    ):
        """Upload and process a step tablet scan."""
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            # Process scan
            reader = StepTabletReader(tablet_type=TabletType(tablet_type))
            result = reader.read(file_path)

            return {
                "success": True,
                "extraction_id": str(result.extraction.id),
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
            # Cleanup
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

    @app.get("/api/calibrations")
    async def list_calibrations(
        paper_type: Optional[str] = None,
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
        raise ImportError(
            "uvicorn is required. Install with: pip install ptpd-calibration[api]"
        )

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
