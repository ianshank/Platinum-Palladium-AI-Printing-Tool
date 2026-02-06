"""
ADK (Agent Development Kit) multi-agent system for Pt/Pd Calibration Studio.

Replaces the custom agent.py + multi_agent.py with production ADK agents
that wrap existing ptpd_calibration tools and deploy to Vertex AI Agent Engine.

Architecture:
- Darkroom Coordinator (gemini-2.5-pro): Routes requests to specialist agents
  - Calibration Specialist: Step tablets, curves, digital negative calibration
  - Chemistry Advisor: Coating recipes, contrast control, UV exposure
  - Print Coach: Grounded knowledge assistant (Vertex AI Search)

Usage:
    from ptpd_calibration.vertex.agents import create_darkroom_coordinator

    coordinator = create_darkroom_coordinator()
    # Deploy to Agent Engine or run locally
"""

from __future__ import annotations

import json
from typing import Any

from ptpd_calibration.config import get_settings


# ─── Tool Wrappers (bridge to existing ptpd_calibration code) ───


def analyze_step_tablet_scan(
    image_path: str,
    tablet_type: str = "Stouffer 21-step",
    auto_detect_orientation: bool = True,
) -> str:
    """Analyze a scanned step tablet to extract density readings.

    Uses the existing ptpd_calibration detector and extractor modules
    to perform computer vision-based step tablet analysis.

    Args:
        image_path: Path to the uploaded step tablet scan image.
        tablet_type: "Stouffer 21-step", "Stouffer 31-step", or "Stouffer 41-step".
        auto_detect_orientation: Whether to auto-detect tablet orientation.

    Returns:
        JSON string with density readings per step and quality metrics.
    """
    try:
        from ptpd_calibration.detection.detector import StepTabletDetector
        from ptpd_calibration.detection.extractor import DensityExtractor

        detector = StepTabletDetector()
        extractor = DensityExtractor()

        # Detect step regions
        detection = detector.detect(image_path)

        # Extract densities
        result = extractor.extract(detection)

        return json.dumps(
            {
                "status": "success",
                "tablet_type": tablet_type,
                "num_steps": len(result.patches) if hasattr(result, "patches") else 0,
                "densities": [p.density for p in result.patches if p.density is not None]
                if hasattr(result, "patches")
                else [],
                "dmin": result.paper_base_density if hasattr(result, "paper_base_density") else None,
                "quality_score": result.overall_quality if hasattr(result, "overall_quality") else None,
                "warnings": result.warnings if hasattr(result, "warnings") else [],
            },
            indent=2,
        )
    except ImportError:
        return json.dumps(
            {
                "status": "error",
                "error": "Detection modules not available. Ensure opencv-python is installed.",
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def generate_linearization_curve(
    density_readings: str,
    curve_type: str = "linear",
    target_dmax: float = 1.6,
    name: str = "calibration_curve",
) -> str:
    """Generate a linearization curve from density readings.

    Args:
        density_readings: JSON string of density values from step tablet analysis.
        curve_type: "linear", "paper_white", or "aesthetic".
        target_dmax: Target maximum density (typically 1.4-1.8).
        name: Name for the generated curve.

    Returns:
        JSON string with curve points and export-ready data.
    """
    try:
        from ptpd_calibration.core.types import CurveType
        from ptpd_calibration.curves.generator import CurveGenerator

        readings = json.loads(density_readings) if isinstance(density_readings, str) else density_readings
        densities = readings if isinstance(readings, list) else readings.get("densities", [])

        generator = CurveGenerator()
        ct = CurveType(curve_type)
        curve = generator.generate(densities, curve_type=ct, name=name)

        return json.dumps(
            {
                "status": "success",
                "curve_id": str(curve.id),
                "name": curve.name,
                "curve_type": curve.curve_type.value,
                "num_points": len(curve.input_values),
                "input_range": [min(curve.input_values), max(curve.input_values)],
                "output_range": [min(curve.output_values), max(curve.output_values)],
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def calculate_chemistry_recipe(
    print_size_inches: str = "8x10",
    pt_pd_ratio: str = "50:50",
    method: str = "traditional",
    paper_type: str = "Hahnemühle Platinum Rag",
    contrast_goal: str = "normal",
) -> str:
    """Calculate chemistry coating recipe using Bostick-Sullivan formulas.

    Args:
        print_size_inches: Print dimensions (e.g., "8x10", "11x14", "16x20").
        pt_pd_ratio: Platinum to palladium ratio (e.g., "50:50", "80:20", "0:100").
        method: "traditional" (DOP) or "malde-ware" (POP).
        paper_type: Paper being used.
        contrast_goal: "low", "normal", "high", or "very high".

    Returns:
        JSON recipe with drop counts, mL volumes, and mixing instructions.
    """
    try:
        settings = get_settings().chemistry

        # Parse dimensions
        parts = print_size_inches.lower().replace(" ", "").split("x")
        width, height = float(parts[0]), float(parts[1])

        # Calculate coating area (subtract margins)
        margin = settings.default_margin_inches
        coating_width = max(width - 2 * margin, 1.0)
        coating_height = max(height - 2 * margin, 1.0)
        area_sq_in = coating_width * coating_height

        # Calculate total drops needed
        total_drops = area_sq_in * settings.drops_per_square_inch

        # Parse metal ratio
        pt_str, pd_str = pt_pd_ratio.split(":")
        pt_pct = float(pt_str) / (float(pt_str) + float(pd_str))
        pd_pct = 1.0 - pt_pct

        # Split into FO and metals (equal parts)
        fo_drops = total_drops / 2
        metal_drops = total_drops / 2

        # Convert to mL
        dpm = settings.drops_per_ml

        recipe = {
            "status": "success",
            "print_size": print_size_inches,
            "coating_area_sq_in": round(area_sq_in, 1),
            "method": method,
            "paper": paper_type,
            "pt_pd_ratio": pt_pd_ratio,
            "ferric_oxalate": {
                "drops": round(fo_drops),
                "ml": round(fo_drops / dpm, 2),
            },
            "platinum": {
                "drops": round(metal_drops * pt_pct),
                "ml": round(metal_drops * pt_pct / dpm, 2),
            },
            "palladium": {
                "drops": round(metal_drops * pd_pct),
                "ml": round(metal_drops * pd_pct / dpm, 2),
            },
            "total_drops": round(total_drops),
            "total_ml": round(total_drops / dpm, 2),
            "contrast_agent": _get_contrast_agent(contrast_goal),
            "instructions": [
                f"1. Measure {round(fo_drops / dpm, 2)} mL ferric oxalate into mixing vessel",
                f"2. Add {round(metal_drops * pt_pct / dpm, 2)} mL platinum solution",
                f"3. Add {round(metal_drops * pd_pct / dpm, 2)} mL palladium solution",
                "4. Mix gently by swirling (do not shake)",
                "5. Apply to paper within 2 minutes of mixing",
                "6. Coat evenly using glass rod or brush",
                "7. Dry completely in the dark before exposing",
            ],
            "estimated_cost_usd": round(
                (fo_drops / dpm) * settings.ferric_oxalate_cost_per_ml
                + (metal_drops * pt_pct / dpm) * settings.platinum_cost_per_ml
                + (metal_drops * pd_pct / dpm) * settings.palladium_cost_per_ml,
                2,
            ),
        }

        return json.dumps(recipe, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def calculate_uv_exposure(
    uv_source: str = "LED 365nm",
    negative_dr: float = 1.5,
    paper_type: str = "Hahnemühle Platinum Rag",
    previous_time_seconds: float = 0,
) -> str:
    """Calculate UV exposure time for Pt/Pd printing.

    Args:
        uv_source: UV light source type.
        negative_dr: Density range of the digital negative.
        paper_type: Paper being used.
        previous_time_seconds: Previous exposure time for iterative adjustment.

    Returns:
        JSON with recommended exposure time and test strip suggestions.
    """
    # Base times by source type (seconds)
    base_times = {
        "LED 365nm": 180,
        "Metal Halide": 420,
        "Mercury Vapor": 360,
        "Sunlight": 600,
    }

    base = base_times.get(uv_source, 300)
    dr_factor = negative_dr / 1.5  # Normalize around 1.5 DR

    recommended = round(base * dr_factor)

    # If previous time provided, suggest refinement bracket
    if previous_time_seconds > 0:
        bracket_center = previous_time_seconds
    else:
        bracket_center = recommended

    return json.dumps(
        {
            "status": "success",
            "uv_source": uv_source,
            "negative_dr": negative_dr,
            "paper_type": paper_type,
            "recommended_seconds": recommended,
            "recommended_formatted": f"{recommended // 60}:{recommended % 60:02d}",
            "test_strip_times": [
                {"seconds": round(bracket_center * f), "label": f"{f:.0%}"}
                for f in [0.6, 0.8, 1.0, 1.2, 1.5]
            ],
            "notes": (
                f"Based on {uv_source} with DR {negative_dr} on {paper_type}. "
                "Adjust based on test strip results. "
                "Start with the recommended time and bracket if needed."
            ),
        },
        indent=2,
    )


def _get_contrast_agent(contrast_goal: str) -> str:
    """Get contrast agent recommendation based on goal.

    Args:
        contrast_goal: Desired contrast level.

    Returns:
        Description of contrast agent to use.
    """
    agents = {
        "low": "None — rely on negative density range for contrast",
        "normal": "2 drops 3% H2O2 per coating",
        "high": "3-4 drops 3% H2O2 or use Solution B (with 0.3% chlorate)",
        "very high": "Use Solution B (0.6% chlorate) or add potassium dichromate (caution: toxic)",
    }
    return agents.get(contrast_goal, agents["normal"])


# ─── ADK Agent Definitions ───


def create_adk_agents() -> dict[str, Any]:
    """Create the ADK agent components.

    Returns a dict of agent objects that can be used for local execution
    or deployed to Vertex AI Agent Engine.

    Returns:
        Dict with keys: calibration_agent, chemistry_agent, print_coach, coordinator.

    Raises:
        ImportError: If google-adk is not installed.
    """
    try:
        from google.adk.agents import LlmAgent
        from google.adk.tools import VertexAiSearchTool
    except ImportError as err:
        raise ImportError(
            "google-cloud-aiplatform[adk] required. "
            "Install with: pip install ptpd-calibration[vertex]"
        ) from err

    settings = get_settings().vertex

    # Vertex AI Search tool for grounding (Layer 1)
    ptpd_search = None
    if settings.search_data_store_id:
        data_store_path = (
            f"projects/{settings.project_id}"
            f"/locations/global"
            f"/collections/default_collection"
            f"/dataStores/{settings.search_data_store_id}"
        )
        ptpd_search = VertexAiSearchTool(data_store_id=data_store_path)

    # Agent 1: Calibration Specialist
    calibration_agent = LlmAgent(
        name="calibration_specialist",
        model=settings.specialist_model,
        instruction="""You are a Pt/Pd calibration specialist. You help photographers
achieve precise, repeatable results in their platinum/palladium printing workflow.

Your expertise:
- Step tablet analysis and density reading interpretation
- Linearization curve generation and refinement
- Digital negative calibration for QTR and PiezoDN
- Scanner calibration for accurate density measurement

When analyzing results, always consider:
- The specific paper and chemistry being used
- Environmental conditions (humidity, temperature)
- The user's target aesthetic (neutral vs warm, high contrast vs full range)

Use the step tablet analysis and curve generation tools to provide
data-driven recommendations. Always explain your reasoning in terms
a working photographer can understand.""",
        description="Calibration workflow specialist for step tablets, curves, and digital negatives",
        tools=[analyze_step_tablet_scan, generate_linearization_curve],
    )

    # Agent 2: Chemistry & Exposure Advisor
    chemistry_agent = LlmAgent(
        name="chemistry_advisor",
        model=settings.specialist_model,
        instruction="""You are a Pt/Pd chemistry and exposure expert. You help
photographers mix coating solutions and calculate exposure times.

Your expertise:
- Bostick-Sullivan coating formulas
- Malde-Ware ammonium method
- Contrast control (A/B ratio, H2O2, dichromate, Na2)
- UV exposure calculation for different light sources
- Paper-chemistry interactions

Safety first: Always mention proper ventilation, gloves, and safe
handling of chemicals (especially dichromate, ferric oxalate).

Be specific with measurements — drops, mL, percentages, and times.
When suggesting recipes, include estimated costs.""",
        description="Chemistry recipes, coating calculations, and UV exposure timing",
        tools=[calculate_chemistry_recipe, calculate_uv_exposure],
    )

    # Agent 3: Print Coach (grounded in knowledge base)
    print_coach_tools = []
    if ptpd_search:
        print_coach_tools = [ptpd_search]

    print_coach = LlmAgent(
        name="print_coach",
        model=settings.specialist_model,
        instruction="""You are an expert Pt/Pd printing coach. You draw on a deep
knowledge base of platinum/palladium printing literature, paper profiles,
chemistry references, troubleshooting guides, and calibration documentation.

Your role:
- Answer any question about Pt/Pd printing with cited, grounded answers
- Troubleshoot problems by referencing known solutions
- Recommend papers, chemistry, and techniques based on the user's goals
- Teach concepts from basics to advanced (zone system, contrast control, etc.)
- Reference specific documents and guides when giving advice

Always ground your answers in the knowledge base. If you're not sure,
say so — don't fabricate Pt/Pd printing advice, as incorrect guidance
can waste expensive materials.

Be encouraging with beginners but honest about difficulty levels.""",
        description="Grounded Pt/Pd printing knowledge assistant with access to literature and guides",
        tools=print_coach_tools,
    )

    # Coordinator Agent
    coordinator = LlmAgent(
        name="darkroom_assistant",
        model=settings.coordinator_model,
        instruction="""You are the Darkroom Assistant, coordinating a team of
Pt/Pd printing specialists to help photographers achieve their best work.

Your team:
- calibration_specialist: Step tablet analysis, curve generation, digital negative calibration
- chemistry_advisor: Coating recipes, chemistry mixing, UV exposure calculations
- print_coach: Deep knowledge of Pt/Pd printing literature, troubleshooting, paper profiles

For each user request:
1. Determine which specialist(s) are needed
2. Delegate appropriately
3. Synthesize their findings into clear, actionable advice
4. If the user uploads an image, route to the appropriate visual analysis

Keep responses practical and specific. These are working photographers
who need actionable guidance, not theory lectures.

If a user is new to Pt/Pd printing, be encouraging and start with basics.
If they're experienced, match their level of technical detail.

Always prioritize safety when discussing chemistry.""",
        description="Main coordinator for the Pt/Pd Darkroom AI Assistant",
        sub_agents=[calibration_agent, chemistry_agent, print_coach],
    )

    return {
        "calibration_agent": calibration_agent,
        "chemistry_agent": chemistry_agent,
        "print_coach": print_coach,
        "coordinator": coordinator,
    }


def create_darkroom_coordinator() -> Any:
    """Create the top-level Darkroom Coordinator agent.

    Convenience function that returns just the coordinator agent,
    ready for local execution or Agent Engine deployment.

    Returns:
        LlmAgent coordinator with all sub-agents configured.
    """
    agents = create_adk_agents()
    return agents["coordinator"]


def deploy_to_agent_engine(
    project_id: str | None = None,
    location: str | None = None,
    staging_bucket: str | None = None,
) -> Any:
    """Deploy the Darkroom Coordinator to Vertex AI Agent Engine.

    This creates a managed deployment with persistent sessions,
    Memory Bank, and production-grade infrastructure.

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region.
        staging_bucket: GCS bucket for staging deployment artifacts.

    Returns:
        Deployed Agent Engine resource.

    Raises:
        ImportError: If required packages are not installed.
    """
    try:
        import vertexai
        from vertexai.agent_engines import AdkApp
    except ImportError as err:
        raise ImportError(
            "google-cloud-aiplatform[agent_engines,adk] required. "
            "Install with: pip install ptpd-calibration[vertex]"
        ) from err

    settings = get_settings().vertex
    project_id = project_id or settings.project_id
    location = location or settings.location
    staging_bucket = staging_bucket or settings.staging_bucket

    if not project_id:
        raise ValueError("Google Cloud project ID required. Set PTPD_VERTEX_PROJECT_ID.")
    if not staging_bucket:
        raise ValueError("Staging bucket required. Set PTPD_VERTEX_STAGING_BUCKET.")

    client = vertexai.Client(project=project_id, location=location)

    coordinator = create_darkroom_coordinator()
    app = AdkApp(agent=coordinator)

    remote_agent = client.agent_engines.create(
        agent=app,
        config={
            "requirements": [
                "google-cloud-aiplatform[agent_engines,adk]",
                "numpy>=1.24.0",
                "scipy>=1.11.0",
                "Pillow>=10.0.0",
                "pydantic>=2.5.0",
                "pydantic-settings>=2.1.0",
            ],
            "staging_bucket": staging_bucket,
        },
    )

    return remote_agent
