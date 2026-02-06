"""
Gemini Vision tools for multimodal Pt/Pd print analysis.

Provides image-understanding capabilities for:
- Step tablet scan analysis (density reading, quality assessment)
- Print quality evaluation from photographs
- Coating defect detection
- Paper texture classification
- Before/after print comparison

These functions are designed to be used as ADK agent tools.

Usage:
    from ptpd_calibration.vertex.vision import GeminiVisionAnalyzer

    analyzer = GeminiVisionAnalyzer(project_id="my-project")
    result = analyzer.analyze_step_tablet("scan.tiff")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ptpd_calibration.config import get_settings


@dataclass
class VisionAnalysisResult:
    """Result from a Gemini Vision analysis.

    Args:
        analysis_type: Type of analysis performed.
        raw_response: Raw text response from Gemini.
        structured_data: Parsed JSON data if available.
        confidence: Overall confidence score (0.0 to 1.0).
        recommendations: List of actionable recommendations.
    """

    analysis_type: str
    raw_response: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)


class GeminiVisionAnalyzer:
    """Multimodal analyzer for Pt/Pd prints using Gemini Vision.

    Args:
        project_id: Google Cloud project ID.
        location: Google Cloud region.
        model: Gemini model to use for vision tasks.
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings().vertex
        self.project_id = project_id or settings.project_id
        self.location = location or settings.location
        self.model = model or settings.vision_model
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as err:
                raise ImportError(
                    "google-genai required. Install with: pip install ptpd-calibration[vertex]"
                ) from err

            self._client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        return self._client

    def analyze_step_tablet(
        self,
        image_path: str,
        tablet_type: str = "Stouffer 21-step",
    ) -> VisionAnalysisResult:
        """Analyze a scanned step tablet image using Gemini Vision.

        Args:
            image_path: Path to the scanned step tablet image (TIFF, PNG, JPEG).
            tablet_type: Type of step tablet (Stouffer 21/31/41 step).

        Returns:
            VisionAnalysisResult with density readings and quality assessment.
        """
        from google.genai import types

        client = self._get_client()
        image_data, mime_type = _load_image(image_path)

        prompt = f"""Analyze this scanned {tablet_type} step tablet from a platinum/palladium print.

For each visible step, estimate:
1. The step number (1 = lightest/paper white, highest = darkest/Dmax)
2. The approximate optical density (0.0 for paper white to ~1.8 for Dmax)
3. Whether the step is well-defined or shows issues

Also assess overall print quality:
- Is the full tonal range represented?
- Are there any blocked highlights or shadows?
- Is the coating even across all steps?
- Any signs of fogging, staining, or contamination?

Return as structured JSON with keys:
- "steps": list of {{"step": int, "density": float, "quality": str}}
- "overall_quality": {{"score": float (1-10), "description": str}}
- "dmin": float
- "dmax": float
- "density_range": float
- "issues": list of str
- "recommendations": list of str"""

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(prompt),
                    ],
                )
            ],
        )

        return _parse_vision_response(response.text or "", "step_tablet_analysis")

    def evaluate_print_quality(
        self,
        image_path: str,
        paper_type: str = "unknown",
        chemistry: str = "unknown",
    ) -> VisionAnalysisResult:
        """Evaluate a finished Pt/Pd print from a photograph.

        Args:
            image_path: Path to photograph of the print.
            paper_type: Paper used (e.g., "Hahnemühle Platinum Rag").
            chemistry: Chemistry used (e.g., "50/50 Pt/Pd, Malde-Ware").

        Returns:
            VisionAnalysisResult with quality scores and recommendations.
        """
        from google.genai import types

        client = self._get_client()
        image_data, mime_type = _load_image(image_path)

        prompt = f"""Evaluate this platinum/palladium print.
Paper: {paper_type}
Chemistry: {chemistry}

Assess:
1. **Tonal range** (1-10): Full range from paper white to deep Dmax?
2. **Highlight quality** (1-10): Clean separation, no blocking?
3. **Shadow quality** (1-10): Deep blacks, detail retention?
4. **Midtone smoothness** (1-10): Gradual transitions, no banding?
5. **Coating quality** (1-10): Even coverage, no streaks or defects?
6. **Overall impression** (1-10): Artistic and technical merit

For any score below 7, provide specific improvement suggestions
referencing Pt/Pd printing techniques.

Return as JSON with keys:
- "scores": {{"tonal_range": float, "highlight_quality": float, "shadow_quality": float, "midtone_smoothness": float, "coating_quality": float, "overall_impression": float}}
- "overall_score": float
- "strengths": list of str
- "improvements": list of str
- "next_steps": list of str"""

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(prompt),
                    ],
                )
            ],
        )

        return _parse_vision_response(response.text or "", "print_quality_evaluation")

    def diagnose_print_problem(
        self,
        image_path: str,
        problem_description: str = "",
    ) -> VisionAnalysisResult:
        """Diagnose a problem visible in a Pt/Pd print photograph.

        Args:
            image_path: Path to photograph showing the problem.
            problem_description: User's description of the issue.

        Returns:
            VisionAnalysisResult with diagnosis and fix suggestions.
        """
        from google.genai import types

        client = self._get_client()
        image_data, mime_type = _load_image(image_path)

        prompt = f"""A platinum/palladium printer is having this problem:
"{problem_description}"

Look at this image of their print and diagnose the issue.

Consider these common Pt/Pd problems:
- Highlight blocking or muddiness
- Weak Dmax / thin shadows
- Grainy or mottled appearance
- Uneven coating (streaks, puddles)
- Orange/brown staining (clearing issues)
- Fogging or unwanted exposure
- Paper-related artifacts
- Chemistry contamination signs

Provide:
1. Most likely diagnosis (with confidence level)
2. Root cause explanation
3. Step-by-step fix
4. Prevention for future prints

Return as JSON with keys:
- "diagnosis": str
- "confidence": float (0.0 to 1.0)
- "root_cause": str
- "fix_steps": list of str
- "prevention": list of str
- "alternative_diagnoses": list of {{"diagnosis": str, "confidence": float}}"""

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(prompt),
                    ],
                )
            ],
        )

        return _parse_vision_response(response.text or "", "print_problem_diagnosis")

    def compare_prints(
        self,
        image_path_before: str,
        image_path_after: str,
        context: str = "",
    ) -> VisionAnalysisResult:
        """Compare two prints to evaluate calibration improvement.

        Args:
            image_path_before: Path to the "before" print image.
            image_path_after: Path to the "after" print image.
            context: Description of what changed between prints.

        Returns:
            VisionAnalysisResult with comparison analysis.
        """
        from google.genai import types

        client = self._get_client()
        before_data, before_mime = _load_image(image_path_before)
        after_data, after_mime = _load_image(image_path_after)

        prompt = f"""Compare these two platinum/palladium prints (before and after calibration adjustment).
Context: {context or 'No additional context provided.'}

The first image is BEFORE and the second is AFTER.

Analyze:
1. What improved?
2. What remained the same?
3. What got worse (if anything)?
4. Overall, is the calibration moving in the right direction?

Return as JSON with keys:
- "improvements": list of str
- "unchanged": list of str
- "regressions": list of str
- "overall_direction": str ("better", "same", "worse")
- "overall_assessment": str
- "next_recommendations": list of str"""

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=before_data, mime_type=before_mime),
                        types.Part.from_bytes(data=after_data, mime_type=after_mime),
                        types.Part.from_text(prompt),
                    ],
                )
            ],
        )

        return _parse_vision_response(response.text or "", "print_comparison")

    def classify_paper(
        self,
        image_path: str,
    ) -> VisionAnalysisResult:
        """Classify paper type from a close-up photograph.

        Args:
            image_path: Path to close-up photograph of paper surface.

        Returns:
            VisionAnalysisResult with paper classification and recommendations.
        """
        from google.genai import types

        client = self._get_client()
        image_data, mime_type = _load_image(image_path)

        prompt = """Analyze this close-up photograph of paper used for platinum/palladium printing.

Identify:
1. Likely paper type or brand
2. Surface texture (smooth, slight tooth, medium tooth, rough)
3. Weight estimate (light, medium, heavy)
4. Fiber type (cotton, cellulose, blend)
5. Sizing characteristics (internally sized, surface sized, both)

Provide printing recommendations:
- Coating method (rod vs brush)
- Estimated coating volume for 8x10
- Humidity sensitivity
- Expected dMax potential

Return as JSON with keys:
- "paper_identification": str
- "confidence": float
- "characteristics": {{"texture": str, "weight": str, "fiber": str, "sizing": str}}
- "printing_recommendations": {{"coating_method": str, "coating_volume_ml": float, "humidity_sensitivity": str, "dmax_potential": str}}"""

        response = client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(prompt),
                    ],
                )
            ],
        )

        return _parse_vision_response(response.text or "", "paper_classification")


# Module-level convenience functions for ADK tool integration

def analyze_step_tablet(
    image_path: str,
    tablet_type: str = "Stouffer 21-step",
) -> str:
    """Analyze a scanned step tablet image. ADK tool wrapper.

    Args:
        image_path: Path to the scanned step tablet image.
        tablet_type: Type of step tablet (Stouffer 21/31/41 step).

    Returns:
        JSON string with density readings and quality assessment.
    """
    analyzer = GeminiVisionAnalyzer()
    result = analyzer.analyze_step_tablet(image_path, tablet_type)
    return json.dumps(result.structured_data, indent=2) if result.structured_data else result.raw_response


def evaluate_print_quality(
    image_path: str,
    paper_type: str = "unknown",
    chemistry: str = "unknown",
) -> str:
    """Evaluate a finished Pt/Pd print. ADK tool wrapper.

    Args:
        image_path: Path to photograph of the print.
        paper_type: Paper used.
        chemistry: Chemistry used.

    Returns:
        JSON string with quality scores and recommendations.
    """
    analyzer = GeminiVisionAnalyzer()
    result = analyzer.evaluate_print_quality(image_path, paper_type, chemistry)
    return json.dumps(result.structured_data, indent=2) if result.structured_data else result.raw_response


def diagnose_print_problem(
    image_path: str,
    problem_description: str = "",
) -> str:
    """Diagnose a problem in a Pt/Pd print. ADK tool wrapper.

    Args:
        image_path: Path to photograph showing the problem.
        problem_description: User's description of the issue.

    Returns:
        JSON string with diagnosis and fix suggestions.
    """
    analyzer = GeminiVisionAnalyzer()
    result = analyzer.diagnose_print_problem(image_path, problem_description)
    return json.dumps(result.structured_data, indent=2) if result.structured_data else result.raw_response


# Internal helpers

def _load_image(image_path: str) -> tuple[bytes, str]:
    """Load image data and determine MIME type.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (image_bytes, mime_type).

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image format is not supported.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_map = {
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    suffix = path.suffix.lower()
    mime_type = mime_map.get(suffix)
    if not mime_type:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: {', '.join(mime_map.keys())}"
        )

    return path.read_bytes(), mime_type


def _parse_vision_response(
    response_text: str,
    analysis_type: str,
) -> VisionAnalysisResult:
    """Parse Gemini Vision response into structured result.

    Args:
        response_text: Raw text response from Gemini.
        analysis_type: Type of analysis for the result.

    Returns:
        VisionAnalysisResult with parsed data.
    """
    structured_data: dict[str, Any] = {}
    confidence = 0.0
    recommendations: list[str] = []

    # Try to extract JSON from the response
    try:
        # Handle responses that wrap JSON in markdown code blocks
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        structured_data = json.loads(text.strip())

        # Extract confidence if present
        if "confidence" in structured_data:
            confidence = float(structured_data["confidence"])
        elif "overall_quality" in structured_data:
            quality = structured_data["overall_quality"]
            if isinstance(quality, dict) and "score" in quality:
                confidence = float(quality["score"]) / 10.0

        # Extract recommendations
        for key in ("recommendations", "next_steps", "improvements", "fix_steps"):
            if key in structured_data and isinstance(structured_data[key], list):
                recommendations.extend(structured_data[key])
                break

    except (json.JSONDecodeError, ValueError, KeyError):
        # If JSON parsing fails, keep raw response
        pass

    return VisionAnalysisResult(
        analysis_type=analysis_type,
        raw_response=response_text,
        structured_data=structured_data,
        confidence=confidence,
        recommendations=recommendations,
    )
