"""
Multi-Modal AI Assistant for Platinum-Palladium printing.

Provides conversational AI assistance with vision-language models,
tool use, and RAG-based knowledge retrieval. Supports Claude and GPT-4V
models with graceful fallback and comprehensive error handling.
"""

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, Field

from ptpd_calibration.deep_learning.config import MultiModalSettings
from ptpd_calibration.deep_learning.models import (
    ImageAnalysis,
    MultiModalResponse,
    ToolCall,
)
from ptpd_calibration.deep_learning.types import (
    AssistantMode,
    ToolType,
    VisionLanguageModel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Base Classes
# =============================================================================


class BaseTool(ABC):
    """Base class for assistant tools."""

    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize the tool.

        Args:
            timeout_seconds: Maximum execution time for the tool
        """
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            dict: Tool execution result
        """
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Get the tool's JSON schema for the LLM.

        Returns:
            dict: JSON schema describing the tool
        """
        pass


class ExposureCalculatorTool(BaseTool):
    """Tool for calculating UV exposure times."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "exposure_calculator",
            "description": "Calculate UV exposure time based on print parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_density": {
                        "type": "number",
                        "description": "Target density value (0.0-3.0)",
                    },
                    "paper_type": {
                        "type": "string",
                        "description": "Paper type (e.g., Arches Platine)",
                    },
                    "chemistry_ratio": {
                        "type": "number",
                        "description": "Platinum to Palladium ratio (0.0-1.0)",
                    },
                    "uv_source": {
                        "type": "string",
                        "description": "UV light source type",
                    },
                },
                "required": ["target_density", "paper_type"],
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute exposure calculation."""
        # Placeholder implementation - would integrate with actual UV prediction
        target_density = kwargs.get("target_density", 1.8)
        paper_type = kwargs.get("paper_type", "Arches Platine")
        chemistry_ratio = kwargs.get("chemistry_ratio", 0.5)

        # Simple heuristic calculation
        base_time = target_density * 60  # seconds
        paper_factor = 1.2 if "platine" in paper_type.lower() else 1.0
        chemistry_factor = 1.0 + (chemistry_ratio * 0.3)

        exposure_seconds = base_time * paper_factor * chemistry_factor

        return {
            "exposure_seconds": exposure_seconds,
            "exposure_minutes": exposure_seconds / 60,
            "parameters": kwargs,
            "recommendations": [
                "Start with a test strip",
                "Ensure consistent coating",
                "Monitor UV source distance",
            ],
        }


class ChemistryCalculatorTool(BaseTool):
    """Tool for calculating chemistry formulas."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "chemistry_calculator",
            "description": "Calculate chemistry formulas and ratios",
            "parameters": {
                "type": "object",
                "properties": {
                    "platinum_ratio": {
                        "type": "number",
                        "description": "Platinum ratio (0.0-1.0)",
                    },
                    "total_volume_ml": {
                        "type": "number",
                        "description": "Total volume in milliliters",
                    },
                    "coating_area_cm2": {
                        "type": "number",
                        "description": "Coating area in square centimeters",
                    },
                },
                "required": ["platinum_ratio", "total_volume_ml"],
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute chemistry calculation."""
        platinum_ratio = kwargs.get("platinum_ratio", 0.5)
        total_volume = kwargs.get("total_volume_ml", 10.0)
        coating_area = kwargs.get("coating_area_cm2")

        platinum_ml = total_volume * platinum_ratio
        palladium_ml = total_volume * (1 - platinum_ratio)

        result = {
            "platinum_ml": platinum_ml,
            "palladium_ml": palladium_ml,
            "total_ml": total_volume,
            "ratio": f"{int(platinum_ratio*100)}:{int((1-platinum_ratio)*100)}",
        }

        if coating_area:
            ml_per_cm2 = total_volume / coating_area
            result["coverage"] = {
                "ml_per_cm2": ml_per_cm2,
                "estimated_prints": int(100 / coating_area) if coating_area > 0 else 0,
            }

        return result


class CurveAdjustmentTool(BaseTool):
    """Tool for curve adjustments."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "curve_adjustment",
            "description": "Suggest curve adjustments for tonal corrections",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_description": {
                        "type": "string",
                        "description": "Description of tonal issue",
                    },
                    "affected_zones": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Affected tonal zones",
                    },
                },
                "required": ["issue_description"],
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute curve adjustment suggestion."""
        issue = kwargs.get("issue_description", "").lower()
        affected_zones = kwargs.get("affected_zones", [])

        suggestions = []

        if "shadow" in issue or "dark" in issue:
            suggestions.append("Lift shadow values in zones 0-III")
            suggestions.append("Reduce exposure time by 5-10%")

        if "highlight" in issue or "light" in issue or "blown" in issue:
            suggestions.append("Pull down highlight values in zones VII-X")
            suggestions.append("Increase exposure time by 5-10%")

        if "contrast" in issue:
            if "low" in issue or "flat" in issue:
                suggestions.append("Steepen curve in midtone region")
            else:
                suggestions.append("Flatten curve to reduce contrast")

        return {
            "issue": kwargs.get("issue_description"),
            "affected_zones": affected_zones,
            "suggestions": suggestions,
            "curve_adjustments": {
                "shadows": "adjust if needed",
                "midtones": "primary adjustment region",
                "highlights": "adjust if needed",
            },
        }


class DefectDiagnosisTool(BaseTool):
    """Tool for defect diagnosis."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "defect_diagnosis",
            "description": "Diagnose print defects and suggest remediation",
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_description": {
                        "type": "string",
                        "description": "Description of the defect",
                    },
                    "defect_location": {
                        "type": "string",
                        "description": "Location of defect (e.g., center, edge)",
                    },
                },
                "required": ["defect_description"],
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute defect diagnosis."""
        description = kwargs.get("defect_description", "").lower()
        location = kwargs.get("defect_location", "unknown")

        defect_type = "unknown"
        causes = []
        remediation = []

        if "streak" in description or "brush" in description:
            defect_type = "brush_marks"
            causes = ["Uneven coating", "Brush quality", "Coating technique"]
            remediation = [
                "Use high-quality hake brush",
                "Apply coating in smooth, even strokes",
                "Ensure paper is properly sized",
            ]

        elif "spot" in description or "stain" in description:
            defect_type = "staining"
            causes = ["Developer residue", "Water quality", "Contamination"]
            remediation = [
                "Use distilled water",
                "Ensure thorough clearing",
                "Check for paper contamination",
            ]

        elif "uneven" in description or "mottl" in description:
            defect_type = "uneven_coating"
            causes = ["Paper sizing", "Humidity", "Coating technique"]
            remediation = [
                "Control humidity (40-60%)",
                "Ensure proper paper sizing",
                "Apply multiple thin coats",
            ]

        else:
            causes = ["Multiple possible causes"]
            remediation = ["Examine under magnification", "Review process steps"]

        return {
            "defect_type": defect_type,
            "location": location,
            "likely_causes": causes,
            "remediation": remediation,
            "severity": "requires_assessment",
        }


class RecipeLookupTool(BaseTool):
    """Tool for recipe lookup."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "recipe_lookup",
            "description": "Look up printing recipes by criteria",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_type": {
                        "type": "string",
                        "description": "Paper type to search for",
                    },
                    "tone_preference": {
                        "type": "string",
                        "description": "Tone preference (warm, neutral, cool)",
                    },
                    "subject_type": {
                        "type": "string",
                        "description": "Subject type (portrait, landscape, etc.)",
                    },
                },
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute recipe lookup."""
        # Placeholder - would integrate with actual recipe database
        paper_type = kwargs.get("paper_type", "any")
        tone = kwargs.get("tone_preference", "neutral")

        recipes = [
            {
                "name": f"Classic {tone.capitalize()} on {paper_type}",
                "platinum_ratio": 0.5 if tone == "neutral" else 0.7,
                "exposure_minutes": 2.5,
                "developer": "Potassium Oxalate",
                "rating": 4.5,
            }
        ]

        return {"recipes": recipes, "total_found": len(recipes), "criteria": kwargs}


class QualityAssessmentTool(BaseTool):
    """Tool for quality assessment."""

    def get_schema(self) -> dict[str, Any]:
        """Get the tool schema."""
        return {
            "name": "quality_assessment",
            "description": "Assess print quality and provide feedback",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "description": "Quality aspect to assess",
                    }
                },
            },
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        """Execute quality assessment."""
        aspect = kwargs.get("aspect", "overall")

        return {
            "aspect": aspect,
            "assessment": "Quality assessment requires image analysis",
            "recommendations": [
                "Ensure even exposure",
                "Check coating consistency",
                "Verify chemistry freshness",
            ],
        }


# =============================================================================
# RAG Retriever
# =============================================================================


class RAGRetriever:
    """Retrieval-augmented generation for knowledge base access."""

    def __init__(
        self,
        knowledge_base_path: Optional[Path] = None,
        chunk_size: int = 512,
        top_k: int = 5,
    ):
        """
        Initialize the RAG retriever.

        Args:
            knowledge_base_path: Path to knowledge base documents
            chunk_size: Size of document chunks
            top_k: Number of top documents to retrieve
        """
        self.knowledge_base_path = knowledge_base_path
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embeddings: dict[str, list[float]] = {}
        self.documents: dict[str, str] = {}

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve (overrides default)

        Returns:
            list: Retrieved document chunks
        """
        k = top_k or self.top_k

        # Placeholder implementation - would use actual vector search
        # In production, this would use:
        # - Embedding model (e.g., sentence-transformers)
        # - Vector database (e.g., FAISS, Chroma, Pinecone)
        # - Semantic search

        if not self.documents:
            return self._get_default_knowledge()[:k]

        # Simplified retrieval for demonstration
        return list(self.documents.values())[:k]

    def _get_default_knowledge(self) -> list[str]:
        """Get default knowledge snippets."""
        return [
            "Platinum-palladium printing uses iron salts sensitized with noble metals.",
            "Typical exposure times range from 2-5 minutes depending on UV source.",
            "Chemistry ratio affects both tone and exposure requirements.",
            "Paper sizing is critical for even coating and good dmax.",
            "Clearing removes iron salts and ensures print permanence.",
            "Humidity should be controlled between 40-60% for best results.",
            "Potassium oxalate is the most common developer for Pt/Pd printing.",
        ]


# =============================================================================
# Multi-Modal Assistant
# =============================================================================


class MultiModalAssistant:
    """
    Multi-modal AI assistant for platinum-palladium printing.

    Provides conversational assistance with vision-language capabilities,
    tool use, and knowledge retrieval. Supports multiple LLM backends
    with graceful fallback.
    """

    def __init__(self, settings: Optional[MultiModalSettings] = None):
        """
        Initialize the multi-modal assistant.

        Args:
            settings: Assistant settings (uses defaults if not provided)
        """
        self.settings = settings or MultiModalSettings()
        self.conversation_history: list[dict[str, Any]] = []
        self.conversation_id = uuid4()
        self.tools = self._initialize_tools()
        self.rag_retriever = self._initialize_rag()

        # Check for API clients
        self._anthropic_client = None
        self._openai_client = None

        try:
            if self.settings.anthropic_api_key:
                try:
                    import anthropic

                    self._anthropic_client = anthropic.AsyncAnthropic(
                        api_key=self.settings.anthropic_api_key
                    )
                except ImportError:
                    logger.warning("anthropic package not available")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {e}")

        try:
            if self.settings.openai_api_key:
                try:
                    import openai

                    self._openai_client = openai.AsyncOpenAI(
                        api_key=self.settings.openai_api_key
                    )
                except ImportError:
                    logger.warning("openai package not available")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")

    def _initialize_tools(self) -> dict[str, BaseTool]:
        """Initialize available tools based on settings."""
        all_tools = {
            "exposure_calculator": ExposureCalculatorTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
            "chemistry_calculator": ChemistryCalculatorTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
            "curve_adjustment": CurveAdjustmentTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
            "defect_diagnosis": DefectDiagnosisTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
            "recipe_lookup": RecipeLookupTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
            "quality_assessment": QualityAssessmentTool(
                timeout_seconds=self.settings.tool_timeout_seconds
            ),
        }

        # Filter to enabled tools only
        enabled_tools = {
            name: tool
            for name, tool in all_tools.items()
            if name in self.settings.enabled_tools
        }

        logger.info(f"Initialized {len(enabled_tools)} tools: {list(enabled_tools.keys())}")
        return enabled_tools

    def _initialize_rag(self) -> RAGRetriever:
        """Initialize RAG retriever if enabled."""
        if not self.settings.use_rag:
            return None

        return RAGRetriever(
            chunk_size=self.settings.rag_chunk_size, top_k=self.settings.rag_top_k
        )

    async def chat(
        self,
        message: str,
        images: Optional[list[np.ndarray]] = None,
        mode: AssistantMode = AssistantMode.CHAT,
        stream: Optional[bool] = None,
    ) -> MultiModalResponse:
        """
        Chat with the assistant.

        Args:
            message: User message
            images: Optional images to include
            mode: Assistant mode
            stream: Whether to stream response (uses settings if None)

        Returns:
            MultiModalResponse: Assistant response
        """
        start_time = datetime.utcnow()
        stream_enabled = stream if stream is not None else self.settings.stream_response

        # Add message to history
        self.conversation_history.append({"role": "user", "content": message})

        # Retrieve context if RAG enabled
        rag_sources = []
        if self.rag_retriever:
            rag_sources = await self.rag_retriever.retrieve(message)

        # Analyze images if provided
        image_analyses = []
        if images:
            for idx, img in enumerate(images):
                analysis = await self._analyze_image_simple(img, idx)
                image_analyses.append(analysis)

        # Prepare context
        context = self._prepare_context(message, rag_sources, image_analyses)

        # Get LLM response
        try:
            if stream_enabled:
                response_text = ""
                async for chunk in self._stream_llm_response(context, images):
                    response_text += chunk
            else:
                response_text = await self._get_llm_response(context, images)
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            response_text = self._get_fallback_response(message)

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response_text})

        # Trim history if needed
        if len(self.conversation_history) > self.settings.max_conversation_history * 2:
            self.conversation_history = self.conversation_history[
                -self.settings.max_conversation_history * 2 :
            ]

        # Build response
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return MultiModalResponse(
            response_text=response_text,
            response_type=mode.value,
            tool_calls=[],
            num_tool_calls=0,
            image_analyses=image_analyses,
            images_analyzed=len(images) if images else 0,
            rag_sources_used=rag_sources,
            conversation_id=self.conversation_id,
            turn_number=len(self.conversation_history) // 2,
            input_tokens=len(message.split()),
            output_tokens=len(response_text.split()),
            inference_time_ms=inference_time,
            device_used="api",
            model_version=self.settings.vision_language_model.value,
        )

    async def analyze_image(
        self,
        image: np.ndarray,
        query: Optional[str] = None,
    ) -> ImageAnalysis:
        """
        Analyze an image with the vision model.

        Args:
            image: Image array to analyze
            query: Optional specific query about the image

        Returns:
            ImageAnalysis: Image analysis result
        """
        default_query = (
            "Analyze this platinum-palladium print. "
            "Describe the tonal range, identify any defects, "
            "and provide recommendations for improvement."
        )

        analysis_query = query or default_query

        # Encode image
        image_b64 = self._encode_image(image)

        # Get analysis from LLM
        try:
            description = await self._get_image_description(image_b64, analysis_query)
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            description = "Unable to analyze image due to API error."

        return ImageAnalysis(
            image_index=0,
            description=description,
            detected_issues=[],
            recommendations=[],
            extracted_data={},
        )

    async def execute_tool(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> ToolCall:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            ToolCall: Tool execution result
        """
        start_time = datetime.utcnow()

        if tool_name not in self.tools:
            return ToolCall(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=None,
                execution_time_ms=0.0,
                success=False,
                error=f"Tool '{tool_name}' not found or not enabled",
            )

        tool = self.tools[tool_name]

        try:
            # Execute with timeout
            output = await asyncio.wait_for(
                tool.execute(**tool_input),
                timeout=self.settings.tool_timeout_seconds,
            )
            success = True
            error = None
        except asyncio.TimeoutError:
            output = None
            success = False
            error = f"Tool execution timed out after {self.settings.tool_timeout_seconds}s"
        except Exception as e:
            output = None
            success = False
            error = str(e)
            logger.error(f"Error executing tool {tool_name}: {e}")

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ToolCall(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output,
            execution_time_ms=execution_time,
            success=success,
            error=error,
        )

    def _prepare_context(
        self,
        message: str,
        rag_sources: list[str],
        image_analyses: list[ImageAnalysis],
    ) -> str:
        """Prepare the context for the LLM."""
        context_parts = []

        # Add RAG context
        if rag_sources:
            context_parts.append("Relevant knowledge:")
            for source in rag_sources[:3]:  # Limit to top 3
                context_parts.append(f"- {source}")
            context_parts.append("")

        # Add image analysis context
        if image_analyses:
            context_parts.append("Image analysis:")
            for analysis in image_analyses:
                context_parts.append(f"- Image {analysis.image_index}: {analysis.description[:100]}")
            context_parts.append("")

        # Add user message
        context_parts.append(f"User: {message}")

        return "\n".join(context_parts)

    async def _get_llm_response(
        self, context: str, images: Optional[list[np.ndarray]] = None
    ) -> str:
        """Get response from LLM (non-streaming)."""
        # Try primary model
        try:
            if self._is_claude_model(self.settings.vision_language_model):
                if self._anthropic_client:
                    return await self._get_claude_response(context, images)
            elif self._is_openai_model(self.settings.vision_language_model):
                if self._openai_client:
                    return await self._get_openai_response(context, images)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")

        # Try fallback model
        try:
            if self._is_claude_model(self.settings.fallback_model):
                if self._anthropic_client:
                    return await self._get_claude_response(context, images)
            elif self._is_openai_model(self.settings.fallback_model):
                if self._openai_client:
                    return await self._get_openai_response(context, images)
        except Exception as e:
            logger.error(f"Fallback model failed: {e}")

        # Final fallback
        return self._get_fallback_response(context)

    async def _stream_llm_response(
        self, context: str, images: Optional[list[np.ndarray]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response from LLM."""
        # Simplified - in production would stream from API
        response = await self._get_llm_response(context, images)
        yield response

    async def _get_claude_response(
        self, context: str, images: Optional[list[np.ndarray]] = None
    ) -> str:
        """Get response from Claude."""
        if not self._anthropic_client:
            raise ValueError("Anthropic client not initialized")

        # Prepare messages
        content = []

        # Add images if provided
        if images:
            for img in images[: self.settings.max_images_per_request]:
                img_b64 = self._encode_image(img)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                })

        # Add text
        content.append({"type": "text", "text": context})

        # Call API
        response = await self._anthropic_client.messages.create(
            model=self._get_model_id(self.settings.vision_language_model),
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text

    async def _get_openai_response(
        self, context: str, images: Optional[list[np.ndarray]] = None
    ) -> str:
        """Get response from OpenAI."""
        if not self._openai_client:
            raise ValueError("OpenAI client not initialized")

        # Prepare messages
        content = [{"type": "text", "text": context}]

        # Add images if provided
        if images:
            for img in images[: self.settings.max_images_per_request]:
                img_b64 = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": self.settings.image_detail,
                    },
                })

        # Call API
        response = await self._openai_client.chat.completions.create(
            model=self._get_model_id(self.settings.vision_language_model),
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            messages=[{"role": "user", "content": content}],
        )

        return response.choices[0].message.content

    async def _get_image_description(
        self, image_b64: str, query: str
    ) -> str:
        """Get image description from vision model."""
        try:
            if self._anthropic_client:
                response = await self._anthropic_client.messages.create(
                    model=self._get_model_id(self.settings.vision_language_model),
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": query},
                        ],
                    }],
                )
                return response.content[0].text
            elif self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model=self._get_model_id(self.settings.vision_language_model),
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }],
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting image description: {e}")

        return "Unable to analyze image."

    async def _analyze_image_simple(
        self, image: np.ndarray, index: int
    ) -> ImageAnalysis:
        """Simple image analysis without API call."""
        # Basic analysis based on image statistics
        mean_value = np.mean(image)
        std_value = np.std(image)

        description = f"Image {index}: Mean intensity {mean_value:.2f}, std {std_value:.2f}"

        return ImageAnalysis(
            image_index=index,
            description=description,
            detected_issues=[],
            recommendations=[],
            extracted_data={"mean": float(mean_value), "std": float(std_value)},
        )

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image as base64 string.

        Args:
            image: Image array

        Returns:
            str: Base64 encoded image
        """
        try:
            from PIL import Image

            # Resize if needed
            if image.shape[0] > self.settings.max_image_size or image.shape[1] > self.settings.max_image_size:
                max_dim = max(image.shape[:2])
                scale = self.settings.max_image_size / max_dim
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

                img = Image.fromarray(image)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                image = np.array(img)

            # Convert to PIL Image
            if len(image.shape) == 2:
                img = Image.fromarray(image, mode="L")
            else:
                img = Image.fromarray(image, mode="RGB")

            # Encode to base64
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return img_b64
        except ImportError:
            logger.error("PIL not available for image encoding")
            return ""

    def _get_fallback_response(self, context: str) -> str:
        """Generate a fallback response when API is unavailable."""
        return (
            "I apologize, but I'm currently unable to access the vision-language "
            "models. This could be due to missing API keys or network issues. "
            "However, I can still help with general platinum-palladium printing "
            "questions based on my built-in knowledge. Please check your API "
            "configuration or try again later."
        )

    @staticmethod
    def _is_claude_model(model: VisionLanguageModel) -> bool:
        """Check if model is a Claude model."""
        return "claude" in model.value.lower()

    @staticmethod
    def _is_openai_model(model: VisionLanguageModel) -> bool:
        """Check if model is an OpenAI model."""
        return "gpt" in model.value.lower()

    @staticmethod
    def _get_model_id(model: VisionLanguageModel) -> str:
        """Get the API model ID from the enum value."""
        # Map enum values to actual API model IDs
        model_map = {
            VisionLanguageModel.CLAUDE_3_OPUS: "claude-3-opus-20240229",
            VisionLanguageModel.CLAUDE_3_SONNET: "claude-3-sonnet-20240229",
            VisionLanguageModel.CLAUDE_3_HAIKU: "claude-3-haiku-20240307",
            VisionLanguageModel.CLAUDE_35_SONNET: "claude-3-5-sonnet-20241022",
            VisionLanguageModel.CLAUDE_4_SONNET: "claude-sonnet-4-20250514",
            VisionLanguageModel.GPT_4_VISION: "gpt-4-vision-preview",
            VisionLanguageModel.GPT_4O: "gpt-4o",
            VisionLanguageModel.GPT_4O_MINI: "gpt-4o-mini",
        }
        return model_map.get(model, model.value)

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        self.conversation_id = uuid4()

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "conversation_id": str(self.conversation_id),
            "num_turns": len(self.conversation_history) // 2,
            "total_messages": len(self.conversation_history),
            "enabled_tools": list(self.tools.keys()),
            "rag_enabled": self.rag_retriever is not None,
        }
