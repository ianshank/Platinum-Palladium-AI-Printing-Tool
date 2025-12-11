"""
MCP resource definitions for exposing calibration data.

Resources in MCP allow servers to expose data and content to LLMs,
such as calibration records, curves, and system information.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID


class ResourceType(str, Enum):
    """Types of MCP resources."""

    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"


@dataclass
class ResourceContent:
    """Content returned from a resource."""

    uri: str
    mime_type: str
    text: Optional[str] = None
    blob: Optional[bytes] = None

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP protocol format."""
        result: dict[str, Any] = {
            "uri": self.uri,
            "mimeType": self.mime_type,
        }
        if self.text is not None:
            result["text"] = self.text
        if self.blob is not None:
            import base64

            result["blob"] = base64.b64encode(self.blob).decode("utf-8")
        return result


@dataclass
class MCPResource:
    """Definition of an MCP resource."""

    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Optional[Callable[..., ResourceContent]] = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_mcp_format(self) -> dict[str, Any]:
        """Convert to MCP protocol format for listing."""
        result: dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }
        if self.annotations:
            result["annotations"] = self.annotations
        return result

    async def read(self, **kwargs: Any) -> ResourceContent:
        """Read the resource content."""
        if self.handler is None:
            raise ValueError(f"No handler defined for resource: {self.uri}")

        result = self.handler(**kwargs)
        if not isinstance(result, ResourceContent):
            # Wrap in ResourceContent if handler returns raw data
            if isinstance(result, (dict, list)):
                return ResourceContent(
                    uri=self.uri,
                    mime_type="application/json",
                    text=json.dumps(result, indent=2, default=str),
                )
            return ResourceContent(
                uri=self.uri,
                mime_type="text/plain",
                text=str(result),
            )
        return result


class ResourceRegistry:
    """Registry of available MCP resources."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._resources: dict[str, MCPResource] = {}
        self._templates: dict[str, MCPResource] = {}

    def register(self, resource: MCPResource) -> None:
        """
        Register a resource.

        Args:
            resource: MCPResource to register.
        """
        # Check if this is a template (contains {})
        if "{" in resource.uri:
            self._templates[resource.uri] = resource
        else:
            self._resources[resource.uri] = resource

    def unregister(self, uri: str) -> bool:
        """
        Unregister a resource.

        Args:
            uri: Resource URI to unregister.

        Returns:
            True if resource was found and removed, False otherwise.
        """
        if uri in self._resources:
            del self._resources[uri]
            return True
        if uri in self._templates:
            del self._templates[uri]
            return True
        return False

    def get(self, uri: str) -> Optional[MCPResource]:
        """
        Get a resource by URI.

        Args:
            uri: Resource URI.

        Returns:
            MCPResource if found, None otherwise.
        """
        # First check exact matches
        if uri in self._resources:
            return self._resources[uri]

        # Then check templates
        for template_uri, resource in self._templates.items():
            if self._matches_template(uri, template_uri):
                return resource

        return None

    def _matches_template(self, uri: str, template: str) -> bool:
        """Check if URI matches a template pattern."""
        # Simple template matching - convert {param} to regex
        import re

        pattern = re.escape(template)
        pattern = re.sub(r"\\{[^}]+\\}", r"[^/]+", pattern)
        return bool(re.fullmatch(pattern, uri))

    def list_resources(self) -> list[MCPResource]:
        """
        List all registered resources.

        Returns:
            List of all registered MCPResource objects.
        """
        return list(self._resources.values()) + list(self._templates.values())

    def to_mcp_format(self) -> list[dict[str, Any]]:
        """
        Convert all resources to MCP protocol format.

        Returns:
            List of resource definitions in MCP format.
        """
        return [r.to_mcp_format() for r in self.list_resources()]


def create_calibration_resources(
    database: Optional[Any] = None,
) -> ResourceRegistry:
    """
    Create the standard set of calibration resources.

    Args:
        database: Optional CalibrationDatabase for data access.

    Returns:
        ResourceRegistry with all calibration resources.
    """
    registry = ResourceRegistry()

    # System information resource
    registry.register(
        MCPResource(
            uri="ptpd://system/info",
            name="System Information",
            description="Information about the PTPD calibration system",
            mime_type="application/json",
            handler=_get_system_info,
        )
    )

    # Calibration capabilities resource
    registry.register(
        MCPResource(
            uri="ptpd://capabilities",
            name="Calibration Capabilities",
            description="Available calibration features and their status",
            mime_type="application/json",
            handler=_get_capabilities,
        )
    )

    # Chemistry reference data
    registry.register(
        MCPResource(
            uri="ptpd://reference/chemistry",
            name="Chemistry Reference",
            description="Reference data for Pt/Pd chemistry calculations",
            mime_type="application/json",
            handler=_get_chemistry_reference,
        )
    )

    # Paper profiles reference
    registry.register(
        MCPResource(
            uri="ptpd://reference/papers",
            name="Paper Profiles",
            description="Common paper profiles and their characteristics",
            mime_type="application/json",
            handler=_get_paper_profiles,
        )
    )

    # Curve types reference
    registry.register(
        MCPResource(
            uri="ptpd://reference/curve-types",
            name="Curve Types",
            description="Available curve types and interpolation methods",
            mime_type="application/json",
            handler=_get_curve_types,
        )
    )

    # Database resources (if available)
    if database:
        # Recent calibrations
        registry.register(
            MCPResource(
                uri="ptpd://calibrations/recent",
                name="Recent Calibrations",
                description="List of recent calibration records",
                mime_type="application/json",
                handler=lambda: _get_recent_calibrations(database),
            )
        )

        # Statistics
        registry.register(
            MCPResource(
                uri="ptpd://calibrations/statistics",
                name="Calibration Statistics",
                description="Aggregate statistics from calibration database",
                mime_type="application/json",
                handler=lambda: _get_calibration_statistics(database),
            )
        )

        # Individual calibration by ID (template)
        registry.register(
            MCPResource(
                uri="ptpd://calibrations/{id}",
                name="Calibration Record",
                description="Get a specific calibration record by ID",
                mime_type="application/json",
                handler=lambda id: _get_calibration_by_id(database, id),
            )
        )

    return registry


# Resource handlers


def _get_system_info() -> ResourceContent:
    """Get system information."""
    try:
        from ptpd_calibration import __version__
    except ImportError:
        __version__ = "unknown"

    info = {
        "name": "PTPD Calibration System",
        "version": __version__,
        "description": "AI-powered calibration system for platinum/palladium printing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": [
            "Step tablet detection",
            "Curve generation",
            "ML-based prediction",
            "LLM integration",
            "MCP server",
        ],
    }

    return ResourceContent(
        uri="ptpd://system/info",
        mime_type="application/json",
        text=json.dumps(info, indent=2),
    )


def _get_capabilities() -> ResourceContent:
    """Get available capabilities."""
    capabilities = {
        "detection": {
            "available": True,
            "description": "Step tablet image detection and density extraction",
        },
        "curves": {
            "available": True,
            "description": "Linearization curve generation and export",
            "formats": ["qtr", "piezography", "csv", "json"],
        },
        "ml": {
            "available": _check_ml_available(),
            "description": "Machine learning-based curve prediction",
        },
        "deep_learning": {
            "available": _check_deep_learning_available(),
            "description": "Deep learning models for curve prediction",
        },
        "llm": {
            "available": _check_llm_available(),
            "description": "LLM-powered assistance and troubleshooting",
        },
    }

    return ResourceContent(
        uri="ptpd://capabilities",
        mime_type="application/json",
        text=json.dumps(capabilities, indent=2),
    )


def _check_ml_available() -> bool:
    """Check if ML features are available."""
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def _check_deep_learning_available() -> bool:
    """Check if deep learning features are available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _check_llm_available() -> bool:
    """Check if LLM features are available."""
    try:
        import anthropic  # noqa: F401

        return True
    except ImportError:
        try:
            import openai  # noqa: F401

            return True
        except ImportError:
            return False


def _get_chemistry_reference() -> ResourceContent:
    """Get chemistry reference data."""
    reference = {
        "metals": {
            "platinum": {
                "symbol": "Pt",
                "standard_solutions": [
                    {"name": "Ferric Oxalate", "concentration": "20%"},
                    {"name": "Potassium Chloroplatinite", "concentration": "15%"},
                ],
                "characteristics": {
                    "tone": "cool/neutral",
                    "dmax_range": "1.8-2.2",
                    "contrast": "high",
                },
            },
            "palladium": {
                "symbol": "Pd",
                "standard_solutions": [
                    {"name": "Sodium Chloropalladite", "concentration": "15%"},
                    {"name": "Ammonium Chloropalladate", "concentration": "15%"},
                ],
                "characteristics": {
                    "tone": "warm/brown",
                    "dmax_range": "1.6-2.0",
                    "contrast": "lower than Pt",
                },
            },
        },
        "contrast_agents": [
            {"name": "NA2 (Sodium Dichromate)", "effect": "increases contrast"},
            {"name": "NA1 (Potassium Dichromate)", "effect": "increases contrast"},
            {"name": "Hydrogen Peroxide", "effect": "increases contrast/clearing"},
        ],
        "developers": [
            {"name": "Potassium Oxalate", "temperature": "68-75F", "tone": "neutral"},
            {"name": "Ammonium Citrate", "temperature": "room temp", "tone": "warm"},
            {"name": "Sodium Citrate", "temperature": "room temp", "tone": "warm"},
        ],
        "clearing_agents": [
            {"name": "EDTA", "concentration": "5%", "baths": 3},
            {"name": "Citric Acid", "concentration": "3%", "baths": 3},
            {"name": "HCl", "concentration": "1%", "baths": 2},
        ],
    }

    return ResourceContent(
        uri="ptpd://reference/chemistry",
        mime_type="application/json",
        text=json.dumps(reference, indent=2),
    )


def _get_paper_profiles() -> ResourceContent:
    """Get common paper profiles."""
    profiles = {
        "categories": ["fine_art", "watercolor", "japanese", "handmade"],
        "papers": [
            {
                "name": "Arches Platine",
                "category": "fine_art",
                "weight": "310gsm",
                "sizing": "internal",
                "characteristics": {
                    "surface": "smooth",
                    "dmax_typical": "2.0-2.2",
                    "recommended_for": ["platinum", "platinum_palladium"],
                },
            },
            {
                "name": "Bergger COT 320",
                "category": "fine_art",
                "weight": "320gsm",
                "sizing": "internal",
                "characteristics": {
                    "surface": "slightly textured",
                    "dmax_typical": "1.9-2.1",
                    "recommended_for": ["platinum", "palladium", "platinum_palladium"],
                },
            },
            {
                "name": "Hahnemuhle Platinum Rag",
                "category": "fine_art",
                "weight": "300gsm",
                "sizing": "internal",
                "characteristics": {
                    "surface": "smooth",
                    "dmax_typical": "2.0-2.2",
                    "recommended_for": ["platinum", "platinum_palladium"],
                },
            },
            {
                "name": "Stonehenge",
                "category": "fine_art",
                "weight": "250gsm",
                "sizing": "surface",
                "characteristics": {
                    "surface": "vellum",
                    "dmax_typical": "1.6-1.9",
                    "recommended_for": ["palladium", "cyanotype"],
                },
            },
            {
                "name": "Washi Kozo",
                "category": "japanese",
                "weight": "varies",
                "sizing": "none",
                "characteristics": {
                    "surface": "textured",
                    "dmax_typical": "1.5-1.8",
                    "recommended_for": ["palladium"],
                    "notes": "May require sizing",
                },
            },
        ],
    }

    return ResourceContent(
        uri="ptpd://reference/papers",
        mime_type="application/json",
        text=json.dumps(profiles, indent=2),
    )


def _get_curve_types() -> ResourceContent:
    """Get available curve types and methods."""
    curve_info = {
        "curve_types": [
            {
                "name": "linear",
                "description": "Standard linear response curve",
                "use_case": "General purpose, predictable gradation",
            },
            {
                "name": "paper_white",
                "description": "Preserves paper white in highlights",
                "use_case": "Images requiring clean highlights",
            },
            {
                "name": "aesthetic",
                "description": "Optimized for visual perception",
                "use_case": "Fine art printing with pleasing tonality",
            },
        ],
        "interpolation_methods": [
            {
                "name": "linear",
                "description": "Simple linear interpolation",
                "smoothness": "low",
            },
            {
                "name": "cubic",
                "description": "Cubic spline interpolation",
                "smoothness": "medium",
            },
            {
                "name": "monotonic",
                "description": "Monotonic cubic interpolation",
                "smoothness": "medium",
                "note": "Prevents curve inversions",
            },
            {
                "name": "pchip",
                "description": "Piecewise cubic Hermite interpolation",
                "smoothness": "high",
                "note": "Recommended for most use cases",
            },
        ],
        "export_formats": [
            {
                "name": "qtr",
                "extension": ".txt",
                "description": "QuadToneRIP format",
            },
            {
                "name": "piezography",
                "extension": ".quad",
                "description": "Piezography quad file format",
            },
            {
                "name": "csv",
                "extension": ".csv",
                "description": "Comma-separated values",
            },
            {
                "name": "json",
                "extension": ".json",
                "description": "JSON format with metadata",
            },
        ],
    }

    return ResourceContent(
        uri="ptpd://reference/curve-types",
        mime_type="application/json",
        text=json.dumps(curve_info, indent=2),
    )


def _get_recent_calibrations(database: Any, limit: int = 10) -> ResourceContent:
    """Get recent calibrations from database."""
    try:
        records = database.query(limit=limit)

        calibrations = []
        for rec in records:
            calibrations.append(
                {
                    "id": str(rec.id),
                    "paper_type": rec.paper_type,
                    "metal_ratio": rec.metal_ratio,
                    "exposure_time": rec.exposure_time,
                    "dmax": max(rec.measured_densities) if rec.measured_densities else None,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                }
            )

        return ResourceContent(
            uri="ptpd://calibrations/recent",
            mime_type="application/json",
            text=json.dumps({"count": len(calibrations), "records": calibrations}, indent=2),
        )
    except Exception as e:
        return ResourceContent(
            uri="ptpd://calibrations/recent",
            mime_type="application/json",
            text=json.dumps({"error": str(e), "records": []}, indent=2),
        )


def _get_calibration_statistics(database: Any) -> ResourceContent:
    """Get calibration statistics from database."""
    try:
        all_records = database.query()

        if not all_records:
            stats = {
                "total_calibrations": 0,
                "paper_types": [],
                "dmax_range": {"min": None, "max": None, "avg": None},
            }
        else:
            paper_types = list(set(r.paper_type for r in all_records))
            dmaxes = [
                max(r.measured_densities) for r in all_records if r.measured_densities
            ]

            stats = {
                "total_calibrations": len(all_records),
                "paper_types": paper_types,
                "unique_papers": len(paper_types),
                "dmax_range": {
                    "min": min(dmaxes) if dmaxes else None,
                    "max": max(dmaxes) if dmaxes else None,
                    "avg": sum(dmaxes) / len(dmaxes) if dmaxes else None,
                },
            }

        return ResourceContent(
            uri="ptpd://calibrations/statistics",
            mime_type="application/json",
            text=json.dumps(stats, indent=2),
        )
    except Exception as e:
        return ResourceContent(
            uri="ptpd://calibrations/statistics",
            mime_type="application/json",
            text=json.dumps({"error": str(e)}, indent=2),
        )


def _get_calibration_by_id(database: Any, calibration_id: str) -> ResourceContent:
    """Get a specific calibration by ID."""
    try:
        uuid_id = UUID(calibration_id)
        record = database.get_record(uuid_id)

        if not record:
            return ResourceContent(
                uri=f"ptpd://calibrations/{calibration_id}",
                mime_type="application/json",
                text=json.dumps({"error": f"Calibration {calibration_id} not found"}, indent=2),
            )

        return ResourceContent(
            uri=f"ptpd://calibrations/{calibration_id}",
            mime_type="application/json",
            text=json.dumps(record.model_dump(mode="json"), indent=2, default=str),
        )
    except Exception as e:
        return ResourceContent(
            uri=f"ptpd://calibrations/{calibration_id}",
            mime_type="application/json",
            text=json.dumps({"error": str(e)}, indent=2),
        )
