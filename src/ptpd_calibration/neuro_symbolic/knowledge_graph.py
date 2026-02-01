"""
Knowledge Graph for Paper/Chemistry Relationships.

This module implements a knowledge graph that captures relationships between
papers, chemistry, and process parameters for analogical reasoning.

The knowledge graph enables:
- Finding similar papers/chemistry combinations
- Inferring optimal settings for new materials
- Explaining recommendations based on known relationships
- Transfer learning from calibration history

Architecture:
- Entities: Papers, Chemistry, Developers, UV Sources, Environmental Conditions
- Relationships: affects, requires, produces, similar_to, incompatible_with
- Embeddings: Learned vector representations for similarity computation
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ptpd_calibration.config import NeuroSymbolicSettings, get_settings


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PAPER = "paper"
    CHEMISTRY = "chemistry"
    METAL_SALT = "metal_salt"
    CONTRAST_AGENT = "contrast_agent"
    DEVELOPER = "developer"
    UV_SOURCE = "uv_source"
    COATING_METHOD = "coating_method"
    ENVIRONMENTAL_CONDITION = "environmental_condition"
    CALIBRATION_RESULT = "calibration_result"
    CURVE = "curve"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    # Causal relationships
    AFFECTS = "affects"
    REQUIRES = "requires"
    PRODUCES = "produces"
    MODIFIES = "modifies"

    # Similarity relationships
    SIMILAR_TO = "similar_to"
    VARIANT_OF = "variant_of"
    ALTERNATIVE_TO = "alternative_to"

    # Compatibility relationships
    COMPATIBLE_WITH = "compatible_with"
    INCOMPATIBLE_WITH = "incompatible_with"
    OPTIMAL_FOR = "optimal_for"

    # Process relationships
    ABSORBS_AT_RATE = "absorbs_at_rate"
    REACTS_WITH = "reacts_with"
    DEVELOPS_IN = "develops_in"
    EXPOSED_BY = "exposed_by"

    # Quantitative relationships
    INCREASES = "increases"
    DECREASES = "decreases"
    PROPORTIONAL_TO = "proportional_to"


class Entity(BaseModel):
    """An entity in the knowledge graph."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=256)
    entity_type: EntityType
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_feature_vector(self) -> NDArray[np.float64]:
        """Convert entity properties to feature vector."""
        if self.embedding is not None:
            return np.array(self.embedding)

        # Create feature vector from properties
        features = []
        for key in sorted(self.properties.keys()):
            value = self.properties[key]
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Simple hash-based encoding
                features.append(hash(value) % 1000 / 1000.0)

        return np.array(features) if features else np.zeros(1)


class Relationship(BaseModel):
    """A relationship between two entities."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    relation_type: RelationType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="manual")  # manual, inferred, learned
    created_at: datetime = Field(default_factory=datetime.now)


class InferenceResult(BaseModel):
    """Result of a knowledge graph inference."""

    query: str
    result_entities: list[UUID] = Field(default_factory=list)
    result_values: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    reasoning_path: list[str] = Field(default_factory=list)


class SimilarityResult(BaseModel):
    """Result of similarity search."""

    entity_id: UUID
    entity_name: str
    entity_type: EntityType
    similarity_score: float = Field(ge=0.0, le=1.0)
    common_relationships: list[str] = Field(default_factory=list)


class KnowledgeGraph:
    """Base knowledge graph implementation.

    Provides entity and relationship storage with query capabilities.
    """

    def __init__(self, settings: NeuroSymbolicSettings | None = None):
        """Initialize knowledge graph.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings().neuro_symbolic
        self._entities: dict[UUID, Entity] = {}
        self._relationships: list[Relationship] = []
        self._entity_index: dict[str, list[UUID]] = {}  # name -> ids
        self._type_index: dict[EntityType, list[UUID]] = {}  # type -> ids
        self._relation_index: dict[UUID, list[Relationship]] = {}  # entity -> relations

    def add_entity(self, entity: Entity) -> UUID:
        """Add an entity to the graph.

        Args:
            entity: Entity to add

        Returns:
            Entity UUID
        """
        self._entities[entity.id] = entity

        # Update indices
        if entity.name not in self._entity_index:
            self._entity_index[entity.name] = []
        self._entity_index[entity.name].append(entity.id)

        if entity.entity_type not in self._type_index:
            self._type_index[entity.entity_type] = []
        self._type_index[entity.entity_type].append(entity.id)

        return entity.id

    def add_relationship(self, relationship: Relationship) -> UUID:
        """Add a relationship to the graph.

        Args:
            relationship: Relationship to add

        Returns:
            Relationship UUID
        """
        self._relationships.append(relationship)

        # Update relation index
        for entity_id in [relationship.source_id, relationship.target_id]:
            if entity_id not in self._relation_index:
                self._relation_index[entity_id] = []
            self._relation_index[entity_id].append(relationship)

        return relationship.id

    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        """Get entity by name (returns first match)."""
        ids = self._entity_index.get(name, [])
        return self._entities.get(ids[0]) if ids else None

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a given type."""
        ids = self._type_index.get(entity_type, [])
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def get_relationships(
        self,
        entity_id: UUID,
        relation_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[Relationship]:
        """Get relationships for an entity.

        Args:
            entity_id: Entity UUID
            relation_type: Optional filter by relation type
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of matching relationships
        """
        relations = self._relation_index.get(entity_id, [])

        filtered = []
        for r in relations:
            if relation_type is not None and r.relation_type != relation_type:
                continue

            if direction == "outgoing" and r.source_id != entity_id:
                continue
            if direction == "incoming" and r.target_id != entity_id:
                continue

            filtered.append(r)

        return filtered

    def find_path(
        self,
        source_id: UUID,
        target_id: UUID,
        max_depth: int | None = None,
    ) -> list[Relationship] | None:
        """Find path between two entities.

        Args:
            source_id: Source entity UUID
            target_id: Target entity UUID
            max_depth: Maximum path length

        Returns:
            List of relationships forming path, or None if not found
        """
        max_depth = max_depth or self.settings.kg_max_inference_depth

        # BFS for shortest path
        from collections import deque

        visited = {source_id}
        queue: deque[tuple[UUID, list[Relationship]]] = deque([(source_id, [])])

        while queue:
            current, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            for rel in self._relation_index.get(current, []):
                # Get the other end of the relationship
                next_id = rel.target_id if rel.source_id == current else rel.source_id

                if next_id == target_id:
                    return path + [rel]

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [rel]))

        return None

    def compute_similarity(
        self,
        entity1_id: UUID,
        entity2_id: UUID,
    ) -> float:
        """Compute similarity between two entities.

        Args:
            entity1_id: First entity UUID
            entity2_id: Second entity UUID

        Returns:
            Similarity score (0-1)
        """
        e1 = self.get_entity(entity1_id)
        e2 = self.get_entity(entity2_id)

        if e1 is None or e2 is None:
            return 0.0

        # Type similarity
        type_sim = 1.0 if e1.entity_type == e2.entity_type else 0.3

        # Embedding similarity (if available)
        embed_sim = 0.5
        if e1.embedding is not None and e2.embedding is not None:
            v1 = np.array(e1.embedding)
            v2 = np.array(e2.embedding)
            if len(v1) == len(v2) and len(v1) > 0:
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    embed_sim = float(np.dot(v1, v2) / (norm1 * norm2))
                    embed_sim = (embed_sim + 1) / 2  # Map from [-1,1] to [0,1]

        # Property similarity
        common_props = set(e1.properties.keys()) & set(e2.properties.keys())
        if common_props:
            matches = sum(1 for k in common_props if e1.properties[k] == e2.properties[k])
            prop_sim = matches / len(common_props)
        else:
            prop_sim = 0.5

        # Relationship similarity
        rel1 = {
            (
                r.relation_type,
                self.get_entity(r.target_id).name if self.get_entity(r.target_id) else None,
            )
            for r in self.get_relationships(entity1_id, direction="outgoing")
        }
        rel2 = {
            (
                r.relation_type,
                self.get_entity(r.target_id).name if self.get_entity(r.target_id) else None,
            )
            for r in self.get_relationships(entity2_id, direction="outgoing")
        }

        if rel1 or rel2:
            rel_sim = len(rel1 & rel2) / len(rel1 | rel2) if rel1 | rel2 else 0.5
        else:
            rel_sim = 0.5

        # Weighted combination
        similarity = 0.2 * type_sim + 0.3 * embed_sim + 0.2 * prop_sim + 0.3 * rel_sim
        return float(similarity)

    def find_similar(
        self,
        entity_id: UUID,
        top_k: int = 5,
        same_type_only: bool = True,
    ) -> list[SimilarityResult]:
        """Find entities most similar to a given entity.

        Args:
            entity_id: Entity to find similar entities for
            top_k: Number of results to return
            same_type_only: Only consider entities of same type

        Returns:
            List of similarity results
        """
        entity = self.get_entity(entity_id)
        if entity is None:
            return []

        candidates = (
            self.get_entities_by_type(entity.entity_type)
            if same_type_only
            else list(self._entities.values())
        )

        results = []
        for candidate in candidates:
            if candidate.id == entity_id:
                continue

            sim = self.compute_similarity(entity_id, candidate.id)

            # Find common relationships
            rel1 = {r.relation_type.value for r in self.get_relationships(entity_id)}
            rel2 = {r.relation_type.value for r in self.get_relationships(candidate.id)}
            common = list(rel1 & rel2)

            results.append(
                SimilarityResult(
                    entity_id=candidate.id,
                    entity_name=candidate.name,
                    entity_type=candidate.entity_type,
                    similarity_score=sim,
                    common_relationships=common,
                )
            )

        # Sort by similarity and return top k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]


class PaperChemistryKnowledgeGraph(KnowledgeGraph):
    """Specialized knowledge graph for paper/chemistry relationships.

    Pre-populated with domain knowledge about platinum/palladium printing.
    """

    def __init__(self, settings: NeuroSymbolicSettings | None = None):
        """Initialize with domain knowledge."""
        super().__init__(settings)
        self._initialize_domain_knowledge()

    def _initialize_domain_knowledge(self) -> None:
        """Initialize graph with platinum/palladium domain knowledge."""
        # Papers
        papers = [
            {
                "name": "Arches Platine",
                "properties": {
                    "manufacturer": "Arches",
                    "weight_gsm": 310,
                    "sizing": "internal",
                    "absorbency": "medium",
                    "texture": "smooth",
                    "coating_efficiency": 0.92,
                    "expected_dmax": 2.0,
                    "warmth": "neutral",
                },
            },
            {
                "name": "Bergger COT320",
                "properties": {
                    "manufacturer": "Bergger",
                    "weight_gsm": 320,
                    "sizing": "internal",
                    "absorbency": "high",
                    "texture": "smooth",
                    "coating_efficiency": 0.88,
                    "expected_dmax": 2.1,
                    "warmth": "warm",
                },
            },
            {
                "name": "Hahnemühle Platinum Rag",
                "properties": {
                    "manufacturer": "Hahnemühle",
                    "weight_gsm": 300,
                    "sizing": "internal",
                    "absorbency": "medium",
                    "texture": "smooth",
                    "coating_efficiency": 0.90,
                    "expected_dmax": 2.0,
                    "warmth": "cool",
                },
            },
            {
                "name": "Fabriano Artistico HP",
                "properties": {
                    "manufacturer": "Fabriano",
                    "weight_gsm": 300,
                    "sizing": "external",
                    "absorbency": "low",
                    "texture": "hot_press",
                    "coating_efficiency": 0.95,
                    "expected_dmax": 1.9,
                    "warmth": "neutral",
                },
            },
            {
                "name": "Fabriano Artistico CP",
                "properties": {
                    "manufacturer": "Fabriano",
                    "weight_gsm": 300,
                    "sizing": "external",
                    "absorbency": "medium",
                    "texture": "cold_press",
                    "coating_efficiency": 0.85,
                    "expected_dmax": 1.85,
                    "warmth": "neutral",
                },
            },
            {
                "name": "Stonehenge",
                "properties": {
                    "manufacturer": "Legion",
                    "weight_gsm": 250,
                    "sizing": "internal",
                    "absorbency": "high",
                    "texture": "smooth",
                    "coating_efficiency": 0.80,
                    "expected_dmax": 1.8,
                    "warmth": "warm",
                },
            },
        ]

        paper_entities = {}
        for paper in papers:
            entity = Entity(
                name=paper["name"],
                entity_type=EntityType.PAPER,
                properties=paper["properties"],
            )
            self.add_entity(entity)
            paper_entities[paper["name"]] = entity

        # Metal salts
        metals = [
            {
                "name": "Palladium Chloride",
                "properties": {
                    "type": "palladium",
                    "tone": "warm",
                    "speed": "fast",
                    "cost_factor": 1.0,
                    "dmax_contribution": 0.9,
                },
            },
            {
                "name": "Platinum Chloride",
                "properties": {
                    "type": "platinum",
                    "tone": "cool",
                    "speed": "slow",
                    "cost_factor": 4.0,
                    "dmax_contribution": 1.0,
                },
            },
            {
                "name": "Ferric Oxalate",
                "properties": {
                    "type": "sensitizer",
                    "function": "light_sensitive",
                    "shelf_life_days": 180,
                },
            },
        ]

        metal_entities = {}
        for metal in metals:
            entity = Entity(
                name=metal["name"],
                entity_type=EntityType.METAL_SALT,
                properties=metal["properties"],
            )
            self.add_entity(entity)
            metal_entities[metal["name"]] = entity

        # Contrast agents
        contrast_agents = [
            {
                "name": "Sodium Chloroplatinate (Na2)",
                "properties": {
                    "contrast_increase": 1.3,
                    "tone_shift": "cooler",
                    "speed_reduction": 0.9,
                    "max_drops_per_ml_metal": 0.5,
                },
            },
            {
                "name": "Potassium Chlorate",
                "properties": {
                    "contrast_increase": 1.2,
                    "tone_shift": "neutral",
                    "speed_reduction": 0.95,
                    "max_drops_per_ml_metal": 0.3,
                },
            },
            {
                "name": "Hydrogen Peroxide",
                "properties": {
                    "contrast_increase": 1.1,
                    "tone_shift": "warmer",
                    "speed_reduction": 0.85,
                    "max_drops_per_ml_metal": 0.2,
                },
            },
        ]

        contrast_entities = {}
        for agent in contrast_agents:
            entity = Entity(
                name=agent["name"],
                entity_type=EntityType.CONTRAST_AGENT,
                properties=agent["properties"],
            )
            self.add_entity(entity)
            contrast_entities[agent["name"]] = entity

        # Developers
        developers = [
            {
                "name": "Potassium Oxalate",
                "properties": {
                    "temperature_c": 20,
                    "development_time_sec": 60,
                    "tone": "neutral",
                    "contrast_effect": "normal",
                },
            },
            {
                "name": "Ammonium Citrate",
                "properties": {
                    "temperature_c": 20,
                    "development_time_sec": 90,
                    "tone": "warm",
                    "contrast_effect": "lower",
                },
            },
            {
                "name": "Sodium Citrate",
                "properties": {
                    "temperature_c": 20,
                    "development_time_sec": 90,
                    "tone": "neutral_warm",
                    "contrast_effect": "normal",
                },
            },
        ]

        developer_entities = {}
        for dev in developers:
            entity = Entity(
                name=dev["name"],
                entity_type=EntityType.DEVELOPER,
                properties=dev["properties"],
            )
            self.add_entity(entity)
            developer_entities[dev["name"]] = entity

        # UV Sources
        uv_sources = [
            {
                "name": "BL Fluorescent",
                "properties": {
                    "wavelength_nm": 365,
                    "speed_factor": 1.0,
                    "cost": "low",
                    "bulb_life_hours": 1000,
                },
            },
            {
                "name": "LED UV 365nm",
                "properties": {
                    "wavelength_nm": 365,
                    "speed_factor": 0.6,
                    "cost": "medium",
                    "bulb_life_hours": 50000,
                },
            },
            {
                "name": "Metal Halide",
                "properties": {
                    "wavelength_nm": 350,
                    "speed_factor": 0.5,
                    "cost": "high",
                    "bulb_life_hours": 500,
                },
            },
            {
                "name": "Sunlight",
                "properties": {
                    "wavelength_nm": 365,
                    "speed_factor": 0.4,
                    "cost": "free",
                    "variability": "high",
                },
            },
        ]

        uv_entities = {}
        for uv in uv_sources:
            entity = Entity(
                name=uv["name"],
                entity_type=EntityType.UV_SOURCE,
                properties=uv["properties"],
            )
            self.add_entity(entity)
            uv_entities[uv["name"]] = entity

        # Add relationships

        # Paper similarity relationships
        self.add_relationship(
            Relationship(
                source_id=paper_entities["Arches Platine"].id,
                target_id=paper_entities["Hahnemühle Platinum Rag"].id,
                relation_type=RelationType.SIMILAR_TO,
                weight=0.85,
                properties={"similarity_aspect": "coating_behavior"},
            )
        )
        self.add_relationship(
            Relationship(
                source_id=paper_entities["Bergger COT320"].id,
                target_id=paper_entities["Stonehenge"].id,
                relation_type=RelationType.SIMILAR_TO,
                weight=0.75,
                properties={"similarity_aspect": "absorbency"},
            )
        )

        # Paper-chemistry compatibility
        for _paper_name, paper in paper_entities.items():
            # All papers compatible with both metals
            for metal_name, metal in metal_entities.items():
                if "Chloride" in metal_name:
                    self.add_relationship(
                        Relationship(
                            source_id=paper.id,
                            target_id=metal.id,
                            relation_type=RelationType.COMPATIBLE_WITH,
                            weight=0.9,
                        )
                    )

        # Metal-contrast agent relationships
        na2 = contrast_entities["Sodium Chloroplatinate (Na2)"]
        for metal_name, metal in metal_entities.items():
            if metal_name == "Platinum Chloride":
                self.add_relationship(
                    Relationship(
                        source_id=na2.id,
                        target_id=metal.id,
                        relation_type=RelationType.OPTIMAL_FOR,
                        weight=0.95,
                    )
                )
            elif metal_name == "Palladium Chloride":
                self.add_relationship(
                    Relationship(
                        source_id=na2.id,
                        target_id=metal.id,
                        relation_type=RelationType.COMPATIBLE_WITH,
                        weight=0.8,
                    )
                )

        # Developer-tone relationships
        for _dev_name, dev in developer_entities.items():
            tone = dev.properties.get("tone", "neutral")
            for _paper_name, paper in paper_entities.items():
                paper_warmth = paper.properties.get("warmth", "neutral")
                # Match warm developers with warm papers, etc.
                if tone == paper_warmth or tone == "neutral" or paper_warmth == "neutral":
                    self.add_relationship(
                        Relationship(
                            source_id=dev.id,
                            target_id=paper.id,
                            relation_type=RelationType.OPTIMAL_FOR,
                            weight=0.85,
                        )
                    )

    def infer_settings_for_paper(self, paper_name: str) -> InferenceResult:
        """Infer optimal settings for a paper.

        Args:
            paper_name: Name of the paper

        Returns:
            Inference result with recommended settings
        """
        paper = self.get_entity_by_name(paper_name)
        if paper is None:
            # Try to find similar paper
            similar = self._find_similar_paper_by_name(paper_name)
            if similar:
                paper = similar
            else:
                return InferenceResult(
                    query=f"Settings for {paper_name}",
                    confidence=0.0,
                    explanation=f"Unknown paper: {paper_name}",
                )

        # Get paper properties
        absorbency = paper.properties.get("absorbency", "medium")
        warmth = paper.properties.get("warmth", "neutral")
        expected_dmax = paper.properties.get("expected_dmax", 2.0)

        # Infer metal ratio based on warmth
        if warmth == "warm":
            metal_ratio = 0.25  # More Pd
        elif warmth == "cool":
            metal_ratio = 0.75  # More Pt
        else:
            metal_ratio = 0.5  # Balanced

        # Infer coating factor based on absorbency
        coating_factors = {"low": 0.8, "medium": 1.0, "high": 1.2}
        coating_factor = coating_factors.get(absorbency, 1.0)

        # Find optimal developer
        optimal_devs = self.get_relationships(
            paper.id, RelationType.OPTIMAL_FOR, direction="incoming"
        )
        developer = "Potassium Oxalate"  # Default
        for rel in optimal_devs:
            dev_entity = self.get_entity(rel.source_id)
            if dev_entity and dev_entity.entity_type == EntityType.DEVELOPER:
                developer = dev_entity.name
                break

        reasoning = [
            f"Paper '{paper.name}' has {absorbency} absorbency → coating factor {coating_factor}",
            f"Paper warmth is {warmth} → metal ratio {metal_ratio} (Pt fraction)",
            f"Expected Dmax: {expected_dmax}",
            f"Recommended developer: {developer}",
        ]

        return InferenceResult(
            query=f"Settings for {paper_name}",
            result_entities=[paper.id],
            result_values={
                "paper": paper.name,
                "metal_ratio": metal_ratio,
                "coating_factor": coating_factor,
                "expected_dmax": expected_dmax,
                "developer": developer,
                "absorbency": absorbency,
            },
            confidence=0.85 if paper.name == paper_name else 0.7,
            explanation=f"Inferred settings for {paper.name} based on paper properties",
            reasoning_path=reasoning,
        )

    def infer_for_new_paper(
        self,
        properties: dict[str, Any],
    ) -> InferenceResult:
        """Infer settings for a new/unknown paper based on properties.

        Uses analogical reasoning to find similar papers and transfer knowledge.

        Args:
            properties: Properties of the new paper

        Returns:
            Inference result with recommended settings
        """
        # Create temporary entity for similarity comparison
        _temp_entity = Entity(  # Reserved for future similarity lookup
            name="Query Paper",
            entity_type=EntityType.PAPER,
            properties=properties,
        )

        # Find most similar papers
        best_match = None
        best_similarity = 0.0

        for paper in self.get_entities_by_type(EntityType.PAPER):
            # Compare properties
            common_keys = set(properties.keys()) & set(paper.properties.keys())
            if not common_keys:
                continue

            matches = sum(1 for k in common_keys if properties.get(k) == paper.properties.get(k))
            similarity = matches / len(common_keys)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = paper

        if best_match is None:
            return InferenceResult(
                query="Settings for new paper",
                confidence=0.0,
                explanation="Could not find similar papers for comparison",
            )

        # Transfer knowledge from best match
        base_result = self.infer_settings_for_paper(best_match.name)

        # Adjust based on property differences
        adjustments = []
        result_values = dict(base_result.result_values)

        if "absorbency" in properties and properties["absorbency"] != best_match.properties.get(
            "absorbency"
        ):
            coating_factors = {"low": 0.8, "medium": 1.0, "high": 1.2}
            result_values["coating_factor"] = coating_factors.get(properties["absorbency"], 1.0)
            adjustments.append(f"Adjusted coating factor for {properties['absorbency']} absorbency")

        if "warmth" in properties and properties["warmth"] != best_match.properties.get("warmth"):
            warmth_ratios = {"warm": 0.25, "neutral": 0.5, "cool": 0.75}
            result_values["metal_ratio"] = warmth_ratios.get(properties["warmth"], 0.5)
            adjustments.append(f"Adjusted metal ratio for {properties['warmth']} tone preference")

        reasoning = (
            [
                f"Most similar paper: {best_match.name} (similarity: {best_similarity:.2f})",
            ]
            + adjustments
            + base_result.reasoning_path
        )

        return InferenceResult(
            query="Settings for new paper",
            result_entities=[best_match.id],
            result_values=result_values,
            confidence=best_similarity * 0.9,  # Reduce confidence for transfer
            explanation=f"Inferred settings by analogy with {best_match.name}",
            reasoning_path=reasoning,
        )

    def _find_similar_paper_by_name(self, name: str) -> Entity | None:
        """Find similar paper by fuzzy name matching."""
        name_lower = name.lower()

        for paper in self.get_entities_by_type(EntityType.PAPER):
            paper_name_lower = paper.name.lower()

            # Check for substring match
            if name_lower in paper_name_lower or paper_name_lower in name_lower:
                return paper

            # Check for manufacturer match
            manufacturer = paper.properties.get("manufacturer", "").lower()
            if manufacturer and manufacturer in name_lower:
                return paper

        return None

    def get_compatibility_explanation(self, entity1_name: str, entity2_name: str) -> str:
        """Get explanation for compatibility between two entities.

        Args:
            entity1_name: First entity name
            entity2_name: Second entity name

        Returns:
            Human-readable explanation
        """
        e1 = self.get_entity_by_name(entity1_name)
        e2 = self.get_entity_by_name(entity2_name)

        if e1 is None or e2 is None:
            return f"Unknown entity: {entity1_name if e1 is None else entity2_name}"

        # Find direct relationship
        path = self.find_path(e1.id, e2.id)

        if path is None:
            return f"No known relationship between {entity1_name} and {entity2_name}"

        # Build explanation
        explanations = []
        _current = e1.name  # Track traversal start
        for rel in path:
            source = self.get_entity(rel.source_id)
            target = self.get_entity(rel.target_id)
            if source and target:
                explanations.append(f"{source.name} --[{rel.relation_type.value}]--> {target.name}")
                _current = target.name  # Track traversal for future use

        return " → ".join(explanations)

    def learn_from_calibration(
        self,
        paper_name: str,
        chemistry_settings: dict[str, Any],
        result_metrics: dict[str, float],
    ) -> None:
        """Learn new relationships from calibration results.

        Args:
            paper_name: Paper used
            chemistry_settings: Chemistry settings used
            result_metrics: Resulting metrics (dmax, linearity, etc.)
        """
        if not self.settings.kg_enable_learning:
            return

        paper = self.get_entity_by_name(paper_name)
        if paper is None:
            # Add new paper entity
            paper = Entity(
                name=paper_name,
                entity_type=EntityType.PAPER,
                properties={
                    "learned": True,
                    "calibration_count": 1,
                },
            )
            self.add_entity(paper)

        # Update paper properties based on results
        if "dmax" in result_metrics:
            paper.properties["observed_dmax"] = result_metrics["dmax"]

        # Learn optimal settings
        calibration = Entity(
            name=f"Calibration_{paper_name}_{datetime.now().isoformat()}",
            entity_type=EntityType.CALIBRATION_RESULT,
            properties={
                **chemistry_settings,
                **result_metrics,
            },
        )
        self.add_entity(calibration)

        # Add relationship
        self.add_relationship(
            Relationship(
                source_id=paper.id,
                target_id=calibration.id,
                relation_type=RelationType.PRODUCES,
                weight=result_metrics.get("quality", 0.8),
                source="learned",
            )
        )
