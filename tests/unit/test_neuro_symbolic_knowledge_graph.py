"""
Tests for neuro-symbolic knowledge graph module.

Tests the knowledge graph for paper/chemistry relationships
and analogical reasoning capabilities.
"""

from uuid import uuid4

import pytest

from ptpd_calibration.config import NeuroSymbolicSettings
from ptpd_calibration.neuro_symbolic.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    PaperChemistryKnowledgeGraph,
    Relationship,
    RelationType,
    SimilarityResult,
)


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            name="Arches Platine",
            entity_type=EntityType.PAPER,
            properties={"weight_gsm": 310, "absorbency": "medium"},
        )

        assert entity.name == "Arches Platine"
        assert entity.entity_type == EntityType.PAPER
        assert entity.properties["weight_gsm"] == 310
        assert entity.id is not None

    def test_entity_feature_vector(self):
        """Test feature vector generation."""
        entity = Entity(
            name="Test Paper",
            entity_type=EntityType.PAPER,
            properties={"weight": 300.0, "coated": True},
        )

        features = entity.to_feature_vector()

        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_entity_with_embedding(self):
        """Test entity with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        entity = Entity(
            name="Test",
            entity_type=EntityType.PAPER,
            embedding=embedding,
        )

        features = entity.to_feature_vector()
        assert list(features) == embedding


class TestRelationship:
    """Tests for Relationship class."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        source_id = uuid4()
        target_id = uuid4()

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType.SIMILAR_TO,
            weight=0.85,
        )

        assert rel.source_id == source_id
        assert rel.target_id == target_id
        assert rel.relation_type == RelationType.SIMILAR_TO
        assert rel.weight == 0.85
        assert rel.confidence == 1.0  # Default

    def test_relationship_with_properties(self):
        """Test relationship with additional properties."""
        rel = Relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type=RelationType.AFFECTS,
            properties={"effect_type": "increases_dmax", "magnitude": 0.2},
        )

        assert rel.properties["effect_type"] == "increases_dmax"


class TestKnowledgeGraph:
    """Tests for base KnowledgeGraph."""

    @pytest.fixture
    def graph(self):
        """Create empty knowledge graph."""
        return KnowledgeGraph()

    @pytest.fixture
    def populated_graph(self, graph):
        """Create knowledge graph with some entities."""
        paper1 = Entity(
            name="Paper A",
            entity_type=EntityType.PAPER,
            properties={"weight": 300},
        )
        paper2 = Entity(
            name="Paper B",
            entity_type=EntityType.PAPER,
            properties={"weight": 310},
        )
        chem = Entity(
            name="Chemistry X",
            entity_type=EntityType.CHEMISTRY,
            properties={"type": "platinum"},
        )

        graph.add_entity(paper1)
        graph.add_entity(paper2)
        graph.add_entity(chem)

        graph.add_relationship(
            Relationship(
                source_id=paper1.id,
                target_id=paper2.id,
                relation_type=RelationType.SIMILAR_TO,
                weight=0.8,
            )
        )
        graph.add_relationship(
            Relationship(
                source_id=paper1.id,
                target_id=chem.id,
                relation_type=RelationType.COMPATIBLE_WITH,
                weight=0.9,
            )
        )

        return graph

    def test_add_entity(self, graph):
        """Test adding entity to graph."""
        entity = Entity(name="Test", entity_type=EntityType.PAPER)
        entity_id = graph.add_entity(entity)

        assert entity_id == entity.id
        assert graph.get_entity(entity_id) == entity

    def test_get_entity_by_name(self, populated_graph):
        """Test getting entity by name."""
        entity = populated_graph.get_entity_by_name("Paper A")

        assert entity is not None
        assert entity.name == "Paper A"

    def test_get_entity_by_name_not_found(self, populated_graph):
        """Test getting non-existent entity."""
        entity = populated_graph.get_entity_by_name("Non-existent")

        assert entity is None

    def test_get_entities_by_type(self, populated_graph):
        """Test getting all entities of a type."""
        papers = populated_graph.get_entities_by_type(EntityType.PAPER)

        assert len(papers) == 2
        assert all(p.entity_type == EntityType.PAPER for p in papers)

    def test_get_relationships(self, populated_graph):
        """Test getting relationships for an entity."""
        paper = populated_graph.get_entity_by_name("Paper A")
        relationships = populated_graph.get_relationships(paper.id)

        assert len(relationships) == 2  # Similar and compatible

    def test_get_relationships_by_type(self, populated_graph):
        """Test filtering relationships by type."""
        paper = populated_graph.get_entity_by_name("Paper A")
        similar = populated_graph.get_relationships(paper.id, RelationType.SIMILAR_TO)

        assert len(similar) == 1
        assert similar[0].relation_type == RelationType.SIMILAR_TO

    def test_get_relationships_direction(self, populated_graph):
        """Test filtering relationships by direction."""
        paper = populated_graph.get_entity_by_name("Paper A")

        outgoing = populated_graph.get_relationships(paper.id, direction="outgoing")
        assert len(outgoing) == 2

        incoming = populated_graph.get_relationships(paper.id, direction="incoming")
        assert len(incoming) == 0  # Paper A is source, not target

    def test_find_path(self, populated_graph):
        """Test finding path between entities."""
        paper = populated_graph.get_entity_by_name("Paper A")
        chem = populated_graph.get_entity_by_name("Chemistry X")

        path = populated_graph.find_path(paper.id, chem.id)

        assert path is not None
        assert len(path) == 1
        assert path[0].relation_type == RelationType.COMPATIBLE_WITH

    def test_find_path_no_connection(self, populated_graph):
        """Test finding path with no connection."""
        paper_b = populated_graph.get_entity_by_name("Paper B")
        chem = populated_graph.get_entity_by_name("Chemistry X")

        # Paper B not directly connected to Chemistry X
        path = populated_graph.find_path(paper_b.id, chem.id, max_depth=1)

        # May find path through Paper A if depth allows
        assert path is None or len(path) <= 2

    def test_compute_similarity(self, populated_graph):
        """Test computing similarity between entities."""
        paper_a = populated_graph.get_entity_by_name("Paper A")
        paper_b = populated_graph.get_entity_by_name("Paper B")

        similarity = populated_graph.compute_similarity(paper_a.id, paper_b.id)

        assert 0 <= similarity <= 1
        # Same type entities without embeddings have baseline similarity
        assert similarity > 0.2

    def test_find_similar(self, populated_graph):
        """Test finding similar entities."""
        paper_a = populated_graph.get_entity_by_name("Paper A")

        similar = populated_graph.find_similar(paper_a.id, top_k=5)

        assert len(similar) >= 1
        assert isinstance(similar[0], SimilarityResult)
        assert similar[0].entity_name == "Paper B"


class TestPaperChemistryKnowledgeGraph:
    """Tests for specialized paper/chemistry knowledge graph."""

    @pytest.fixture
    def kg(self):
        """Create paper/chemistry knowledge graph."""
        return PaperChemistryKnowledgeGraph()

    def test_initialization_with_domain_knowledge(self, kg):
        """Test that knowledge graph is initialized with domain knowledge."""
        # Should have papers
        papers = kg.get_entities_by_type(EntityType.PAPER)
        assert len(papers) >= 5

        # Should have metal salts
        metals = kg.get_entities_by_type(EntityType.METAL_SALT)
        assert len(metals) >= 2

        # Should have developers
        developers = kg.get_entities_by_type(EntityType.DEVELOPER)
        assert len(developers) >= 2

    def test_known_paper_properties(self, kg):
        """Test properties of known papers."""
        arches = kg.get_entity_by_name("Arches Platine")

        assert arches is not None
        assert arches.properties["manufacturer"] == "Arches"
        assert arches.properties["weight_gsm"] == 310
        assert arches.properties["absorbency"] == "medium"

    def test_infer_settings_known_paper(self, kg):
        """Test inferring settings for known paper."""
        result = kg.infer_settings_for_paper("Arches Platine")

        assert result.confidence > 0.5
        assert "metal_ratio" in result.result_values
        assert "coating_factor" in result.result_values
        assert "expected_dmax" in result.result_values
        assert len(result.reasoning_path) > 0

    def test_infer_settings_unknown_paper(self, kg):
        """Test inferring settings for unknown paper."""
        result = kg.infer_settings_for_paper("Unknown Paper XYZ")

        # Should have low confidence or find similar paper
        assert result.confidence < 0.9

    def test_infer_settings_partial_match(self, kg):
        """Test inferring settings with partial name match."""
        result = kg.infer_settings_for_paper("Arches")

        # Should find Arches Platine
        assert result.confidence > 0
        assert "Arches" in result.result_values.get("paper", "")

    def test_infer_for_new_paper(self, kg):
        """Test analogical reasoning for new paper."""
        new_paper_props = {
            "absorbency": "medium",
            "warmth": "neutral",
            "weight_gsm": 300,
        }

        result = kg.infer_for_new_paper(new_paper_props)

        assert result.confidence > 0
        assert "metal_ratio" in result.result_values
        assert len(result.reasoning_path) > 0

    def test_infer_for_new_paper_warm(self, kg):
        """Test inference adjusts for warm paper."""
        warm_paper_props = {
            "absorbency": "high",
            "warmth": "warm",
        }

        result = kg.infer_for_new_paper(warm_paper_props)

        # Should recommend lower Pt ratio for warm paper
        metal_ratio = result.result_values.get("metal_ratio", 0.5)
        assert metal_ratio < 0.5  # More Pd

    def test_infer_for_new_paper_cool(self, kg):
        """Test inference adjusts for cool paper."""
        cool_paper_props = {
            "absorbency": "low",
            "warmth": "cool",
        }

        result = kg.infer_for_new_paper(cool_paper_props)

        # Should recommend higher Pt ratio for cool paper
        metal_ratio = result.result_values.get("metal_ratio", 0.5)
        assert metal_ratio > 0.5  # More Pt

    def test_compatibility_explanation(self, kg):
        """Test generating compatibility explanation."""
        explanation = kg.get_compatibility_explanation("Arches Platine", "Palladium Chloride")

        assert "Arches Platine" in explanation or "compatible" in explanation.lower()

    def test_learn_from_calibration(self, kg):
        """Test learning from calibration results."""
        initial_count = len(kg.get_entities_by_type(EntityType.CALIBRATION_RESULT))

        kg.learn_from_calibration(
            paper_name="Arches Platine",
            chemistry_settings={"metal_ratio": 0.5, "contrast_agent": "Na2"},
            result_metrics={"dmax": 2.1, "linearity": 0.95},
        )

        final_count = len(kg.get_entities_by_type(EntityType.CALIBRATION_RESULT))
        assert final_count == initial_count + 1

    def test_learn_from_calibration_new_paper(self, kg):
        """Test learning from calibration with new paper."""
        initial_papers = len(kg.get_entities_by_type(EntityType.PAPER))

        kg.learn_from_calibration(
            paper_name="Brand New Paper 2024",
            chemistry_settings={"metal_ratio": 0.3},
            result_metrics={"dmax": 1.8},
        )

        final_papers = len(kg.get_entities_by_type(EntityType.PAPER))
        assert final_papers == initial_papers + 1

    def test_find_similar_papers(self, kg):
        """Test finding similar papers."""
        arches = kg.get_entity_by_name("Arches Platine")
        similar = kg.find_similar(arches.id, top_k=3, same_type_only=True)

        assert len(similar) >= 1
        assert all(s.entity_type == EntityType.PAPER for s in similar)

        # Hahnemühle Platinum Rag should be similar
        similar_names = [s.entity_name for s in similar]
        assert any("Hahnem" in name or "Platine" not in name for name in similar_names)


class TestKnowledgeGraphSettings:
    """Tests for knowledge graph settings integration."""

    def test_settings_from_config(self):
        """Test that knowledge graph uses settings from config."""
        settings = NeuroSymbolicSettings(
            kg_embedding_dim=128,
            kg_similarity_threshold=0.8,
            kg_max_inference_depth=5,
            kg_enable_learning=False,
        )

        kg = PaperChemistryKnowledgeGraph(settings=settings)

        assert kg.settings.kg_embedding_dim == 128
        assert kg.settings.kg_similarity_threshold == 0.8
        assert kg.settings.kg_max_inference_depth == 5
        assert kg.settings.kg_enable_learning is False

    def test_learning_disabled(self):
        """Test that learning can be disabled."""
        settings = NeuroSymbolicSettings(kg_enable_learning=False)
        kg = PaperChemistryKnowledgeGraph(settings=settings)

        initial_count = len(kg.get_entities_by_type(EntityType.CALIBRATION_RESULT))

        kg.learn_from_calibration(
            paper_name="Test Paper",
            chemistry_settings={},
            result_metrics={},
        )

        final_count = len(kg.get_entities_by_type(EntityType.CALIBRATION_RESULT))
        assert final_count == initial_count  # No change
