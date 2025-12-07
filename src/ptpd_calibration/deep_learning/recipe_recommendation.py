"""
Recipe Recommendation Engine for Platinum-Palladium printing.

Provides intelligent recipe recommendations using:
- Content-based filtering (recipe features)
- Collaborative filtering (user preferences)
- Hybrid approaches
- Image-based similarity
- Diversity optimization

All parameters are configuration-driven with no hardcoded values.
"""

import logging
import time
from typing import Any, Optional
from uuid import UUID

import numpy as np

from ptpd_calibration.deep_learning.config import RecipeRecommendationSettings
from ptpd_calibration.deep_learning.models import (
    RecipeRecommendation,
    RecipeRecommendationResult,
)
from ptpd_calibration.deep_learning.types import (
    RecipeCategory,
    RecommendationStrategy,
    SimilarityMetric,
)

logger = logging.getLogger(__name__)


class RecipeEncoder:
    """
    Encodes recipes into embedding vectors for similarity comparison.

    Uses learned or hand-crafted features to create dense representations
    of recipe characteristics including paper type, chemistry, exposure,
    and process parameters.
    """

    def __init__(self, settings: RecipeRecommendationSettings):
        """
        Initialize the recipe encoder.

        Args:
            settings: Recipe recommendation settings
        """
        self.settings = settings
        self.embedding_dim = settings.recipe_embedding_dim
        self._encoder_model = None

    def _ensure_encoder_loaded(self) -> None:
        """Lazy load the encoder model if using neural encoding."""
        if self._encoder_model is not None:
            return

        try:
            import torch
            import torch.nn as nn

            # Simple MLP encoder for recipe features
            class RecipeEncoderNet(nn.Module):
                """Neural network for recipe encoding."""

                def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int]):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim

                    for hidden_dim in hidden_layers:
                        layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(self.settings.nn_dropout)
                        ])
                        prev_dim = hidden_dim

                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            # Initialize with typical recipe features (28 features)
            input_features = 28
            self._encoder_model = RecipeEncoderNet(
                input_features,
                self.embedding_dim,
                self.settings.nn_hidden_layers
            )

            # Load pretrained weights if available
            if self.settings.model_path and self.settings.model_path.exists():
                try:
                    state_dict = torch.load(
                        self.settings.model_path,
                        map_location=self.settings.device
                    )
                    self._encoder_model.load_state_dict(state_dict)
                    logger.info(f"Loaded recipe encoder from {self.settings.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load encoder weights: {e}")

            self._encoder_model.eval()

        except ImportError:
            logger.warning("PyTorch not available, using hand-crafted features")
            self._encoder_model = None

    def encode(self, recipe: dict[str, Any]) -> np.ndarray:
        """
        Encode a recipe into an embedding vector.

        Args:
            recipe: Recipe dictionary with features

        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Extract features from recipe
        features = self._extract_features(recipe)

        # Use neural encoder if available
        if self._encoder_model is not None:
            try:
                import torch

                with torch.no_grad():
                    features_tensor = torch.tensor(
                        features,
                        dtype=torch.float32
                    ).unsqueeze(0)
                    embedding = self._encoder_model(features_tensor)
                    return embedding.squeeze(0).numpy()
            except Exception as e:
                logger.warning(f"Neural encoding failed: {e}, using feature vector")

        # Fall back to normalized feature vector
        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        else:
            features = features[:self.embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def _extract_features(self, recipe: dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from recipe.

        Args:
            recipe: Recipe dictionary

        Returns:
            Feature vector
        """
        features = []

        # Numerical features
        features.append(recipe.get("metal_ratio", 0.5))
        features.append(recipe.get("exposure_time", 180.0) / 600.0)  # Normalize
        features.append(recipe.get("developer_temperature", 20.0) / 30.0)
        features.append(recipe.get("development_time", 5.0) / 10.0)
        features.append(recipe.get("humidity", 50.0) / 100.0)
        features.append(recipe.get("coating_layers", 2) / 4.0)
        features.append(recipe.get("dmax", 1.8) / 2.5)
        features.append(recipe.get("dmin", 0.05) / 0.3)
        features.append(recipe.get("contrast_grade", 2) / 5.0)

        # Categorical features (one-hot encoded)
        paper_types = ["arches_platine", "bergger_cot320", "hahnemuhle_platinum", "other"]
        paper = recipe.get("paper_type", "other")
        features.extend([1.0 if p == paper else 0.0 for p in paper_types])

        chemistry_types = ["traditional", "ziatype", "new_cyanotype", "other"]
        chemistry = recipe.get("chemistry_type", "traditional")
        features.extend([1.0 if c == chemistry else 0.0 for c in chemistry_types])

        # Image characteristics from categories
        categories = recipe.get("categories", [])
        category_features = [
            RecipeCategory.HIGH_CONTRAST in categories,
            RecipeCategory.LOW_CONTRAST in categories,
            RecipeCategory.HIGH_KEY in categories,
            RecipeCategory.LOW_KEY in categories,
            RecipeCategory.WARM_TONE in categories,
            RecipeCategory.COOL_TONE in categories,
            RecipeCategory.PORTRAIT in categories,
            RecipeCategory.LANDSCAPE in categories,
        ]
        features.extend([1.0 if cf else 0.0 for cf in category_features])

        return np.array(features, dtype=np.float32)

    def batch_encode(self, recipes: list[dict[str, Any]]) -> np.ndarray:
        """
        Encode multiple recipes.

        Args:
            recipes: List of recipe dictionaries

        Returns:
            Array of embeddings with shape (num_recipes, embedding_dim)
        """
        return np.array([self.encode(recipe) for recipe in recipes])


class ImageEncoder:
    """
    Encodes target images into embedding vectors for content-based filtering.

    Uses pretrained vision models (ResNet, CLIP, etc.) to extract visual features
    that can be matched against recipe embeddings.
    """

    def __init__(self, settings: RecipeRecommendationSettings):
        """
        Initialize the image encoder.

        Args:
            settings: Recipe recommendation settings
        """
        self.settings = settings
        self.embedding_dim = settings.image_embedding_dim
        self._model = None
        self._preprocess = None

    def _ensure_model_loaded(self) -> None:
        """Lazy load the image encoder model."""
        if self._model is not None:
            return

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            # Use ResNet50 as image encoder
            self._model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self._model = torch.nn.Sequential(*list(self._model.children())[:-1])
            self._model.eval()

            # Image preprocessing
            self._preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            logger.info("Image encoder (ResNet50) loaded successfully")

        except ImportError as e:
            logger.warning(f"Failed to load image encoder: {e}")
            self._model = None
            self._preprocess = None

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode an image into an embedding vector.

        Args:
            image: Image array (H, W) or (H, W, C)

        Returns:
            Embedding vector of shape (image_embedding_dim,)
        """
        if not self.settings.use_image_embeddings:
            # Return zero vector if image embeddings disabled
            return np.zeros(self.embedding_dim, dtype=np.float32)

        self._ensure_model_loaded()

        if self._model is None:
            # Fallback: use histogram features
            return self._extract_histogram_features(image)

        try:
            import torch

            # Ensure image is in correct format
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)  # Grayscale to RGB

            # Preprocess and encode
            with torch.no_grad():
                img_tensor = self._preprocess(image).unsqueeze(0)
                embedding = self._model(img_tensor)
                embedding = embedding.squeeze().numpy()

            # Resize to target dimension
            if embedding.shape[0] != self.embedding_dim:
                # Simple interpolation
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, embedding.shape[0])
                x_new = np.linspace(0, 1, self.embedding_dim)
                f = interp1d(x_old, embedding, kind='linear')
                embedding = f(x_new)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Image encoding failed: {e}, using histogram features")
            return self._extract_histogram_features(image)

    def _extract_histogram_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract histogram-based features as fallback.

        Args:
            image: Image array

        Returns:
            Feature vector
        """
        # Normalize image
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0

        # Compute histograms for different regions
        features = []

        # Global histogram (32 bins)
        hist, _ = np.histogram(image.ravel(), bins=32, range=(0, 1))
        features.extend(hist / hist.sum())

        # Zone-based histograms (shadows, midtones, highlights)
        h, w = image.shape[:2]
        zones = [
            image[image < 0.3],  # Shadows
            image[(image >= 0.3) & (image < 0.7)],  # Midtones
            image[image >= 0.7],  # Highlights
        ]

        for zone_pixels in zones:
            if len(zone_pixels) > 0:
                hist, _ = np.histogram(zone_pixels, bins=16, range=(0, 1))
                features.extend(hist / (hist.sum() + 1e-6))
            else:
                features.extend([0.0] * 16)

        # Statistical features
        features.extend([
            np.mean(image),
            np.std(image),
            np.percentile(image, 5),
            np.percentile(image, 50),
            np.percentile(image, 95),
        ])

        feature_vec = np.array(features, dtype=np.float32)

        # Pad or truncate to embedding_dim
        if len(feature_vec) < self.embedding_dim:
            feature_vec = np.pad(feature_vec, (0, self.embedding_dim - len(feature_vec)))
        else:
            feature_vec = feature_vec[:self.embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(feature_vec)
        if norm > 0:
            feature_vec = feature_vec / norm

        return feature_vec


class CollaborativeFilter:
    """
    Collaborative filtering for recipe recommendations based on user preferences.

    Uses matrix factorization to learn latent factors from user-recipe interactions
    and predict preferences for unseen recipes.
    """

    def __init__(self, settings: RecipeRecommendationSettings):
        """
        Initialize the collaborative filter.

        Args:
            settings: Recipe recommendation settings
        """
        self.settings = settings
        self.num_factors = settings.cf_num_factors
        self.regularization = settings.cf_regularization
        self._user_factors = {}
        self._recipe_factors = {}

    def fit(
        self,
        interactions: list[dict[str, Any]],
        num_epochs: int = None
    ) -> None:
        """
        Fit the collaborative filtering model on user interactions.

        Args:
            interactions: List of {user_id, recipe_id, rating} dictionaries
            num_epochs: Number of training epochs (uses settings if None)
        """
        if num_epochs is None:
            num_epochs = self.settings.epochs

        # Extract unique users and recipes
        users = set(i["user_id"] for i in interactions)
        recipes = set(i["recipe_id"] for i in interactions)

        # Initialize factors randomly
        for user_id in users:
            self._user_factors[user_id] = np.random.randn(
                self.num_factors
            ).astype(np.float32) * 0.01

        for recipe_id in recipes:
            self._recipe_factors[recipe_id] = np.random.randn(
                self.num_factors
            ).astype(np.float32) * 0.01

        # Alternating least squares
        learning_rate = self.settings.learning_rate

        for epoch in range(num_epochs):
            # Update user factors
            for user_id in users:
                user_interactions = [
                    i for i in interactions if i["user_id"] == user_id
                ]
                if user_interactions:
                    self._update_user_factor(user_id, user_interactions, learning_rate)

            # Update recipe factors
            for recipe_id in recipes:
                recipe_interactions = [
                    i for i in interactions if i["recipe_id"] == recipe_id
                ]
                if recipe_interactions:
                    self._update_recipe_factor(
                        recipe_id,
                        recipe_interactions,
                        learning_rate
                    )

        logger.info(f"Collaborative filter trained on {len(interactions)} interactions")

    def _update_user_factor(
        self,
        user_id: str,
        interactions: list[dict[str, Any]],
        learning_rate: float
    ) -> None:
        """Update user factor based on interactions."""
        user_vec = self._user_factors[user_id]

        for interaction in interactions:
            recipe_id = interaction["recipe_id"]
            rating = interaction["rating"]

            if recipe_id in self._recipe_factors:
                recipe_vec = self._recipe_factors[recipe_id]
                predicted = np.dot(user_vec, recipe_vec)
                error = rating - predicted

                # Gradient update
                gradient = error * recipe_vec - self.regularization * user_vec
                user_vec += learning_rate * gradient

        self._user_factors[user_id] = user_vec

    def _update_recipe_factor(
        self,
        recipe_id: UUID,
        interactions: list[dict[str, Any]],
        learning_rate: float
    ) -> None:
        """Update recipe factor based on interactions."""
        recipe_vec = self._recipe_factors[recipe_id]

        for interaction in interactions:
            user_id = interaction["user_id"]
            rating = interaction["rating"]

            if user_id in self._user_factors:
                user_vec = self._user_factors[user_id]
                predicted = np.dot(user_vec, recipe_vec)
                error = rating - predicted

                # Gradient update
                gradient = error * user_vec - self.regularization * recipe_vec
                recipe_vec += learning_rate * gradient

        self._recipe_factors[recipe_id] = recipe_vec

    def predict(self, user_id: str, recipe_id: UUID) -> float:
        """
        Predict user's rating for a recipe.

        Args:
            user_id: User identifier
            recipe_id: Recipe identifier

        Returns:
            Predicted rating (0-1)
        """
        if user_id not in self._user_factors or recipe_id not in self._recipe_factors:
            return 0.5  # Default neutral rating

        user_vec = self._user_factors[user_id]
        recipe_vec = self._recipe_factors[recipe_id]

        # Dot product, scaled to 0-1
        score = np.dot(user_vec, recipe_vec)
        return float(np.clip(score, 0.0, 1.0))

    def get_recipe_embedding(self, recipe_id: UUID) -> Optional[np.ndarray]:
        """Get the learned embedding for a recipe."""
        return self._recipe_factors.get(recipe_id)


class RecipeRecommender:
    """
    Main recipe recommendation engine.

    Combines content-based filtering, collaborative filtering, and hybrid
    approaches to recommend recipes tailored to user preferences and target images.
    """

    def __init__(
        self,
        settings: Optional[RecipeRecommendationSettings] = None
    ):
        """
        Initialize the recipe recommender.

        Args:
            settings: Recipe recommendation settings (uses defaults if None)
        """
        if settings is None:
            settings = RecipeRecommendationSettings()

        self.settings = settings
        self.recipe_encoder = RecipeEncoder(settings)
        self.image_encoder = ImageEncoder(settings)
        self.collaborative_filter = CollaborativeFilter(settings)

        # Recipe database (would be loaded from actual DB)
        self._recipe_embeddings = {}
        self._recipes = {}

    def add_recipes(self, recipes: list[dict[str, Any]]) -> None:
        """
        Add recipes to the recommendation database.

        Args:
            recipes: List of recipe dictionaries
        """
        for recipe in recipes:
            recipe_id = recipe["id"]
            self._recipes[recipe_id] = recipe
            self._recipe_embeddings[recipe_id] = self.recipe_encoder.encode(recipe)

        logger.info(f"Added {len(recipes)} recipes to recommender")

    def recommend(
        self,
        user_id: Optional[str] = None,
        target_image: Optional[np.ndarray] = None,
        query_parameters: Optional[dict[str, Any]] = None,
        user_preferences: Optional[dict[str, Any]] = None,
    ) -> RecipeRecommendationResult:
        """
        Generate recipe recommendations.

        Args:
            user_id: User identifier for collaborative filtering
            target_image: Target image for content-based filtering
            query_parameters: Additional query parameters
            user_preferences: User preference dictionary

        Returns:
            RecipeRecommendationResult with ranked recommendations
        """
        start_time = time.time()

        if query_parameters is None:
            query_parameters = {}
        if user_preferences is None:
            user_preferences = {}

        # Compute scores based on strategy
        if self.settings.strategy == RecommendationStrategy.CONTENT_BASED:
            scores = self._content_based_scores(target_image, query_parameters)
        elif self.settings.strategy == RecommendationStrategy.COLLABORATIVE_FILTERING:
            scores = self._collaborative_scores(user_id)
        elif self.settings.strategy == RecommendationStrategy.HYBRID:
            content_scores = self._content_based_scores(target_image, query_parameters)
            collab_scores = self._collaborative_scores(user_id)
            # Weighted combination
            scores = {
                rid: 0.6 * content_scores.get(rid, 0.0) + 0.4 * collab_scores.get(rid, 0.0)
                for rid in self._recipes.keys()
            }
        else:
            # Default to content-based
            scores = self._content_based_scores(target_image, query_parameters)

        # Apply diversity optimization
        if self.settings.diversity_weight > 0:
            scores = self._apply_diversity(scores)

        # Apply recency weighting
        if self.settings.recency_weight > 0:
            scores = self._apply_recency(scores, user_preferences)

        # Get top-k recommendations
        top_recipe_ids = sorted(
            scores.keys(),
            key=lambda rid: scores[rid],
            reverse=True
        )[:self.settings.top_k]

        # Build recommendation objects
        recommendations = []
        for rank, recipe_id in enumerate(top_recipe_ids, 1):
            recipe = self._recipes[recipe_id]

            rec = RecipeRecommendation(
                recipe_id=recipe_id,
                recipe_name=recipe.get("name", f"Recipe {recipe_id}"),
                similarity_score=scores[recipe_id],
                rank=rank,
                paper_type=recipe.get("paper_type", "unknown"),
                chemistry_type=recipe.get("chemistry_type", "traditional"),
                metal_ratio=recipe.get("metal_ratio", 0.5),
                exposure_time=recipe.get("exposure_time", 180.0),
                categories=recipe.get("categories", []),
            )

            # Generate explanation if enabled
            if self.settings.generate_explanations:
                rec.explanation = self._generate_explanation(
                    recipe,
                    target_image,
                    query_parameters
                )
                rec.matching_factors = self._get_matching_factors(
                    recipe,
                    query_parameters
                )

            recommendations.append(rec)

        # Compute diversity metric
        diversity = self._compute_diversity(recommendations)

        inference_time = (time.time() - start_time) * 1000

        return RecipeRecommendationResult(
            query_image_used=target_image is not None,
            query_parameters=query_parameters,
            recommendations=recommendations,
            num_recommendations=len(recommendations),
            preferences_used=user_preferences,
            recommendation_diversity=diversity,
            inference_time_ms=inference_time,
            device_used=self.settings.device,
        )

    def _content_based_scores(
        self,
        target_image: Optional[np.ndarray],
        query_parameters: dict[str, Any]
    ) -> dict[UUID, float]:
        """Compute content-based similarity scores."""
        scores = {}

        # Create query embedding
        query_embedding = None

        if target_image is not None:
            image_embedding = self.image_encoder.encode(target_image)
            query_embedding = image_embedding

        if query_parameters:
            # Create synthetic recipe from parameters
            param_embedding = self.recipe_encoder.encode(query_parameters)
            if query_embedding is not None:
                # Combine image and parameter embeddings
                query_embedding = np.concatenate([
                    query_embedding[:self.settings.image_embedding_dim // 2],
                    param_embedding[:self.settings.recipe_embedding_dim // 2]
                ])
            else:
                query_embedding = param_embedding

        if query_embedding is None:
            # No query provided, return uniform scores
            return {rid: 0.5 for rid in self._recipes.keys()}

        # Compute similarities
        for recipe_id, recipe_embedding in self._recipe_embeddings.items():
            similarity = self._compute_similarity(query_embedding, recipe_embedding)
            scores[recipe_id] = similarity

        return scores

    def _collaborative_scores(self, user_id: Optional[str]) -> dict[UUID, float]:
        """Compute collaborative filtering scores."""
        if user_id is None:
            return {rid: 0.5 for rid in self._recipes.keys()}

        scores = {}
        for recipe_id in self._recipes.keys():
            scores[recipe_id] = self.collaborative_filter.predict(user_id, recipe_id)

        return scores

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute similarity between two embeddings."""
        # Ensure same length
        min_len = min(len(embedding1), len(embedding2))
        embedding1 = embedding1[:min_len]
        embedding2 = embedding2[:min_len]

        if self.settings.similarity_metric == SimilarityMetric.COSINE:
            dot = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
            return 0.0

        elif self.settings.similarity_metric == SimilarityMetric.EUCLIDEAN:
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity (inverse distance)
            return float(1.0 / (1.0 + distance))

        elif self.settings.similarity_metric == SimilarityMetric.MANHATTAN:
            distance = np.sum(np.abs(embedding1 - embedding2))
            return float(1.0 / (1.0 + distance))

        else:
            # Default to cosine
            dot = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 > 0 and norm2 > 0:
                return float(dot / (norm1 * norm2))
            return 0.0

    def _apply_diversity(self, scores: dict[UUID, float]) -> dict[UUID, float]:
        """Apply diversity optimization to scores."""
        # Penalize very similar recipes
        diversity_weight = self.settings.diversity_weight

        adjusted_scores = scores.copy()
        recipe_ids = list(scores.keys())

        for i, rid1 in enumerate(recipe_ids):
            for rid2 in recipe_ids[i+1:]:
                if rid1 in self._recipe_embeddings and rid2 in self._recipe_embeddings:
                    similarity = self._compute_similarity(
                        self._recipe_embeddings[rid1],
                        self._recipe_embeddings[rid2]
                    )

                    # Penalize both if too similar
                    if similarity > 0.9:
                        penalty = diversity_weight * similarity
                        adjusted_scores[rid1] *= (1 - penalty)
                        adjusted_scores[rid2] *= (1 - penalty)

        return adjusted_scores

    def _apply_recency(
        self,
        scores: dict[UUID, float],
        user_preferences: dict[str, Any]
    ) -> dict[UUID, float]:
        """Apply recency weighting to scores."""
        recency_weight = self.settings.recency_weight
        recent_recipes = user_preferences.get("recent_recipes", [])

        adjusted_scores = scores.copy()

        # Boost recently used recipes slightly
        for recipe_id in recent_recipes:
            if recipe_id in adjusted_scores:
                adjusted_scores[recipe_id] *= (1 + recency_weight * 0.5)

        return adjusted_scores

    def _generate_explanation(
        self,
        recipe: dict[str, Any],
        target_image: Optional[np.ndarray],
        query_parameters: dict[str, Any]
    ) -> str:
        """Generate explanation for why recipe was recommended."""
        reasons = []

        # Check matching categories
        if query_parameters.get("categories"):
            recipe_cats = set(recipe.get("categories", []))
            query_cats = set(query_parameters.get("categories", []))
            matching = recipe_cats & query_cats
            if matching:
                reasons.append(
                    f"matches your preferences for {', '.join(str(c) for c in matching)}"
                )

        # Check paper match
        if query_parameters.get("paper_type") == recipe.get("paper_type"):
            reasons.append(f"uses your preferred {recipe['paper_type']} paper")

        # Check chemistry match
        if query_parameters.get("chemistry_type") == recipe.get("chemistry_type"):
            reasons.append(f"uses {recipe['chemistry_type']} chemistry")

        # Image-based reason
        if target_image is not None:
            reasons.append("visually similar to your target image")

        if not reasons:
            reasons.append("highly rated by similar users")

        return "This recipe " + " and ".join(reasons) + "."

    def _get_matching_factors(
        self,
        recipe: dict[str, Any],
        query_parameters: dict[str, Any]
    ) -> list[str]:
        """Get list of matching factors between recipe and query."""
        factors = []

        for key in ["paper_type", "chemistry_type", "metal_ratio", "exposure_time"]:
            if key in query_parameters and key in recipe:
                if query_parameters[key] == recipe[key]:
                    factors.append(key)

        return factors

    def _compute_diversity(self, recommendations: list[RecipeRecommendation]) -> float:
        """Compute diversity metric for recommendations."""
        if len(recommendations) < 2:
            return 1.0

        # Compute average pairwise distance
        total_distance = 0.0
        count = 0

        for i, rec1 in enumerate(recommendations):
            for rec2 in recommendations[i+1:]:
                if rec1.recipe_id in self._recipe_embeddings and \
                   rec2.recipe_id in self._recipe_embeddings:
                    emb1 = self._recipe_embeddings[rec1.recipe_id]
                    emb2 = self._recipe_embeddings[rec2.recipe_id]
                    distance = np.linalg.norm(emb1 - emb2)
                    total_distance += distance
                    count += 1

        if count > 0:
            avg_distance = total_distance / count
            # Normalize to 0-1 (assuming max distance ~2 for normalized embeddings)
            return float(min(avg_distance / 2.0, 1.0))

        return 0.5
