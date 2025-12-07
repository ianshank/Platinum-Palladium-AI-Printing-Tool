"""
Configuration for deep learning components.

All settings are environment-configurable with PTPD_DL_ prefix.
No hardcoded values - everything is configurable.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ptpd_calibration.deep_learning.types import (
    AggregationStrategy,
    ComparisonMode,
    ControlNetType,
    CurveLossFunction,
    CurvePredictorArchitecture,
    DefectDetectorArchitecture,
    DetectionBackend,
    DiffusionModelType,
    DiffusionScheduler,
    EnhancementMode,
    IQAMetric,
    PerceptualMetric,
    PrivacyLevel,
    RecommendationStrategy,
    SegmentationBackend,
    SimilarityMetric,
    UncertaintyMethod,
    VisionLanguageModel,
)


class DetectionModelSettings(BaseSettings):
    """Settings for deep learning-based detection (YOLOv8 + SAM)."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_DETECTION_")

    # Model selection
    detection_backend: DetectionBackend = Field(
        default=DetectionBackend.YOLOV8,
        description="Object detection model backend",
    )
    segmentation_backend: SegmentationBackend = Field(
        default=SegmentationBackend.SAM,
        description="Instance segmentation model backend",
    )

    # YOLOv8 settings
    yolo_model_size: str = Field(
        default="m",
        description="YOLO model size: n (nano), s (small), m (medium), l (large), x (xlarge)",
    )
    yolo_confidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for detections",
    )
    yolo_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )
    yolo_max_detections: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of detections per image",
    )

    # SAM settings
    sam_model_type: str = Field(
        default="vit_h",
        description="SAM model type: vit_b, vit_l, vit_h",
    )
    sam_points_per_side: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Points per side for automatic mask generation",
    )
    sam_pred_iou_threshold: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Predicted IoU threshold for masks",
    )
    sam_stability_score_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Stability score threshold for masks",
    )

    # Hardware settings
    device: str = Field(
        default="auto",
        description="Device: auto, cpu, cuda, cuda:0, mps",
    )
    half_precision: bool = Field(
        default=True,
        description="Use FP16 for inference (faster on GPU)",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Batch size for inference",
    )

    # Model paths (optional custom models)
    custom_yolo_weights: Optional[Path] = Field(
        default=None,
        description="Path to custom YOLO weights",
    )
    custom_sam_checkpoint: Optional[Path] = Field(
        default=None,
        description="Path to custom SAM checkpoint",
    )

    # Preprocessing
    image_size: int = Field(
        default=640,
        ge=320,
        le=1280,
        description="Input image size for detection",
    )
    normalize_input: bool = Field(
        default=True,
        description="Normalize input images",
    )

    # Post-processing
    min_patch_area_ratio: float = Field(
        default=0.005,
        ge=0.001,
        le=0.1,
        description="Minimum patch area as ratio of image",
    )
    max_patch_area_ratio: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Maximum patch area as ratio of image",
    )
    merge_overlapping_masks: bool = Field(
        default=True,
        description="Merge overlapping segmentation masks",
    )

    # Fallback settings
    fallback_to_classical: bool = Field(
        default=True,
        description="Fall back to classical CV if DL fails",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries before fallback",
    )


class ImageQualitySettings(BaseSettings):
    """Settings for Vision Transformer-based image quality assessment."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_IQA_")

    # Model selection
    primary_metric: IQAMetric = Field(
        default=IQAMetric.MANIQA,
        description="Primary IQA metric",
    )
    secondary_metrics: list[IQAMetric] = Field(
        default_factory=lambda: [IQAMetric.CLIP_IQA, IQAMetric.MUSIQ],
        description="Additional metrics to compute",
    )

    # Model settings
    vit_model_name: str = Field(
        default="vit_base_patch16_224",
        description="Vision Transformer model architecture",
    )
    pretrained_weights: str = Field(
        default="imagenet",
        description="Pretrained weights: imagenet, clip, custom",
    )
    custom_weights_path: Optional[Path] = Field(
        default=None,
        description="Path to custom IQA model weights",
    )

    # Inference settings
    input_size: int = Field(
        default=224,
        ge=224,
        le=512,
        description="Input size for ViT",
    )
    num_crops: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of crops for multi-crop evaluation",
    )
    use_multi_scale: bool = Field(
        default=True,
        description="Use multi-scale evaluation",
    )

    # Zone-based analysis
    analyze_zones: bool = Field(
        default=True,
        description="Perform zone-based quality analysis",
    )
    zone_count: int = Field(
        default=11,
        ge=5,
        le=21,
        description="Number of tonal zones to analyze",
    )

    # Quality thresholds
    excellent_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Threshold for excellent quality",
    )
    good_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Threshold for good quality",
    )
    acceptable_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for acceptable quality",
    )
    poor_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Threshold for poor quality",
    )

    # Hardware
    device: str = Field(default="auto")
    batch_size: int = Field(default=8, ge=1, le=64)

    # Caching
    cache_embeddings: bool = Field(
        default=True,
        description="Cache image embeddings for faster comparison",
    )
    cache_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum cached embeddings",
    )


class DiffusionSettings(BaseSettings):
    """Settings for diffusion model-based enhancement."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_DIFFUSION_")

    # Model selection
    model_type: DiffusionModelType = Field(
        default=DiffusionModelType.STABLE_DIFFUSION_XL,
        description="Base diffusion model",
    )
    scheduler: DiffusionScheduler = Field(
        default=DiffusionScheduler.EULER,
        description="Noise scheduler",
    )

    # Enhancement mode
    default_mode: EnhancementMode = Field(
        default=EnhancementMode.TONAL_ENHANCEMENT,
        description="Default enhancement mode",
    )

    # ControlNet settings
    use_controlnet: bool = Field(
        default=True,
        description="Use ControlNet for structure preservation",
    )
    controlnet_type: ControlNetType = Field(
        default=ControlNetType.CANNY,
        description="ControlNet conditioning type",
    )
    controlnet_conditioning_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="ControlNet influence strength",
    )

    # Generation settings
    num_inference_steps: int = Field(
        default=30,
        ge=10,
        le=150,
        description="Number of denoising steps",
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Denoising strength for img2img",
    )

    # LoRA settings
    use_custom_lora: bool = Field(
        default=False,
        description="Use custom LoRA for Pt/Pd aesthetic",
    )
    lora_weights_path: Optional[Path] = Field(
        default=None,
        description="Path to custom LoRA weights",
    )
    lora_scale: float = Field(
        default=0.8,
        ge=0.0,
        le=1.5,
        description="LoRA influence scale",
    )

    # Inpainting settings
    inpaint_mask_blur: int = Field(
        default=4,
        ge=0,
        le=64,
        description="Blur radius for inpainting mask edges",
    )
    inpaint_mask_padding: int = Field(
        default=32,
        ge=0,
        le=128,
        description="Padding around inpainting mask",
    )

    # Style transfer
    style_prompt_template: str = Field(
        default="platinum palladium print, {style}, high quality, detailed tones",
        description="Style prompt template with {style} placeholder",
    )

    # Hardware
    device: str = Field(default="auto")
    half_precision: bool = Field(default=True)
    enable_attention_slicing: bool = Field(
        default=True,
        description="Enable attention slicing for memory efficiency",
    )
    enable_vae_slicing: bool = Field(
        default=True,
        description="Enable VAE slicing for memory efficiency",
    )

    # Safety
    safety_checker: bool = Field(
        default=False,
        description="Enable NSFW safety checker",
    )

    # Output
    output_format: str = Field(
        default="png",
        description="Output format: png, tiff, tiff_16bit",
    )


class NeuralCurveSettings(BaseSettings):
    """Settings for Transformer-based curve prediction."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_CURVE_")

    # Architecture
    architecture: CurvePredictorArchitecture = Field(
        default=CurvePredictorArchitecture.TRANSFORMER,
        description="Neural network architecture",
    )

    # Transformer settings
    d_model: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Model dimension",
    )
    n_heads: int = Field(
        default=8,
        ge=2,
        le=32,
        description="Number of attention heads",
    )
    n_layers: int = Field(
        default=6,
        ge=2,
        le=24,
        description="Number of transformer layers",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate",
    )
    max_sequence_length: int = Field(
        default=64,
        ge=16,
        le=512,
        description="Maximum sequence length",
    )

    # Input/Output
    input_features: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Number of input features",
    )
    output_points: int = Field(
        default=256,
        ge=21,
        le=4096,
        description="Number of output curve points",
    )
    include_conditioning: bool = Field(
        default=True,
        description="Include process conditioning",
    )

    # Training settings
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Learning rate",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Weight decay for regularization",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Training batch size",
    )
    epochs: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Training epochs",
    )
    early_stopping_patience: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Early stopping patience",
    )

    # Loss function
    loss_function: CurveLossFunction = Field(
        default=CurveLossFunction.MONOTONIC_MSE,
        description="Loss function for training",
    )
    monotonicity_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for monotonicity penalty",
    )
    smoothness_weight: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Weight for smoothness penalty",
    )

    # Uncertainty
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.ENSEMBLE,
        description="Uncertainty quantification method",
    )
    ensemble_size: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of models in ensemble",
    )
    mc_dropout_samples: int = Field(
        default=30,
        ge=10,
        le=100,
        description="MC dropout samples for uncertainty",
    )

    # Model paths
    pretrained_model_path: Optional[Path] = Field(
        default=None,
        description="Path to pretrained model",
    )
    save_model_path: Optional[Path] = Field(
        default=None,
        description="Path to save trained model",
    )

    # Hardware
    device: str = Field(default="auto")
    mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision training",
    )


class UVExposureSettings(BaseSettings):
    """Settings for neural UV exposure prediction."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_EXPOSURE_")

    # Model architecture
    model_architecture: str = Field(
        default="mlp_with_residual",
        description="Model architecture: mlp, mlp_with_residual, transformer",
    )
    hidden_layers: list[int] = Field(
        default_factory=lambda: [256, 128, 64],
        description="Hidden layer sizes",
    )
    activation: str = Field(
        default="gelu",
        description="Activation function: relu, gelu, silu, mish",
    )

    # Input features
    input_features: list[str] = Field(
        default_factory=lambda: [
            "target_density",
            "paper_type",
            "chemistry_ratio",
            "uv_source",
            "humidity",
            "temperature",
            "coating_thickness",
            "negative_dmax",
        ],
        description="Input feature names",
    )
    categorical_features: list[str] = Field(
        default_factory=lambda: ["paper_type", "uv_source"],
        description="Categorical feature names (will be embedded)",
    )
    embedding_dim: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Embedding dimension for categorical features",
    )
    max_paper_types: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum number of paper types for embedding layer",
    )

    # Output
    predict_confidence_interval: bool = Field(
        default=True,
        description="Predict confidence interval",
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for intervals",
    )

    # Training
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=64, ge=1, le=512)
    epochs: int = Field(default=200, ge=10, le=1000)

    # Uncertainty
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.ENSEMBLE,
    )
    ensemble_size: int = Field(default=5, ge=2, le=20)

    # Model paths
    model_path: Optional[Path] = Field(default=None)

    # Hardware
    device: str = Field(default="auto")


class DefectDetectionSettings(BaseSettings):
    """Settings for U-Net + ResNet defect detection."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_DEFECT_")

    # Segmentation model
    segmentation_architecture: DefectDetectorArchitecture = Field(
        default=DefectDetectorArchitecture.UNET_PLUS_PLUS,
        description="Segmentation architecture",
    )
    segmentation_encoder: str = Field(
        default="resnet50",
        description="Encoder backbone: resnet34, resnet50, efficientnet-b4",
    )
    pretrained_encoder: bool = Field(
        default=True,
        description="Use pretrained encoder",
    )

    # Classification model
    classifier_architecture: DefectDetectorArchitecture = Field(
        default=DefectDetectorArchitecture.RESNET_CLASSIFIER,
        description="Classification architecture",
    )
    classifier_backbone: str = Field(
        default="resnet34",
        description="Classifier backbone",
    )
    num_classes: int = Field(
        default=25,
        ge=2,
        le=100,
        description="Number of defect classes",
    )

    # Detection settings
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections",
    )
    min_defect_area: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Minimum defect area in pixels",
    )
    max_defects_per_image: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum defects per image",
    )

    # Multi-scale detection
    use_multi_scale: bool = Field(
        default=True,
        description="Use multi-scale detection",
    )
    scales: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 1.5],
        description="Scale factors for multi-scale detection",
    )

    # Post-processing
    apply_morphological_cleanup: bool = Field(
        default=True,
        description="Apply morphological cleanup to masks",
    )
    merge_nearby_defects: bool = Field(
        default=True,
        description="Merge nearby defects of same type",
    )
    merge_distance: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Distance for merging defects",
    )

    # Severity estimation
    estimate_severity: bool = Field(
        default=True,
        description="Estimate defect severity",
    )

    # Model paths
    segmentation_model_path: Optional[Path] = Field(default=None)
    classifier_model_path: Optional[Path] = Field(default=None)

    # Training
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=16, ge=1, le=64)
    epochs: int = Field(default=100, ge=10, le=500)

    # Hardware
    device: str = Field(default="auto")
    mixed_precision: bool = Field(default=True)


class RecipeRecommendationSettings(BaseSettings):
    """Settings for recipe recommendation engine."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_RECIPE_")

    # Strategy
    strategy: RecommendationStrategy = Field(
        default=RecommendationStrategy.HYBRID,
        description="Recommendation strategy",
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE,
        description="Similarity metric",
    )

    # Embedding settings
    recipe_embedding_dim: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Recipe embedding dimension",
    )
    image_embedding_dim: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Image embedding dimension",
    )
    use_image_embeddings: bool = Field(
        default=True,
        description="Include image embeddings in recommendations",
    )

    # Collaborative filtering
    cf_num_factors: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Number of latent factors for CF",
    )
    cf_regularization: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Regularization for CF",
    )

    # Neural network settings
    nn_hidden_layers: list[int] = Field(
        default_factory=lambda: [256, 128, 64],
        description="Hidden layer sizes for neural model",
    )
    nn_dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Dropout rate",
    )

    # Recommendation settings
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return",
    )
    diversity_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for recommendation diversity",
    )
    recency_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for recipe recency",
    )

    # Explanation
    generate_explanations: bool = Field(
        default=True,
        description="Generate explanations for recommendations",
    )

    # Model paths
    model_path: Optional[Path] = Field(default=None)
    embeddings_path: Optional[Path] = Field(default=None)

    # Training
    learning_rate: float = Field(default=1e-3, ge=1e-6, le=1e-2)
    batch_size: int = Field(default=128, ge=1, le=512)
    epochs: int = Field(default=50, ge=10, le=500)

    # Hardware
    device: str = Field(default="auto")


class PrintComparisonSettings(BaseSettings):
    """Settings for LPIPS-based print comparison."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_COMPARISON_")

    # Primary metric
    primary_metric: PerceptualMetric = Field(
        default=PerceptualMetric.LPIPS_ALEX,
        description="Primary perceptual metric",
    )
    additional_metrics: list[PerceptualMetric] = Field(
        default_factory=lambda: [PerceptualMetric.SSIM, PerceptualMetric.DISTS],
        description="Additional metrics to compute",
    )

    # Comparison mode
    comparison_mode: ComparisonMode = Field(
        default=ComparisonMode.ZONE_BASED,
        description="Comparison mode",
    )

    # Zone-based settings
    zone_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "shadows": 1.0,
            "midtones": 1.5,
            "highlights": 1.0,
        },
        description="Weights for zone-based comparison",
    )

    # LPIPS settings
    lpips_net: str = Field(
        default="alex",
        description="LPIPS network: alex, vgg, squeeze",
    )
    lpips_spatial: bool = Field(
        default=True,
        description="Return spatial LPIPS map",
    )

    # Multi-scale
    use_multi_scale: bool = Field(
        default=True,
        description="Use multi-scale comparison",
    )
    scales: list[float] = Field(
        default_factory=lambda: [1.0, 0.5, 0.25],
        description="Scale factors",
    )

    # Thresholds
    identical_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Threshold for identical classification",
    )
    similar_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Threshold for similar classification",
    )
    different_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Threshold for different classification",
    )

    # Attention visualization
    generate_attention_maps: bool = Field(
        default=True,
        description="Generate attention/difference maps",
    )
    highlight_differences: bool = Field(
        default=True,
        description="Highlight difference regions",
    )

    # Hardware
    device: str = Field(default="auto")
    batch_size: int = Field(default=8, ge=1, le=32)


class MultiModalSettings(BaseSettings):
    """Settings for multi-modal AI assistant."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_MULTIMODAL_")

    # Model selection
    vision_language_model: VisionLanguageModel = Field(
        default=VisionLanguageModel.CLAUDE_4_SONNET,
        description="Vision-language model to use",
    )
    fallback_model: VisionLanguageModel = Field(
        default=VisionLanguageModel.GPT_4O,
        description="Fallback model if primary fails",
    )

    # API settings (inherit from LLM settings if not specified)
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

    # Image processing
    max_image_size: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Maximum image dimension",
    )
    image_detail: str = Field(
        default="high",
        description="Image detail level: low, high, auto",
    )
    max_images_per_request: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum images per request",
    )

    # Tool settings
    enabled_tools: list[str] = Field(
        default_factory=lambda: [
            "exposure_calculator",
            "chemistry_calculator",
            "curve_adjustment",
            "defect_diagnosis",
            "recipe_lookup",
            "quality_assessment",
        ],
        description="Enabled assistant tools",
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Tool execution timeout",
    )

    # Conversation
    max_conversation_history: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum conversation turns to retain",
    )
    include_image_history: bool = Field(
        default=False,
        description="Include images in conversation history",
    )

    # RAG settings
    use_rag: bool = Field(
        default=True,
        description="Use retrieval-augmented generation",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    rag_chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Document chunk size",
    )

    # Response settings
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32000,
        description="Maximum response tokens",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Response temperature",
    )
    stream_response: bool = Field(
        default=True,
        description="Stream responses",
    )


class FederatedLearningSettings(BaseSettings):
    """Settings for federated community learning."""

    model_config = SettingsConfigDict(env_prefix="PTPD_DL_FEDERATED_")

    # Enable/disable
    enabled: bool = Field(
        default=False,
        description="Enable federated learning",
    )

    # Aggregation
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.FEDAVG,
        description="Model aggregation strategy",
    )
    min_clients_per_round: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Minimum clients per aggregation round",
    )
    max_clients_per_round: int = Field(
        default=10,
        ge=2,
        le=1000,
        description="Maximum clients per aggregation round",
    )

    # Privacy
    privacy_level: PrivacyLevel = Field(
        default=PrivacyLevel.DIFFERENTIAL,
        description="Privacy level",
    )
    differential_privacy_epsilon: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Differential privacy epsilon",
    )
    differential_privacy_delta: float = Field(
        default=1e-5,
        ge=1e-10,
        le=1e-3,
        description="Differential privacy delta",
    )
    clip_norm: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Gradient clipping norm",
    )

    # Communication
    server_address: str = Field(
        default="localhost:8080",
        description="Federation server address",
    )
    communication_rounds: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of communication rounds",
    )
    local_epochs: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Local training epochs per round",
    )

    # Data settings
    min_local_samples: int = Field(
        default=10,
        ge=5,
        le=1000,
        description="Minimum local samples to participate",
    )
    share_sample_count: bool = Field(
        default=True,
        description="Share sample count with server",
    )

    # Model settings
    model_type: str = Field(
        default="curve_predictor",
        description="Model type to federate",
    )
    gradient_compression: bool = Field(
        default=True,
        description="Compress gradients before sending",
    )
    compression_ratio: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Gradient compression ratio",
    )

    # Client settings
    client_id: Optional[str] = Field(
        default=None,
        description="Unique client identifier",
    )
    auto_participate: bool = Field(
        default=False,
        description="Automatically participate in training",
    )


class DeepLearningSettings(BaseSettings):
    """Main settings aggregating all deep learning subsettings."""

    model_config = SettingsConfigDict(
        env_prefix="PTPD_DL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Global settings
    enabled: bool = Field(
        default=True,
        description="Enable deep learning features",
    )
    default_device: str = Field(
        default="auto",
        description="Default device: auto, cpu, cuda, mps",
    )
    cache_models: bool = Field(
        default=True,
        description="Cache loaded models in memory",
    )
    model_cache_dir: Optional[Path] = Field(
        default=None,
        description="Directory for model cache",
    )
    download_models: bool = Field(
        default=True,
        description="Auto-download missing models",
    )
    offline_mode: bool = Field(
        default=False,
        description="Operate in offline mode",
    )

    # Logging
    log_inference_time: bool = Field(
        default=True,
        description="Log inference times",
    )
    log_memory_usage: bool = Field(
        default=False,
        description="Log memory usage",
    )

    # Subsettings
    detection: DetectionModelSettings = Field(
        default_factory=DetectionModelSettings
    )
    image_quality: ImageQualitySettings = Field(
        default_factory=ImageQualitySettings
    )
    diffusion: DiffusionSettings = Field(
        default_factory=DiffusionSettings
    )
    neural_curve: NeuralCurveSettings = Field(
        default_factory=NeuralCurveSettings
    )
    uv_exposure: UVExposureSettings = Field(
        default_factory=UVExposureSettings
    )
    defect_detection: DefectDetectionSettings = Field(
        default_factory=DefectDetectionSettings
    )
    recipe_recommendation: RecipeRecommendationSettings = Field(
        default_factory=RecipeRecommendationSettings
    )
    print_comparison: PrintComparisonSettings = Field(
        default_factory=PrintComparisonSettings
    )
    multimodal: MultiModalSettings = Field(
        default_factory=MultiModalSettings
    )
    federated: FederatedLearningSettings = Field(
        default_factory=FederatedLearningSettings
    )

    @field_validator("model_cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: Optional[Path]) -> Optional[Path]:
        """Resolve cache directory path."""
        if v is None:
            return Path.home() / ".ptpd" / "models"
        return Path(v)

    def get_device(self) -> str:
        """Get the appropriate device string."""
        if self.default_device != "auto":
            return self.default_device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"


# Global settings instance
_deep_learning_settings: Optional[DeepLearningSettings] = None


def get_deep_learning_settings() -> DeepLearningSettings:
    """Get the global deep learning settings instance."""
    global _deep_learning_settings
    if _deep_learning_settings is None:
        _deep_learning_settings = DeepLearningSettings()
    return _deep_learning_settings


def configure_deep_learning(
    settings: Optional[DeepLearningSettings] = None,
    **kwargs,
) -> DeepLearningSettings:
    """Configure deep learning settings."""
    global _deep_learning_settings
    if settings is not None:
        _deep_learning_settings = settings
    elif kwargs:
        _deep_learning_settings = DeepLearningSettings(**kwargs)
    else:
        _deep_learning_settings = DeepLearningSettings()
    return _deep_learning_settings
