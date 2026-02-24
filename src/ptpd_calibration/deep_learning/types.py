"""
Type definitions for deep learning components.

Provides enums and type aliases for all AI/ML features.
All types are string-based enums for JSON serialization compatibility.
"""

from enum import Enum
from typing import TypeAlias

import numpy as np

# =============================================================================
# Array Type Aliases
# =============================================================================

ImageArray: TypeAlias = np.ndarray  # Shape: (H, W) or (H, W, C)
FeatureVector: TypeAlias = np.ndarray  # Shape: (D,)
CurvePoints: TypeAlias = np.ndarray  # Shape: (N, 2) or (N,)
BoundingBox: TypeAlias = tuple[int, int, int, int]  # (x, y, width, height)
Mask: TypeAlias = np.ndarray  # Shape: (H, W), dtype: bool or uint8


# =============================================================================
# Detection Types (YOLOv8 + SAM)
# =============================================================================


class DetectionBackend(str, Enum):
    """Backend for object detection models."""

    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLO_WORLD = "yolo_world"
    DETR = "detr"
    RT_DETR = "rt_detr"
    GROUNDING_DINO = "grounding_dino"
    OWL_VIT = "owl_vit"
    CLASSICAL = "classical"  # Fallback to classical CV


class SegmentationBackend(str, Enum):
    """Backend for instance segmentation."""

    SAM = "sam"  # Segment Anything Model
    SAM2 = "sam2"  # SAM 2.0
    MOBILE_SAM = "mobile_sam"
    FAST_SAM = "fast_sam"
    SEMANTIC_SAM = "semantic_sam"
    MASK_RCNN = "mask_rcnn"
    CLASSICAL = "classical"  # Fallback to contour-based


class DetectionConfidence(str, Enum):
    """Detection confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# Image Quality Assessment Types (Vision Transformer)
# =============================================================================


class IQAMetric(str, Enum):
    """Image quality assessment metrics."""

    # No-reference metrics
    NIQE = "niqe"
    BRISQUE = "brisque"
    CLIP_IQA = "clip_iqa"
    MUSIQ = "musiq"
    MANIQA = "maniqa"
    TOPIQ = "topiq"

    # Full-reference metrics
    PSNR = "psnr"
    SSIM = "ssim"
    MS_SSIM = "ms_ssim"
    LPIPS = "lpips"
    DISTS = "dists"

    # Domain-specific
    PTPD_QUALITY = "ptpd_quality"  # Custom for Pt/Pd prints


class QualityLevel(str, Enum):
    """Print quality level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class ZoneQuality(str, Enum):
    """Quality level for specific tonal zones."""

    HIGHLIGHTS = "highlights"
    UPPER_MIDTONES = "upper_midtones"
    MIDTONES = "midtones"
    LOWER_MIDTONES = "lower_midtones"
    SHADOWS = "shadows"
    DEEP_SHADOWS = "deep_shadows"


# =============================================================================
# Diffusion Model Types
# =============================================================================


class DiffusionScheduler(str, Enum):
    """Noise scheduler for diffusion models."""

    DDPM = "ddpm"
    DDIM = "ddim"
    PNDM = "pndm"
    LMS = "lms"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    DPM_SOLVER = "dpm_solver"
    DPM_SOLVER_PP = "dpm_solver++"
    UNIPC = "unipc"


class EnhancementMode(str, Enum):
    """Enhancement mode for diffusion-based processing."""

    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    SUPER_RESOLUTION = "super_resolution"
    DENOISING = "denoising"
    STYLE_TRANSFER = "style_transfer"
    TONAL_ENHANCEMENT = "tonal_enhancement"
    DEFECT_REMOVAL = "defect_removal"
    DETAIL_RECOVERY = "detail_recovery"


class DiffusionModelType(str, Enum):
    """Type of diffusion model to use."""

    STABLE_DIFFUSION_15 = "sd_1.5"
    STABLE_DIFFUSION_21 = "sd_2.1"
    STABLE_DIFFUSION_XL = "sdxl"
    STABLE_DIFFUSION_3 = "sd_3"
    CONTROLNET = "controlnet"
    IP_ADAPTER = "ip_adapter"
    CUSTOM_LORA = "custom_lora"


class ControlNetType(str, Enum):
    """ControlNet conditioning type."""

    CANNY = "canny"
    DEPTH = "depth"
    NORMAL = "normal"
    OPENPOSE = "openpose"
    SOFTEDGE = "softedge"
    LINEART = "lineart"
    MLSD = "mlsd"
    SCRIBBLE = "scribble"
    TILE = "tile"
    INPAINT = "inpaint"


# =============================================================================
# Neural Curve Prediction Types
# =============================================================================


class CurvePredictorArchitecture(str, Enum):
    """Neural network architecture for curve prediction."""

    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    MLP_MIXER = "mlp_mixer"
    RESNET = "resnet"
    LSTM = "lstm"
    GRU = "gru"
    HYBRID = "hybrid"
    GRADIENT_BOOSTING = "gradient_boosting"  # Fallback to sklearn


class UncertaintyMethod(str, Enum):
    """Method for uncertainty quantification."""

    ENSEMBLE = "ensemble"
    MC_DROPOUT = "mc_dropout"
    DEEP_ENSEMBLE = "deep_ensemble"
    BAYESIAN = "bayesian"
    EVIDENTIAL = "evidential"
    CONFORMAL = "conformal"


class CurveLossFunction(str, Enum):
    """Loss function for curve prediction training."""

    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    MONOTONIC_MSE = "monotonic_mse"
    PERCEPTUAL = "perceptual"


# =============================================================================
# UV Exposure Prediction Types
# =============================================================================


class UVSourceType(str, Enum):
    """Types of UV light sources."""

    NUARC_26_1K = "nuarc_26_1k"
    NUARC_26_1KS = "nuarc_26_1ks"
    FLUORESCENT_BL = "fluorescent_bl"
    FLUORESCENT_BLB = "fluorescent_blb"
    LED_365NM = "led_365nm"
    LED_385NM = "led_385nm"
    LED_405NM = "led_405nm"
    MERCURY_VAPOR = "mercury_vapor"
    METAL_HALIDE = "metal_halide"
    SUNLIGHT = "sunlight"
    CUSTOM = "custom"


class ExposureConfidence(str, Enum):
    """Confidence level for exposure predictions."""

    EXPERIMENTAL = "experimental"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CALIBRATED = "calibrated"


# =============================================================================
# Defect Detection Types
# =============================================================================


class DefectType(str, Enum):
    """Types of print defects."""

    # Coating defects
    BRUSH_MARK = "brush_mark"
    POOLING = "pooling"
    STREAKING = "streaking"
    UNEVEN_COATING = "uneven_coating"
    COATING_GAP = "coating_gap"

    # Chemical defects
    DEVELOPER_STAIN = "developer_stain"
    FIXER_RESIDUE = "fixer_residue"
    OXIDATION = "oxidation"
    BRONZING = "bronzing"
    SOLARIZATION = "solarization"

    # Paper defects
    PAPER_INCLUSION = "paper_inclusion"
    FIBER_DAMAGE = "fiber_damage"
    TEAR = "tear"
    CREASE = "crease"

    # Environmental defects
    DUST = "dust"
    FINGERPRINT = "fingerprint"
    WATER_SPOT = "water_spot"
    HUMIDITY_DAMAGE = "humidity_damage"

    # Exposure defects
    LIGHT_LEAK = "light_leak"
    UNEVEN_EXPOSURE = "uneven_exposure"
    UNDEREXPOSURE = "underexposure"
    OVEREXPOSURE = "overexposure"

    # Other
    SCRATCH = "scratch"
    UNKNOWN = "unknown"


class DefectSeverity(str, Enum):
    """Severity level of detected defects."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class DefectDetectorArchitecture(str, Enum):
    """Architecture for defect detection models."""

    UNET = "unet"
    UNET_PLUS_PLUS = "unet++"
    DEEPLABV3 = "deeplabv3"
    SEGFORMER = "segformer"
    MASK2FORMER = "mask2former"
    RESNET_CLASSIFIER = "resnet_classifier"
    EFFICIENTNET = "efficientnet"
    VISION_TRANSFORMER = "vision_transformer"


# =============================================================================
# Recipe Recommendation Types
# =============================================================================


class RecommendationStrategy(str, Enum):
    """Strategy for recipe recommendations."""

    CONTENT_BASED = "content_based"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    HYBRID = "hybrid"
    KNOWLEDGE_BASED = "knowledge_based"
    NEURAL_COLLABORATIVE = "neural_collaborative"
    GRAPH_NEURAL = "graph_neural"
    TRANSFORMER = "transformer"


class SimilarityMetric(str, Enum):
    """Metric for computing recipe similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    PEARSON = "pearson"
    JACCARD = "jaccard"
    LEARNED = "learned"


class RecipeCategory(str, Enum):
    """Categories for recipe organization."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    STILL_LIFE = "still_life"
    ABSTRACT = "abstract"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"
    HIGH_CONTRAST = "high_contrast"
    LOW_CONTRAST = "low_contrast"
    WARM_TONE = "warm_tone"
    NEUTRAL_TONE = "neutral_tone"
    COOL_TONE = "cool_tone"


# =============================================================================
# Deep Print Comparison Types (LPIPS)
# =============================================================================


class PerceptualMetric(str, Enum):
    """Perceptual similarity metrics."""

    LPIPS_ALEX = "lpips_alex"
    LPIPS_VGG = "lpips_vgg"
    LPIPS_SQUEEZE = "lpips_squeeze"
    DISTS = "dists"
    PIEAPP = "pieapp"
    FSIM = "fsim"
    VSI = "vsi"
    GMSD = "gmsd"
    MDSI = "mdsi"


class ComparisonMode(str, Enum):
    """Mode for print comparison."""

    GLOBAL = "global"
    ZONE_BASED = "zone_based"
    REGION_OF_INTEREST = "region_of_interest"
    ADAPTIVE = "adaptive"
    MULTI_SCALE = "multi_scale"


class ComparisonResult(str, Enum):
    """Result classification for print comparison."""

    IDENTICAL = "identical"
    VERY_SIMILAR = "very_similar"
    SIMILAR = "similar"
    DIFFERENT = "different"
    VERY_DIFFERENT = "very_different"


# =============================================================================
# Multi-Modal AI Assistant Types
# =============================================================================


class VisionLanguageModel(str, Enum):
    """Vision-language model options."""

    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_35_SONNET = "claude-3.5-sonnet"
    CLAUDE_4_SONNET = "claude-4-sonnet"
    GPT_4_VISION = "gpt-4-vision"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    GEMINI_15_PRO = "gemini-1.5-pro"
    LLAVA = "llava"
    IDEFICS = "idefics"
    QWEN_VL = "qwen-vl"
    INTERNVL = "internvl"


class ToolType(str, Enum):
    """Types of tools available to the AI assistant."""

    EXPOSURE_CALCULATOR = "exposure_calculator"
    CHEMISTRY_CALCULATOR = "chemistry_calculator"
    CURVE_ADJUSTMENT = "curve_adjustment"
    DEFECT_DIAGNOSIS = "defect_diagnosis"
    RECIPE_LOOKUP = "recipe_lookup"
    PAPER_RECOMMENDATION = "paper_recommendation"
    PROCESS_SIMULATION = "process_simulation"
    COST_ESTIMATION = "cost_estimation"
    ZONE_ANALYSIS = "zone_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"


class AssistantMode(str, Enum):
    """Operating mode for the assistant."""

    CHAT = "chat"
    ANALYSIS = "analysis"
    GUIDED_WORKFLOW = "guided_workflow"
    TROUBLESHOOTING = "troubleshooting"
    EDUCATION = "education"


# =============================================================================
# Federated Learning Types
# =============================================================================


class AggregationStrategy(str, Enum):
    """Strategy for federated model aggregation."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDADAM = "fedadam"
    SCAFFOLD = "scaffold"
    FEDBN = "fedbn"
    FEDOPT = "fedopt"
    WEIGHTED_AVERAGE = "weighted_average"


class PrivacyLevel(str, Enum):
    """Privacy level for federated learning."""

    NONE = "none"
    DIFFERENTIAL = "differential"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC = "homomorphic"
    MAXIMUM = "maximum"


class FederatedRole(str, Enum):
    """Role in federated learning system."""

    CLIENT = "client"
    SERVER = "server"
    AGGREGATOR = "aggregator"
    COORDINATOR = "coordinator"


class CommunicationProtocol(str, Enum):
    """Protocol for federated communication."""

    GRPC = "grpc"
    HTTP = "http"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"


class DataContribution(str, Enum):
    """Level of data contribution in federated learning."""

    GRADIENTS_ONLY = "gradients_only"
    COMPRESSED_GRADIENTS = "compressed_gradients"
    MODEL_UPDATES = "model_updates"
    STATISTICS_ONLY = "statistics_only"
