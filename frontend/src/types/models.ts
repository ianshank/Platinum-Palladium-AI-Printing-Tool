/**
 * Domain-specific types and enumerations for Pt/Pd printing.
 * Mirrored from src/ptpd_calibration/core/models.py and types.py
 */

export type UUID = string;
export type DateTime = string; // ISO 8601 string

// --- Enums ---

export enum ChemistryType {
    PURE_PLATINUM = 'pure_platinum',
    PURE_PALLADIUM = 'pure_palladium',
    PLATINUM_PALLADIUM = 'platinum_palladium',
    ZIATYPE = 'ziatype',
    NA2_PROCESS = 'na2_process',
}

export enum ContrastAgent {
    NONE = 'none',
    NA2 = 'na2',
    POTASSIUM_CHLORATE = 'potassium_chlorate',
    HYDROGEN_PEROXIDE = 'hydrogen_peroxide',
    DICHROMATE = 'dichromate',
}

export enum DeveloperType {
    POTASSIUM_OXALATE = 'potassium_oxalate',
    AMMONIUM_CITRATE = 'ammonium_citrate',
    SODIUM_CITRATE = 'sodium_citrate',
    EDTA = 'edta',
}

export enum PaperSizing {
    NONE = 'none',
    INTERNAL = 'internal',
    GELATIN = 'gelatin',
    STARCH = 'starch',
    ARROWROOT = 'arrowroot',
}

export enum CurveType {
    LINEAR = 'linear',
    PAPER_WHITE = 'paper_white',
    AESTHETIC = 'aesthetic',
    CUSTOM = 'custom',
}

export enum MeasurementUnit {
    VISUAL_DENSITY = 'visual_density',
    STATUS_A = 'status_a',
    STATUS_M = 'status_m',
    LAB = 'lab',
    RGB = 'rgb',
}

// --- Interfaces ---

export interface PatchData {
    index: number;
    // x, y, width, height
    position: [number, number, number, number];
    // r, g, b
    rgb_mean: [number, number, number];
    rgb_std: [number, number, number];
    // L, a, b
    lab_mean?: [number, number, number] | null;
    density?: number | null;
    uniformity: number;
}

export interface DensityMeasurement {
    step: number;
    input_value: number; // 0-1
    density: number;
    lab?: [number, number, number] | null;
    unit: MeasurementUnit;
}

export interface ExtractionResult {
    id: UUID;
    timestamp: DateTime;

    // Image info
    source_path?: string | null;
    image_size: [number, number]; // width, height
    image_dpi?: number | null;

    // Detection info
    tablet_bounds: [number, number, number, number]; // x, y, w, h
    rotation_angle: number;
    orientation: string;

    // Patch data
    patches: PatchData[];
    num_patches: number;

    // Paper base reference
    paper_base_rgb?: [number, number, number] | null;
    paper_base_density?: number | null;

    // Quality metrics
    overall_quality: number; // 0-1
    warnings: string[];
}

export interface StepTabletResult {
    extraction: ExtractionResult;
    measurements: DensityMeasurement[];
}

export interface CurveData {
    id: UUID;
    name: string;
    created_at: DateTime;

    // Curve type and metadata
    curve_type: CurveType;
    paper_type?: string | null;
    chemistry?: string | null;
    notes?: string | null;

    // Input/output points
    input_values: number[];
    output_values: number[];

    // Source calibration
    source_extraction_id?: UUID | null;
    target_curve_type?: string | null;
}

export interface PaperProfile {
    id: UUID;
    name: string;
    manufacturer?: string | null;
    weight_gsm?: number | null;
    sizing: PaperSizing;

    // Characteristics
    base_density?: number | null;
    max_density?: number | null;
    recommended_exposure_factor: number;

    // Notes and metadata
    notes?: string | null;
    calibration_ids: UUID[];
}

export interface CalibrationRecord {
    id: UUID;
    timestamp: DateTime;
    name?: string | null;

    // Paper info
    paper_type: string;
    paper_weight?: number | null;
    paper_sizing: PaperSizing;

    // Chemistry
    chemistry_type: ChemistryType;
    metal_ratio: number;
    contrast_agent: ContrastAgent;
    contrast_amount: number;
    developer: DeveloperType;

    // Process parameters
    exposure_time: number;
    uv_source?: string | null;
    humidity?: number | null;
    temperature?: number | null;

    // Results
    measured_densities: number[];
    extraction_id?: UUID | null;
    curve_id?: UUID | null;

    // Metadata
    notes?: string | null;
    tags: string[];
}

// API Response Wrappers
export interface ApiSuccessResponse<T> {
    success: boolean;
    data: T;
    message?: string;
}

export interface CurveGenerationRequest {
    measurements: number[];
    type?: string;
    name?: string;
}

export interface ScanUploadResponse {
    success: boolean;
    extraction_id: UUID;
    num_patches: number;
    densities: number[];
    dmin: number | null;
    dmax: number | null;
    range: number | null;
    quality: number;
    warnings: string[];
}

export interface CurveGenerationResponse {
    success: boolean;
    curve_id: UUID;
    name: string;
    num_points: number;
    input_values: number[];
    output_values: number[];
}

export interface CurveModificationRequest {
    name: string;
    input_values: number[];
    output_values: number[];
    adjustment_type: string;
    amount: number;
    pivot?: number;
    black_point?: number;
    white_point?: number;
}

export interface CurveModificationResponse {
    success: boolean;
    curve_id: UUID;
    name: string;
    adjustment_applied: string;
    input_values: number[];
    output_values: number[];
}

export interface CurveSmoothRequest {
    name: string;
    input_values: number[];
    output_values: number[];
    method: string;
    strength?: number; // defaults differ by method, handled on server
    preserve_endpoints?: boolean;
}

export interface CurveSmoothingResponse {
    success: boolean;
    curve_id: UUID;
    name: string;
    method_applied: string;
    input_values: number[];
    output_values: number[];
}

export interface CurveEnhanceResponse {
    success: boolean;
    curve_id: UUID;
    name: string;
    goal: string;
    confidence: number;
    analysis: string;
    changes_made: string[];
    input_values: number[];
    output_values: number[];
}

export interface EnforceMonotonicityResponse {
    success: boolean;
    curve_id: UUID;
    name: string;
    input_values: number[];
    output_values: number[];
}

export interface CalibrationSummary {
    id: UUID;
    paper_type: string;
    exposure_time: number;
    metal_ratio: number;
    timestamp: DateTime;
    dmax: number;
}

export interface CalibrationListResponse {
    count: number;
    records: CalibrationSummary[];
}

export interface CalibrationCreateResponse {
    success: boolean;
    id: UUID;
    message: string;
}

export interface ChatResponse {
    response: string;
    context_used?: string[];
}

export interface StatisticsResponse {
    total_records: number;
    paper_types: string[];
    chemistry_types: string[];
    date_range?: [string, string] | null;
    exposure_range?: [number, number] | null;
}

export interface AnalysisResponse {
    analysis: unknown; // Generic for now
}
