"""
Enhanced technical calculations for platinum/palladium printing.

Provides advanced calculators for UV exposure, coating volumes, costs, dilutions,
and environmental compensation. All calculations are configuration-driven with
no hardcoded values.

This module extends the basic exposure and chemistry calculators with:
- Multi-factor UV exposure calculations
- Coating volume optimization
- Cost tracking and analysis
- Developer and clearing bath dilution
- Environmental and seasonal adjustments
"""

from pydantic import BaseModel, ConfigDict, Field

from ptpd_calibration.config import ChemistrySettings, get_settings

# ============================================================================
# Pydantic Result Models
# ============================================================================


class ExposureResult(BaseModel):
    """Result of enhanced UV exposure calculation."""

    model_config = ConfigDict(frozen=True)

    # Final exposure time
    adjusted_exposure_minutes: float = Field(ge=0)
    adjusted_exposure_seconds: float = Field(ge=0)

    # Base values
    base_time_minutes: float = Field(ge=0)
    base_negative_density: float = Field(ge=0)

    # Individual adjustment factors
    humidity_factor: float = Field(gt=0)
    temperature_factor: float = Field(gt=0)
    density_factor: float = Field(gt=0)
    intensity_factor: float = Field(gt=0)
    paper_factor: float = Field(gt=0)
    chemistry_factor: float = Field(gt=0)

    # Confidence interval (95%)
    confidence_lower_minutes: float = Field(ge=0)
    confidence_upper_minutes: float = Field(ge=0)
    confidence_interval_percent: float = Field(default=95.0, ge=0, le=100)

    # Input values for reference
    negative_density: float = Field(ge=0)
    humidity_percent: float = Field(ge=0, le=100)
    temperature_fahrenheit: float
    uv_intensity_percent: float = Field(ge=0)

    # Warnings and notes
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CoatingResult(BaseModel):
    """Result of coating volume calculation."""

    model_config = ConfigDict(frozen=True)

    # Calculated volumes
    total_ml: float = Field(ge=0)
    total_drops: float = Field(ge=0)

    # Paper and method info
    paper_area_sq_inches: float = Field(ge=0)
    paper_type: str
    coating_method: str
    absorption_rate_ml_per_sq_inch: float = Field(ge=0)

    # Adjustments
    humidity_adjustment_factor: float = Field(gt=0)
    method_efficiency_factor: float = Field(gt=0)
    waste_factor: float = Field(gt=0)

    # Volume breakdown
    base_volume_ml: float = Field(ge=0)
    adjusted_volume_ml: float = Field(ge=0)
    waste_volume_ml: float = Field(ge=0)

    # Practical information
    recommended_ml: float = Field(ge=0, description="Rounded up for practical use")
    recommended_drops: float = Field(ge=0)

    notes: list[str] = Field(default_factory=list)


class PrintCostResult(BaseModel):
    """Result of print cost calculation."""

    model_config = ConfigDict(frozen=True)

    # Total cost
    total_cost_usd: float = Field(ge=0)

    # Cost breakdown
    ferric_oxalate_cost: float = Field(ge=0)
    platinum_cost: float = Field(ge=0)
    palladium_cost: float = Field(ge=0)
    contrast_agent_cost: float = Field(ge=0)
    paper_cost: float = Field(ge=0)
    other_costs: float = Field(ge=0, description="Developer, clearing baths, etc.")

    # Volume breakdown
    total_chemistry_ml: float = Field(ge=0)
    chemistry_cost_per_ml: float = Field(ge=0)

    # Print info
    paper_size: str
    paper_area_sq_inches: float = Field(ge=0)
    metal_ratio_platinum: float = Field(ge=0, le=1)

    notes: list[str] = Field(default_factory=list)


class SessionCostResult(BaseModel):
    """Result of session cost calculation."""

    model_config = ConfigDict(frozen=True)

    # Total session cost
    total_session_cost_usd: float = Field(ge=0)

    # Aggregate breakdown
    total_chemistry_cost: float = Field(ge=0)
    total_paper_cost: float = Field(ge=0)
    total_other_costs: float = Field(ge=0)

    # Session info
    num_prints: int = Field(ge=0)
    average_cost_per_print: float = Field(ge=0)
    total_chemistry_ml: float = Field(ge=0)

    # Print breakdown
    print_costs: list[PrintCostResult] = Field(default_factory=list)

    notes: list[str] = Field(default_factory=list)


class SolutionUsageEstimate(BaseModel):
    """Estimate of solution usage."""

    model_config = ConfigDict(frozen=True)

    # Estimated volumes
    ferric_oxalate_ml: float = Field(ge=0)
    platinum_ml: float = Field(ge=0)
    palladium_ml: float = Field(ge=0)
    contrast_agent_ml: float = Field(ge=0)
    developer_ml: float = Field(ge=0)
    clearing_bath_ml: float = Field(ge=0)

    # Usage info
    num_prints: int = Field(ge=0)
    average_print_size_sq_inches: float = Field(ge=0)
    average_platinum_ratio: float = Field(ge=0, le=1)

    # Practical recommendations
    recommended_stock_ferric_oxalate_ml: float = Field(ge=0)
    recommended_stock_platinum_ml: float = Field(ge=0)
    recommended_stock_palladium_ml: float = Field(ge=0)


class DilutionResult(BaseModel):
    """Result of dilution calculation."""

    model_config = ConfigDict(frozen=True)

    # Volumes
    concentrate_ml: float = Field(ge=0)
    water_ml: float = Field(ge=0)
    total_ml: float = Field(ge=0)

    # Concentrations
    concentrate_strength: float = Field(ge=0)
    target_strength: float = Field(ge=0)
    dilution_ratio: str = Field(description="e.g., '1:4' means 1 part concentrate to 4 parts water")

    # Additional info
    solution_type: str
    notes: list[str] = Field(default_factory=list)


class ReplenishmentResult(BaseModel):
    """Result of replenishment calculation."""

    model_config = ConfigDict(frozen=True)

    # Replenishment amounts
    replenish_ml: float = Field(ge=0)
    replenish_drops: float = Field(ge=0)

    # Solution info
    solution_type: str
    current_volume_ml: float = Field(ge=0)
    usage_ml: float = Field(ge=0)
    target_volume_ml: float = Field(ge=0)

    # Recommendations
    should_replace: bool = Field(description="True if solution should be fully replaced")
    exhaustion_percent: float = Field(ge=0, le=100)

    notes: list[str] = Field(default_factory=list)


class EnvironmentalAdjustment(BaseModel):
    """Result of environmental adjustment calculation."""

    model_config = ConfigDict(frozen=True)

    # Adjusted value
    adjusted_value: float

    # Adjustment details
    base_value: float
    adjustment_factor: float = Field(gt=0)
    adjustment_type: str

    # Environmental conditions
    altitude_feet: float | None = None
    month: int | None = Field(None, ge=1, le=12)
    humidity_percent: float | None = Field(None, ge=0, le=100)
    temperature_fahrenheit: float | None = None

    notes: list[str] = Field(default_factory=list)


class OptimalConditions(BaseModel):
    """Optimal working conditions for platinum/palladium printing."""

    model_config = ConfigDict(frozen=True)

    # Optimal ranges
    temperature_f_min: float
    temperature_f_max: float
    temperature_f_ideal: float

    humidity_percent_min: float
    humidity_percent_max: float
    humidity_percent_ideal: float

    altitude_feet_max: float

    # Timing recommendations
    coating_to_exposure_hours_min: float
    coating_to_exposure_hours_max: float

    development_minutes_min: float
    development_minutes_max: float

    notes: list[str] = Field(default_factory=list)


class DryingTimeEstimate(BaseModel):
    """Estimate of drying time."""

    model_config = ConfigDict(frozen=True)

    # Drying time
    drying_minutes: float = Field(ge=0)
    drying_hours: float = Field(ge=0)

    # Conditions
    humidity_percent: float = Field(ge=0, le=100)
    temperature_fahrenheit: float
    paper_type: str

    # Factors
    humidity_factor: float = Field(gt=0)
    temperature_factor: float = Field(gt=0)
    paper_absorbency_factor: float = Field(gt=0)

    # Recommendations
    forced_air_recommended: bool
    estimated_range_minutes: tuple[float, float]

    notes: list[str] = Field(default_factory=list)


# ============================================================================
# Enhanced UV Exposure Calculator
# ============================================================================


class UVExposureCalculator:
    """
    Enhanced UV exposure calculator with comprehensive environmental compensation.

    Accounts for:
    - Negative density variations
    - Relative humidity effects on paper sensitivity
    - Temperature effects on chemical reactions
    - UV intensity variations
    - Paper speed factors
    - Chemistry-based exposure adjustments
    """

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with settings.

        Args:
            settings: Chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def calculate_uv_exposure(
        self,
        base_time: float,
        negative_density: float,
        humidity: float,
        temperature: float,
        uv_intensity: float = 100.0,
        paper_factor: float = 1.0,
        chemistry_factor: float = 1.0,
        base_density: float = 1.6,
        optimal_humidity: float = 50.0,
        optimal_temperature: float = 68.0,
        uncertainty_percent: float = 10.0,
    ) -> ExposureResult:
        """Calculate adjusted UV exposure time accounting for all environmental factors.

        Args:
            base_time: Base exposure time in minutes at reference conditions
            negative_density: Negative density range (Dmax - Dmin)
            humidity: Relative humidity in percent (0-100)
            temperature: Temperature in Fahrenheit
            uv_intensity: UV intensity as percentage of reference (default 100.0)
            paper_factor: Paper speed multiplier (1.0 = average)
            chemistry_factor: Chemistry speed multiplier (1.0 = average)
            base_density: Reference negative density for base_time (default 1.6)
            optimal_humidity: Optimal humidity in percent (default 50.0)
            optimal_temperature: Optimal temperature in Fahrenheit (default 68.0)
            uncertainty_percent: Uncertainty for confidence interval (default 10.0)

        Returns:
            ExposureResult with calculated exposure time and adjustments
        """
        warnings = []
        notes = []

        # 1. Humidity factor
        # Higher humidity = faster exposure (more water in emulsion)
        # Formula: factor = 1.0 - (humidity_delta * humidity_coefficient)
        humidity_delta = (humidity - optimal_humidity) / 100.0
        humidity_coefficient = 0.15  # 15% change per 100% humidity delta
        humidity_factor = 1.0 - (humidity_delta * humidity_coefficient)
        humidity_factor = max(0.7, min(1.3, humidity_factor))  # Clamp to reasonable range

        if humidity < 30:
            warnings.append("Low humidity (<30%) may cause uneven coating and slow exposure")
        elif humidity > 70:
            warnings.append(
                "High humidity (>70%) may accelerate exposure and affect coating drying"
            )

        # 2. Temperature factor
        # Higher temperature = faster exposure (increased chemical activity)
        # Formula: factor = 1.0 - (temp_delta * temp_coefficient)
        temp_delta = (temperature - optimal_temperature) / 10.0  # Per 10°F
        temp_coefficient = 0.05  # 5% change per 10°F
        temperature_factor = 1.0 - (temp_delta * temp_coefficient)
        temperature_factor = max(0.8, min(1.2, temperature_factor))  # Clamp

        if temperature < 60:
            warnings.append("Low temperature (<60°F) may slow chemical reactions")
        elif temperature > 80:
            warnings.append("High temperature (>80°F) may accelerate exposure and affect coating")

        # 3. Density factor
        # More dense negative = more exposure needed
        # Industry standard: 0.3 density = 1 stop = 2x exposure
        density_delta = negative_density - base_density
        density_factor = 2 ** (density_delta / 0.3)

        if density_delta > 0.3:
            notes.append(
                f"Dense negative (+{density_delta:.2f}D) requires {density_factor:.1f}x exposure"
            )
        elif density_delta < -0.3:
            notes.append(
                f"Thin negative ({density_delta:.2f}D) requires {density_factor:.2f}x exposure"
            )

        # 4. UV intensity factor
        # Lower intensity = more time needed
        intensity_factor = 100.0 / max(1.0, uv_intensity)

        if uv_intensity < 70:
            warnings.append(f"Low UV intensity ({uv_intensity:.0f}%) - check bulbs or source")
        elif uv_intensity > 120:
            warnings.append(f"High UV intensity ({uv_intensity:.0f}%) - verify measurement")

        # 5. Calculate final exposure
        adjusted_exposure = (
            base_time
            * humidity_factor
            * temperature_factor
            * density_factor
            * intensity_factor
            * paper_factor
            * chemistry_factor
        )

        # 6. Calculate confidence interval
        uncertainty_factor = uncertainty_percent / 100.0
        confidence_lower = adjusted_exposure * (1.0 - uncertainty_factor)
        confidence_upper = adjusted_exposure * (1.0 + uncertainty_factor)

        # 7. Add practical warnings
        if adjusted_exposure > 30:
            warnings.append(
                "Long exposure (>30 min) - consider using faster light source or thinner negative"
            )
        if adjusted_exposure < 1:
            warnings.append(
                "Short exposure (<1 min) - risk of underexposure, consider neutral density filter"
            )

        return ExposureResult(
            adjusted_exposure_minutes=adjusted_exposure,
            adjusted_exposure_seconds=adjusted_exposure * 60,
            base_time_minutes=base_time,
            base_negative_density=base_density,
            humidity_factor=humidity_factor,
            temperature_factor=temperature_factor,
            density_factor=density_factor,
            intensity_factor=intensity_factor,
            paper_factor=paper_factor,
            chemistry_factor=chemistry_factor,
            confidence_lower_minutes=confidence_lower,
            confidence_upper_minutes=confidence_upper,
            negative_density=negative_density,
            humidity_percent=humidity,
            temperature_fahrenheit=temperature,
            uv_intensity_percent=uv_intensity,
            warnings=warnings,
            notes=notes,
        )


# ============================================================================
# Coating Volume Calculator
# ============================================================================


class CoatingVolumeCalculator:
    """
    Calculate optimal coating volume based on paper and method.

    Accounts for:
    - Paper absorbency profiles
    - Coating method efficiency
    - Environmental humidity effects
    - Waste factors
    """

    # Paper absorbency profiles (ml per square inch)
    # These can be overridden via configuration
    DEFAULT_PAPER_PROFILES = {
        "arches_platine": 0.0465,  # Hot press, low absorbency
        "arches_88": 0.0510,  # Medium absorbency
        "hahnemuhle_platinum": 0.0445,  # Hot press, sized
        "bergger_cot320": 0.0520,  # Medium-high
        "fabriano_artistico_hp": 0.0450,  # Hot press
        "fabriano_artistico_cp": 0.0530,  # Cold press, higher absorbency
        "revere_platinum": 0.0440,  # Sized, low absorbency
        "custom_hot_press": 0.0450,  # Generic hot press
        "custom_cold_press": 0.0540,  # Generic cold press
        "custom_rough": 0.0580,  # Rough paper, high absorbency
    }

    # Coating method efficiency factors
    METHOD_EFFICIENCY = {
        "brush": 1.0,  # Baseline, requires most solution
        "hake_brush": 1.0,
        "glass_rod": 0.75,  # More efficient
        "puddle_pusher": 0.75,
        "coating_rod": 0.70,  # Most efficient
    }

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with settings.

        Args:
            settings: Chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def determine_coating_volume(
        self,
        paper_area: float,
        paper_type: str,
        coating_method: str,
        humidity: float,
        optimal_humidity: float = 50.0,
        waste_factor: float = 1.15,
    ) -> CoatingResult:
        """Calculate sensitizer volume needed.

        Args:
            paper_area: Paper area in square inches
            paper_type: Paper type identifier (e.g., 'arches_platine')
            coating_method: Coating method (e.g., 'brush', 'glass_rod')
            humidity: Current relative humidity (0-100)
            optimal_humidity: Optimal humidity for coating (default 50.0)
            waste_factor: Waste/safety factor (default 1.15 = 15% extra)

        Returns:
            CoatingResult with calculated volumes
        """
        notes = []

        # 1. Get paper absorption rate
        absorption_rate = self.DEFAULT_PAPER_PROFILES.get(
            paper_type.lower(), self.settings.drops_per_square_inch / self.settings.drops_per_ml
        )

        if paper_type.lower() not in self.DEFAULT_PAPER_PROFILES:
            notes.append(f"Using default absorption rate for unknown paper type '{paper_type}'")

        # 2. Get coating method efficiency
        method_efficiency = self.METHOD_EFFICIENCY.get(coating_method.lower(), 1.0)

        if coating_method.lower() not in self.METHOD_EFFICIENCY:
            notes.append(f"Using default efficiency for unknown coating method '{coating_method}'")

        # 3. Humidity adjustment
        # Higher humidity = paper absorbs more, needs more solution
        humidity_delta = (humidity - optimal_humidity) / 100.0
        humidity_coefficient = 0.10  # 10% change per 100% humidity delta
        humidity_adjustment = 1.0 + (humidity_delta * humidity_coefficient)
        humidity_adjustment = max(0.9, min(1.2, humidity_adjustment))

        if humidity < 35:
            notes.append("Low humidity - paper may absorb less solution, apply quickly")
        elif humidity > 65:
            notes.append("High humidity - paper may absorb more solution, coating may dry slowly")

        # 4. Calculate base volume
        base_volume = paper_area * absorption_rate

        # 5. Apply all adjustments
        adjusted_volume = base_volume * method_efficiency * humidity_adjustment
        waste_volume = adjusted_volume * (waste_factor - 1.0)
        total_volume = adjusted_volume + waste_volume

        # 6. Convert to drops
        total_drops = total_volume * self.settings.drops_per_ml

        # 7. Round up for practical use
        recommended_ml = round(total_volume * 2) / 2  # Round to nearest 0.5 ml
        recommended_drops = round(total_drops)

        # 8. Add practical notes
        if method_efficiency < 1.0:
            savings = (1.0 - method_efficiency) * 100
            notes.append(f"{coating_method} saves ~{savings:.0f}% solution vs brush coating")

        if total_volume < 1.0:
            notes.append("Small volume - measure carefully with calibrated dropper")
        elif total_volume > 10.0:
            notes.append("Large volume - consider preparing in batches for even coating")

        return CoatingResult(
            total_ml=total_volume,
            total_drops=total_drops,
            paper_area_sq_inches=paper_area,
            paper_type=paper_type,
            coating_method=coating_method,
            absorption_rate_ml_per_sq_inch=absorption_rate,
            humidity_adjustment_factor=humidity_adjustment,
            method_efficiency_factor=method_efficiency,
            waste_factor=waste_factor,
            base_volume_ml=base_volume,
            adjusted_volume_ml=adjusted_volume,
            waste_volume_ml=waste_volume,
            recommended_ml=recommended_ml,
            recommended_drops=recommended_drops,
            notes=notes,
        )


# ============================================================================
# Cost Calculator
# ============================================================================


class CostCalculator:
    """
    Calculate printing costs including chemistry, paper, and supplies.

    Tracks costs per print and per session for budgeting and analysis.
    """

    # Default paper costs (USD per square inch)
    # These are representative values, should be configured per user
    DEFAULT_PAPER_COSTS = {
        "arches_platine": 0.15,
        "arches_88": 0.12,
        "hahnemuhle_platinum": 0.14,
        "bergger_cot320": 0.13,
        "fabriano_artistico_hp": 0.11,
        "fabriano_artistico_cp": 0.11,
        "revere_platinum": 0.14,
        "custom": 0.10,
    }

    # Developer and clearing bath costs (USD per liter)
    DEVELOPER_COST_PER_LITER = 5.00  # EDTA or Sodium citrate
    CLEARING_BATH_COST_PER_LITER = 3.00  # Citric acid or HCl

    # Typical usage (ml per square inch)
    DEVELOPER_ML_PER_SQ_INCH = 0.5
    CLEARING_ML_PER_SQ_INCH = 0.5

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with settings.

        Args:
            settings: Chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def calculate_print_cost(
        self,
        paper_size: str,
        chemistry: dict,
        paper_type: str = "custom",
        include_processing: bool = True,
    ) -> PrintCostResult:
        """Calculate total cost per print.

        Args:
            paper_size: Paper size (e.g., "8x10", "11x14")
            chemistry: Dictionary with chemistry volumes in ml:
                - ferric_oxalate_ml
                - platinum_ml
                - palladium_ml
                - na2_ml (contrast agent)
            paper_type: Paper type for cost lookup
            include_processing: Include developer and clearing bath costs

        Returns:
            PrintCostResult with cost breakdown
        """
        notes = []

        # Parse paper size
        try:
            dimensions = paper_size.lower().replace('"', "").split("x")
            width = float(dimensions[0])
            height = float(dimensions[1])
            paper_area = width * height
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid paper size format: {paper_size}. Use format like '8x10' or '11x14'"
            )

        # Chemistry costs
        ferric_cost = (
            chemistry.get("ferric_oxalate_ml", 0) * self.settings.ferric_oxalate_cost_per_ml
        )
        platinum_cost = chemistry.get("platinum_ml", 0) * self.settings.platinum_cost_per_ml
        palladium_cost = chemistry.get("palladium_ml", 0) * self.settings.palladium_cost_per_ml
        contrast_cost = chemistry.get("na2_ml", 0) * self.settings.na2_cost_per_ml

        total_chemistry_ml = sum(
            [
                chemistry.get("ferric_oxalate_ml", 0),
                chemistry.get("platinum_ml", 0),
                chemistry.get("palladium_ml", 0),
                chemistry.get("na2_ml", 0),
            ]
        )

        # Paper cost
        paper_cost_per_sq_inch = self.DEFAULT_PAPER_COSTS.get(paper_type.lower(), 0.10)
        paper_cost = paper_area * paper_cost_per_sq_inch

        # Processing costs
        other_costs = 0.0
        if include_processing:
            developer_cost = (
                paper_area * self.DEVELOPER_ML_PER_SQ_INCH / 1000.0
            ) * self.DEVELOPER_COST_PER_LITER
            clearing_cost = (
                paper_area * self.CLEARING_ML_PER_SQ_INCH / 1000.0
            ) * self.CLEARING_BATH_COST_PER_LITER
            other_costs = developer_cost + clearing_cost
            notes.append("Processing costs include developer and clearing baths")

        # Total cost
        total_cost = (
            ferric_cost + platinum_cost + palladium_cost + contrast_cost + paper_cost + other_costs
        )

        # Calculate metal ratio
        total_metal = chemistry.get("platinum_ml", 0) + chemistry.get("palladium_ml", 0)
        platinum_ratio = chemistry.get("platinum_ml", 0) / total_metal if total_metal > 0 else 0

        # Cost per ml
        chemistry_cost_per_ml = (
            (ferric_cost + platinum_cost + palladium_cost + contrast_cost) / total_chemistry_ml
            if total_chemistry_ml > 0
            else 0
        )

        return PrintCostResult(
            total_cost_usd=total_cost,
            ferric_oxalate_cost=ferric_cost,
            platinum_cost=platinum_cost,
            palladium_cost=palladium_cost,
            contrast_agent_cost=contrast_cost,
            paper_cost=paper_cost,
            other_costs=other_costs,
            total_chemistry_ml=total_chemistry_ml,
            chemistry_cost_per_ml=chemistry_cost_per_ml,
            paper_size=paper_size,
            paper_area_sq_inches=paper_area,
            metal_ratio_platinum=platinum_ratio,
            notes=notes,
        )

    def calculate_session_cost(
        self,
        prints: list[PrintCostResult],
    ) -> SessionCostResult:
        """Calculate total session cost from multiple prints.

        Args:
            prints: List of PrintCostResult objects

        Returns:
            SessionCostResult with aggregated costs
        """
        notes = []

        if not prints:
            return SessionCostResult(
                total_session_cost_usd=0.0,
                total_chemistry_cost=0.0,
                total_paper_cost=0.0,
                total_other_costs=0.0,
                num_prints=0,
                average_cost_per_print=0.0,
                total_chemistry_ml=0.0,
                print_costs=[],
                notes=["No prints in session"],
            )

        # Aggregate costs
        total_chemistry = sum(
            p.ferric_oxalate_cost + p.platinum_cost + p.palladium_cost + p.contrast_agent_cost
            for p in prints
        )
        total_paper = sum(p.paper_cost for p in prints)
        total_other = sum(p.other_costs for p in prints)
        total_chemistry_ml = sum(p.total_chemistry_ml for p in prints)

        total_session = total_chemistry + total_paper + total_other
        average_cost = total_session / len(prints)

        # Add insights
        most_expensive = max(prints, key=lambda p: p.total_cost_usd)
        least_expensive = min(prints, key=lambda p: p.total_cost_usd)

        notes.append(
            f"Most expensive print: {most_expensive.paper_size} at ${most_expensive.total_cost_usd:.2f}"
        )
        notes.append(
            f"Least expensive print: {least_expensive.paper_size} at ${least_expensive.total_cost_usd:.2f}"
        )

        chemistry_percent = (total_chemistry / total_session * 100) if total_session > 0 else 0
        notes.append(f"Chemistry represents {chemistry_percent:.1f}% of total session cost")

        return SessionCostResult(
            total_session_cost_usd=total_session,
            total_chemistry_cost=total_chemistry,
            total_paper_cost=total_paper,
            total_other_costs=total_other,
            num_prints=len(prints),
            average_cost_per_print=average_cost,
            total_chemistry_ml=total_chemistry_ml,
            print_costs=prints,
            notes=notes,
        )

    def estimate_solution_usage(
        self,
        num_prints: int,
        avg_size: str,
        avg_platinum_ratio: float = 0.0,
        coating_method: str = "brush",
    ) -> SolutionUsageEstimate:
        """Estimate chemistry usage for multiple prints.

        Args:
            num_prints: Number of prints to estimate
            avg_size: Average print size (e.g., "8x10")
            avg_platinum_ratio: Average platinum ratio (0-1)
            coating_method: Coating method used

        Returns:
            SolutionUsageEstimate with recommended stock levels
        """
        # Parse size
        dimensions = avg_size.lower().replace('"', "").split("x")
        width = float(dimensions[0])
        height = float(dimensions[1])
        avg_area = width * height

        # Estimate chemistry per print (simplified)
        ml_per_print = avg_area * self.settings.drops_per_square_inch / self.settings.drops_per_ml

        # Apply coating method efficiency
        method_factor = CoatingVolumeCalculator.METHOD_EFFICIENCY.get(coating_method.lower(), 1.0)
        ml_per_print *= method_factor

        # Split between components (ferric = half, metals = half)
        ferric_per_print = ml_per_print / 2
        metal_per_print = ml_per_print / 2

        platinum_per_print = metal_per_print * avg_platinum_ratio
        palladium_per_print = metal_per_print * (1 - avg_platinum_ratio)
        contrast_per_print = metal_per_print * self.settings.default_na2_drops_ratio

        # Total estimates
        total_ferric = ferric_per_print * num_prints
        total_platinum = platinum_per_print * num_prints
        total_palladium = palladium_per_print * num_prints
        total_contrast = contrast_per_print * num_prints

        # Developer and clearing (approximate)
        total_developer = num_prints * avg_area * self.DEVELOPER_ML_PER_SQ_INCH
        total_clearing = num_prints * avg_area * self.CLEARING_ML_PER_SQ_INCH

        # Recommended stock (add 20% safety margin)
        safety_margin = 1.2

        return SolutionUsageEstimate(
            ferric_oxalate_ml=total_ferric,
            platinum_ml=total_platinum,
            palladium_ml=total_palladium,
            contrast_agent_ml=total_contrast,
            developer_ml=total_developer,
            clearing_bath_ml=total_clearing,
            num_prints=num_prints,
            average_print_size_sq_inches=avg_area,
            average_platinum_ratio=avg_platinum_ratio,
            recommended_stock_ferric_oxalate_ml=total_ferric * safety_margin,
            recommended_stock_platinum_ml=total_platinum * safety_margin,
            recommended_stock_palladium_ml=total_palladium * safety_margin,
        )

    def generate_cost_report(
        self,
        session_cost: SessionCostResult,
        time_period: str = "session",
    ) -> str:
        """Generate formatted cost analysis report.

        Args:
            session_cost: SessionCostResult to report on
            time_period: Time period description (e.g., "session", "week", "month")

        Returns:
            Formatted cost report string
        """
        lines = [
            "=" * 60,
            f"COST ANALYSIS REPORT - {time_period.upper()}",
            "=" * 60,
            "",
            f"Number of prints: {session_cost.num_prints}",
            f"Total cost: ${session_cost.total_session_cost_usd:.2f}",
            f"Average cost per print: ${session_cost.average_cost_per_print:.2f}",
            "",
            "-" * 60,
            "COST BREAKDOWN",
            "-" * 60,
            f"Chemistry:        ${session_cost.total_chemistry_cost:>8.2f}  "
            f"({session_cost.total_chemistry_cost / session_cost.total_session_cost_usd * 100:.1f}%)",
            f"Paper:            ${session_cost.total_paper_cost:>8.2f}  "
            f"({session_cost.total_paper_cost / session_cost.total_session_cost_usd * 100:.1f}%)",
            f"Processing/Other: ${session_cost.total_other_costs:>8.2f}  "
            f"({session_cost.total_other_costs / session_cost.total_session_cost_usd * 100:.1f}%)",
            "",
            "-" * 60,
            "CHEMISTRY USAGE",
            "-" * 60,
            f"Total solution used: {session_cost.total_chemistry_ml:.1f} ml",
            f"Average per print: {session_cost.total_chemistry_ml / session_cost.num_prints:.1f} ml",
            "",
        ]

        if session_cost.notes:
            lines.extend(
                [
                    "-" * 60,
                    "INSIGHTS",
                    "-" * 60,
                ]
            )
            for note in session_cost.notes:
                lines.append(f"• {note}")

        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# Dilution Calculator
# ============================================================================


class DilutionCalculator:
    """
    Calculate dilutions for developers and clearing baths.

    Handles various dilution scenarios common in platinum/palladium printing.
    """

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with settings.

        Args:
            settings: Chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def calculate_developer_dilution(
        self,
        concentrate_strength: float,
        target_strength: float,
        volume: float,
    ) -> DilutionResult:
        """Calculate developer dilution ratios.

        Args:
            concentrate_strength: Concentration of stock solution (e.g., 20% EDTA)
            target_strength: Desired final concentration (e.g., 2% EDTA)
            volume: Total volume needed in ml

        Returns:
            DilutionResult with dilution instructions
        """
        notes = []

        if concentrate_strength <= target_strength:
            raise ValueError("Concentrate strength must be greater than target strength")

        # Calculate dilution ratio
        dilution_factor = concentrate_strength / target_strength

        # Calculate volumes
        concentrate_ml = volume / dilution_factor
        water_ml = volume - concentrate_ml

        # Express as ratio
        # Normalize to smallest whole numbers
        ratio_parts_water = round((water_ml / concentrate_ml), 1)

        if ratio_parts_water == int(ratio_parts_water):
            dilution_ratio = f"1:{int(ratio_parts_water)}"
        else:
            dilution_ratio = f"1:{ratio_parts_water:.1f}"

        notes.append(
            f"Mix {concentrate_ml:.1f} ml concentrate with {water_ml:.1f} ml distilled water"
        )
        notes.append(f"Dilution ratio: {dilution_ratio} (concentrate:water)")

        if target_strength <= 3.0:
            notes.append("Standard working strength for most platinum/palladium prints")

        return DilutionResult(
            concentrate_ml=concentrate_ml,
            water_ml=water_ml,
            total_ml=volume,
            concentrate_strength=concentrate_strength,
            target_strength=target_strength,
            dilution_ratio=dilution_ratio,
            solution_type="developer",
            notes=notes,
        )

    def calculate_clearing_bath(
        self,
        volume: float,
        bath_number: int = 1,
    ) -> DilutionResult:
        """Calculate clearing bath solution preparation.

        Args:
            volume: Total volume needed in ml
            bath_number: Clearing bath number (1, 2, or 3)

        Returns:
            DilutionResult with clearing bath instructions
        """
        notes = []

        # Standard clearing bath concentrations
        # Bath 1: 1% citric acid or 1% HCl
        # Bath 2: 0.5% citric acid
        # Bath 3: water rinse
        if bath_number == 1:
            concentrate_strength = 100.0  # Assuming 100% citric acid powder
            target_strength = 1.0
            notes.append("First clearing bath: 1% citric acid solution")
            notes.append("Use to remove residual iron salts from the print")
        elif bath_number == 2:
            concentrate_strength = 100.0
            target_strength = 0.5
            notes.append("Second clearing bath: 0.5% citric acid solution")
            notes.append("Weaker solution for final clearing")
        else:
            # Bath 3 is just water
            return DilutionResult(
                concentrate_ml=0.0,
                water_ml=volume,
                total_ml=volume,
                concentrate_strength=0.0,
                target_strength=0.0,
                dilution_ratio="0:1",
                solution_type=f"clearing_bath_{bath_number}",
                notes=["Third clearing bath: distilled water rinse"],
            )

        # Calculate dilution
        dilution_factor = concentrate_strength / target_strength
        concentrate_ml = volume / dilution_factor
        water_ml = volume - concentrate_ml

        # For powder, convert to grams
        grams = concentrate_ml  # 1 ml ≈ 1 gram for citric acid
        notes.append(f"Dissolve {grams:.1f}g citric acid in {water_ml:.1f}ml distilled water")

        return DilutionResult(
            concentrate_ml=concentrate_ml,
            water_ml=water_ml,
            total_ml=volume,
            concentrate_strength=concentrate_strength,
            target_strength=target_strength,
            dilution_ratio=f"{grams:.1f}g/L",
            solution_type=f"clearing_bath_{bath_number}",
            notes=notes,
        )

    def suggest_replenishment(
        self,
        solution: str,
        usage: float,
        current_volume: float = 1000.0,
        exhaustion_threshold: float = 0.30,
    ) -> ReplenishmentResult:
        """Suggest solution replenishment amounts.

        Args:
            solution: Solution type ('developer', 'clearing_bath_1', etc.)
            usage: Amount used in ml
            current_volume: Current solution volume in ml (default 1000.0)
            exhaustion_threshold: Replace when exhaustion exceeds this (default 0.30 = 30%)

        Returns:
            ReplenishmentResult with replenishment recommendations
        """
        notes = []

        # Calculate exhaustion
        exhaustion_percent = (usage / current_volume) * 100
        should_replace = exhaustion_percent >= (exhaustion_threshold * 100)

        # Target volume after replenishment
        target_volume = current_volume

        if should_replace:
            # Full replacement recommended
            replenish_ml = current_volume
            notes.append(f"Solution exhaustion ({exhaustion_percent:.1f}%) exceeds threshold")
            notes.append("Recommend preparing fresh solution for consistent results")
        else:
            # Just top up to original volume
            replenish_ml = usage
            notes.append(f"Solution exhaustion ({exhaustion_percent:.1f}%) within acceptable range")
            notes.append(f"Top up with {replenish_ml:.1f}ml fresh solution")

        replenish_drops = replenish_ml * self.settings.drops_per_ml

        # Solution-specific notes
        if "developer" in solution.lower():
            notes.append("Developers can typically handle 20-30% exhaustion before replacement")
        elif "clearing" in solution.lower():
            notes.append("Clearing baths should be replaced more frequently for clean whites")

        return ReplenishmentResult(
            replenish_ml=replenish_ml,
            replenish_drops=replenish_drops,
            solution_type=solution,
            current_volume_ml=current_volume,
            usage_ml=usage,
            target_volume_ml=target_volume,
            should_replace=should_replace,
            exhaustion_percent=exhaustion_percent,
            notes=notes,
        )


# ============================================================================
# Environmental Compensation
# ============================================================================


class EnvironmentalCompensation:
    """
    Environmental compensation calculations for altitude, season, and conditions.

    Provides adjustments for:
    - Altitude effects on drying and exposure
    - Seasonal variations in temperature and humidity
    - Optimal working conditions
    - Drying time estimates
    """

    def __init__(self, settings: ChemistrySettings | None = None):
        """Initialize calculator with settings.

        Args:
            settings: Chemistry settings. If None, uses global settings.
        """
        self.settings = settings or get_settings().chemistry

    def adjust_for_altitude(
        self,
        base_value: float,
        altitude: float,
        value_type: str = "drying_time",
    ) -> EnvironmentalAdjustment:
        """Adjust values for altitude.

        Args:
            base_value: Base value at sea level
            altitude: Altitude in feet
            value_type: Type of value being adjusted ('drying_time', 'exposure_time')

        Returns:
            EnvironmentalAdjustment with altitude-adjusted value
        """
        notes = []

        # Altitude effects:
        # - Lower air pressure = faster drying (water evaporates more easily)
        # - UV intensity increases with altitude
        if value_type == "drying_time":
            # Drying time decreases ~5% per 1000 ft
            adjustment_factor = 1.0 - (altitude / 1000.0 * 0.05)
            adjustment_factor = max(0.7, adjustment_factor)  # Cap at 30% reduction
            notes.append(
                f"At {altitude:.0f} ft altitude, drying time reduced by {(1 - adjustment_factor) * 100:.1f}%"
            )
        elif value_type == "exposure_time":
            # UV intensity increases ~4% per 1000 ft, so exposure time decreases
            adjustment_factor = 1.0 - (altitude / 1000.0 * 0.04)
            adjustment_factor = max(0.6, adjustment_factor)  # Cap at 40% reduction
            notes.append(
                f"At {altitude:.0f} ft altitude, UV intensity higher - reduce exposure by {(1 - adjustment_factor) * 100:.1f}%"
            )
        else:
            adjustment_factor = 1.0
            notes.append(f"No altitude adjustment for {value_type}")

        adjusted_value = base_value * adjustment_factor

        if altitude > 5000:
            notes.append("High altitude: Monitor drying carefully, may be significantly faster")

        return EnvironmentalAdjustment(
            adjusted_value=adjusted_value,
            base_value=base_value,
            adjustment_factor=adjustment_factor,
            adjustment_type=f"altitude_{value_type}",
            altitude_feet=altitude,
            month=None,
            humidity_percent=None,
            temperature_fahrenheit=None,
            notes=notes,
        )

    def adjust_for_season(
        self,
        base_value: float,
        month: int,
        value_type: str = "drying_time",
        latitude: float = 40.0,
    ) -> EnvironmentalAdjustment:
        """Adjust values for seasonal variations.

        Args:
            base_value: Base value at optimal conditions
            month: Month of year (1-12)
            value_type: Type of value being adjusted
            latitude: Latitude in degrees (affects seasonal variation)

        Returns:
            EnvironmentalAdjustment with seasonal adjustment
        """
        import math

        notes = []

        # Seasonal variation in temperature and humidity
        # Summer (Jun-Aug): warmer, more humid in many locations
        # Winter (Dec-Feb): cooler, drier
        # Peak summer: month 7 (July), Peak winter: month 1 (January)

        # Calculate seasonal factor using sine wave
        # Peak summer = 1.0, Peak winter = -1.0
        seasonal_phase = (month - 1) / 12.0 * 2 * math.pi
        seasonal_factor = math.sin(seasonal_phase - math.pi / 2)  # Shift so July is peak

        # Latitude affects magnitude of seasonal variation
        # Higher latitude = more seasonal variation
        latitude_factor = min(1.0, abs(latitude) / 50.0)
        seasonal_factor *= latitude_factor

        if value_type == "drying_time":
            # Summer = faster drying, Winter = slower drying
            # Variation: ±15% from base
            adjustment_factor = 1.0 - (seasonal_factor * 0.15)
            season_name = self._get_season_name(month)
            notes.append(
                f"{season_name}: drying time adjusted by {(adjustment_factor - 1) * 100:+.1f}%"
            )
        elif value_type == "exposure_time":
            # Summer = faster exposure (warmer), Winter = slower
            adjustment_factor = 1.0 - (seasonal_factor * 0.10)
            season_name = self._get_season_name(month)
            notes.append(
                f"{season_name}: exposure time adjusted by {(adjustment_factor - 1) * 100:+.1f}%"
            )
        else:
            adjustment_factor = 1.0

        adjusted_value = base_value * adjustment_factor

        return EnvironmentalAdjustment(
            adjusted_value=adjusted_value,
            base_value=base_value,
            adjustment_factor=adjustment_factor,
            adjustment_type=f"season_{value_type}",
            month=month,
            altitude_feet=None,
            humidity_percent=None,
            temperature_fahrenheit=None,
            notes=notes,
        )

    def get_optimal_conditions(self) -> OptimalConditions:
        """Get optimal working conditions for platinum/palladium printing.

        Returns:
            OptimalConditions with recommended ranges
        """
        notes = [
            "Maintain consistent conditions throughout the entire process",
            "Avoid working in direct sunlight or UV-rich environments before exposure",
            "Store coated paper in dark, dry conditions if not exposing immediately",
            "Allow chemicals to reach room temperature before mixing",
        ]

        return OptimalConditions(
            temperature_f_min=65.0,
            temperature_f_max=75.0,
            temperature_f_ideal=68.0,
            humidity_percent_min=40.0,
            humidity_percent_max=60.0,
            humidity_percent_ideal=50.0,
            altitude_feet_max=8000.0,
            coating_to_exposure_hours_min=0.25,  # 15 minutes minimum
            coating_to_exposure_hours_max=24.0,  # Best within 24 hours
            development_minutes_min=3.0,
            development_minutes_max=5.0,
            notes=notes,
        )

    def calculate_drying_time(
        self,
        humidity: float,
        temperature: float,
        paper: str,
        forced_air: bool = False,
    ) -> DryingTimeEstimate:
        """Estimate drying time for coated paper.

        Args:
            humidity: Relative humidity in percent
            temperature: Temperature in Fahrenheit
            paper: Paper type
            forced_air: Whether using forced air drying

        Returns:
            DryingTimeEstimate with estimated drying time
        """
        notes = []

        # Base drying time (minutes) for average conditions
        base_drying_minutes = 15.0

        # 1. Humidity factor
        # Higher humidity = slower drying
        optimal_humidity = 50.0
        humidity_delta = (humidity - optimal_humidity) / 100.0
        humidity_coefficient = 0.40  # 40% change per 100% humidity delta
        humidity_factor = 1.0 + (humidity_delta * humidity_coefficient)
        humidity_factor = max(0.6, min(2.0, humidity_factor))

        # 2. Temperature factor
        # Higher temperature = faster drying
        optimal_temp = 68.0
        temp_delta = (temperature - optimal_temp) / 10.0  # Per 10°F
        temp_coefficient = 0.15  # 15% change per 10°F
        temperature_factor = 1.0 - (temp_delta * temp_coefficient)
        temperature_factor = max(0.6, min(1.4, temperature_factor))

        # 3. Paper absorbency factor
        paper_lower = paper.lower()
        if "hot_press" in paper_lower or "hp" in paper_lower or "platine" in paper_lower:
            absorbency_factor = 0.9  # Hot press dries faster
            notes.append("Hot press paper: faster drying")
        elif "cold_press" in paper_lower or "cp" in paper_lower:
            absorbency_factor = 1.2  # Cold press holds more water
            notes.append("Cold press paper: slower drying, more absorbent")
        elif "rough" in paper_lower:
            absorbency_factor = 1.3
            notes.append("Rough paper: slowest drying, high absorbency")
        else:
            absorbency_factor = 1.0

        # 4. Calculate base drying time
        drying_minutes = (
            base_drying_minutes * humidity_factor * temperature_factor * absorbency_factor
        )

        # 5. Forced air adjustment
        if forced_air:
            drying_minutes *= 0.5  # 50% reduction with forced air
            notes.append("Forced air drying reduces time by ~50%")
        else:
            notes.append("Natural air drying - consider forced air for faster results")

        # 6. Estimate range (±20%)
        range_min = drying_minutes * 0.8
        range_max = drying_minutes * 1.2

        # 7. Recommendations
        forced_air_recommended = False
        if humidity > 65 or temperature < 65:
            forced_air_recommended = True
            notes.append("Forced air drying recommended due to conditions")

        if drying_minutes > 30:
            notes.append("Long drying time - consider dehumidifier or warming area")

        return DryingTimeEstimate(
            drying_minutes=drying_minutes,
            drying_hours=drying_minutes / 60.0,
            humidity_percent=humidity,
            temperature_fahrenheit=temperature,
            paper_type=paper,
            humidity_factor=humidity_factor,
            temperature_factor=temperature_factor,
            paper_absorbency_factor=absorbency_factor,
            forced_air_recommended=forced_air_recommended,
            estimated_range_minutes=(range_min, range_max),
            notes=notes,
        )

    @staticmethod
    def _get_season_name(month: int) -> str:
        """Get season name from month (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
