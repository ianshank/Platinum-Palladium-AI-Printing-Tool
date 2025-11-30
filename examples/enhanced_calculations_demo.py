#!/usr/bin/env python
"""
Demonstration of the enhanced calculations module.

This script shows how to use all the advanced calculators for
platinum/palladium printing calculations.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ptpd_calibration.calculations import (
    UVExposureCalculator,
    CoatingVolumeCalculator,
    CostCalculator,
    DilutionCalculator,
    EnvironmentalCompensation,
)


def demo_uv_exposure():
    """Demonstrate UV exposure calculations."""
    print("\n" + "=" * 70)
    print("UV EXPOSURE CALCULATION WITH ENVIRONMENTAL COMPENSATION")
    print("=" * 70)

    calc = UVExposureCalculator()

    # Calculate exposure for a dense negative on a humid summer day
    result = calc.calculate_uv_exposure(
        base_time=10.0,  # 10 minutes at reference conditions
        negative_density=1.9,  # Dense negative
        humidity=65.0,  # 65% RH (humid)
        temperature=75.0,  # 75°F (warm)
        uv_intensity=95.0,  # UV source at 95% intensity
        paper_factor=1.1,  # Slightly slower paper
        chemistry_factor=1.2,  # High platinum ratio
        base_density=1.6,  # Reference density
        optimal_humidity=50.0,
        optimal_temperature=68.0,
    )

    print(f"\nBase exposure time: {result.base_time_minutes:.1f} minutes")
    print(f"Adjusted exposure time: {result.adjusted_exposure_minutes:.2f} minutes")
    print(f"                        ({result.adjusted_exposure_seconds:.0f} seconds)")
    print(f"\nConfidence interval (95%): {result.confidence_lower_minutes:.2f} - {result.confidence_upper_minutes:.2f} minutes")

    print("\nAdjustment Factors:")
    print(f"  Humidity:     {result.humidity_factor:.3f}x")
    print(f"  Temperature:  {result.temperature_factor:.3f}x")
    print(f"  Density:      {result.density_factor:.3f}x")
    print(f"  UV Intensity: {result.intensity_factor:.3f}x")
    print(f"  Paper:        {result.paper_factor:.3f}x")
    print(f"  Chemistry:    {result.chemistry_factor:.3f}x")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")

    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  • {note}")


def demo_coating_volume():
    """Demonstrate coating volume calculations."""
    print("\n" + "=" * 70)
    print("COATING VOLUME CALCULATION")
    print("=" * 70)

    calc = CoatingVolumeCalculator()

    # Calculate coating for 11x14 print on Arches Platine with glass rod
    result = calc.determine_coating_volume(
        paper_area=154.0,  # 11x14 inches = 154 sq in
        paper_type="arches_platine",
        coating_method="glass_rod",
        humidity=55.0,
    )

    print(f"\nPaper: {result.paper_type}")
    print(f"Area: {result.paper_area_sq_inches:.1f} square inches")
    print(f"Coating method: {result.coating_method}")

    print(f"\nCalculated volume: {result.total_ml:.2f} ml ({result.total_drops:.0f} drops)")
    print(f"Recommended volume: {result.recommended_ml:.1f} ml ({result.recommended_drops:.0f} drops)")

    print(f"\nBreakdown:")
    print(f"  Base volume:     {result.base_volume_ml:.2f} ml")
    print(f"  After method:    {result.adjusted_volume_ml:.2f} ml ({result.method_efficiency_factor:.2f}x)")
    print(f"  Waste allowance: {result.waste_volume_ml:.2f} ml ({result.waste_factor:.2f}x)")

    print(f"\nAbsorption rate: {result.absorption_rate_ml_per_sq_inch:.4f} ml/sq in")
    print(f"Humidity adjustment: {result.humidity_adjustment_factor:.3f}x")

    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  • {note}")


def demo_cost_calculation():
    """Demonstrate cost calculations."""
    print("\n" + "=" * 70)
    print("PRINT COST CALCULATION")
    print("=" * 70)

    calc = CostCalculator()

    # Calculate cost for 8x10 print with 50/50 Pt/Pd mix
    chemistry = {
        'ferric_oxalate_ml': 1.8,
        'platinum_ml': 0.9,
        'palladium_ml': 0.9,
        'na2_ml': 0.45,
    }

    result = calc.calculate_print_cost(
        paper_size="8x10",
        chemistry=chemistry,
        paper_type="arches_platine",
        include_processing=True,
    )

    print(f"\nPrint size: {result.paper_size} ({result.paper_area_sq_inches:.0f} sq in)")
    print(f"Metal ratio: {result.metal_ratio_platinum * 100:.0f}% Platinum / {(1 - result.metal_ratio_platinum) * 100:.0f}% Palladium")

    print(f"\nTotal cost: ${result.total_cost_usd:.2f}")

    print("\nCost breakdown:")
    print(f"  Ferric Oxalate:  ${result.ferric_oxalate_cost:.2f}")
    print(f"  Platinum:        ${result.platinum_cost:.2f}")
    print(f"  Palladium:       ${result.palladium_cost:.2f}")
    print(f"  Contrast Agent:  ${result.contrast_agent_cost:.2f}")
    print(f"  Paper:           ${result.paper_cost:.2f}")
    print(f"  Processing:      ${result.other_costs:.2f}")

    print(f"\nChemistry usage: {result.total_chemistry_ml:.2f} ml")
    print(f"Chemistry cost per ml: ${result.chemistry_cost_per_ml:.2f}")


def demo_session_cost():
    """Demonstrate session cost calculations."""
    print("\n" + "=" * 70)
    print("SESSION COST CALCULATION")
    print("=" * 70)

    calc = CostCalculator()

    # Create multiple prints for a session
    prints = []

    # Print 1: 5x7
    prints.append(calc.calculate_print_cost(
        "5x7",
        {'ferric_oxalate_ml': 0.8, 'platinum_ml': 0.0, 'palladium_ml': 0.8, 'na2_ml': 0.2},
        "arches_platine"
    ))

    # Print 2: 8x10
    prints.append(calc.calculate_print_cost(
        "8x10",
        {'ferric_oxalate_ml': 1.8, 'platinum_ml': 0.9, 'palladium_ml': 0.9, 'na2_ml': 0.45},
        "arches_platine"
    ))

    # Print 3: 11x14
    prints.append(calc.calculate_print_cost(
        "11x14",
        {'ferric_oxalate_ml': 3.5, 'platinum_ml': 2.6, 'palladium_ml': 0.9, 'na2_ml': 0.9},
        "arches_platine"
    ))

    session = calc.calculate_session_cost(prints)

    print(f"\nSession summary:")
    print(f"  Number of prints: {session.num_prints}")
    print(f"  Total cost: ${session.total_session_cost_usd:.2f}")
    print(f"  Average per print: ${session.average_cost_per_print:.2f}")

    print(f"\nCost breakdown:")
    print(f"  Chemistry:  ${session.total_chemistry_cost:.2f} ({session.total_chemistry_cost / session.total_session_cost_usd * 100:.1f}%)")
    print(f"  Paper:      ${session.total_paper_cost:.2f} ({session.total_paper_cost / session.total_session_cost_usd * 100:.1f}%)")
    print(f"  Processing: ${session.total_other_costs:.2f} ({session.total_other_costs / session.total_session_cost_usd * 100:.1f}%)")

    print(f"\nTotal chemistry used: {session.total_chemistry_ml:.1f} ml")

    if session.notes:
        print("\nInsights:")
        for note in session.notes:
            print(f"  • {note}")


def demo_solution_usage():
    """Demonstrate solution usage estimation."""
    print("\n" + "=" * 70)
    print("SOLUTION USAGE ESTIMATION")
    print("=" * 70)

    calc = CostCalculator()

    # Estimate for 10 prints, 8x10 average, 50% Pt
    result = calc.estimate_solution_usage(
        num_prints=10,
        avg_size="8x10",
        avg_platinum_ratio=0.5,
        coating_method="glass_rod",
    )

    print(f"\nEstimate for {result.num_prints} prints (average {result.average_print_size_sq_inches:.0f} sq in)")
    print(f"Average platinum ratio: {result.average_platinum_ratio * 100:.0f}%")

    print("\nEstimated usage:")
    print(f"  Ferric Oxalate:  {result.ferric_oxalate_ml:.1f} ml")
    print(f"  Platinum:        {result.platinum_ml:.1f} ml")
    print(f"  Palladium:       {result.palladium_ml:.1f} ml")
    print(f"  Contrast Agent:  {result.contrast_agent_ml:.1f} ml")
    print(f"  Developer:       {result.developer_ml:.1f} ml")
    print(f"  Clearing Bath:   {result.clearing_bath_ml:.1f} ml")

    print("\nRecommended stock levels (with 20% safety margin):")
    print(f"  Ferric Oxalate:  {result.recommended_stock_ferric_oxalate_ml:.0f} ml")
    print(f"  Platinum:        {result.recommended_stock_platinum_ml:.0f} ml")
    print(f"  Palladium:       {result.recommended_stock_palladium_ml:.0f} ml")


def demo_dilution():
    """Demonstrate dilution calculations."""
    print("\n" + "=" * 70)
    print("DILUTION CALCULATIONS")
    print("=" * 70)

    calc = DilutionCalculator()

    # Developer dilution
    print("\n--- Developer Dilution ---")
    result = calc.calculate_developer_dilution(
        concentrate_strength=20.0,  # 20% EDTA stock
        target_strength=3.0,  # 3% working solution
        volume=500.0,  # 500 ml needed
    )

    print(f"\nDiluting {result.concentrate_strength:.1f}% to {result.target_strength:.1f}%")
    print(f"Total volume: {result.total_ml:.0f} ml")
    print(f"Dilution ratio: {result.dilution_ratio}")
    print(f"\nMix:")
    print(f"  Concentrate: {result.concentrate_ml:.1f} ml")
    print(f"  Water:       {result.water_ml:.1f} ml")

    if result.notes:
        print("\nInstructions:")
        for note in result.notes:
            print(f"  • {note}")

    # Clearing bath
    print("\n--- Clearing Bath Preparation ---")
    for bath_num in [1, 2, 3]:
        result = calc.calculate_clearing_bath(
            volume=1000.0,  # 1 liter
            bath_number=bath_num,
        )

        print(f"\nClearing Bath #{bath_num}:")
        if bath_num < 3:
            print(f"  Citric acid: {result.concentrate_ml:.1f}g")
            print(f"  Water:       {result.water_ml:.0f} ml")
        else:
            print(f"  Distilled water rinse")

        for note in result.notes:
            print(f"  • {note}")


def demo_replenishment():
    """Demonstrate replenishment calculations."""
    print("\n" + "=" * 70)
    print("SOLUTION REPLENISHMENT")
    print("=" * 70)

    calc = DilutionCalculator()

    # Check if developer needs replenishment
    result = calc.suggest_replenishment(
        solution="developer",
        usage=250.0,  # Used 250 ml
        current_volume=1000.0,  # 1 liter working solution
        exhaustion_threshold=0.30,  # Replace at 30% exhaustion
    )

    print(f"\nSolution: {result.solution_type}")
    print(f"Current volume: {result.current_volume_ml:.0f} ml")
    print(f"Usage: {result.usage_ml:.0f} ml")
    print(f"Exhaustion: {result.exhaustion_percent:.1f}%")

    if result.should_replace:
        print(f"\n⚠ REPLACE SOLUTION ({result.replenish_ml:.0f} ml fresh solution)")
    else:
        print(f"\n✓ Top up with {result.replenish_ml:.0f} ml ({result.replenish_drops:.0f} drops)")

    if result.notes:
        print("\nRecommendations:")
        for note in result.notes:
            print(f"  • {note}")


def demo_environmental():
    """Demonstrate environmental compensation."""
    print("\n" + "=" * 70)
    print("ENVIRONMENTAL COMPENSATION")
    print("=" * 70)

    calc = EnvironmentalCompensation()

    # Altitude adjustment
    print("\n--- Altitude Adjustment ---")
    result = calc.adjust_for_altitude(
        base_value=15.0,  # 15 minutes base drying time
        altitude=5000.0,  # 5000 ft altitude
        value_type="drying_time",
    )

    print(f"\nBase drying time: {result.base_value:.1f} minutes")
    print(f"At {result.altitude_feet:.0f} ft: {result.adjusted_value:.1f} minutes")
    print(f"Adjustment factor: {result.adjustment_factor:.3f}x")

    for note in result.notes:
        print(f"  • {note}")

    # Seasonal adjustment
    print("\n--- Seasonal Adjustment ---")
    result = calc.adjust_for_season(
        base_value=10.0,  # 10 minutes base exposure
        month=7,  # July (summer)
        value_type="exposure_time",
        latitude=40.0,
    )

    print(f"\nBase exposure: {result.base_value:.1f} minutes")
    print(f"Seasonal adjustment: {result.adjusted_value:.1f} minutes")
    print(f"Adjustment factor: {result.adjustment_factor:.3f}x")

    for note in result.notes:
        print(f"  • {note}")

    # Optimal conditions
    print("\n--- Optimal Conditions ---")
    conditions = calc.get_optimal_conditions()

    print(f"\nTemperature: {conditions.temperature_f_ideal:.0f}°F (range: {conditions.temperature_f_min:.0f}-{conditions.temperature_f_max:.0f}°F)")
    print(f"Humidity: {conditions.humidity_percent_ideal:.0f}% RH (range: {conditions.humidity_percent_min:.0f}-{conditions.humidity_percent_max:.0f}%)")
    print(f"Max altitude: {conditions.altitude_feet_max:.0f} ft")
    print(f"\nTiming:")
    print(f"  Coating to exposure: {conditions.coating_to_exposure_hours_min:.2f}-{conditions.coating_to_exposure_hours_max:.0f} hours")
    print(f"  Development: {conditions.development_minutes_min:.0f}-{conditions.development_minutes_max:.0f} minutes")

    print("\nGuidelines:")
    for note in conditions.notes:
        print(f"  • {note}")

    # Drying time
    print("\n--- Drying Time Estimate ---")
    result = calc.calculate_drying_time(
        humidity=60.0,
        temperature=72.0,
        paper="arches_platine",
        forced_air=False,
    )

    print(f"\nConditions: {result.humidity_percent:.0f}% RH, {result.temperature_fahrenheit:.0f}°F")
    print(f"Paper: {result.paper_type}")
    print(f"\nEstimated drying time: {result.drying_minutes:.1f} minutes ({result.drying_hours:.2f} hours)")
    print(f"Range: {result.estimated_range_minutes[0]:.1f} - {result.estimated_range_minutes[1]:.1f} minutes")

    print(f"\nFactors:")
    print(f"  Humidity:    {result.humidity_factor:.3f}x")
    print(f"  Temperature: {result.temperature_factor:.3f}x")
    print(f"  Paper:       {result.paper_absorbency_factor:.3f}x")

    if result.forced_air_recommended:
        print("\n⚠ Forced air drying recommended")

    for note in result.notes:
        print(f"  • {note}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("ENHANCED CALCULATIONS MODULE DEMONSTRATION")
    print("Platinum/Palladium Printing Technical Calculations")
    print("=" * 70)

    demo_uv_exposure()
    demo_coating_volume()
    demo_cost_calculation()
    demo_session_cost()
    demo_solution_usage()
    demo_dilution()
    demo_replenishment()
    demo_environmental()

    print("\n" + "=" * 70)
    print("END OF DEMONSTRATION")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
