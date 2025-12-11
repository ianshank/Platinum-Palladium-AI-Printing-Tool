"""
Test data builders for PTPD Calibration tests.

Provides builder classes and factories for creating test data.
"""

import random
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np


class DensityBuilder:
    """Builder for density measurement data."""

    def __init__(self):
        self._num_steps = 21
        self._dmin = 0.1
        self._dmax = 2.0
        self._gamma = 0.85
        self._noise = 0.0
        self._monotonic = True

    def with_steps(self, num_steps: int) -> "DensityBuilder":
        """Set the number of density steps."""
        self._num_steps = num_steps
        return self

    def with_range(self, dmin: float, dmax: float) -> "DensityBuilder":
        """Set the density range."""
        self._dmin = dmin
        self._dmax = dmax
        return self

    def with_gamma(self, gamma: float) -> "DensityBuilder":
        """Set the gamma curve."""
        self._gamma = gamma
        return self

    def with_noise(self, noise: float) -> "DensityBuilder":
        """Add random noise to measurements."""
        self._noise = noise
        return self

    def non_monotonic(self) -> "DensityBuilder":
        """Allow non-monotonic values."""
        self._monotonic = False
        return self

    def build(self) -> list[float]:
        """Build the density list."""
        steps = np.linspace(0, 1, self._num_steps)
        densities = self._dmin + (self._dmax - self._dmin) * (steps ** self._gamma)

        if self._noise > 0:
            noise = np.random.normal(0, self._noise, self._num_steps)
            densities = densities + noise

            if self._monotonic:
                # Ensure monotonicity
                for i in range(1, len(densities)):
                    if densities[i] < densities[i - 1]:
                        densities[i] = densities[i - 1] + 0.001

        return list(densities)


class CurveBuilder:
    """Builder for curve data."""

    def __init__(self):
        self._num_points = 256
        self._gamma = 0.9
        self._brightness = 0.0
        self._contrast = 0.0
        self._name = "Test Curve"

    def with_points(self, num_points: int) -> "CurveBuilder":
        """Set the number of curve points."""
        self._num_points = num_points
        return self

    def with_gamma(self, gamma: float) -> "CurveBuilder":
        """Set the gamma value."""
        self._gamma = gamma
        return self

    def with_brightness(self, brightness: float) -> "CurveBuilder":
        """Set brightness offset (-1 to 1)."""
        self._brightness = brightness
        return self

    def with_contrast(self, contrast: float) -> "CurveBuilder":
        """Set contrast adjustment (-1 to 1)."""
        self._contrast = contrast
        return self

    def with_name(self, name: str) -> "CurveBuilder":
        """Set the curve name."""
        self._name = name
        return self

    def build(self) -> dict:
        """Build the curve data."""
        input_values = list(np.linspace(0, 1, self._num_points))

        # Apply gamma
        output_values = [x ** self._gamma for x in input_values]

        # Apply brightness
        output_values = [min(1, max(0, y + self._brightness)) for y in output_values]

        # Apply contrast
        if self._contrast != 0:
            pivot = 0.5
            factor = (1 + self._contrast) if self._contrast > 0 else (1 / (1 - self._contrast))
            output_values = [
                min(1, max(0, (y - pivot) * factor + pivot))
                for y in output_values
            ]

        return {
            "name": self._name,
            "input_values": input_values,
            "output_values": output_values,
        }

    def build_model(self):
        """Build a CurveData model instance."""
        from ptpd_calibration.core.models import CurveData

        data = self.build()
        return CurveData(
            name=data["name"],
            input_values=data["input_values"],
            output_values=data["output_values"],
        )


class CalibrationRecordBuilder:
    """Builder for CalibrationRecord instances."""

    def __init__(self):
        from ptpd_calibration.core.types import (
            ChemistryType,
            ContrastAgent,
            DeveloperType,
        )

        self._paper_type = "Test Paper"
        self._paper_weight = 300
        self._exposure_time = 180.0
        self._metal_ratio = 0.5
        self._chemistry_type = ChemistryType.PLATINUM_PALLADIUM
        self._contrast_agent = ContrastAgent.NA2
        self._contrast_amount = 5.0
        self._developer = DeveloperType.POTASSIUM_OXALATE
        self._humidity = 50.0
        self._temperature = 21.0
        self._densities = None
        self._notes = None

    def with_paper(self, paper_type: str, weight: int = 300) -> "CalibrationRecordBuilder":
        """Set paper type and weight."""
        self._paper_type = paper_type
        self._paper_weight = weight
        return self

    def with_exposure(self, exposure_time: float) -> "CalibrationRecordBuilder":
        """Set exposure time."""
        self._exposure_time = exposure_time
        return self

    def with_metal_ratio(self, ratio: float) -> "CalibrationRecordBuilder":
        """Set metal ratio (0-1)."""
        self._metal_ratio = ratio
        return self

    def with_contrast(self, agent: str, amount: float) -> "CalibrationRecordBuilder":
        """Set contrast agent and amount."""
        from ptpd_calibration.core.types import ContrastAgent

        self._contrast_agent = ContrastAgent(agent)
        self._contrast_amount = amount
        return self

    def with_environment(
        self, humidity: float, temperature: float
    ) -> "CalibrationRecordBuilder":
        """Set environment conditions."""
        self._humidity = humidity
        self._temperature = temperature
        return self

    def with_densities(self, densities: list[float]) -> "CalibrationRecordBuilder":
        """Set measured densities."""
        self._densities = densities
        return self

    def with_notes(self, notes: str) -> "CalibrationRecordBuilder":
        """Set notes."""
        self._notes = notes
        return self

    def build(self):
        """Build a CalibrationRecord instance."""
        from ptpd_calibration.core.models import CalibrationRecord

        densities = self._densities or DensityBuilder().build()

        return CalibrationRecord(
            paper_type=self._paper_type,
            paper_weight=self._paper_weight,
            exposure_time=self._exposure_time,
            metal_ratio=self._metal_ratio,
            chemistry_type=self._chemistry_type,
            contrast_agent=self._contrast_agent,
            contrast_amount=self._contrast_amount,
            developer=self._developer,
            humidity=self._humidity,
            temperature=self._temperature,
            measured_densities=densities,
            notes=self._notes,
        )


class ImageBuilder:
    """Builder for test images."""

    def __init__(self):
        self._width = 100
        self._height = 100
        self._mode = "L"
        self._fill = "gradient"

    def with_size(self, width: int, height: int) -> "ImageBuilder":
        """Set image dimensions."""
        self._width = width
        self._height = height
        return self

    def with_mode(self, mode: str) -> "ImageBuilder":
        """Set color mode (L, RGB, RGBA)."""
        self._mode = mode
        return self

    def with_gradient(self) -> "ImageBuilder":
        """Fill with a gradient."""
        self._fill = "gradient"
        return self

    def with_random(self) -> "ImageBuilder":
        """Fill with random noise."""
        self._fill = "random"
        return self

    def with_solid(self, value: int = 128) -> "ImageBuilder":
        """Fill with solid color."""
        self._fill = ("solid", value)
        return self

    def with_step_tablet(self, num_patches: int = 21) -> "ImageBuilder":
        """Create a step tablet pattern."""
        self._fill = ("step_tablet", num_patches)
        return self

    def build(self) -> "Image":
        """Build the PIL Image."""
        from PIL import Image

        if self._fill == "gradient":
            if self._mode == "L":
                arr = np.tile(
                    np.linspace(0, 255, self._width).astype(np.uint8),
                    (self._height, 1),
                )
            else:
                gray = np.tile(
                    np.linspace(0, 255, self._width).astype(np.uint8),
                    (self._height, 1),
                )
                arr = np.stack([gray, gray, gray], axis=-1)

        elif self._fill == "random":
            if self._mode == "L":
                arr = np.random.randint(0, 255, (self._height, self._width), dtype=np.uint8)
            else:
                arr = np.random.randint(
                    0, 255, (self._height, self._width, 3), dtype=np.uint8
                )

        elif isinstance(self._fill, tuple) and self._fill[0] == "solid":
            value = self._fill[1]
            if self._mode == "L":
                arr = np.full((self._height, self._width), value, dtype=np.uint8)
            else:
                arr = np.full((self._height, self._width, 3), value, dtype=np.uint8)

        elif isinstance(self._fill, tuple) and self._fill[0] == "step_tablet":
            num_patches = self._fill[1]
            patch_width = self._width // num_patches
            arr = np.zeros((self._height, self._width), dtype=np.uint8)
            for i in range(num_patches):
                value = 255 - int(255 * (i / (num_patches - 1)) ** 0.8)
                x_start = i * patch_width
                x_end = (i + 1) * patch_width
                arr[:, x_start:x_end] = value

        else:
            raise ValueError(f"Unknown fill type: {self._fill}")

        return Image.fromarray(arr, mode=self._mode)

    def build_path(self, tmp_path, filename: str = "test_image.png") -> "Path":
        """Build and save image, returning the path."""
        from pathlib import Path

        img = self.build()
        path = tmp_path / filename
        img.save(path)
        return path


def random_paper_type() -> str:
    """Generate a random paper type name."""
    papers = [
        "Arches Platine",
        "Bergger COT320",
        "Hahnemuhle Platinum Rag",
        "Revere Platinum",
        "Crane's Platinotype",
    ]
    return random.choice(papers)


def random_chemistry_type() -> str:
    """Generate a random chemistry type."""
    types = ["platinum_palladium", "palladium", "platinum", "kallitype"]
    return random.choice(types)


def random_densities(num_steps: int = 21) -> list[float]:
    """Generate random but realistic density measurements."""
    return DensityBuilder().with_steps(num_steps).with_noise(0.02).build()


class CyanotypeRecipeBuilder:
    """Builder for cyanotype chemistry recipe test data."""

    def __init__(self):
        self._width = 8.0
        self._height = 10.0
        self._formula = "classic"
        self._paper_type = "cotton_rag"
        self._concentration = 1.0
        self._margin = 0.5

    def with_dimensions(self, width: float, height: float) -> "CyanotypeRecipeBuilder":
        """Set print dimensions in inches."""
        self._width = width
        self._height = height
        return self

    def with_formula(self, formula: str) -> "CyanotypeRecipeBuilder":
        """Set cyanotype formula (classic, new, ware, rex)."""
        self._formula = formula
        return self

    def with_paper_type(self, paper_type: str) -> "CyanotypeRecipeBuilder":
        """Set paper type."""
        self._paper_type = paper_type
        return self

    def with_concentration(self, factor: float) -> "CyanotypeRecipeBuilder":
        """Set concentration factor."""
        self._concentration = factor
        return self

    def with_margin(self, margin: float) -> "CyanotypeRecipeBuilder":
        """Set margin in inches."""
        self._margin = margin
        return self

    def build(self) -> dict:
        """Build the recipe input data."""
        return {
            "width_inches": self._width,
            "height_inches": self._height,
            "formula": self._formula,
            "paper_type": self._paper_type,
            "concentration_factor": self._concentration,
            "margin_inches": self._margin,
        }

    def build_model(self):
        """Build a CyanotypeRecipe model instance."""
        from ptpd_calibration.chemistry.cyanotype_calculator import (
            CyanotypeCalculator,
            CyanotypeRecipe,
        )
        from ptpd_calibration.core.types import CyanotypeFormula

        calculator = CyanotypeCalculator()
        formula_map = {
            "classic": CyanotypeFormula.CLASSIC,
            "new": CyanotypeFormula.NEW,
            "ware": CyanotypeFormula.WARE,
            "rex": CyanotypeFormula.REX,
        }

        return calculator.calculate(
            width_inches=self._width,
            height_inches=self._height,
            formula=formula_map.get(self._formula, CyanotypeFormula.CLASSIC),
            concentration_factor=self._concentration,
            margin_inches=self._margin if self._margin else None,
        )


class CyanotypeExposureBuilder:
    """Builder for cyanotype exposure test data."""

    def __init__(self):
        self._negative_density = 1.6
        self._uv_source = "bl_tubes"
        self._formula = "classic"
        self._humidity = 50.0
        self._paper_factor = 1.0
        self._distance = 4.0

    def with_density(self, density: float) -> "CyanotypeExposureBuilder":
        """Set negative density."""
        self._negative_density = density
        return self

    def with_uv_source(self, source: str) -> "CyanotypeExposureBuilder":
        """Set UV source type."""
        self._uv_source = source
        return self

    def with_formula(self, formula: str) -> "CyanotypeExposureBuilder":
        """Set cyanotype formula."""
        self._formula = formula
        return self

    def with_humidity(self, humidity: float) -> "CyanotypeExposureBuilder":
        """Set humidity percentage."""
        self._humidity = humidity
        return self

    def with_distance(self, distance: float) -> "CyanotypeExposureBuilder":
        """Set UV source distance in inches."""
        self._distance = distance
        return self

    def build(self) -> dict:
        """Build the exposure input data."""
        return {
            "negative_density": self._negative_density,
            "uv_source": self._uv_source,
            "formula": self._formula,
            "humidity_percent": self._humidity,
            "paper_factor": self._paper_factor,
            "distance_inches": self._distance,
        }

    def build_model(self):
        """Build a CyanotypeExposureResult model instance."""
        from ptpd_calibration.exposure.alternative_calculators import (
            CyanotypeExposureCalculator,
            UVSource,
        )
        from ptpd_calibration.core.types import CyanotypeFormula

        calculator = CyanotypeExposureCalculator()
        uv_map = {
            "sunlight": UVSource.SUNLIGHT,
            "bl_tubes": UVSource.BL_TUBES,
            "led_uv": UVSource.LED_UV,
            "mercury_vapor": UVSource.MERCURY_VAPOR,
        }
        formula_map = {
            "classic": CyanotypeFormula.CLASSIC,
            "new": CyanotypeFormula.NEW,
            "ware": CyanotypeFormula.WARE,
            "rex": CyanotypeFormula.REX,
        }

        return calculator.calculate(
            negative_density=self._negative_density,
            uv_source=uv_map.get(self._uv_source, UVSource.BL_TUBES),
            formula=formula_map.get(self._formula, CyanotypeFormula.CLASSIC),
            humidity_percent=self._humidity,
            paper_factor=self._paper_factor,
            distance_inches=self._distance,
        )


class SilverGelatinChemistryBuilder:
    """Builder for silver gelatin processing chemistry test data."""

    def __init__(self):
        self._width = 8.0
        self._height = 10.0
        self._paper_base = "fiber"
        self._developer = "dektol"
        self._dilution = "1:2"
        self._temperature = 20.0
        self._fixer = "sodium_thiosulfate"
        self._include_hypo_clear = True
        self._tray_size = "8x10"
        self._num_prints = 1

    def with_dimensions(self, width: float, height: float) -> "SilverGelatinChemistryBuilder":
        """Set print dimensions in inches."""
        self._width = width
        self._height = height
        return self

    def with_paper_base(self, base: str) -> "SilverGelatinChemistryBuilder":
        """Set paper base type (fiber or rc)."""
        self._paper_base = base
        return self

    def with_developer(self, developer: str) -> "SilverGelatinChemistryBuilder":
        """Set developer type."""
        self._developer = developer
        return self

    def with_dilution(self, dilution: str) -> "SilverGelatinChemistryBuilder":
        """Set developer dilution."""
        self._dilution = dilution
        return self

    def with_temperature(self, temp_c: float) -> "SilverGelatinChemistryBuilder":
        """Set development temperature in Celsius."""
        self._temperature = temp_c
        return self

    def with_fixer(self, fixer: str) -> "SilverGelatinChemistryBuilder":
        """Set fixer type."""
        self._fixer = fixer
        return self

    def with_hypo_clear(self, include: bool) -> "SilverGelatinChemistryBuilder":
        """Set whether to include hypo clear."""
        self._include_hypo_clear = include
        return self

    def with_tray_size(self, size: str) -> "SilverGelatinChemistryBuilder":
        """Set tray size."""
        self._tray_size = size
        return self

    def with_num_prints(self, num: int) -> "SilverGelatinChemistryBuilder":
        """Set number of prints."""
        self._num_prints = num
        return self

    def for_fiber_paper(self) -> "SilverGelatinChemistryBuilder":
        """Configure for fiber-based paper."""
        self._paper_base = "fiber"
        self._include_hypo_clear = True
        return self

    def for_rc_paper(self) -> "SilverGelatinChemistryBuilder":
        """Configure for resin-coated paper."""
        self._paper_base = "rc"
        self._include_hypo_clear = False
        return self

    def build(self) -> dict:
        """Build the chemistry input data."""
        return {
            "width_inches": self._width,
            "height_inches": self._height,
            "paper_base": self._paper_base,
            "developer": self._developer,
            "dilution": self._dilution,
            "temperature_c": self._temperature,
            "fixer": self._fixer,
            "include_hypo_clear": self._include_hypo_clear,
            "tray_size": self._tray_size,
            "num_prints": self._num_prints,
        }

    def build_model(self):
        """Build a ProcessingChemistry model instance."""
        from ptpd_calibration.chemistry.silver_gelatin_calculator import (
            SilverGelatinCalculator,
            DilutionRatio,
            TraySize,
        )
        from ptpd_calibration.core.types import (
            DeveloperType,
            FixerType,
            PaperBase,
        )

        calculator = SilverGelatinCalculator()
        paper_map = {
            "fiber": PaperBase.FIBER,
            "rc": PaperBase.RESIN_COATED,
        }
        developer_map = {
            "dektol": DeveloperType.DEKTOL,
            "d_72": DeveloperType.D_72,
            "d_76": DeveloperType.D_76,
        }
        dilution_map = {
            "stock": DilutionRatio.STOCK,
            "1:1": DilutionRatio.ONE_TO_ONE,
            "1:2": DilutionRatio.ONE_TO_TWO,
            "1:3": DilutionRatio.ONE_TO_THREE,
        }
        fixer_map = {
            "sodium_thiosulfate": FixerType.SODIUM_THIOSULFATE,
            "ammonium_thiosulfate": FixerType.AMMONIUM_THIOSULFATE,
        }
        tray_map = {
            "8x10": TraySize.TRAY_8X10,
            "11x14": TraySize.TRAY_11X14,
            "16x20": TraySize.TRAY_16X20,
        }

        return calculator.calculate(
            width_inches=self._width,
            height_inches=self._height,
            paper_base=paper_map.get(self._paper_base, PaperBase.FIBER),
            developer=developer_map.get(self._developer),
            dilution=dilution_map.get(self._dilution),
            temperature_c=self._temperature,
            fixer=fixer_map.get(self._fixer, FixerType.SODIUM_THIOSULFATE),
            include_hypo_clear=self._include_hypo_clear,
            tray_size=tray_map.get(self._tray_size),
            num_prints=self._num_prints,
        )


class SilverGelatinExposureBuilder:
    """Builder for silver gelatin exposure test data."""

    def __init__(self):
        self._enlarger_height = 30.0
        self._f_stop = 8.0
        self._paper_grade = "grade_2"
        self._paper_speed_iso = 250.0
        self._filter_factor = 1.0
        self._negative_density = 1.0
        self._light_source = "diffusion"

    def with_enlarger_height(self, height_cm: float) -> "SilverGelatinExposureBuilder":
        """Set enlarger height in cm."""
        self._enlarger_height = height_cm
        return self

    def with_f_stop(self, f_stop: float) -> "SilverGelatinExposureBuilder":
        """Set lens f-stop."""
        self._f_stop = f_stop
        return self

    def with_paper_grade(self, grade: str) -> "SilverGelatinExposureBuilder":
        """Set paper grade."""
        self._paper_grade = grade
        return self

    def with_paper_speed(self, iso: float) -> "SilverGelatinExposureBuilder":
        """Set paper speed (ISO)."""
        self._paper_speed_iso = iso
        return self

    def with_filter_factor(self, factor: float) -> "SilverGelatinExposureBuilder":
        """Set filter factor."""
        self._filter_factor = factor
        return self

    def with_negative_density(self, density: float) -> "SilverGelatinExposureBuilder":
        """Set negative density."""
        self._negative_density = density
        return self

    def with_light_source(self, source: str) -> "SilverGelatinExposureBuilder":
        """Set enlarger light source type."""
        self._light_source = source
        return self

    def build(self) -> dict:
        """Build the exposure input data."""
        return {
            "enlarger_height_cm": self._enlarger_height,
            "f_stop": self._f_stop,
            "paper_grade": self._paper_grade,
            "paper_speed_iso": self._paper_speed_iso,
            "filter_factor": self._filter_factor,
            "negative_density": self._negative_density,
            "light_source": self._light_source,
        }

    def build_model(self):
        """Build a SilverGelatinExposureResult model instance."""
        from ptpd_calibration.exposure.alternative_calculators import (
            SilverGelatinExposureCalculator,
            EnlargerLightSource,
        )
        from ptpd_calibration.core.types import PaperGrade

        calculator = SilverGelatinExposureCalculator()
        grade_map = {
            "grade_0": PaperGrade.GRADE_0,
            "grade_1": PaperGrade.GRADE_1,
            "grade_2": PaperGrade.GRADE_2,
            "grade_3": PaperGrade.GRADE_3,
            "grade_4": PaperGrade.GRADE_4,
            "grade_5": PaperGrade.GRADE_5,
        }


        return calculator.calculate(
            enlarger_height_cm=self._enlarger_height,
            f_stop=self._f_stop,
            paper_grade=grade_map.get(self._paper_grade, PaperGrade.GRADE_2),
            paper_speed_iso=self._paper_speed_iso,
            filter_factor=self._filter_factor,
            negative_density=self._negative_density,
        )


def random_cyanotype_formula() -> str:
    """Generate a random cyanotype formula type."""
    formulas = ["classic", "new", "ware", "rex"]
    return random.choice(formulas)


def random_uv_source() -> str:
    """Generate a random UV source type."""
    sources = ["sunlight", "bl_tubes", "led_uv", "mercury_vapor"]
    return random.choice(sources)


def random_developer() -> str:
    """Generate a random developer type."""
    developers = ["dektol", "d_72", "d_76", "xtol", "rodinal"]
    return random.choice(developers)


def random_paper_base() -> str:
    """Generate a random paper base type."""
    bases = ["fiber", "rc"]
    return random.choice(bases)


def random_paper_grade() -> str:
    """Generate a random paper grade."""
    grades = ["grade_0", "grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
    return random.choice(grades)
