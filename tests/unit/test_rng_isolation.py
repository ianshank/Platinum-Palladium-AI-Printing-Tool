"""Library code must not seed or consume the process-global numpy RNG.

Two separate defects shared this cause. Rendering helpers called
``np.random.seed(42)`` for repeatable grain, which made the grain repeatable by
resetting the RNG of whatever called them. And the simulated spectrophotometer
seeded from ``hash(patch_id)``, which is salted per process by
``PYTHONHASHSEED``, so readings its own comment called "consistent" differed
between runs while still clobbering the global RNG for everyone else.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

GLOBAL_SEED = 7
DRAWS = 4


def _global_draws_around(work) -> tuple[list[float], list[float]]:
    """Return numbers drawn from the global RNG with and without ``work`` run."""
    np.random.seed(GLOBAL_SEED)
    without = np.random.rand(DRAWS).tolist()

    np.random.seed(GLOBAL_SEED)
    work()
    with_work = np.random.rand(DRAWS).tolist()
    return without, with_work


class TestRenderingLeavesTheGlobalRngAlone:
    @pytest.fixture
    def gray(self) -> Image.Image:
        return Image.fromarray(np.full((8, 8), 128, np.uint8))

    @staticmethod
    def _textured_settings():
        """Paper texture is off by default, and it is the path that drew noise."""
        from ptpd_calibration.proofing.simulation import ProofSettings

        settings = ProofSettings()
        settings.add_paper_texture = True
        assert settings.texture_strength > 0
        return settings

    def test_soft_proofing(self, gray: Image.Image) -> None:
        from ptpd_calibration.proofing.simulation import SoftProofer

        settings = self._textured_settings()
        without, with_work = _global_draws_around(
            lambda: SoftProofer().proof(gray, settings=settings)
        )

        assert without == with_work

    def test_style_transfer_texture(self, gray: Image.Image) -> None:
        from ptpd_calibration.advanced.features import HistoricStyle, StyleTransfer

        transfer = StyleTransfer()
        # A style whose texture_strength is above zero, so the noise path runs.
        style = HistoricStyle.PICTORIALIST_1890S
        assert transfer.styles[style].texture_strength > 0

        without, with_work = _global_draws_around(lambda: transfer.apply_style(gray, style))

        assert without == with_work

    def test_the_texture_is_still_repeatable(self, gray: Image.Image) -> None:
        """Dropping the global seed must not cost the determinism it bought."""
        from ptpd_calibration.proofing.simulation import SoftProofer

        proofer = SoftProofer()
        settings = self._textured_settings()
        first = np.array(proofer.proof(gray, settings=settings).image)
        second = np.array(proofer.proof(gray, settings=settings).image)

        assert np.array_equal(first, second)


class TestSimulatedPatchReadingsAreReproducible:
    def test_the_same_patch_reads_the_same_way_twice(self) -> None:
        from ptpd_calibration.integrations.spectrophotometer import _patch_rng

        assert _patch_rng("patch_A1").uniform(0, 1) == _patch_rng("patch_A1").uniform(0, 1)

    def test_different_patches_read_differently(self) -> None:
        from ptpd_calibration.integrations.spectrophotometer import _patch_rng

        assert _patch_rng("patch_A1").uniform(0, 1) != _patch_rng("patch_B2").uniform(0, 1)

    def test_the_seed_survives_a_new_process(self) -> None:
        """``hash()`` is salted per process, so this is the test that matters.

        The two child processes are given different ``PYTHONHASHSEED`` values,
        which is exactly what made the old readings irreproducible.
        """
        script = (
            "from ptpd_calibration.integrations.spectrophotometer import _patch_rng;"
            "print(_patch_rng('patch_A1').uniform(0, 1))"
        )
        runs = [
            subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                # The package logs on import, so the value is the last line.
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed},
            )
            .stdout.strip()
            .splitlines()[-1]
            for seed in ("1", "2")
        ]

        assert runs[0] == runs[1]


class TestMeasurementsAreReproducible:
    """The two draws that moved a number the user is shown.

    A calibration tool that reports a different density for the same scan, or a
    different accuracy for the same records, is reporting noise. Both of these
    drew from the process-global RNG unseeded, so the value changed run to run
    and any unrelated caller that drew first changed it again.
    """

    @pytest.fixture
    def margin(self) -> np.ndarray:
        """A margin strip with the gradient and dust a flatbed actually gives."""
        rng = np.random.default_rng(0)
        strip = np.clip(
            np.linspace(225, 238, 60)[:, None] + rng.normal(0, 1.2, (60, 400)), 0, 255
        ).astype(np.uint8)
        strip[rng.integers(0, 60, 40), rng.integers(0, 400, 40)] = 90  # specks
        return strip

    def test_the_paper_base_reads_the_same_every_time(self, margin: np.ndarray) -> None:
        """This reference divides every patch density in the scan."""
        from ptpd_calibration.detection.extractor import DensityExtractor

        extractor = DensityExtractor()
        reads = [extractor._sample_region(margin).mean(axis=0).tolist() for _ in range(8)]

        assert all(r == reads[0] for r in reads)

    def test_the_paper_base_ignores_the_global_rng(self, margin: np.ndarray) -> None:
        from ptpd_calibration.detection.extractor import DensityExtractor

        extractor = DensityExtractor()

        np.random.seed(GLOBAL_SEED)
        first = extractor._sample_region(margin).mean(axis=0).tolist()
        np.random.seed(GLOBAL_SEED)
        np.random.rand(DRAWS)
        second = extractor._sample_region(margin).mean(axis=0).tolist()

        assert first == second

    def test_a_large_margin_is_bounded_without_sampling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounded work, but by an even stride rather than a random draw."""
        from ptpd_calibration.config import ExtractionSettings
        from ptpd_calibration.detection.extractor import DensityExtractor

        limit = 500
        monkeypatch.setenv("PTPD_EXTRACTION_PAPER_SAMPLE_PIXELS", str(limit))
        extractor = DensityExtractor(settings=ExtractionSettings())
        assert extractor.settings.paper_sample_pixels == limit

        big = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
        taken = extractor._sample_region(big)

        assert len(taken) == limit
        # An even stride, so the same rows every time.
        assert np.array_equal(taken, extractor._sample_region(big))

    def test_training_reports_the_same_error_twice(self) -> None:
        """The split was the only nondeterminism left; the estimators were pinned."""
        from ptpd_calibration.ml.predictor import CurvePredictor

        database = _training_database()
        maes = [CurvePredictor().train(database)["validation_mae"] for _ in range(3)]

        assert len(set(maes)) == 1

    def test_the_training_seed_is_a_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two different seeds must actually give two different splits."""
        from ptpd_calibration.config import MLSettings
        from ptpd_calibration.ml.predictor import CurvePredictor

        database = _training_database()

        monkeypatch.setenv("PTPD_ML_RANDOM_SEED", "1")
        first = CurvePredictor(settings=MLSettings()).train(database)["validation_mae"]
        monkeypatch.setenv("PTPD_ML_RANDOM_SEED", "999")
        second = CurvePredictor(settings=MLSettings()).train(database)["validation_mae"]

        assert first != second

    def test_training_ignores_the_global_rng(self) -> None:
        from ptpd_calibration.ml.predictor import CurvePredictor

        database = _training_database()
        without, with_work = _global_draws_around(lambda: CurvePredictor().train(database))

        assert without == with_work


def _training_database():
    """A small database with enough spread to train on."""
    from ptpd_calibration.core.models import CalibrationRecord
    from ptpd_calibration.core.types import ChemistryType
    from ptpd_calibration.ml.database import CalibrationDatabase

    rng = np.random.default_rng(3)
    database = CalibrationDatabase()
    for i in range(14):
        database.add_record(
            CalibrationRecord(
                paper_type=["arches", "platine", "revere"][i % 3],
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                exposure_time=float(60 + i),
                humidity=float(45 + i),
                temperature=float(20 + i % 5),
                measured_densities=list(
                    np.clip(np.linspace(0.05, 1.9, 21) + rng.normal(0, 0.02, 21), 0, None)
                ),
            )
        )
    return database
