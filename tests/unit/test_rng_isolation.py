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
