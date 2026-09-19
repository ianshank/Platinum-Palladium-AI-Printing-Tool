"""Two defects the ADR-0006 provenance work and the metrics plumbing left behind.

A checkpoint is written with ``torch.save`` and always read back with
``weights_only=True``, which accepts only a small set of globals. A numpy scalar
is not one of them, and one reaches a checkpoint very easily: ``np.mean``
returns ``np.float64``, which satisfies ``isinstance(x, float)`` and a
``dict[str, float]`` annotation, so neither review nor mypy sees it.

Separately, marking generated records ``provenance="simulated"`` is what stops a
model training on simulated data believing it measured. It also meant the
dataset excluded them with no way to say otherwise, so deliberately training on
generated data produced an empty dataset.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pytest

from ptpd_calibration.core.artifacts import to_plain_python


class TestCheckpointValuesAreLoadable:
    """Nothing numpy may reach a file read back with ``weights_only=True``."""

    @dataclass
    class _Metrics:
        epoch: int = 0
        additional_metrics: dict[str, float] = field(default_factory=dict)

    def _metrics_with_reductions(self) -> _Metrics:
        samples = {"mae": [1.0, 2.0], "rmse": [3.0, 5.0]}
        return self._Metrics(
            epoch=1,
            additional_metrics={k: np.mean(v) for k, v in samples.items()},
        )

    def test_asdict_alone_leaves_numpy_behind(self) -> None:
        """Pin the trap: the dataclass is unwrapped, its values are not."""
        raw = asdict(self._metrics_with_reductions())

        assert isinstance(raw["additional_metrics"]["mae"], np.floating)
        # And it passes the annotation that was supposed to prevent this.
        assert isinstance(raw["additional_metrics"]["mae"], float)
        assert b"numpy" in pickle.dumps(raw)

    def test_normalising_removes_every_numpy_global(self) -> None:
        converted = to_plain_python(asdict(self._metrics_with_reductions()))

        assert type(converted["additional_metrics"]["mae"]) is float
        assert b"numpy" not in pickle.dumps(converted)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.float64(1.5), 1.5),
            (np.int64(7), 7),
            (np.arange(3), [0, 1, 2]),
            ({"a": {"b": np.float32(0.5)}}, {"a": {"b": 0.5}}),
            ([np.int32(1), 2], [1, 2]),
            ("text", "text"),
            (None, None),
        ],
    )
    def test_conversion_is_recursive_and_value_preserving(self, value, expected) -> None:
        converted = to_plain_python(value)

        assert converted == expected
        assert b"numpy" not in pickle.dumps(converted)

    def test_a_tuple_stays_a_tuple(self) -> None:
        """Checkpoint shapes are often tuples; converting must not retype them."""
        converted = to_plain_python((np.int64(1), np.int64(2)))

        assert converted == (1, 2)
        assert isinstance(converted, tuple)

    def test_the_training_reductions_produce_builtin_floats(self) -> None:
        """Fixed at the source too, so the annotation stops being a lie.

        The file is read rather than imported: the module decorates with
        ``@torch.no_grad()`` at class-definition time, so importing it -- or
        anything that pulls its package in -- needs a torch install this check
        does not otherwise require.
        """
        import ptpd_calibration

        module = (
            Path(ptpd_calibration.__file__).parent / "deep_learning" / "training" / "pipelines.py"
        )
        source = module.read_text()

        assert "float(np.mean(v))" in source
        assert "{k: np.mean(v) for" not in source


class TestGeneratedDataCanStillBeTrainedOn:
    """Excluded by default, reachable on request."""

    @pytest.fixture
    def synthetic_db(self):
        from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

        return generate_training_data(num_records=12, seed=42)

    def test_the_default_still_hides_simulated_records(self, synthetic_db) -> None:
        """ADR-0006: this is the property the provenance field exists to give."""
        assert len(synthetic_db) == 12
        assert synthetic_db.get_all_records() == []

    def test_an_explicit_caller_sees_them(self, synthetic_db) -> None:
        assert len(synthetic_db.get_all_records(include_simulated=True)) == 12

    def test_the_encoder_can_be_built_from_generated_data(self, synthetic_db) -> None:
        """The dataset builds its encoder from the same database."""
        from ptpd_calibration.ml.deep.dataset import FeatureEncoder

        encoder = FeatureEncoder.from_database(synthetic_db, include_simulated=True)

        assert encoder.num_features > 0

    def test_the_encoder_default_still_refuses(self, synthetic_db) -> None:
        from ptpd_calibration.ml.deep.dataset import DatasetError, FeatureEncoder

        with pytest.raises(DatasetError, match="empty database"):
            FeatureEncoder.from_database(synthetic_db)

    def test_the_opt_in_reaches_every_layer(self) -> None:
        """A parameter that stops halfway is the same bug one level down."""
        import inspect

        from ptpd_calibration.ml.deep.dataset import CalibrationDataset, create_dataloaders
        from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor

        for func in (
            CalibrationDataset.__init__,
            create_dataloaders,
            DeepCurvePredictor.train,
            DeepCurvePredictor._train_ensemble,
        ):
            assert "include_simulated" in inspect.signature(func).parameters, func.__qualname__

    def test_the_route_opts_in_only_when_it_generated_the_data(self) -> None:
        """The measured-only branch must keep the default."""
        import inspect

        from ptpd_calibration.api import deep_learning

        source = inspect.getsource(deep_learning)

        assert "include_simulated=uses_simulated" in source
        assert "uses_simulated = request.use_synthetic_data" in source
