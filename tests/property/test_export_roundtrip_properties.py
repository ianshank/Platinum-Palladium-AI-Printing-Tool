"""Property-based round-trip tests for ``curves/export.py`` and ``curves/parser.py`` (TST-07).

The QTR round trip is only exact to within 2/255: ``QTRExporter`` writes
16-bit values and ``QuadFileParser`` quantises them to 8-bit (one step of
16->8 bit truncation plus ``int()`` truncation on export).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ptpd_calibration.core.models import CurveData
from ptpd_calibration.curves.export import QTRExporter, load_curve, save_curve
from ptpd_calibration.curves.parser import QuadFileParser
from tests.property.strategies import curves

pytestmark = pytest.mark.property

CSV_TOL = 1e-6
"""CSVExporter writes six decimals."""
QUAD_TOL = 2 / 255
"""8-bit quantisation in the parser plus int() truncation in the exporter."""
QTR_POINTS = 256
QTR_MAX_VALUE = 65535
QTR_CHANNEL_MAX_8BIT = 255
INT_TRUNCATION_SLACK = 1
INK_LIMIT_MIN, INK_LIMIT_MAX = 1.0, 100.0
PERCENT = 100.0
QTR_SUFFIXES = [".quad", ".txt"]
PRIMARY_CHANNEL = "K"


def _arr(curve: CurveData) -> np.ndarray:
    return np.asarray(curve.output_values, dtype=float)


def _roundtrip(curve: CurveData, suffix: str) -> CurveData:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"curve{suffix}"
        save_curve(curve, path)
        return load_curve(path)


def _expected_qtr(curve: CurveData) -> np.ndarray:
    return np.interp(np.linspace(0.0, 1.0, QTR_POINTS), curve.input_values, curve.output_values)


def _channel_values(path: Path, channel: str) -> list[int]:
    values: list[int] = []
    active = False
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            active = line.strip() == f"# {channel} Curve"
            continue
        if active and line.strip():
            values.append(int(line))
    return values


@given(curve=curves())
def test_json_roundtrip_is_exact(curve: CurveData) -> None:
    loaded = _roundtrip(curve, ".json")

    assert loaded.name == curve.name
    assert loaded.input_values == curve.input_values
    assert loaded.output_values == curve.output_values


@given(curve=curves())
def test_csv_roundtrip_within_tolerance(curve: CurveData) -> None:
    loaded = _roundtrip(curve, ".csv")

    assert len(loaded.output_values) == len(curve.output_values)
    np.testing.assert_allclose(
        np.asarray(loaded.input_values), np.asarray(curve.input_values), atol=CSV_TOL, rtol=0.0
    )
    np.testing.assert_allclose(_arr(loaded), _arr(curve), atol=CSV_TOL, rtol=0.0)


@pytest.mark.parametrize("suffix", QTR_SUFFIXES)
@given(curve=curves())
def test_qtr_roundtrip_within_quantisation(suffix: str, curve: CurveData) -> None:
    """F3: save_curve -> load_curve works for QTR files and matches within 2/255."""
    loaded = _roundtrip(curve, suffix)

    assert len(loaded.output_values) == QTR_POINTS
    assert np.max(np.abs(_arr(loaded) - _expected_qtr(curve))) <= QUAD_TOL


@pytest.mark.parametrize("suffix", QTR_SUFFIXES)
@given(curve=curves(monotone=True))
def test_qtr_roundtrip_preserves_monotonicity_and_endpoints(suffix: str, curve: CurveData) -> None:
    loaded = _roundtrip(curve, suffix)
    out = _arr(loaded)

    assert np.all(np.diff(out) >= 0.0)
    assert abs(out[0] - curve.output_values[0]) <= QUAD_TOL
    assert abs(out[-1] - curve.output_values[-1]) <= QUAD_TOL


@given(curve=curves(), ink_limit=st.floats(INK_LIMIT_MIN, INK_LIMIT_MAX))
def test_ink_limit_scales_primary_channel(curve: CurveData, ink_limit: float) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "limited.quad"
        QTRExporter(primary_channel=PRIMARY_CHANNEL, ink_limit=ink_limit).export(
            curve, path, format="quad"
        )
        values = _channel_values(path, PRIMARY_CHANNEL)

    ceiling = QTR_MAX_VALUE * ink_limit / PERCENT
    assert len(values) == QTR_POINTS
    assert max(values) <= ceiling + INT_TRUNCATION_SLACK
    assert max(values) >= float(np.max(_expected_qtr(curve))) * ceiling - INT_TRUNCATION_SLACK


@given(text=st.text())
def test_parse_string_never_raises(text: str) -> None:
    profile = QuadFileParser().parse_string(text)

    assert profile is not None


@given(
    values=st.dictionaries(
        st.integers(0, QTR_CHANNEL_MAX_8BIT),
        st.integers(0, QTR_CHANNEL_MAX_8BIT),
        max_size=QTR_POINTS,
    )
)
def test_bracket_section_values_are_reproduced(values: dict[int, int]) -> None:
    content = "[General]\nProfileName=prop\n[K]\n" + "\n".join(
        f"{index}={value}" for index, value in values.items()
    )

    channel = QuadFileParser().parse_string(content).get_channel(PRIMARY_CHANNEL)

    assert channel is not None
    for index in range(QTR_POINTS):
        assert channel.values[index] == values.get(index, 0)
    assert channel.enabled == any(v > 0 for v in values.values())
