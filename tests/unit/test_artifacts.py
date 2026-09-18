"""Tests for safe artifact deserialization (SEC-13)."""

from __future__ import annotations

import importlib.util
import logging
import os
import pickle
from pathlib import Path

import pytest

from ptpd_calibration.core.artifacts import (
    ArtifactPolicy,
    UnsafeArtifactError,
    compute_sha256,
    load_safetensors,
    load_torch_checkpoint,
    manifest_path_for,
    resolve_artifact_path,
    safetensors_available,
    verify_manifest,
    write_manifest,
)

pytestmark = pytest.mark.unit

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

MARKER_CALLS: list[str] = []


def marker() -> str:
    """Would be executed by an unrestricted unpickler; must never run."""
    MARKER_CALLS.append("executed")
    return "pwned"


class MaliciousPayload:
    """Pickles to a call of :func:`marker` on load."""

    def __reduce__(self) -> tuple:
        return (marker, ())


@pytest.fixture(autouse=True)
def _reset_marker() -> None:
    MARKER_CALLS.clear()


@pytest.fixture
def allowed(tmp_path: Path) -> Path:
    path = tmp_path / "allowed"
    path.mkdir()
    return path


@pytest.fixture
def policy(allowed: Path) -> ArtifactPolicy:
    return ArtifactPolicy(allowed_dirs=[allowed])


class TestArtifactPolicy:
    def test_default_allowed_dirs_come_from_settings(self) -> None:
        from ptpd_calibration.config import get_settings

        dirs = ArtifactPolicy().allowed_dirs
        assert Path(get_settings().data_dir).expanduser().resolve() in dirs
        assert all(path.is_absolute() for path in dirs)

    def test_env_json_list(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("PTPD_ARTIFACTS_ALLOWED_DIRS", f'["{tmp_path}"]')
        assert ArtifactPolicy().allowed_dirs == [tmp_path]

    def test_env_pathsep_list(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("PTPD_ARTIFACTS_ALLOWED_DIRS", os.pathsep.join([str(a), str(b)]))
        assert ArtifactPolicy().allowed_dirs == [a, b]

    def test_env_require_manifest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PTPD_ARTIFACTS_REQUIRE_MANIFEST", "true")
        assert ArtifactPolicy().require_manifest is True

    def test_defaults(self) -> None:
        policy = ArtifactPolicy(allowed_dirs=[])
        assert policy.require_manifest is False
        assert policy.manifest_suffix == ".sha256"


class TestResolveArtifactPath:
    def test_inside_allowed_dir(self, allowed: Path, policy: ArtifactPolicy) -> None:
        target = allowed / "sub" / "model.pt"
        assert resolve_artifact_path(target, policy) == target.resolve()

    def test_outside_allowed_dir_rejected(self, tmp_path: Path, policy: ArtifactPolicy) -> None:
        with pytest.raises(UnsafeArtifactError, match="outside the allowed"):
            resolve_artifact_path(tmp_path / "elsewhere.pt", policy)

    def test_dot_dot_escape_rejected(self, allowed: Path, policy: ArtifactPolicy) -> None:
        with pytest.raises(UnsafeArtifactError):
            resolve_artifact_path(allowed / ".." / "escape.pt", policy)

    def test_symlink_pointing_outside_rejected(
        self, tmp_path: Path, allowed: Path, policy: ArtifactPolicy
    ) -> None:
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"x")
        link = allowed / "link.pt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):  # pragma: no cover - platform without symlinks
            pytest.skip("symlinks not supported")
        with pytest.raises(UnsafeArtifactError):
            resolve_artifact_path(link, policy)

    def test_no_policy_only_resolves(self, tmp_path: Path) -> None:
        assert resolve_artifact_path(tmp_path / "x.pt") == (tmp_path / "x.pt").resolve()

    def test_empty_allowlist_unrestricted_unless_required(self, tmp_path: Path) -> None:
        policy = ArtifactPolicy(allowed_dirs=[])
        assert resolve_artifact_path(tmp_path / "x.pt", policy) == (tmp_path / "x.pt").resolve()
        with pytest.raises(UnsafeArtifactError, match="no artifact directories"):
            resolve_artifact_path(tmp_path / "x.pt", policy, require_allowlist=True)

    def test_error_is_value_error(self) -> None:
        assert issubclass(UnsafeArtifactError, ValueError)


class TestManifest:
    def test_round_trip(self, allowed: Path, policy: ArtifactPolicy) -> None:
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"hello world")
        manifest = write_manifest(artifact, policy)
        assert manifest == manifest_path_for(artifact, policy)
        assert manifest.name == "model.bin.sha256"
        digest, name = manifest.read_text().split()
        assert digest == compute_sha256(artifact)
        assert name == "model.bin"
        assert verify_manifest(artifact, policy) is True

    def test_mismatch_rejected(self, allowed: Path, policy: ArtifactPolicy) -> None:
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"original")
        write_manifest(artifact, policy)
        artifact.write_bytes(b"tampered")
        with pytest.raises(UnsafeArtifactError, match="does not match"):
            verify_manifest(artifact, policy)

    def test_missing_manifest(self, allowed: Path, policy: ArtifactPolicy) -> None:
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"data")
        assert verify_manifest(artifact, policy) is False
        with pytest.raises(UnsafeArtifactError, match="missing"):
            verify_manifest(artifact, policy, required=True)

    def test_policy_require_manifest_is_honoured(self, allowed: Path) -> None:
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"data")
        strict = ArtifactPolicy(allowed_dirs=[allowed], require_manifest=True)
        with pytest.raises(UnsafeArtifactError):
            verify_manifest(artifact, strict)

    def test_malformed_manifest_rejected(self, allowed: Path, policy: ArtifactPolicy) -> None:
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"data")
        manifest_path_for(artifact, policy).write_text("not-a-digest\n")
        with pytest.raises(UnsafeArtifactError, match="SHA-256"):
            verify_manifest(artifact, policy)

    def test_custom_suffix(self, allowed: Path) -> None:
        policy = ArtifactPolicy(allowed_dirs=[allowed], manifest_suffix=".sum")
        artifact = allowed / "model.bin"
        artifact.write_bytes(b"data")
        assert write_manifest(artifact, policy).name == "model.bin.sum"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.deep
class TestTorchLoader:
    def test_state_dict_round_trip(self, allowed: Path, policy: ArtifactPolicy) -> None:
        import torch

        path = allowed / "weights.pt"
        torch.save({"w": torch.arange(4.0), "epoch": 3, "cfg": {"lr": 0.1}}, path)
        loaded = load_torch_checkpoint(path, policy=policy)
        assert torch.equal(loaded["w"], torch.arange(4.0))
        assert loaded["epoch"] == 3

    def test_crafted_pickle_refused_and_never_executed(
        self, allowed: Path, policy: ArtifactPolicy
    ) -> None:
        import torch

        path = allowed / "evil.pt"
        torch.save({"payload": MaliciousPayload()}, path)
        # The policy is explicit so this exercises the unpickler guard rather
        # than the location guard, which the next test covers.
        with pytest.raises(UnsafeArtifactError, match="weights_only"):
            load_torch_checkpoint(path, policy=policy)
        assert MARKER_CALLS == []

    def test_outside_allowlist_refused_before_reading(
        self, tmp_path: Path, policy: ArtifactPolicy
    ) -> None:
        import torch

        path = tmp_path / "evil.pt"
        torch.save({"payload": MaliciousPayload()}, path)
        with pytest.raises(UnsafeArtifactError, match="outside"):
            load_torch_checkpoint(path, policy=policy)
        assert MARKER_CALLS == []

    def test_manifest_mismatch_refused(self, allowed: Path, policy: ArtifactPolicy) -> None:
        import torch

        path = allowed / "weights.pt"
        torch.save({"w": torch.zeros(1)}, path)
        write_manifest(path, policy)
        torch.save({"w": torch.ones(1)}, path)
        with pytest.raises(UnsafeArtifactError, match="does not match"):
            load_torch_checkpoint(path, policy=policy)

    def test_missing_file(self, allowed: Path, policy: ArtifactPolicy) -> None:
        with pytest.raises(FileNotFoundError):
            load_torch_checkpoint(allowed / "nope.pt", policy=policy)


class TestSafetensors:
    def test_availability_flag_matches_import(self) -> None:
        assert safetensors_available() == (importlib.util.find_spec("safetensors") is not None)

    @pytest.mark.skipif(safetensors_available(), reason="safetensors installed")
    def test_clear_import_error_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ImportError, match="safetensors"):
            load_safetensors(tmp_path / "x.safetensors")

    @pytest.mark.skipif(
        not (safetensors_available() and TORCH_AVAILABLE), reason="safetensors+torch needed"
    )
    @pytest.mark.deep
    def test_round_trip(self, allowed: Path, policy: ArtifactPolicy) -> None:
        import torch
        from safetensors.torch import save_file

        path = allowed / "w.safetensors"
        save_file({"w": torch.arange(3.0)}, str(path))
        assert torch.equal(load_safetensors(path, policy=policy)["w"], torch.arange(3.0))


class TestLoaderPolicyDefaults:
    """``policy=None`` means the configured policy, not "unrestricted".

    Ten production call sites load checkpoints without passing a policy. If
    ``None`` meant no allow-list, ``PTPD_ARTIFACTS_ALLOWED_DIRS`` would have no
    effect on any of them.
    """

    def test_env_allowlist_applies_when_no_policy_is_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside.pt"
        torch.save({"w": torch.zeros(1)}, outside)
        monkeypatch.setenv("PTPD_ARTIFACTS_ALLOWED_DIRS", str(allowed))

        with pytest.raises(UnsafeArtifactError, match="outside"):
            load_torch_checkpoint(outside)

    def test_explicitly_empty_allowlist_still_disables_the_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        torch = pytest.importorskip("torch")
        monkeypatch.setenv("PTPD_ARTIFACTS_ALLOWED_DIRS", str(tmp_path / "elsewhere"))
        target = tmp_path / "model.pt"
        torch.save({"w": torch.zeros(1)}, target)

        loaded = load_torch_checkpoint(target, policy=ArtifactPolicy(allowed_dirs=[]))

        assert "w" in loaded


class TestCurvePredictorLoader:
    """``CurvePredictor.load`` only unpickles allow-listed, manifest-verified files."""

    @staticmethod
    def _write_evil(path: Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"model_type": "random_forest", "payload": MaliciousPayload()}, handle)

    def test_outside_allowlist_refused(self, tmp_path: Path, policy: ArtifactPolicy) -> None:
        from ptpd_calibration.ml.predictor import CurvePredictor

        evil = tmp_path / "evil.pkl"
        self._write_evil(evil)
        with pytest.raises(UnsafeArtifactError, match="outside"):
            CurvePredictor.load(evil, policy=policy)
        assert MARKER_CALLS == []

    def test_missing_manifest_refused(self, allowed: Path, policy: ArtifactPolicy) -> None:
        from ptpd_calibration.ml.predictor import CurvePredictor

        evil = allowed / "evil.pkl"
        self._write_evil(evil)
        with pytest.raises(UnsafeArtifactError, match="manifest"):
            CurvePredictor.load(evil, policy=policy)
        assert MARKER_CALLS == []

    def test_manifest_mismatch_refused(self, allowed: Path, policy: ArtifactPolicy) -> None:
        from ptpd_calibration.ml.predictor import CurvePredictor

        evil = allowed / "evil.pkl"
        evil.write_bytes(b"placeholder")
        write_manifest(evil, policy)
        self._write_evil(evil)  # swap the payload after the manifest was written
        with pytest.raises(UnsafeArtifactError, match="does not match"):
            CurvePredictor.load(evil, policy=policy)
        assert MARKER_CALLS == []

    def test_default_policy_refuses_tmp_paths(self, tmp_path: Path) -> None:
        from ptpd_calibration.ml.predictor import CurvePredictor

        evil = tmp_path / "evil.pkl"
        self._write_evil(evil)
        write_manifest(evil)
        with pytest.raises(UnsafeArtifactError):
            CurvePredictor.load(evil)
        assert MARKER_CALLS == []

    def test_env_allowlist_admits_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A directory added via the environment is accepted by the default policy."""
        monkeypatch.setenv("PTPD_ARTIFACTS_ALLOWED_DIRS", str(tmp_path))
        assert resolve_artifact_path(tmp_path / "m.pkl", ArtifactPolicy(), require_allowlist=True)

    def test_save_outside_allowlist_warns_that_load_will_refuse_it(
        self, tmp_path: Path, policy: ArtifactPolicy, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Saving is allowed anywhere, but the caller is told loading will refuse it."""
        pytest.importorskip("sklearn")
        from ptpd_calibration.ml.predictor import CurvePredictor

        predictor = CurvePredictor(model_type="random_forest")
        predictor.model = object()  # save() only needs is_trained and a picklable model
        predictor.is_trained = True
        outside = tmp_path / "outside" / "predictor.pkl"

        with caplog.at_level(logging.WARNING, logger="ptpd_calibration.ml.predictor"):
            predictor.save(outside, policy=policy)

        assert outside.is_file()
        assert manifest_path_for(outside, policy).is_file()
        assert "PTPD_ARTIFACTS_ALLOWED_DIRS" in caplog.text

        with pytest.raises(UnsafeArtifactError):
            CurvePredictor.load(outside, policy=policy)

    def test_save_and_load_round_trip(self, allowed: Path, policy: ArtifactPolicy) -> None:
        pytest.importorskip("sklearn")
        from ptpd_calibration.config import MLSettings
        from ptpd_calibration.core.models import CalibrationRecord
        from ptpd_calibration.core.types import ChemistryType, ContrastAgent, DeveloperType
        from ptpd_calibration.ml.database import CalibrationDatabase
        from ptpd_calibration.ml.predictor import CurvePredictor

        database = CalibrationDatabase()
        for i in range(8):
            database.add_record(
                CalibrationRecord(
                    paper_type="Arches Platine" if i % 2 else "Bergger COT320",
                    exposure_time=120.0 + 10 * i,
                    metal_ratio=0.2 + 0.05 * i,
                    chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                    contrast_agent=ContrastAgent.NA2,
                    contrast_amount=float(i % 3),
                    developer=DeveloperType.POTASSIUM_OXALATE,
                    measured_densities=[0.05 + k * (0.08 + 0.002 * i) for k in range(21)],
                )
            )
        settings = MLSettings(n_estimators=10, max_depth=2, min_training_samples=3)
        predictor = CurvePredictor(model_type="random_forest", settings=settings)
        predictor.train(database, validation_split=0.0)
        record = database.get_all_records()[0]
        expected = predictor.predict(record)

        target = allowed / "predictor.pkl"
        predictor.save(target, policy=policy)
        assert manifest_path_for(target, policy).is_file()

        loaded = CurvePredictor.load(target, policy=policy)
        assert loaded.is_trained
        assert loaded.feature_names == predictor.feature_names
        assert loaded.predict(record) == pytest.approx(expected)
