"""
Safe deserialization of model artifacts (plan item SEC-13).

Model checkpoints and fitted estimators are untrusted input: a ``torch.load``
or ``pickle.load`` of an attacker-controlled file executes arbitrary code.
This module centralises the defences so every loader in the code base behaves
the same way:

* :func:`resolve_artifact_path` resolves symlinks and, when an
  :class:`ArtifactPolicy` with ``allowed_dirs`` is in force, refuses paths
  outside those directories.
* :func:`load_torch_checkpoint` always calls ``torch.load(weights_only=True)``.
  A checkpoint that needs the full unpickler (it holds Python objects rather
  than tensors and primitives) raises :class:`UnsafeArtifactError` with
  instructions to re-save it; there is deliberately no fallback to
  ``weights_only=False``.
* :func:`load_safetensors` loads a ``.safetensors`` file when the optional
  ``safetensors`` package is installed.
* :func:`write_manifest` / :func:`verify_manifest` maintain a ``<file>.sha256``
  side-car (``sha256sum -c`` compatible) so an artifact that changed on disk
  since it was written is refused.

``torch`` and ``safetensors`` are imported lazily so the module is importable
in the API-only install.

Environment overrides use the ``PTPD_ARTIFACTS_`` prefix, e.g.
``PTPD_ARTIFACTS_ALLOWED_DIRS=/srv/models:/home/me/.ptpd``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST_SUFFIX = ".sha256"


class UnsafeArtifactError(ValueError):
    """An artifact was refused: bad location, failed integrity check or unsafe payload."""


def default_allowed_dirs() -> list[Path]:
    """Directories the application itself writes artifacts to.

    Derived from the global settings (``data_dir``, ``ml.model_cache_dir`` and
    ``deep_learning.checkpoint_dir``) when they can be loaded; an empty list
    otherwise. ``config`` is imported lazily to keep ``core`` free of an import
    cycle and importable without the full settings stack.
    """
    dirs: list[Path] = []
    try:
        from ptpd_calibration.config import get_settings

        settings = get_settings()
        candidates: list[Path | None] = [
            settings.data_dir,
            settings.ml.model_cache_dir,
            settings.deep_learning.checkpoint_dir,
        ]
    except Exception as exc:  # pragma: no cover - only when settings are broken
        logger.debug("Could not derive artifact directories from settings: %s", exc)
        candidates = []

    for candidate in candidates:
        if candidate is None:
            continue
        resolved = Path(candidate).expanduser().resolve()
        if resolved not in dirs:
            dirs.append(resolved)
    logger.debug("Default artifact allow-list: %s", [str(d) for d in dirs])
    return dirs


def _parse_dir_list(value: Any) -> Any:
    """Accept a JSON list or an ``os.pathsep``-separated string for ``allowed_dirs``."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            return [Path(item) for item in json.loads(text)]
        return [Path(item) for item in text.split(os.pathsep) if item]
    if isinstance(value, Path):
        return [value]
    return value


class ArtifactPolicy(BaseSettings):
    """Where model artifacts may be loaded from and how they are verified.

    ``allowed_dirs`` defaults to the directories the application writes to
    (see :func:`default_allowed_dirs`). An explicit empty list disables the
    location check; loaders that must never run unrestricted pass
    ``require_allowlist=True`` to :func:`resolve_artifact_path`.
    """

    model_config = SettingsConfigDict(env_prefix="PTPD_ARTIFACTS_")

    allowed_dirs: Annotated[list[Path], NoDecode] = Field(
        default_factory=default_allowed_dirs,
        description="Directories artifacts may be loaded from (JSON list or os.pathsep string)",
    )
    require_manifest: bool = Field(
        default=False,
        description="Refuse artifacts that have no <file>.sha256 manifest",
    )
    manifest_suffix: str = Field(
        default=_DEFAULT_MANIFEST_SUFFIX,
        min_length=1,
        description="Suffix appended to the artifact filename for its manifest",
    )
    hash_chunk_size: int = Field(
        default=1024 * 1024,
        ge=4096,
        description="Read size used while hashing artifacts",
    )

    @field_validator("allowed_dirs", mode="before")
    @classmethod
    def _coerce_allowed_dirs(cls, value: Any) -> Any:
        return _parse_dir_list(value)


def resolve_artifact_path(
    path: Path | str,
    policy: ArtifactPolicy | None = None,
    *,
    require_allowlist: bool = False,
) -> Path:
    """Resolve ``path`` (following symlinks) and enforce the policy's allow-list.

    Args:
        path: Candidate artifact location.
        policy: Policy to enforce; ``None`` only resolves the path.
        require_allowlist: Refuse to proceed when no allowed directories are
            configured instead of treating that as "unrestricted".

    Raises:
        UnsafeArtifactError: The resolved path is outside every allowed
            directory, or no allow-list exists and one is required.
    """
    resolved = Path(path).expanduser().resolve()
    allowed = [] if policy is None else [d.expanduser().resolve() for d in policy.allowed_dirs]

    if not allowed:
        if require_allowlist:
            raise UnsafeArtifactError(
                f"Refusing to load {resolved}: no artifact directories are configured. "
                "Set PTPD_ARTIFACTS_ALLOWED_DIRS or pass an ArtifactPolicy with allowed_dirs."
            )
        logger.debug("No artifact allow-list configured; resolved %s without restriction", resolved)
        return resolved

    for base in allowed:
        if resolved == base or resolved.is_relative_to(base):
            logger.debug("Artifact %s accepted under %s", resolved, base)
            return resolved

    logger.debug("Artifact %s rejected; allowed dirs: %s", resolved, [str(d) for d in allowed])
    raise UnsafeArtifactError(
        f"Refusing to load {resolved}: it is outside the allowed artifact directories "
        f"({', '.join(str(d) for d in allowed)}). "
        "Move the file or extend PTPD_ARTIFACTS_ALLOWED_DIRS."
    )


def compute_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest of a file, streamed in ``chunk_size`` blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(path: Path | str, policy: ArtifactPolicy | None = None) -> Path:
    """Location of the manifest side-car for ``path``."""
    suffix = policy.manifest_suffix if policy is not None else _DEFAULT_MANIFEST_SUFFIX
    path = Path(path)
    return path.with_name(path.name + suffix)


def write_manifest(path: Path | str, policy: ArtifactPolicy | None = None) -> Path:
    """Write ``<file>.sha256`` next to ``path`` (``sha256sum -c`` format) and return it."""
    path = Path(path)
    chunk = policy.hash_chunk_size if policy is not None else 1024 * 1024
    digest = compute_sha256(path, chunk)
    manifest = manifest_path_for(path, policy)
    manifest.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    logger.debug("Wrote manifest %s (%s)", manifest, digest[:12])
    return manifest


def verify_manifest(
    path: Path | str,
    policy: ArtifactPolicy | None = None,
    *,
    required: bool | None = None,
) -> bool:
    """Check ``path`` against its ``<file>.sha256`` manifest.

    Returns ``True`` when the manifest exists and matches, ``False`` when it is
    absent and not required.

    Raises:
        UnsafeArtifactError: The manifest is missing while required, is
            unreadable, or its digest does not match the file.
    """
    path = Path(path)
    if required is None:
        required = policy.require_manifest if policy is not None else False
    manifest = manifest_path_for(path, policy)

    if not manifest.is_file():
        if required:
            raise UnsafeArtifactError(
                f"Refusing to load {path}: manifest {manifest.name} is missing. "
                "Re-save the artifact with the current version to generate one."
            )
        logger.debug("No manifest for %s; integrity not verified", path)
        return False

    tokens = manifest.read_text(encoding="utf-8").split()
    expected = tokens[0].lower() if tokens else ""
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise UnsafeArtifactError(f"Manifest {manifest} does not contain a SHA-256 digest")

    chunk = policy.hash_chunk_size if policy is not None else 1024 * 1024
    actual = compute_sha256(path, chunk)
    if actual != expected:
        logger.debug("Manifest mismatch for %s: expected %s got %s", path, expected, actual)
        raise UnsafeArtifactError(
            f"Refusing to load {path}: SHA-256 does not match its manifest "
            "(the file changed after it was saved)"
        )
    logger.debug("Manifest verified for %s", path)
    return True


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required to load this checkpoint. Install with: pip install torch"
        ) from exc
    return torch


def _is_weights_only_failure(exc: BaseException) -> bool:
    """Whether ``exc`` came from the restricted unpickler rather than a corrupt file."""
    if isinstance(exc, pickle.UnpicklingError):
        return True
    message = str(exc)
    return (
        "weights_only" in message
        or "Unsupported global" in message
        or "WeightsUnpickler" in message
    )


def load_torch_checkpoint(
    path: Path | str,
    map_location: Any = None,
    *,
    policy: ArtifactPolicy | None = None,
) -> Any:
    """Load a checkpoint with ``torch.load(..., weights_only=True)``.

    Args:
        path: Checkpoint file (``.pt`` / ``.pth``).
        map_location: Passed straight to ``torch.load``.
        policy: Optional location/manifest policy. With ``None`` the path is
            only resolved; an existing manifest is still verified.

    Raises:
        UnsafeArtifactError: The path is outside the allow-list, the manifest
            does not match, or the checkpoint contains objects the restricted
            unpickler refuses (re-save it as a ``state_dict`` or safetensors).
        FileNotFoundError: The checkpoint does not exist.
    """
    torch = _import_torch()
    resolved = resolve_artifact_path(path, policy)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    verify_manifest(resolved, policy)

    logger.debug("Loading torch checkpoint %s (weights_only=True)", resolved)
    try:
        return torch.load(resolved, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not _is_weights_only_failure(exc):
            raise
        raise UnsafeArtifactError(
            f"Refusing to load {resolved}: it contains Python objects that the restricted "
            "(weights_only) unpickler does not allow. Re-save the checkpoint as a plain "
            "state_dict with JSON-serialisable metadata, or as a .safetensors file; "
            "this application never falls back to torch.load(weights_only=False)."
        ) from exc


def safetensors_available() -> bool:
    """Whether the optional ``safetensors`` package can be imported."""
    try:
        import safetensors  # noqa: F401
    except ImportError:
        return False
    return True


def load_safetensors(
    path: Path | str,
    *,
    policy: ArtifactPolicy | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a ``.safetensors`` file into a ``{name: tensor}`` dict.

    Raises:
        ImportError: The ``safetensors`` package is not installed.
        UnsafeArtifactError: The path fails the policy or manifest checks.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "The safetensors package is required for .safetensors checkpoints. "
            "Install with: pip install safetensors"
        ) from exc

    resolved = resolve_artifact_path(path, policy)
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    verify_manifest(resolved, policy)
    logger.debug("Loading safetensors file %s", resolved)
    tensors: dict[str, Any] = load_file(str(resolved), device=device)
    return tensors


__all__ = [
    "ArtifactPolicy",
    "UnsafeArtifactError",
    "compute_sha256",
    "default_allowed_dirs",
    "load_safetensors",
    "load_torch_checkpoint",
    "manifest_path_for",
    "resolve_artifact_path",
    "safetensors_available",
    "verify_manifest",
    "write_manifest",
]
