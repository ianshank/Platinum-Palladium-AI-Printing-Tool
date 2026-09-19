# ADR-0003: uv with a committed lockfile

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

Three manifests (`pyproject.toml`, `requirements.txt`, `requirements-dl.txt`) disagreed; no lockfile existed; matplotlib, tifffile, python-dotenv, psutil and pyyaml were imported unconditionally but never declared, so the package could not be imported from its own metadata.

## Decision

`pyproject.toml` is the only manifest: every unconditional import is a declared dependency; optional stacks are extras (`ml`, `llm`, `api`, `gcp`, `ui`, `hardware`, `qr`, `dl`, `server`, `all`); tooling lives in PEP 735 dependency groups. `uv.lock` is committed and checked in CI (`uv lock --check`); `pylock.toml` is exported for interoperability. `.python-version` pins 3.12 for development and CI. `requirements-dl.txt` is deleted; `requirements.txt` is generated only while the Gradio Space still installs it. The `requires-python` floor stays at 3.10 until the owner decides on 3.11 (plan §9 Q7).

## Consequences

`pip install -e .[dev]` keeps working for one release. Torch resolves from PyPI in the lock; CI passes the CPU index at sync time.
