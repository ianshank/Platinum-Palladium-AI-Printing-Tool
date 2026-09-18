# Contributing

## Setup

```bash
uv sync --group dev --extra api --extra ml --extra llm   # backend + tooling (Python from .python-version)
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
cd frontend && pnpm install --frozen-lockfile
```

Optional extras: `--extra dl` (torch stack; on Linux add
`UV_INDEX="pytorch-cpu=https://download.pytorch.org/whl/cpu"` for CPU wheels),
`--extra ui` (legacy Gradio), `--extra gcp`, `--extra hardware`, `--extra qr`.

## Verification loop

Run after every change, then the full set before committing:

```bash
uv run ruff check src tests app.py scripts && uv run ruff format --check src tests app.py scripts
uv run mypy
uv run pytest tests/unit/<area> -q          # targeted
uv run pytest                               # full backend (unit, api, integration, property)
cd frontend && pnpm check:all && pnpm test:run && pnpm build
```

`uv run pre-commit run --all-files` runs the same checks CI runs, plus gitleaks,
actionlint, zizmor, and the documentation placement rules.

## What CI requires

`.github/workflows/ci.yml` has one required status check, `all-green`, which
today requires the `changes`, `backend`, `frontend`, and `security` jobs.
`e2e`, `dependency-review`, and `codeql` are advisory until the phase-2 ratchet
(see `docs/plans/2026-09-review/expert-devops.md` §2).

Backend gates: ruff check and format, mypy on the allowlisted packages
(`[tool.mypy] files` in `pyproject.toml`; expansion order: packages with fewer
than ten errors first, then `curves`, `detection`, `llm`), pytest with
`--strict-markers`, per-package coverage floors (mcts 70, chemistry 85,
curves 80, whole package 60; rising to 80 / 90 / 85 / 70), and `diff-cover`
at 90% on changed lines. Frontend gates: `tsc`, eslint, prettier, vitest with
the thresholds in `vitest.config.ts`, and a production build.

## Branches, commits, PRs

- Trunk-based on `main`; branches live less than a week.
- One concern per PR, under about 400 changed lines excluding generated and
  lock files; larger changes come with a design note in `docs/adr/` or
  `docs/plans/`.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `ci:`).
- Never force-push a shared branch; never merge red.

## Adding an endpoint (API-first, ADR-0013)

1. Put the domain logic in the Python package (`curves/`, `chemistry/`, ...),
   with unit and property tests.
2. Add a pydantic request/response model with explicit bounds
   (`max_length`, `ge`/`le`), reading limits from `APISettings`.
3. Add the route in `src/ptpd_calibration/api/`, guarded by the helpers in
   `api/security.py` for anything that touches files, and a test in `tests/api/`.
4. Regenerate the TypeScript types from the OpenAPI schema and consume them in
   `frontend/src/api/`; never re-implement the formula in TypeScript.

## Dependencies

`pyproject.toml` is the only manifest. Add runtime dependencies to
`[project.dependencies]` or an extra, tooling to `[dependency-groups]`, then
run `uv lock` and commit `uv.lock`. `requirements.txt` is generated
(`uv export`) for the legacy Gradio Space and will be deleted with it.

## Experimental code

Packages with no wired importer are moved to `experimental/` or `contrib/`
(ADR-0011); they are excluded from coverage floors and required CI, and are
deleted after one release unless an ADR adopts them.

## Releases

Tag `vX.Y.Z` on `main`. CI builds the sdist/wheel, exports `pylock.toml`,
generates an SPDX SBOM, attests provenance, and (once the Dockerfile lands)
publishes the container image and deploys the Space behind the `huggingface`
environment approval. Rollback: `workflow_dispatch` with `deploy_ref` set to
the previous tag.

## Documentation

See `AGENTS.md` for placement rules. Decisions go in `docs/adr/`, plans in
`docs/plans/`, the roadmap in `docs/roadmap.md`.
