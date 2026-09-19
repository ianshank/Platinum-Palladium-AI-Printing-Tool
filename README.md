---
title: Pt/Pd Calibration Studio
emoji: 📷
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - photography
  - calibration
  - platinum
  - palladium
  - alternative-process
  - digital-negative
  - curve-editor
short_description: AI-powered calibration for platinum/palladium printing
---

# Platinum/Palladium Calibration Studio

[![CI](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/actions/workflows/ci.yml)

A calibration toolkit for platinum/palladium alternative photographic printing:
step-tablet densitometry, linearization curves for digital negatives, curve
editing and export (QuadTone RIP, Piezography, CSV, JSON), chemistry and
exposure calculators, and an LLM-backed printing assistant.

Test counts and coverage are published by CI on every run (job summary and the
`backend-reports` / `frontend-coverage` artifacts), never typed into this file.

## Status

The project is executing a validation-first plan, see
[docs/plans/2026-09-validation-sdlc-plan.md](docs/plans/2026-09-validation-sdlc-plan.md)
and the decision records in [docs/adr/](docs/adr/). In particular:

- The parameter-search module under `src/ptpd_calibration/mcts/` is
  experimental. Its value is being established by a pre-registered ablation
  against standard optimizers (ADR-0005); nothing in the product depends on it
  until that result is in.
- Recommendations derived from the process simulator are labelled
  `provenance: simulated` and are never mixed with measured prints (ADR-0006).
- The Gradio UI (`app.py`, `src/ptpd_calibration/ui/`) is being retired in
  favour of the FastAPI + React application once the remaining feature gaps
  are closed (ADR-0004). The Hugging Face Space still runs the Gradio build.

## Architecture

| Layer | Stack | Purpose |
| --- | --- | --- |
| Frontend | React 18, TypeScript, Vite, Zustand, TanStack Query | Curve editor, calibration wizard, chemistry calculator, assistant |
| Backend API | FastAPI, Pydantic | REST endpoints for scans, curves, calibrations, chat, export |
| Scientific core | NumPy, SciPy, scikit-learn | Detection, curves, chemistry, exposure, optional simulation |

See [docs/architecture.md](docs/architecture.md).

## Quick start

Backend (Python 3.12 pinned in `.python-version`; [uv](https://docs.astral.sh/uv/) manages the environment):

```bash
uv sync --extra server          # api + ml + llm; add --extra dl for torch
uv run ptpd-server              # FastAPI on http://localhost:8000
```

Frontend:

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev                        # http://localhost:3000, proxies /api to :8000
```

LLM features read `PTPD_LLM_PROVIDER` and `PTPD_LLM_ANTHROPIC_API_KEY` /
`PTPD_LLM_OPENAI_API_KEY` from the environment (or a local `.env`). Never commit
keys; see [SECURITY.md](SECURITY.md).

## Tests

```bash
uv run pytest                                   # backend, strict markers, 120 s timeout per test
uv run pytest tests/property                    # Hypothesis property and metamorphic suites
cd frontend && pnpm test:run                    # vitest
cd frontend && pnpm exec playwright test        # e2e (needs the backend running)
```

The CI workflow (`.github/workflows/ci.yml`) is the single source of truth for
what must pass: ruff, mypy on the allowlisted packages, pytest with per-package
coverage floors and diff coverage, vitest with thresholds, build, gitleaks, and
dependency audits. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository layout

```
frontend/                  React application
src/ptpd_calibration/      Python package
  api/                     FastAPI routes and request guards
  core/                    models, types, settings
  curves/ chemistry/ exposure/ detection/ imaging/ papers/ session/
  mcts/                    experimental parameter search (ADR-0005)
  llm/ agents/             assistant and (unexposed) agent framework
tests/                     pytest suites: unit, api, integration, property, e2e
docs/                      architecture, plans, ADRs, agent protocol
kb/                        agent session knowledge base (docs/agents/kb-protocol.md)
```

## Deployment

The Hugging Face Space deploys only from `v*` tags after the CI gate, behind a
GitHub Environment approval (`deploy-hf` job). The Docker Space cutover is
tracked by ADR-0004.

## Requirements

- Python 3.10 to 3.13 (CI uses 3.12)
- Node 20 and pnpm (see `frontend/package.json` `packageManager`)
- A step-tablet scan (Stouffer 21/31/41 step or similar)

## License

MIT, see [LICENSE](LICENSE). Created by Ian Cruickshank.
