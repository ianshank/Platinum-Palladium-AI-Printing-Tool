# CLAUDE.md

Instructions for agent sessions in this repository. Keep this file under 6 KB;
architecture prose lives in `docs/`, decisions in `docs/adr/`, plans in
`docs/plans/`. Numbers (test counts, coverage) are never written here.

@AGENTS.md

## What this is

A platinum/palladium print-calibration toolkit: FastAPI backend
(`src/ptpd_calibration/`), React 18 + TypeScript frontend (`frontend/`), and a
scientific core (curves, chemistry, exposure, detection, imaging). The
parameter-search module `mcts/` is experimental and gated by ADR-0005. The
Gradio UI (`app.py`, `src/ptpd_calibration/ui/`) is frozen pending retirement
(ADR-0004).

## Commands

```bash
uv sync --group dev --extra api --extra ml --extra llm   # backend dev environment
uv run pytest                                            # strict markers, 120 s timeout
uv run pytest tests/property                             # Hypothesis suites
uv run ruff check src tests app.py scripts && uv run ruff format --check src tests app.py scripts
uv run mypy                                              # allowlisted packages only
cd frontend && pnpm install --frozen-lockfile && pnpm check:all && pnpm test:run && pnpm build
uv run pre-commit run --all-files                        # same checks as CI, locally
```

Run the relevant subset after every change and the full set before a commit.
CI (`.github/workflows/ci.yml`) is the single definition of green; the one
required check is `all-green`.

## Hard constraints

- Never modify `src/ptpd_calibration/ui/` (frozen legacy UI; excluded from lint).
- Never add markdown at the repository root beyond the allowlist in `AGENTS.md`;
  never create `AGENT.md`, `*_SUMMARY.md`, `*_NEXT_STEPS*.md`, or `plan.md`.
- Never type a test count or coverage figure into documentation; link the CI run.
- Never commit secrets. LLM keys are read from `PTPD_LLM_ANTHROPIC_API_KEY`,
  `PTPD_LLM_OPENAI_API_KEY`, `PTPD_LLM_API_KEY` (prefix `PTPD_LLM_`).
- Domain math (chemistry, exposure, linearization, imaging) lives in Python only;
  the frontend consumes generated OpenAPI types and never re-implements formulas
  (ADR-0013).
- Simulated data never enters a field named `measured_*`; records carry
  `provenance` (ADR-0006).
- New limits and tunables are fields on the pydantic settings classes in
  `src/ptpd_calibration/config.py` (env prefix `PTPD_`), never literals.
- Use `logging.getLogger(__name__)`; add debug logging on guarded paths.
- Tests accompany every change; keep existing tests passing; never skip, disable,
  or quarantine a test to get green.
- `kb/` changes ride in their own commit (see `docs/agents/kb-protocol.md`).

## Conventions

- Python: explicit return types, pydantic models at boundaries, `ruff` clean,
  markers registered in `pyproject.toml` (`--strict-markers`).
- TypeScript: strict mode, no `any`, co-located `*.test.tsx`, Zustand slices
  under `frontend/src/stores/slices/`, hooks under `frontend/src/hooks/`.
- Branches live less than a week; one concern per PR; Conventional Commits.
- Backwards compatible by default: new fields optional with defaults, old JSON
  records must still load, API responses gain fields but never lose them.

## Where things are

- Plan and expert reviews: `docs/plans/2026-09-validation-sdlc-plan.md`,
  `docs/plans/2026-09-review/`
- Decisions: `docs/adr/`; roadmap: `docs/roadmap.md`
- Architecture: `docs/architecture.md`
- Agent knowledge base and handoff protocol: `docs/agents/kb-protocol.md`,
  skills in `.claude/skills/<role>-handoff/SKILL.md`
- Security policy: `SECURITY.md`; contribution guide: `CONTRIBUTING.md`

## Session start

`.claude/hooks/kb-start.sh` prints a capped knowledge-base digest with a
staleness banner. Set `CLAUDE_ROLE=planning|dev-sqe|pre-pr` for role context.
Run the matching handoff skill before ending a session that did meaningful work.
