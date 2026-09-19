# ADR-0015: Knowledge-base protocol minimization

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

The SessionStart hook injected about 36 KB of February-2026 state on every startup, resume, and compaction, asserting facts that were false; a blocking Stop prompt hook forced `kb/` writes into every PR; three inconsistent protocol specifications coexisted.

## Decision

`.claude/hooks/kb-start.sh` prints a digest capped at `KB_MAX_INJECT_BYTES` (4096) with a staleness banner, on startup only; role context is opt-in via `CLAUDE_ROLE`. The Stop hook is a non-blocking reminder; handoff is an explicit skill. One protocol document, `docs/agents/kb-protocol.md`. Stale summaries and state are archived under `kb/archive/`. The Antigravity configuration moved to `docs/agents/antigravity/`, out of any load path. `kb/` changes ride in their own commit.

## Consequences

`tests/unit/test_kb_protocol.py` continues to validate the structure; a JSON-schema check for ledger entries is added next.
