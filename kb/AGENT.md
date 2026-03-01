# Knowledge Base Directory

## Purpose
Structured knowledge base for session continuity across the three-role workflow pipeline (Planning → DEV-SQE → Pre-PR). Stores handoff documents, event logs, session state, and rolling summaries.

## Key Subdirectories
- `handoffs/` — Immutable cross-role handoff documents (NEVER modify after creation)
- `ledger/ledger.jsonl` — Append-only event log shared across all roles (NEVER overwrite or truncate)
- `sessions/` — Mutable per-role session state: `planning.state.json`, `dev-sqe.state.json`, `pre-pr.state.json`
- `summaries/` — Rolling summaries per role + `design-contract-readiness.md`

## Critical Safety Rules
1. **NEVER** overwrite or truncate `ledger/ledger.jsonl` — ONLY append new lines
2. **NEVER** modify files in `handoffs/` — they are immutable once created
3. **ALWAYS** validate JSONL entries with `jq` after writing to `ledger.jsonl`
4. Session state files (`sessions/*.state.json`) are the ONLY mutable state per role
5. Summaries are append-at-top with a cap of 20 entries
6. In Agent Teams, designate ONE teammate as the sole ledger writer to avoid race conditions

## Event Types
- `PLANNING-HANDOFF` — Planning → DEV-SQE
- `DEV-SQE-HANDOFF` — DEV-SQE → Pre-PR
- `DESIGN-GAP` — DEV-SQE → Planning (feedback loop)
- `QA-GAP` — Pre-PR → DEV-SQE (feedback loop)
- `SCOPE-CHANGE` — Pre-PR → Planning (feedback loop)

## Environment Variables
See CLAUDE.md "Knowledge Base Protocol" section for: `CLAUDE_ROLE`, `KB_DIR`, `KB_MAX_SUMMARY_BYTES`, `KB_MAX_CROSS_ROLE_BYTES`, `KB_MAX_LEDGER_LINES`, `KB_MAX_HANDOFFS`, `KB_DEBUG`.

## Pitfalls
- Race conditions on `ledger.jsonl` if multiple agents write simultaneously
- Summary files can exceed `KB_MAX_SUMMARY_BYTES` if not pruned — check before appending
- Handoff skill must be run before stopping any session that did meaningful work

## Related
- `.claude/hooks/kb-start.sh` — Dispatcher hook that loads KB context on session start
- `.claude/skills/` — Handoff skills for each role
- `.agent/rules/kb-protocol.md` — Full KB protocol rules
