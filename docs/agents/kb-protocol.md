# Knowledge-base protocol for agent sessions

This is the single source of truth for `kb/` (ADR-0015). It replaces the protocol
sections that used to live in `CLAUDE.md`, `kb/AGENT.md`, and `.agent/rules/`.

## Purpose

`kb/` carries state between agent sessions in a three-role pipeline: Planning →
DEV-SQE → Pre-PR. It records what was decided and done, so a later session can
resume without re-deriving it. It is not documentation for humans; humans read
`docs/`.

## Layout

```
kb/
├── ledger/ledger.jsonl        append-only event log (one JSON object per line)
├── sessions/<role>.state.json mutable resume pointers, one per role
├── summaries/<role>.md        rolling, prepend-at-top, capped at 20 entries
├── handoffs/                  immutable cross-role handoff documents
└── archive/<yyyy-mm>/         superseded summaries and state, kept for history
```

## Rules

1. `ledger.jsonl` is append-only. Never rewrite or truncate it. Every line is valid
   JSON with at least `timestamp` (ISO-8601 UTC), `event`, and `session_id`; events
   that make numeric claims (tests passed, coverage) carry `git_commit` and
   `ci_run_url`.
2. Event vocabulary: `PLANNING`, `PLANNING-HANDOFF`, `DEV-SQE`, `DEV-SQE-HANDOFF`,
   `PRE-PR`, `PR-READY`, `DESIGN-GAP`, `QA-GAP`, `SCOPE-CHANGE`. `tests/unit/test_kb_protocol.py`
   validates the file.
3. Handoff files are immutable once created and named
   `YYYYMMDD-HHMMSS_<from>_to_<to>.md`.
4. State files are the only mutable per-role state and must keep the keys
   `last_session_id`, `last_timestamp`, `active_branch`, `resume_pointers`,
   `context_files` at the top level.
5. Summaries are prepended and capped at 20 entries; when a summary describes a
   state that no longer holds, archive it under `kb/archive/` rather than editing
   history.
6. Session start injects a digest only: `.claude/hooks/kb-start.sh` prints at most
   `KB_MAX_INJECT_BYTES` (default 4096) with a staleness banner that compares the
   last ledger commit against `HEAD`. Role context is opt-in via `CLAUDE_ROLE`.
7. Handoff is explicit: run the matching skill in `.claude/skills/<role>-handoff/`
   before ending a session that did meaningful work. The Stop hook only reminds;
   it never blocks.
8. `kb/` changes ride in their own commit and never mix with unrelated code
   changes in a PR.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_ROLE` | unset (digest) | `planning`, `dev-sqe`, `pre-pr`, or `all` |
| `KB_DIR` | `$CLAUDE_PROJECT_DIR/kb` | knowledge-base location |
| `KB_MAX_INJECT_BYTES` | `4096` | hard cap on injected bytes |
| `KB_MAX_LEDGER_LINES` | `5` | ledger events shown in the digest |
| `KB_MAX_HANDOFFS` | `3` | handoff paths listed in the digest |
| `KB_DEBUG` | `false` | debug output on stderr |

Role hooks (`planning-start.sh`, `dev-sqe-start.sh`, `pre-pr-start.sh`) keep their
own `KB_MAX_SUMMARY_BYTES`, `KB_MAX_CROSS_ROLE_BYTES`, and `KB_MAX_LEDGER_LINES`
settings; their output is still subject to the global cap.

## History

The February 2026 summaries and state files asserted "726 passed, 0 failed" and
"ready for PR" for months after they stopped being true, and were injected on every
session start, resume, and compaction (about 36 KB). They are preserved under
`kb/archive/2026-02/`.
