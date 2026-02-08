# Knowledge Base Protocol (Full Lifecycle)

## CRITICAL: Context Loading & Role Inference

Before starting, read the last 5 lines of `kb/ledger/ledger.jsonl`. Your role depends on the **last event type**:

### 1. Planning Phase

**Trigger:** Last event is `PR-READY` (Project Idle) or user explicitly runs `/plan`.
**Goal:** Create an immutable Spec in `kb/specs/`.
**Read:** `kb/summaries/planning.md`, `kb/sessions/planning.state.json`.

### 2. Dev/SQE Phase

**Trigger:** Last event is `PLAN-LOCKED` or `DEV-PROGRESS`.
**Goal:** Implement the `spec_file` defined in the last event.
**Read:**

- The Active Spec: `kb/specs/<filename>.md`
- Session State: `kb/sessions/dev-sqe.state.json`
- **Rule:** You are in a TDD loop. Do not ask for new requirements; follow the Spec.

### 3. Pre-PR Phase

**Trigger:** Last event is `DEV-COMPLETE` or user runs `/pre-pr`.
**Goal:** Verify code quality and generate PR artifacts.
**Read:** `kb/sessions/pre-pr.state.json`, `kb/specs/<active>.md`.

## Universal Safety Rules

1. **Atomic Locking:** NEVER write to `kb/ledger/` without `mkdir kb/ledger/lock.dir`.
2. **Immutable Specs:** Once a file is written to `kb/specs/`, it is read-only.

## Multi-Agent Coordination

When multiple agents are working on this project simultaneously:

1. **Atomic Locking**: Before writing to `kb/ledger/`, you MUST acquire a lock.
Run: `mkdir kb/ledger/lock.dir`

- **If exit code is 0 (Success)**: You have the lock. Write to the ledger, then immediately run `rmdir kb/ledger/lock.dir` to release it.
- **If exit code is non-zero (Failure)**: The lock exists. Wait 2 seconds and retry.
- **Stale Lock**: If `kb/ledger/lock.dir` is older than 60 seconds, assume the previous agent crashed. Remove it and try again.

1. **Per-agent state files**: Each agent should use its own state file
(e.g., `kb/sessions/dev-sqe-frontend.state.json`,
`kb/sessions/dev-sqe-backend.state.json`) rather than sharing one.

2. **Handoff documents as coordination signals**: When one agent
discovers a dependency on another agent's work, create a handoff
document in `kb/handoffs/`.
