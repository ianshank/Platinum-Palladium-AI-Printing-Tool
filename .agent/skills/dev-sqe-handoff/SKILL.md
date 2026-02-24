---
name: dev-sqe-handoff
description: >
  DEV-SQE session handoff protocol. Captures files changed (committed and
  uncommitted), tests added, coverage data, and report paths. Updates
  ledger, summary, and state files. Detects design gaps.
---

# DEV-SQE Handoff Protocol

Execute ALL of the following steps. Do not skip any step.

## Step 1: Gather Session Metrics

If the `/handoff` workflow provided metrics (Files Changed, Tests Affected), use them.
If running autonomously, run the git commands manually:

```bash
LAST_TS=$(jq -r '.last_timestamp // "1970-01-01"' kb/sessions/dev-sqe.state.json)
git log --since="$LAST_TS" --name-only --pretty=format:"" HEAD
git diff --name-only HEAD
```

## Step 2: Generate Session Identifier

Use the IDs provided by the workflow. If running autonomously:

* Logical ID: `session-$(date +%Y%m%d-%H%M%S)`
* Native UUID: Find the latest directory in `~/.gemini/antigravity/brain/`

## Step 3: Append Ledger Event

Append a single JSON line to `kb/ledger/ledger.jsonl`. Note the inclusion of `native_session_id` to link the KB to Antigravity's internal brain.
Concurrency Rule: You must acquire a lock before writing.

* Run `mkdir kb/ledger/lock.dir`
* If successful (exit code 0), write the line.
* If failed (directory exists), wait 2 seconds and retry.
* After writing, run `rmdir kb/ledger/lock.dir`.

```json
{
  "event": "DEV-SQE",
  "timestamp": "<ISO-8601>",
  "session_id": "<logical identifier>",
  "native_session_id": "<antigravity brain uuid>",
  "files_changed": ["<list of files from git log + diff>"],
  "tests_added": ["<list of test files>"],
  "coverage": "<percentage or 'unknown'>",
  "report_paths": ["<any generated reports>"],
  "summary": "<one-line summary of what was accomplished>"
}
```

## Step 4: Update Summary

Update `kb/summaries/dev-sqe.md` with a new entry containing:

* Date and session identifier
* Concise summary of what was accomplished
* Paths to any artifacts produced
* Current status: completed, in-progress, or blocked
Append the new entry to the bottom of the file. Do not delete old entries; rely on the context window to handle history.

## Step 5: Update Session State

Write to `kb/sessions/dev-sqe.state.json`. Update `last_timestamp` to the current time so the next session tracks changes starting from now.

```json
{
  "last_session_id": "<logical identifier>",
  "last_timestamp": "<current ISO-8601 time>",
  "active_branch": "<current git branch>",
  "resume_pointers": {
    "current_task": "<what was being worked on>",
    "blocked_on": ["<any blockers>"],
    "next_steps": ["<recommended next actions>"]
  },
  "context_files": ["<key files the next session should read>"]
}
```

## Step 6: Determine Completion Status

Compare current state against the active Spec in `kb/specs/`.

### Scenario A: Work In Progress (Loop)

If items remain unchecked or tests are failing:

1. Update `kb/sessions/dev-sqe.state.json` with `resume_pointers`.
2. Append `DEV-PROGRESS` event to Ledger (include coverage metrics).

### Scenario B: Spec Complete (Exit)

If **ALL** tests pass and Spec requirements are met:

1. **Run Regression Check:** Ensure no existing tests broke.
2. Update `kb/sessions/dev-sqe.state.json` (clear tasks).
3. Append `DEV-COMPLETE` event to Ledger:

```json
{
  "event": "DEV-COMPLETE",
  "timestamp": "<ISO-8601>",
  "spec_file": "<path-to-spec>",
  "native_session_id": "<brain-uuid>",
  "handoff_to": "pre-pr"
}
```

* Suggest: "Feature complete. Run /pre-pr to package."

## Step 7: Detect Design Gaps (Conditional)

If during this session you encountered missing specs, ambiguous contracts, or undefined behavior:

* Append a `DESIGN-GAP` ledger event:

```json
{
  "event": "DESIGN-GAP",
  "timestamp": "<ISO-8601>",
  "session_id": "<logical identifier>",
  "gaps": ["<description of each gap>"],
  "blocking": "<what work is blocked>",
  "handoff_to": "design-contract-readiness"
}
```

* Create a handoff document:
Create `kb/handoffs/<YYYYMMDD-HHmmss>_dev-sqe_to_design-contract-readiness.md`. Handoff files are immutable once created.
