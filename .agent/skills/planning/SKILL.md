---
name: planning-protocol
description: Finalize architecture and lock the spec for development.
---

# Planning Protocol

## Step 1: Write the Spec

Work with the user to create a new file `kb/specs/<YYYYMMDD>-<slug>.md`.
**Content Requirements:**

- **Objective:** One sentence summary.
- **Files to Touch:** List of paths.
- **Verification Steps:** Specific tests that must pass.
- **Definition of Done:** Checklist.

## Step 2: Clear Planning State

Reset `kb/sessions/planning.state.json` to idle.

## Step 3: Ledger Event (The Lock)

Acquire Atomic Lock. Append to `kb/ledger/ledger.jsonl`:

```json
{
  "event": "PLAN-LOCKED",
  "timestamp": "<ISO-8601>",
  "spec_file": "kb/specs/<filename>.md",
  "native_session_id": "<brain-uuid>",
  "handoff_to": "dev-sqe"
}
```

Release Lock.
