# Pre-PR Protocol

## Step 1: Verification (The Gauntlet)

Run the project's strict verification suite:

- `npm run lint` (or equivalent)
- `npm test` (Full suite, no skipping)

## Step 2: Mine the Brain (Forensics)

1. Read the last 5 `DEV-PROGRESS` events to find `native_session_id`s.
2. Locate the `walkthrough.md` files in `~/.gemini/antigravity/brain/` for those sessions.
3. Extract "Challenges" and "Key Decisions" to explain *why* changes were made.

## Step 3: Generate PR Description

Create `kb/pr-drafts/PR-<date>.md`:

- **Title:** `feat/fix: <Spec Title>`
- **Linked Spec:** `kb/specs/<filename>.md`
- **Changes:** derived from `git log` since `PLAN-LOCKED`.
- **Learnings:** (From Brain mining).

## Step 4: Final Ledger Entry

Acquire Lock. Append:

```json
{
  "event": "PR-READY",
  "timestamp": "<ISO-8601>",
  "spec_file": "<path>",
  "pr_draft": "kb/pr-drafts/PR-<date>.md",
  "status": "Awaiting Human Push"
}
```

Release Lock.
