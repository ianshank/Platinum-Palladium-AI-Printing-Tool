---
description: Run the DEV-SQE handoff protocol. Captures session metrics (committed + uncommitted), updates the ledger, summary, and state files. Run this before ending any DEV-SQE session.
---

1. Read the DEV-SQE handoff skill
Read `.agent/skills/dev-sqe-handoff/SKILL.md` and follow every step
in the protocol. Do not skip any step.

2. Gather git metrics (Committed + Uncommitted)
// turbo

```bash
# Retrieve start time of previous session
LAST_TS=$(jq -r '.last_timestamp // "1970-01-01"' kb/sessions/dev-sqe.state.json)

echo "=== FILES CHANGED (Since $LAST_TS) ==="
# Combine git log (committed) and git diff (working tree)
{
git log --since="$LAST_TS" --name-only --pretty=format:"" HEAD 2>/dev/null
git diff --name-only HEAD 2>/dev/null
} | sort | uniq | grep -vE "^$" || echo "no changes"

echo "=== TESTS AFFECTED ==="
{
git log --since="$LAST_TS" --name-only --pretty=format:"" HEAD 2>/dev/null
git diff --name-only HEAD 2>/dev/null
} | grep -iE '(test_|_test\.|spec\.|\.test\.)' | sort | uniq || echo "none"

echo "=== CURRENT BRANCH ==="
git branch --show-current 2>/dev/null || echo "unknown"
```

1. Generate session identifiers
// turbo

```bash
# Generate logical Session ID
echo "LOGICAL_ID: session-$(date +%Y%m%d-%H%M%S)"

# Heuristic to find Antigravity's Native Brain UUID (most recently modified brain dir)
NATIVE_UUID=$(ls -td ~/.gemini/antigravity/brain/*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "unknown-uuid")
echo "NATIVE_UUID: $NATIVE_UUID"
```

1. Execute handoff protocol
Using the metrics and IDs from the previous steps, execute ALL steps
in the DEV-SQE handoff skill:

* Append a DEV-SQE event to kb/ledger/ledger.jsonl (include both IDs)
* Update kb/summaries/dev-sqe.md
* Update kb/sessions/dev-sqe.state.json
* If design gaps were found, append a DESIGN-GAP event and create
a handoff document in kb/handoffs/

1. Validate the ledger entry
// turbo

```bash
[ -s kb/ledger/ledger.jsonl ] && tail -1 kb/ledger/ledger.jsonl | python3 -m json.tool > /dev/null 2>&1 && echo "OK: valid JSON" || echo "ERROR: invalid JSON or empty file"
```
