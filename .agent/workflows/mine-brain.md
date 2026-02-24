---
description: Extract learnings from Antigravity brain walkthroughs into project KB rules. Mines recent session artifacts for patterns and gotchas.
---

1. Find recent walkthroughs
// turbo

```bash
find ~/.gemini/antigravity/brain -name "walkthrough.md" -type f -mtime -7 2>/dev/null | head -10
```

1. Extract and categorize learnings
For each walkthrough found:

* Check if its parent folder UUID matches any native_session_id in kb/ledger/ledger.jsonl.
* Extract “Challenges & Learnings” and “Key Patterns”.
* Categorize findings by domain (architecture, testing, API design).

1. Append to KB summary
Add a dated entry to kb/summaries/dev-sqe.md with extracted patterns.

2. Suggest new rules
If patterns repeat across 3+ walkthroughs, suggest creating a new rule file.
