---
description: Trigger the Pre-PR verification sequence. Runs final tests, linters, and generates PR text.
---

1. Safety Check
// turbo

```bash
if [[ -n $(git status --porcelain) ]]; then
echo "❌ ERROR: Uncommitted changes. Run /handoff first."
exit 1
fi
```

1. Execute Pre-PR Skill
Read .agent/skills/pre-pr-verification/SKILL.md and execute.
