---
description: Run automated gap analysis comparing CLAUDE.md checklist vs actual implementation
---

# Gap Analysis Workflow

Audits the codebase against the gap checklist in `CLAUDE.md` and `.claude/agents/gap-remediation-agent.md`.

## Steps

1. **Read Gap Checklist**
   Parse all `- [ ]` items from `CLAUDE.md` lines 396–422 (Gap Analysis Checklist section).

2. **Scan Implementation**
   For each gap category, search the codebase for matching implementations:
   - **Calibration Workflow**: Check `frontend/src/components/wizard/`, `frontend/src/components/calibration/`
   - **AI Integration**: Check `src/ptpd_calibration/ai/`, `src/ptpd_calibration/llm/`, `src/ptpd_calibration/ml/`
   - **Data Persistence**: Check `src/ptpd_calibration/gcp/`, `src/ptpd_calibration/ml/database.py`
   - **Missing Features**: Check `frontend/src/hooks/`, `frontend/src/stores/slices/`

3. **Generate Gap Report**
   Output structured report per the template in `gap-remediation-agent.md`:

   ```
   ## Gap Analysis Report — {DATE}
   ### Summary
   - Total Gaps: X
   - Resolved: X
   - Remaining: X
   ### By Category
   | Gap ID | Feature | Status | Evidence |
   ```

4. **Update CLAUDE.md**
   Mark resolved gaps with `- [x]` and add implementation references.

5. **Create Issues** (optional)
   Use `mcp_GitKraken_issues_add_comment` to track outstanding gaps as GitHub issues.

## Output

- Gap report written to `docs/gap-analysis-{DATE}.md`
- Updated `CLAUDE.md` checklist
- Optional GitHub issues for remaining gaps
