---
name: dev-sqe-handoff
description: >
  DEV-SQE session handoff protocol. Captures files changed, tests added,
  coverage data, and report paths. Updates ledger, summary, and state files.
  Detects design gaps and creates handoff documents when blocked.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# DEV-SQE Handoff Skill

## Purpose
Captures development and testing session artifacts, tracks implementation progress, and creates structured handoff documents for upstream (PLANNING) or downstream (PRE-PR) phases.

## Protocol Steps

### 1. Gather Session Metrics
Collect metrics from the current DEV-SQE session:

**Files Changed:**
```bash
git diff --name-only HEAD
```

**Tests Added:**
```bash
git diff --name-only HEAD | grep -E '\.test\.(ts|tsx|py)$|\.spec\.(ts|tsx|py)$|^tests/'
```

**Source Changes:**
```bash
git diff --name-only HEAD | grep -vE '\.test\.|\.spec\.|^tests/'
```

**Current Branch:**
```bash
git branch --show-current
```

**Test Summary (Python):**
```bash
pytest --tb=short --maxfail=1 -q 2>&1 | tail -10
```

**Test Summary (Frontend):**
```bash
cd frontend && pnpm test -- --run --reporter=verbose 2>&1 | tail -20
```

**Linting:**
```bash
ruff check --output-format=text 2>&1
```

### 2. Generate Session ID
```bash
session_id=$(date +%s-%N)
```

### 3. Append DEV-SQE Ledger Event
Append to `kb/ledger/ledger.jsonl`:

```json
{
  "event": "DEV-SQE",
  "timestamp": "2026-02-08T14:22:33.456Z",
  "session_id": "1707402153-987654321",
  "git_commit": "abc123def456",
  "git_branch": "feature/implement-image-slice",
  "files_changed": [
    "frontend/src/stores/slices/imageSlice.ts",
    "frontend/src/stores/slices/imageSlice.test.ts"
  ],
  "tests_added": [
    "frontend/src/stores/slices/imageSlice.test.ts"
  ],
  "coverage": {
    "lines": 85.2,
    "branches": 78.5,
    "functions": 90.0,
    "statements": 85.2
  },
  "report_paths": {
    "coverage": "frontend/coverage/lcov-report/index.html",
    "test_results": "frontend/test-results.json",
    "lint": "ruff-report.txt"
  },
  "tasks_completed": ["TASK-001"],
  "tasks_in_progress": ["TASK-002"],
  "tasks_blocked": [],
  "design_gaps_found": [],
  "summary": "Implemented imageSlice with 85% coverage"
}
```

**Critical Rules:**
- ONLY append to ledger.jsonl, NEVER overwrite
- Create `kb/ledger/` directory if it doesn't exist
- Each event must be valid JSON on a single line

### 4. Validate Ledger Entry
```bash
tail -1 kb/ledger/ledger.jsonl | jq empty
```
If validation fails, log error and abort handoff.

### 5. Update DEV-SQE Summary
Update `kb/summaries/dev-sqe.md` (create if missing):
- Prepend new entry at TOP
- Cap at 20 most recent entries
- Format:
```markdown
## [YYYY-MM-DD HH:mm:ss] Session 1707402153-987654321

**Branch**: feature/implement-image-slice
**Files Changed**: 2 (1 source, 1 test)
**Coverage**: 85.2% lines, 78.5% branches
**Tasks Completed**: TASK-001
**Tasks In Progress**: TASK-002

### Summary
Implemented imageSlice with Zustand immer middleware. Added comprehensive tests covering state updates, image upload, and error handling.

### Files Modified
- `frontend/src/stores/slices/imageSlice.ts` (+120 lines)
- `frontend/src/stores/slices/imageSlice.test.ts` (+95 lines)

### Test Results
- 12 tests passed
- Coverage: 85.2% (target: 80%)
- No lint errors

---
```

### 6. Update DEV-SQE State
Write `kb/sessions/dev-sqe.state.json`:
```json
{
  "last_session_id": "1707402153-987654321",
  "last_timestamp": "2026-02-08T14:22:33.456Z",
  "active_branch": "feature/implement-image-slice",
  "git_commit": "abc123def456",
  "resume_pointers": {
    "last_file_modified": "frontend/src/stores/slices/imageSlice.ts",
    "last_test_file": "frontend/src/stores/slices/imageSlice.test.ts",
    "context_files": [
      "CLAUDE.md",
      "frontend/src/stores/index.ts",
      "migration/component-map.json"
    ]
  },
  "implementation_status": {
    "completed": ["TASK-001"],
    "in_progress": ["TASK-002"],
    "blocked": [],
    "ready_for_pr": []
  },
  "test_status": {
    "total_tests": 12,
    "passing": 12,
    "failing": 0,
    "coverage_percent": 85.2,
    "last_test_run": "2026-02-08T14:22:33.456Z"
  },
  "quality_checks": {
    "lint": "pass",
    "typecheck": "pass",
    "format": "pass"
  }
}
```

### 7. CONDITIONAL: Upstream Handoff (Design Gap)
**Trigger**: Design gaps, unclear requirements, missing specifications

**Design Gap Indicators:**
- Missing acceptance criteria
- Unclear API contracts
- Ambiguous behavior specifications
- Conflicting requirements
- Missing design decisions

Actions:
1. Append DESIGN-GAP ledger event:
```json
{
  "event": "DESIGN-GAP",
  "timestamp": "2026-02-08T14:22:33.456Z",
  "session_id": "1707402153-987654321",
  "from_phase": "DEV-SQE",
  "to_phase": "PLANNING",
  "task_id": "TASK-002",
  "gap_type": "missing-specification|unclear-contract|conflicting-requirements",
  "gaps_identified": [
    {
      "id": "GAP-001",
      "description": "Missing error handling specification for file upload failures",
      "blocking_task": "TASK-002",
      "needs_clarification": true
    }
  ],
  "handoff_doc": "kb/handoffs/20260208-142233_dev-sqe_to_planning.md",
  "summary": "Implementation blocked by missing error handling specs"
}
```

2. Create handoff document at `kb/handoffs/YYYYMMDD-HHmmss_dev-sqe_to_planning.md`:
```markdown
# DEV-SQE → PLANNING Handoff (Design Gap)

**Date**: 2026-02-08 14:22:33
**Session ID**: 1707402153-987654321
**Branch**: feature/implement-image-slice
**Commit**: abc123def456

## Design Gaps Identified

### GAP-001: Missing Error Handling Specification
**Blocking Task**: TASK-002
**Gap Type**: missing-specification

**Context**:
While implementing file upload in imageSlice, discovered that error handling behavior is not specified:
1. Should upload errors clear the current image?
2. Should partial uploads be retained?
3. What retry strategy should be used?

**Legacy Behavior** (from Gradio):
- Errors display toast notification
- Previous image is retained
- No automatic retry

**Questions for Planning**:
- [ ] Confirm error handling strategy matches legacy
- [ ] Define retry policy (if any)
- [ ] Specify error state persistence

**Impact**: Blocks TASK-002 implementation

**Files Affected**:
- frontend/src/stores/slices/imageSlice.ts (incomplete)
- frontend/src/components/ImageUpload.tsx (pending)

### GAP-002: ...

## Current Implementation Status
- TASK-001: ✓ Complete
- TASK-002: ⚠ Blocked by GAP-001

## Next Steps for Planning
1. Review error handling requirements
2. Document acceptance criteria for edge cases
3. Update task specifications

---
**Handoff Document**: Immutable after creation
**Next Phase**: PLANNING (gap resolution)
```

**Handoff files are immutable** - never modify after creation.

### 8. CONDITIONAL: Downstream Handoff (Implementation Complete)
**Trigger**: All tasks complete, quality checks pass, ready for pre-PR validation

**Ready for PR Indicators:**
- All assigned tasks completed
- Tests passing (coverage ≥ 80%)
- No lint/typecheck errors
- No design gaps
- Documentation updated

Actions:
1. Append DEV-SQE-HANDOFF ledger event:
```json
{
  "event": "DEV-SQE-HANDOFF",
  "timestamp": "2026-02-08T15:30:00.123Z",
  "session_id": "1707402153-987654321",
  "from_phase": "DEV-SQE",
  "to_phase": "PRE-PR",
  "tasks_completed": ["TASK-001", "TASK-002"],
  "quality_metrics": {
    "coverage": 85.2,
    "tests_passing": 24,
    "tests_total": 24,
    "lint_errors": 0,
    "typecheck_errors": 0
  },
  "handoff_doc": "kb/handoffs/20260208-153000_dev-sqe_to_pre-pr.md",
  "summary": "Implementation complete, ready for PR validation"
}
```

2. Create handoff document at `kb/handoffs/YYYYMMDD-HHmmss_dev-sqe_to_pre-pr.md`:
```markdown
# DEV-SQE → PRE-PR Handoff

**Date**: 2026-02-08 15:30:00
**Session ID**: 1707402153-987654321
**Branch**: feature/implement-image-slice
**Commit**: abc123def456

## Implementation Summary
Completed implementation of imageSlice and related components. All tests passing, coverage exceeds target.

## Tasks Completed
- ✓ TASK-001: Implement Zustand imageSlice
- ✓ TASK-002: Create ImageUpload component

## Quality Metrics
- **Test Coverage**: 85.2% (target: 80%)
- **Tests**: 24/24 passing
- **Lint**: 0 errors
- **Typecheck**: 0 errors
- **Format**: Pass

## Files Changed
### Source Files
- `frontend/src/stores/slices/imageSlice.ts` (+120 lines)
- `frontend/src/components/ImageUpload.tsx` (+85 lines)
- `frontend/src/stores/index.ts` (modified)

### Test Files
- `frontend/src/stores/slices/imageSlice.test.ts` (+95 lines)
- `frontend/src/components/ImageUpload.test.tsx` (+75 lines)

## Verification Checklist
- [x] All acceptance criteria met
- [x] Tests pass (frontend: pnpm test)
- [x] Coverage ≥ 80%
- [x] Lint passes (pnpm lint)
- [x] Typecheck passes (pnpm typecheck)
- [x] Format passes (pnpm format:check)
- [x] Documentation updated
- [x] Migration equivalence verified

## Next Steps for PRE-PR
1. Run full validation suite
2. Verify changelog entry
3. Run security audit (bandit/snyk)
4. Generate PR description
5. Validate commit messages

---
**Handoff Document**: Immutable after creation
**Next Phase**: PRE-PR validation
```

## Backwards Compatibility

### Missing Directories
If `kb/` structure doesn't exist, create:
```bash
mkdir -p kb/ledger kb/summaries kb/sessions kb/handoffs
```

### Missing Ledger
If `kb/ledger/ledger.jsonl` doesn't exist, create with header comment:
```bash
echo '{"_comment":"Persistent Knowledge-Base Ledger - Append-only event log"}' > kb/ledger/ledger.jsonl
```

### Missing Tools
- If `pytest` not available, skip Python test metrics
- If `pnpm` not available, skip frontend metrics
- If `jq` not available, skip validation but log warning
- If `git` not available, use filesystem timestamps

### Missing Test Runners
Gracefully handle missing test runners:
```bash
if command -v pytest &> /dev/null; then
  # Run pytest metrics
else
  echo "pytest not found, skipping Python test metrics"
fi
```

## Logging Structure

All ledger events include:
```json
{
  "event": "EVENT_TYPE",
  "timestamp": "ISO-8601 timestamp",
  "session_id": "unique-session-id",
  "git_commit": "git SHA (if available)",
  "git_branch": "branch name (if available)",
  "user": "environment user (if available)",
  "..." : "event-specific fields"
}
```

## Success Criteria
- Ledger event appended successfully
- Ledger validates with jq
- Summary updated with new entry (max 20)
- State file written with correct structure
- Handoff docs created when conditions met
- Quality metrics captured accurately
- No files overwritten (append-only for ledger)
