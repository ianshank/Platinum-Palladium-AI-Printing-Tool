---
name: pre-pr-handoff
description: >
  Pre-PR session handoff protocol. Validates code quality, runs checks,
  verifies documentation, and prepares PR artifacts. Updates ledger, summary,
  and state files.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Pre-PR Handoff Skill

## Purpose
Validates implementation quality before creating pull requests. Runs comprehensive checks, identifies issues, and creates handoff documents for upstream (DEV-SQE, PLANNING) or downstream (PR creation) phases.

## Protocol Steps

### 1. Run Validation Checks
Execute all quality checks and capture results:

**Python Lint:**
```bash
ruff check --output-format=text src/ tests/ 2>&1
lint_status=$?
```

**Python Format:**
```bash
ruff format --check src/ tests/ 2>&1
format_status=$?
```

**Python Type Check:**
```bash
mypy src/ tests/ --show-error-codes 2>&1
mypy_status=$?
```

**Python Tests:**
```bash
pytest tests/ -v --tb=short 2>&1
pytest_status=$?
```

**Python Coverage:**
```bash
pytest --cov=src --cov-report=term --cov-report=html 2>&1
coverage_status=$?
```

**Python Security:**
```bash
bandit -r src/ -f json -o bandit-report.json 2>&1
bandit_status=$?
```

**Frontend Lint:**
```bash
cd frontend && pnpm lint 2>&1
frontend_lint_status=$?
```

**Frontend Typecheck:**
```bash
cd frontend && pnpm typecheck 2>&1
frontend_typecheck_status=$?
```

**Frontend Format:**
```bash
cd frontend && pnpm format:check 2>&1
frontend_format_status=$?
```

**Frontend Tests:**
```bash
cd frontend && pnpm test -- --run 2>&1
frontend_test_status=$?
```

**Frontend Coverage:**
```bash
cd frontend && pnpm test:coverage 2>&1
frontend_coverage_status=$?
```

**Documentation Changes:**
```bash
git diff --name-only HEAD | grep -E '\.md$|docs/' | wc -l
docs_changed=$?
```

**Changelog Check:**
```bash
git diff --name-only HEAD | grep -E 'CHANGELOG\.md|changelog\.md'
changelog_status=$?
```

**Git Status:**
```bash
git status --porcelain
git_status=$?
```

### 2. Generate Session ID
```bash
session_id=$(date +%s-%N)
```

### 3. Append PRE-PR Ledger Event
Append to `kb/ledger/ledger.jsonl`:

```json
{
  "event": "PRE-PR",
  "timestamp": "2026-02-08T16:45:12.789Z",
  "session_id": "1707406512-456789123",
  "git_commit": "def789abc123",
  "git_branch": "feature/implement-image-slice",
  "checks": {
    "python": {
      "lint": {"status": "pass|fail|skip", "errors": 0, "warnings": 2},
      "format": {"status": "pass|fail|skip", "files_changed": 0},
      "typecheck": {"status": "pass|fail|skip", "errors": 0},
      "tests": {"status": "pass|fail|skip", "passed": 45, "failed": 0, "skipped": 2},
      "coverage": {"status": "pass|fail|skip", "percent": 88.5, "threshold": 80.0},
      "security": {"status": "pass|fail|skip", "issues": 0, "severity": "none"}
    },
    "frontend": {
      "lint": {"status": "pass|fail|skip", "errors": 0, "warnings": 1},
      "format": {"status": "pass|fail|skip", "files_changed": 0},
      "typecheck": {"status": "pass|fail|skip", "errors": 0},
      "tests": {"status": "pass|fail|skip", "passed": 24, "failed": 0},
      "coverage": {"status": "pass|fail|skip", "percent": 85.2, "threshold": 80.0}
    },
    "documentation": {
      "status": "pass|fail|skip",
      "files_changed": 3,
      "changelog_updated": true
    },
    "git": {
      "status": "pass|fail|skip",
      "uncommitted_files": 0,
      "untracked_files": 0
    }
  },
  "all_passing": true,
  "issues_found": [
    {
      "severity": "warning",
      "check": "python.lint",
      "message": "Line too long (100 > 88 characters)",
      "file": "src/example.py",
      "line": 42
    }
  ],
  "fixes_applied": [
    "Auto-formatted 3 Python files",
    "Auto-formatted 2 TypeScript files"
  ],
  "summary": "All checks passed, ready for PR"
}
```

**Critical Rules:**
- ONLY append to ledger.jsonl, NEVER overwrite
- Create `kb/ledger/` directory if it doesn't exist
- Each event must be valid JSON on a single line
- Capture both pass/fail status AND metrics for each check

### 4. Validate Ledger Entry
```bash
tail -1 kb/ledger/ledger.jsonl | jq empty
```
If validation fails, log error and abort handoff.

### 5. Update PRE-PR Summary
Update `kb/summaries/pre-pr.md` (create if missing):
- Prepend new entry at TOP
- Cap at 20 most recent entries
- Format:
```markdown
## [YYYY-MM-DD HH:mm:ss] Session 1707406512-456789123

**Branch**: feature/implement-image-slice
**Commit**: def789abc123
**Overall Status**: ✓ PASS

### Check Results
| Check | Status | Details |
|-------|--------|---------|
| Python Lint | ✓ PASS | 0 errors, 2 warnings |
| Python Format | ✓ PASS | All files formatted |
| Python Typecheck | ✓ PASS | 0 errors |
| Python Tests | ✓ PASS | 45/45 passing |
| Python Coverage | ✓ PASS | 88.5% (threshold: 80%) |
| Python Security | ✓ PASS | 0 issues |
| Frontend Lint | ✓ PASS | 0 errors, 1 warning |
| Frontend Format | ✓ PASS | All files formatted |
| Frontend Typecheck | ✓ PASS | 0 errors |
| Frontend Tests | ✓ PASS | 24/24 passing |
| Frontend Coverage | ✓ PASS | 85.2% (threshold: 80%) |
| Documentation | ✓ PASS | 3 files updated, changelog updated |
| Git Status | ✓ PASS | Clean working directory |

### Issues Found
- 2 warnings in Python lint (non-blocking)
- 1 warning in Frontend lint (non-blocking)

### Fixes Applied
- Auto-formatted 3 Python files
- Auto-formatted 2 TypeScript files

### PR Readiness
**Status**: READY
**Next Step**: Create PR

---
```

### 6. Update PRE-PR State
Write `kb/sessions/pre-pr.state.json`:
```json
{
  "last_session_id": "1707406512-456789123",
  "last_timestamp": "2026-02-08T16:45:12.789Z",
  "active_branch": "feature/implement-image-slice",
  "git_commit": "def789abc123",
  "resume_pointers": {
    "last_check_run": "2026-02-08T16:45:12.789Z",
    "context_files": [
      "CLAUDE.md",
      "frontend/package.json",
      "pyproject.toml"
    ]
  },
  "pr_readiness": {
    "status": "ready|blocked|needs-fixes",
    "blocking_issues": [],
    "warnings": [
      "2 Python lint warnings",
      "1 Frontend lint warning"
    ]
  },
  "check_results": {
    "python_lint": "pass",
    "python_format": "pass",
    "python_typecheck": "pass",
    "python_tests": "pass",
    "python_coverage": "pass",
    "python_security": "pass",
    "frontend_lint": "pass",
    "frontend_format": "pass",
    "frontend_typecheck": "pass",
    "frontend_tests": "pass",
    "frontend_coverage": "pass",
    "documentation": "pass",
    "git_status": "pass"
  },
  "last_pr_doc": null
}
```

### 7. CONDITIONAL: QA Gap Handoff (Unfixable Issues)
**Trigger**: Pre-PR detects issues that require code changes (not auto-fixable)

**QA Gap Indicators:**
- Failing tests
- Coverage below threshold
- Type errors
- Security issues
- Broken functionality

Actions:
1. Append QA-GAP ledger event:
```json
{
  "event": "QA-GAP",
  "timestamp": "2026-02-08T16:45:12.789Z",
  "session_id": "1707406512-456789123",
  "from_phase": "PRE-PR",
  "to_phase": "DEV-SQE",
  "issues_found": [
    {
      "severity": "error",
      "check": "python.tests",
      "message": "Test test_image_upload_error failed",
      "file": "tests/test_image_slice.py",
      "line": 85,
      "requires_code_change": true
    }
  ],
  "handoff_doc": "kb/handoffs/20260208-164512_pre-pr_to_dev-sqe.md",
  "summary": "3 test failures require code fixes"
}
```

2. Create handoff document at `kb/handoffs/YYYYMMDD-HHmmss_pre-pr_to_dev-sqe.md`:
```markdown
# PRE-PR → DEV-SQE Handoff (QA Gap)

**Date**: 2026-02-08 16:45:12
**Session ID**: 1707406512-456789123
**Branch**: feature/implement-image-slice
**Commit**: def789abc123

## QA Issues Requiring Code Fixes

### Issue 1: Test Failure - test_image_upload_error
**Severity**: ERROR
**Check**: python.tests
**File**: tests/test_image_slice.py:85

**Error Message**:
```
AssertionError: Expected error state to be set, but got None
```

**Context**:
Test expects imageSlice to set error state when upload fails, but current implementation doesn't handle this case.

**Required Fix**:
Update imageSlice.uploadImage to set error state on failure.

**Files to Modify**:
- src/stores/slices/imageSlice.ts

### Issue 2: Coverage Below Threshold
**Severity**: WARNING
**Check**: python.coverage
**Current**: 75.2%
**Threshold**: 80.0%

**Uncovered Lines**:
- src/stores/slices/imageSlice.ts: Lines 92-98 (error handling)

**Required Fix**:
Add tests for error handling paths.

**Files to Modify**:
- tests/test_image_slice.py (add error case tests)

## Check Results Summary
| Check | Status | Notes |
|-------|--------|-------|
| Python Tests | ✗ FAIL | 3 failures |
| Python Coverage | ✗ FAIL | 75.2% < 80% |
| All Others | ✓ PASS | No issues |

## Next Steps for DEV-SQE
1. Fix test failures in test_image_slice.py
2. Add error handling tests to reach 80% coverage
3. Re-run verification loop
4. Return to PRE-PR when fixed

---
**Handoff Document**: Immutable after creation
**Next Phase**: DEV-SQE (fix implementation)
```

### 8. CONDITIONAL: Scope Change Handoff (Architecture Issues)
**Trigger**: Pre-PR detects architectural or scope issues

**Scope Change Indicators:**
- Major refactoring needed
- Missing features discovered
- Architecture violations
- Breaking changes required

Actions:
1. Append SCOPE-CHANGE ledger event:
```json
{
  "event": "SCOPE-CHANGE",
  "timestamp": "2026-02-08T16:45:12.789Z",
  "session_id": "1707406512-456789123",
  "from_phase": "PRE-PR",
  "to_phase": "PLANNING",
  "scope_issues": [
    {
      "type": "missing-feature",
      "description": "Batch upload functionality not implemented",
      "impact": "high",
      "requires_planning": true
    }
  ],
  "handoff_doc": "kb/handoffs/20260208-164512_pre-pr_to_planning.md",
  "summary": "Discovered missing batch upload feature during validation"
}
```

2. Create handoff document at `kb/handoffs/YYYYMMDD-HHmmss_pre-pr_to_planning.md`:
```markdown
# PRE-PR → PLANNING Handoff (Scope Change)

**Date**: 2026-02-08 16:45:12
**Session ID**: 1707406512-456789123
**Branch**: feature/implement-image-slice
**Commit**: def789abc123

## Scope Issues Identified

### Issue 1: Missing Batch Upload Feature
**Type**: missing-feature
**Impact**: High
**Requires Planning**: Yes

**Context**:
During validation, discovered that legacy Gradio UI supports batch upload of multiple images, but this was not included in original task definition (TASK-001).

**Current Implementation**:
- Single image upload only
- No batch queue management

**Legacy Behavior**:
- Supports drag-drop of multiple files
- Processes images sequentially
- Shows progress for each image

**Questions for Planning**:
- [ ] Should batch upload be part of TASK-001 or separate task?
- [ ] What is the priority for this feature?
- [ ] Are there dependencies on other components?

**Impact on Current PR**:
- Current implementation is incomplete without batch support
- May need to split into multiple PRs

## Next Steps for Planning
1. Review original requirements for batch upload
2. Decide on task breakdown
3. Update acceptance criteria
4. Provide guidance on PR strategy

---
**Handoff Document**: Immutable after creation
**Next Phase**: PLANNING (scope resolution)
```

### 9. CONDITIONAL: PR Ready Handoff (All Checks Pass)
**Trigger**: All validation checks pass, no blocking issues

Actions:
1. Append PR-READY ledger event:
```json
{
  "event": "PR-READY",
  "timestamp": "2026-02-08T17:00:00.123Z",
  "session_id": "1707406512-456789123",
  "from_phase": "PRE-PR",
  "to_phase": "PR-CREATION",
  "all_checks_passing": true,
  "quality_summary": {
    "python_coverage": 88.5,
    "frontend_coverage": 85.2,
    "total_tests": 69,
    "lint_errors": 0,
    "security_issues": 0
  },
  "pr_description_doc": "kb/handoffs/20260208-170000_pr-description.md",
  "summary": "All validations passed, ready for PR"
}
```

2. Create PR description draft at `kb/handoffs/YYYYMMDD-HHmmss_pr-description.md`:
```markdown
# PR Description: Implement Image Slice with Zustand

## Summary
Implements imageSlice using Zustand with immer middleware for state management. Includes comprehensive tests and achieves 85%+ coverage.

## Related Tasks
- TASK-001: Implement Zustand imageSlice
- TASK-002: Create ImageUpload component

## Changes Made
### Source Files
- `frontend/src/stores/slices/imageSlice.ts` - Core image state management
- `frontend/src/components/ImageUpload.tsx` - Image upload component
- `frontend/src/stores/index.ts` - Store integration

### Test Files
- `frontend/src/stores/slices/imageSlice.test.ts` - Slice tests (95% coverage)
- `frontend/src/components/ImageUpload.test.tsx` - Component tests (87% coverage)

## Quality Metrics
- **Test Coverage**: 85.2% (threshold: 80%)
- **Tests**: 24/24 passing
- **Lint**: 0 errors, 1 warning
- **Typecheck**: 0 errors
- **Security**: 0 issues

## Migration Equivalence
- [x] Functional equivalence verified with legacy Gradio UI
- [x] All acceptance criteria met
- [x] Edge cases tested (upload errors, file validation, size limits)

## Testing Checklist
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Coverage threshold met
- [x] Manual testing completed
- [x] Accessibility audit passed

## Breaking Changes
None

## Dependencies
None

## Rollback Plan
Revert commit def789abc123 if issues discovered post-merge.

## Screenshots
(Add screenshots of ImageUpload component if applicable)

---
**Generated**: 2026-02-08 17:00:00
**Session**: 1707406512-456789123
**Branch**: feature/implement-image-slice
**Commit**: def789abc123
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
Gracefully handle missing validation tools:
```bash
# Python checks
if ! command -v ruff &> /dev/null; then
  echo "ruff not found, skipping Python lint/format"
  lint_status="skip"
  format_status="skip"
fi

if ! command -v mypy &> /dev/null; then
  echo "mypy not found, skipping type check"
  mypy_status="skip"
fi

if ! command -v pytest &> /dev/null; then
  echo "pytest not found, skipping tests"
  pytest_status="skip"
fi

if ! command -v bandit &> /dev/null; then
  echo "bandit not found, skipping security scan"
  bandit_status="skip"
fi

# Frontend checks
if ! command -v pnpm &> /dev/null; then
  echo "pnpm not found, skipping frontend checks"
  frontend_lint_status="skip"
  frontend_typecheck_status="skip"
  frontend_test_status="skip"
fi

# Validation tool
if ! command -v jq &> /dev/null; then
  echo "jq not found, skipping ledger validation"
fi
```

### Missing Frontend Directory
If `frontend/` doesn't exist, skip all frontend checks:
```bash
if [ ! -d "frontend" ]; then
  echo "frontend/ directory not found, skipping frontend checks"
  frontend_lint_status="skip"
  frontend_typecheck_status="skip"
  frontend_format_status="skip"
  frontend_test_status="skip"
  frontend_coverage_status="skip"
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
- All validation checks executed
- Check results captured accurately
- Handoff docs created based on results
- No files overwritten (append-only for ledger)
- Graceful handling of missing tools
