---
name: planning-handoff
description: >
  Planning session handoff protocol. Captures task breakdowns, architecture
  decisions, acceptance criteria, and dependency maps. Updates ledger, summary,
  and state files. Produces handoff documents to DEV-SQE when tasks are ready
  for implementation.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Planning Handoff Skill

## Purpose
Captures planning session artifacts and creates structured handoff documents for downstream phases (DEV-SQE). Maintains persistent knowledge about task definitions, architectural decisions, and dependencies.

## Protocol Steps

### 1. Gather Planning Artifacts
Collect artifacts from the current planning session:
```bash
git diff --name-only HEAD -- \
  plan/ spec/ rfc/ design/ arch/ adr/ \
  migration/ \
  CLAUDE.md \
  .claude/agents/
```

Capture:
- Task definitions and breakdowns
- Architectural decisions (ADRs)
- Design specifications
- Acceptance criteria
- Dependency maps
- RFC documents
- Migration plans

### 2. Generate Session ID
```bash
session_id=$(date +%s-%N)
```

### 3. Append PLANNING Ledger Event
Append to `kb/ledger/ledger.jsonl` (create if missing):

```json
{
  "event": "PLANNING",
  "timestamp": "2026-02-08T12:34:56.789Z",
  "session_id": "1707395696-123456789",
  "tasks_defined": [
    {
      "id": "TASK-001",
      "title": "Task description",
      "status": "ready-for-dev|in-progress|blocked",
      "acceptance_criteria": ["criterion 1", "criterion 2"]
    }
  ],
  "decisions_made": [
    {
      "id": "ADR-001",
      "title": "Decision title",
      "status": "accepted|proposed|deprecated",
      "rationale": "Brief explanation"
    }
  ],
  "dependencies_identified": {
    "upstream": ["dependency-1"],
    "downstream": ["dependent-1"],
    "blocked_by": []
  },
  "docs_produced": [
    "migration/component-map.json",
    "adr/001-zustand-architecture.md"
  ],
  "summary": "Session summary (max 200 chars)"
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

### 5. Update Planning Summary
Update `kb/summaries/planning.md` (create if missing):
- Prepend new entry at TOP
- Cap at 20 most recent entries
- Format:
```markdown
## [YYYY-MM-DD HH:mm:ss] Session 1707395696-123456789

**Tasks Defined**: 3 (2 ready-for-dev, 1 in-progress)
**Decisions Made**: 2 (ADR-001, ADR-002)
**Dependencies**: 4 identified
**Artifacts**: migration/component-map.json, adr/001-zustand-architecture.md

### Summary
Brief session summary...

### Tasks Ready for Dev
- TASK-001: Implement Zustand imageSlice
- TASK-002: Create CurveEditor component

---
```

### 6. Update Planning State
Write `kb/sessions/planning.state.json`:
```json
{
  "last_session_id": "1707395696-123456789",
  "last_timestamp": "2026-02-08T12:34:56.789Z",
  "active_branch": "feature/planning-phase-1",
  "resume_pointers": {
    "last_planning_doc": "migration/component-map.json",
    "last_adr": "adr/001-zustand-architecture.md",
    "context_files": [
      "CLAUDE.md",
      "migration/progress.json"
    ]
  },
  "tasks_ready_for_dev": [
    "TASK-001",
    "TASK-002"
  ],
  "tasks_in_progress": [
    "TASK-003"
  ],
  "tasks_blocked": []
}
```

### 7. CONDITIONAL: Planning-to-Dev Handoff
**Trigger**: One or more tasks reached "ready-for-dev" status

Actions:
1. Append PLANNING-HANDOFF ledger event:
```json
{
  "event": "PLANNING-HANDOFF",
  "timestamp": "2026-02-08T12:34:56.789Z",
  "session_id": "1707395696-123456789",
  "from_phase": "PLANNING",
  "to_phase": "DEV-SQE",
  "tasks_handed_off": ["TASK-001", "TASK-002"],
  "handoff_doc": "kb/handoffs/20260208-123456_planning_to_dev-sqe.md",
  "summary": "Handoff summary"
}
```

2. Create handoff document at `kb/handoffs/YYYYMMDD-HHmmss_planning_to_dev-sqe.md`:
```markdown
# Planning → DEV-SQE Handoff

**Date**: 2026-02-08 12:34:56
**Session ID**: 1707395696-123456789
**Branch**: feature/planning-phase-1

## Tasks Ready for Implementation

### TASK-001: Implement Zustand imageSlice
**Priority**: High
**Acceptance Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2

**Files to Create/Modify**:
- frontend/src/stores/slices/imageSlice.ts
- frontend/src/stores/slices/imageSlice.test.ts

**Dependencies**: None

**Architectural Decisions**:
- ADR-001: Use Zustand with immer middleware

### TASK-002: Create CurveEditor component
...

## Context Files
- CLAUDE.md
- migration/component-map.json
- adr/001-zustand-architecture.md

## Dependencies
- TASK-001 must complete before TASK-005

## Notes for DEV-SQE
- Follow test-first approach
- Verify functional equivalence with legacy
- Run verification loop after each change

---
**Handoff Document**: Immutable after creation
**Next Phase**: DEV-SQE implementation
```

**Handoff files are immutable** - never modify after creation.

### 8. CONDITIONAL: Design Gap or Scope Change Handoff
**Triggers**:
- DESIGN-GAP: Missing specifications, unclear requirements
- SCOPE-CHANGE: Requirements changed, new features identified

Actions:
1. Append appropriate ledger event (DESIGN-GAP or SCOPE-CHANGE)
2. Update `kb/decisions/design-contract-readiness.md` with gap details
3. Create handoff doc if gap requires upstream input

## Backwards Compatibility

### Missing Directories
If `kb/` structure doesn't exist, create:
```bash
mkdir -p kb/ledger kb/summaries kb/sessions kb/handoffs kb/decisions
```

### Missing Ledger
If `kb/ledger/ledger.jsonl` doesn't exist, create with header comment:
```bash
echo '{"_comment":"Persistent Knowledge-Base Ledger - Append-only event log"}' > kb/ledger/ledger.jsonl
```

### Missing Tools
- If `jq` not available, skip validation but log warning
- If `git` not available, use filesystem timestamps instead

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
- No files overwritten (append-only for ledger)
