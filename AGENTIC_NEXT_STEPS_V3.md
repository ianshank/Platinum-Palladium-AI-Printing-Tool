# Agentic Next Steps Investigation V3

> **Date:** 2026-02-08
> **Branch:** `claude/investigate-agentic-next-steps-Wr5Rv`
> **Purpose:** Updated investigation of next steps based on current codebase state, aligned with universal-dev-agent template milestones

---

## Executive Summary

This investigation consolidates findings from a thorough analysis of the Platinum-Palladium AI Printing Tool codebase as of 2026-02-08. The project has progressed significantly beyond what previous tracking documents reflect. The `progress.json` migration tracker shows 0/15 components complete, but the actual state is approximately **12/15 components implemented** with 592 passing frontend tests and 0 TypeScript errors.

### Current State at a Glance

| Dimension | Status | Evidence |
|-----------|--------|----------|
| **Frontend React Migration** | ~80% complete | 61 TSX files, 7 pages, 7 Zustand slices, 44 test files |
| **Backend API** | Production-ready | 30+ FastAPI endpoints, 168 Python modules |
| **Agentic System** | Milestones 1-3 complete | 6,155 LOC, 4 subagents, orchestrator, message bus |
| **Deep Learning** | 10+ AI systems | Detection, IQA, diffusion, neural curves, federated learning |
| **Test Suite** | Comprehensive | 592 frontend tests passing, 153 backend test files |
| **Code Quality** | Phase B hardened | 0 TypeScript errors, 0 ruff errors, Pydantic V2 migrated |
| **Migration Tracker** | Stale | `progress.json` shows 0/15 - does not reflect reality |

### Delta from Previous Investigations

| Document | Date | Key Gap Identified | Current Status |
|----------|------|--------------------|----------------|
| `AGENTIC_NEXT_STEPS.md` | 2026-02-01 | Agent unit tests (5% coverage) | 78% coverage achieved |
| `INVESTIGATION_SUMMARY.md` | 2026-02-02 | Observability (health, metrics, circuit breakers) | Still a gap |
| `NEXT_STEPS_AGENTIC_DEVELOPMENT.md` | 2026-02 | Real hardware integration | Still critical gap |
| This document | 2026-02-08 | Migration tracker stale + 3 new focus areas | See below |

---

## Consolidated Gap Analysis

### Category A: Migration Infrastructure Debt

These are housekeeping gaps where documentation and tracking have fallen behind actual implementation.

| Gap ID | Description | Impact | Effort |
|--------|-------------|--------|--------|
| **A1** | `progress.json` shows 0/15 components, reality is ~12/15 | Misleading project status | Low |
| **A2** | No equivalence tests in `migration/equivalence-tests/` | Cannot verify Gradio-React parity | Medium |
| **A3** | No `legacy/` symlink created (referenced in CLAUDE.md) | Missing reference during migration | Low |
| **A4** | 1 TODO in CurveEditor.tsx (line 136: save via API) | Incomplete save functionality | Low |

### Category B: Observability & Production Readiness

The agentic system (Milestones 1-3) is architecturally complete but lacks production instrumentation.

| Gap ID | Description | Impact | Effort |
|--------|-------------|--------|--------|
| **B1** | Agent test coverage at 78% (target: 90%) | Risk of regressions | Medium |
| **B2** | No agent health check endpoints | Cannot monitor in production | Medium |
| **B3** | No circuit breaker for LLM calls | No graceful degradation on LLM failure | Medium |
| **B4** | No Prometheus metrics for agent performance | Cannot optimize agent behavior | Medium |
| **B5** | No workflow persistence/checkpoint-resume | Long workflows lost on restart | High |

### Category C: Hardware Integration (Critical Production Gap)

All hardware code is simulated. This is the single largest gap preventing real-world usage.

| Gap ID | Description | Impact | Effort |
|--------|-------------|--------|--------|
| **C1** | Spectrophotometer integration is simulated only | Cannot do real calibrations | High |
| **C2** | Printer integration is simulated only | Cannot print real targets | High |
| **C3** | No device discovery mechanism | Manual device configuration | Medium |
| **C4** | Agent tools cannot control physical devices | Agents cannot run autonomously | Medium |

### Category D: Frontend Feature Gaps

Features referenced in CLAUDE.md but not yet implemented in the React frontend.

| Gap ID | Description | Impact | Effort |
|--------|-------------|--------|--------|
| **D1** | Undo/redo stack for curve editing | Missing UX for iterative editing | Medium |
| **D2** | PWA offline mode | No offline capability | High |
| **D3** | i18n support | English only | High |
| **D4** | Mobile responsive layout (partial) | Desktop-focused only | Medium |
| **D5** | Standalone ExportPanel component | Export embedded in other components | Low |
| **D6** | Standalone ImageUpload/ImagePreview components | Upload embedded in wizard steps | Low |
| **D7** | Standalone Histogram component | No dedicated histogram view | Low |

---

## Prioritized Next Steps Plan

### Sprint 0: Housekeeping (Immediate - Low Risk)

**Goal:** Align project tracking with reality. No functional changes.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Update `progress.json` to reflect actual component status | A1 | migration-coordinator | All 12 implemented components marked complete with accurate test coverage |
| Create equivalence test structure | A2 | testing-agent | At least 3 critical components have equivalence test shells |
| Resolve CurveEditor TODO | A4 | ui-migration-agent | Save via API connected to `/api/curves/{id}` PUT/POST endpoint |

**Definition of Done:**
- `progress.json` accurately reflects current state
- `pnpm migrate:status` shows correct dashboard
- CurveEditor save functionality works end-to-end

---

### Sprint 1: Observability Foundation (Priority 0)

**Goal:** Enable production monitoring for the agentic system.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Implement agent health checks | B2 | gap-remediation-agent | `/api/health/agents` returns status, LLM connectivity, queue depth; response <100ms |
| Implement circuit breakers | B3 | gap-remediation-agent | 3 failures = OPEN, 30s cooldown, auto-recovery; fallback to cached responses |
| Expand agent test coverage to 90% | B1 | testing-agent | `pytest --cov=agents` reports 90%+; edge cases for orchestrator, message bus |
| Add performance metrics | B4 | gap-remediation-agent | Request latency, success rate, token usage tracked; exportable format |

**Definition of Done:**
- Agent system has health monitoring
- LLM failures degrade gracefully (no user-facing errors)
- 90% test coverage for agents module
- Metrics visible in structured logs

---

### Sprint 2: Migration Completion (Priority 1)

**Goal:** Reach 15/15 components migrated and verified.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Extract standalone ExportPanel component | D5 | ui-migration-agent | Dedicated component with format selection, download, progress; tests passing |
| Extract standalone ImageUpload component | D6 | ui-migration-agent | Dropzone, preview, progress tracking; accessible; tests passing |
| Extract standalone ImagePreview component | D6 | ui-migration-agent | Zoom/pan/pinch, overlays; react-zoom-pan-pinch integration; tests passing |
| Create equivalence tests for critical paths | A2 | testing-agent | CurveEditor, CalibrationWizard, ChemistryCalculator equivalence verified |
| Add undo/redo to CurveEditor | D1 | ui-migration-agent | Ctrl+Z/Y works, stack maintained in Zustand; tests passing |

**Definition of Done:**
- 15/15 components in `progress.json` marked complete
- Equivalence tests passing for top 3 critical components
- Undo/redo functional in CurveEditor
- `pnpm check:all && pnpm test && pnpm build` passes

---

### Sprint 3: Hardware Integration (Priority 0 - Critical Path)

**Goal:** Enable real-world calibration workflows with physical devices.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| X-Rite i1 SDK integration | C1 | gap-remediation-agent | USB/serial communication; white/black calibration; density readings within +/- 0.02 |
| Device discovery mechanism | C3 | gap-remediation-agent | Auto-detect connected spectrophotometers; graceful fallback to simulation |
| Hardware agent tools | C4 | gap-remediation-agent | `measure_density` and `print_target` tools in agent registry; mock-tested |
| CUPS/Windows printer integration | C2 | gap-remediation-agent | Print via CUPS (macOS/Linux) or Windows Print API; status querying |

**Definition of Done:**
- Real spectrophotometer readings can be taken
- Agents can autonomously trigger measurements
- Printer can output test targets
- All hardware code has mock-based tests in CI

---

### Sprint 4: Production Readiness (Priority 1)

**Goal:** Workflow persistence, GPU acceleration, streaming.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Workflow persistence | B5 | gap-remediation-agent | State saved per step; resume on restart; cleanup after completion |
| GPU acceleration (CUDA/MPS) | - | gap-remediation-agent | Auto-detect GPU; batch processing 10x speedup; CPU fallback |
| Streaming LLM responses | - | ui-migration-agent | Token streaming to chat UI; progressive updates; SSE transport |
| Tile-based image processing | - | gap-remediation-agent | Handle >100MP images; peak memory <4GB |

---

### Sprint 5+: Enhancement (Priority 2-3)

| Task | Gap IDs | Priority | Notes |
|------|---------|----------|-------|
| Mobile responsive layout | D4 | P2 | Tailwind breakpoint audit |
| PWA offline mode | D2 | P3 | Service worker, IndexedDB cache |
| i18n support | D3 | P3 | react-intl or similar |
| Federated learning activation | - | P3 | Community knowledge sharing |
| External integrations (Lightroom, Photoshop) | - | P4 | Plugin development |

---

## Architecture Alignment with Universal-Dev-Agent Template

### Subagent Mapping

| Template Subagent | Project Implementation | Status |
|-------------------|----------------------|--------|
| **Planner** | `agents/subagents/planner.py` (403 LOC) | Complete |
| **SQE** | `agents/subagents/sqa.py` (503 LOC) | Complete |
| **Coder** | `agents/subagents/coder.py` (518 LOC) | Complete |
| **Reviewer** | `agents/subagents/reviewer.py` (555 LOC) | Complete |
| **Orchestrator** | `agents/orchestrator.py` (572 LOC) | Complete |

### Template Principles Compliance

| Principle | Status | Evidence |
|-----------|--------|----------|
| **No hardcoding** | Compliant | Pydantic-settings, env vars, `config.py` |
| **Backward compatible** | Compliant | Gradio UI preserved, React is additive |
| **Reusable** | Compliant | Modular components, shared libs, tool registry |
| **Observable** | Partial | Structured logging done, metrics/health pending |
| **Test-first** | Partial | 78% agents, 592 frontend tests; target 90% |
| **Self-healing** | Pending | Circuit breakers not yet implemented |

### C4 Architecture Levels

| Level | Status | Document |
|-------|--------|----------|
| **Context (L1)** | Documented | `ARCHITECTURE.md` |
| **Container (L2)** | Documented | `ARCHITECTURE.md` |
| **Component (L3)** | Documented | `ARCHITECTURE.md` |
| **Code (L4)** | Partially documented | Mermaid diagrams in investigation docs |

---

## Success Metrics Dashboard

| Metric | Current | Sprint 0 Target | Sprint 2 Target | Sprint 4 Target |
|--------|---------|-----------------|-----------------|-----------------|
| Components migrated | ~12/15 | 12/15 (accurate) | 15/15 | 15/15 |
| Frontend test count | 592 | 592 | 650+ | 700+ |
| Agent test coverage | 78% | 78% | 90%+ | 95%+ |
| TypeScript errors | 0 | 0 | 0 | 0 |
| Ruff errors | 0 | 0 | 0 | 0 |
| Equivalence tests | 0 | 3 shells | 5 passing | 10 passing |
| Health endpoints | 0 | 0 | 1 (`/health/agents`) | 3 |
| Hardware devices supported | 0 (simulated) | 0 | 0 | 2+ (real) |
| Bundle size (gzipped) | Unknown | Measured | <500KB | <500KB |

---

## Quick Reference: Verification Commands

```bash
# Frontend verification loop
cd frontend && pnpm check:all && pnpm test && pnpm build

# Backend verification loop
PYTHONPATH=src ruff check src/ && PYTHONPATH=src mypy src/ptpd_calibration --ignore-missing-imports && PYTHONPATH=src pytest tests/ -v

# Agent-specific coverage
PYTHONPATH=src pytest tests/unit/test_agent*.py tests/unit/test_subagents.py \
  --cov=src/ptpd_calibration/agents --cov-report=term-missing

# Migration status
cd frontend && pnpm migrate:status
```

---

## Recommended Execution Order

```
Sprint 0 (Housekeeping)
    |
    v
Sprint 1 (Observability) -------> Sprint 3 (Hardware)
    |                                   |
    v                                   v
Sprint 2 (Migration Completion)    Sprint 4 (Production)
    |                                   |
    +-----------------------------------+
    |
    v
Sprint 5+ (Enhancements)
```

Sprints 1 and 3 can run in parallel since they target different subsystems (agents vs. hardware). Sprint 2 depends on Sprint 0's tracking updates. Sprint 4 depends on both Sprint 1 (observability) and Sprint 3 (hardware).

---

## References

| Document | Path | Purpose |
|----------|------|---------|
| Project Guide | `CLAUDE.md` | Coding conventions, migration rules |
| Gap Analysis V1 | `AGENTIC_NEXT_STEPS.md` | Initial agent gap analysis |
| Investigation V1 | `INVESTIGATION_SUMMARY.md` | Detailed implementation findings |
| Next Steps V2 | `NEXT_STEPS_AGENTIC_DEVELOPMENT.md` | Hardware + performance roadmap |
| Deep Learning | `DEEP_LEARNING_IMPLEMENTATION.md` | AI/ML module documentation |
| Migration Tracker | `docs/migration/progress.json` | Component tracking (stale) |
| Component Map | `docs/migration/component-map.json` | Gradio to React mapping |

---

*Generated: 2026-02-08*
*Status: Investigation Complete - Ready for Sprint 0 execution*
