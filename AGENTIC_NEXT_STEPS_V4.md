# Agentic Next Steps Investigation V4

> **Date:** 2026-02-15
> **Branch:** `claude/investigate-agentic-next-steps-MoPtZ`
> **Purpose:** Post-migration investigation of next steps, aligned with universal-dev-agent template milestones
> **Supersedes:** V3 (2026-02-08), V2 (NEXT_STEPS_AGENTIC_DEVELOPMENT.md), V1 (AGENTIC_NEXT_STEPS.md)

---

## Executive Summary

The Platinum-Palladium AI Printing Tool React migration is **100% complete**. All 15 components are migrated, 726 tests pass with ~80% coverage, 0 TypeScript errors, and the production bundle is 280KB gzipped. All 5 identified migration gaps are closed. The project is now in **Phase 4: Post-Migration Hardening & Feature Development**.

This document maps the remaining work against the universal-dev-agent template's 4-milestone structure and identifies concrete next sprints with acceptance criteria.

### What Changed Since V3 (2026-02-08)

| V3 Status | Current Status | Delta |
|-----------|---------------|-------|
| ~12/15 components migrated | **15/15 complete** | +3 components (ImageUpload, ImagePreview, ExportPanel) |
| 592 frontend tests | **726 tests passing** | +134 tests |
| gap-1 open (no equivalence tests) | **CLOSED** | CurveEditor + ScanAnalysis equiv tests |
| gap-3 open (no undo/redo) | **CLOSED** | useUndoRedo hook + CurveEditor integration |
| CurveEditor save TODO | **CLOSED** | Wired to useSaveCurve API mutation |
| progress.json stale (0/15) | **Accurate (15/15)** | Fully reconciled |
| No keyboard shortcuts | **Ctrl+1-5, Ctrl+Z/Y** | Layout-wired, memoized, select-exclusion |
| Desktop-only layout | **Responsive** | px-4 sm:px-6 lg:px-8 on all 7 pages |

### Current Milestone Alignment

| Template Milestone | Status | Notes |
|-------------------|--------|-------|
| **M1: Project Initialization** | COMPLETE | Repo, C4 architecture, CI/CD, KB system |
| **M2: Core Implementation** | COMPLETE | 15/15 components, 7 store slices, API layer |
| **M3: Review and Polish** | **75% COMPLETE** | Code review done; testing gaps remain |
| **M4: Deployment Readiness** | **20% COMPLETE** | CI/CD exists but needs frontend integration |

---

## Current State Dashboard

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Components migrated | 15/15 | 15/15 | DONE |
| Frontend tests | 726 | 700+ | DONE |
| Test coverage | ~80% | 80% | DONE |
| TypeScript errors | 0 | 0 | DONE |
| Bundle size (gzipped) | ~280KB | <500KB | DONE |
| Migration gaps | 0/5 | 0/5 | DONE |
| E2E test coverage | ~10% | 60%+ | GAP |
| Accessibility audit | ~30% | 100% | GAP |
| Visual regression baselines | 0 | 15+ | GAP |
| Performance benchmarks | 0 | 5+ | GAP |
| Agent health checks | 0 | 3+ | GAP |
| Hardware integration | Simulated | Real | GAP |

---

## Remaining Gap Analysis

### Category A: Testing Quality Gaps (Milestone 3 Completion)

The unit/component test layer is solid. The gaps are in higher-order testing tiers.

| Gap ID | Description | Impact | Effort | Priority |
|--------|-------------|--------|--------|----------|
| **A1** | E2E tests limited to 1 basic file (12 test cases) | Cannot verify multi-step workflows end-to-end | Medium | P1 |
| **A2** | No accessibility audit integration (jest-axe/axe-core unused) | WCAG compliance unverified | Low | P1 |
| **A3** | No visual regression baselines (Playwright configured but unused) | Cannot detect visual regressions | Medium | P1 |
| **A4** | No performance benchmarks (Lighthouse, Core Web Vitals) | Cannot track performance regressions | Low | P2 |
| **A5** | `config/tablet.config.ts` has no test file | Untested tablet calibration config | Low | P2 |
| **A6** | `api/client.ts` has no isolated unit tests | Interceptor/error handling untested | Low | P2 |
| **A7** | WebSocket/SSE streaming not tested (chat) | Streaming reliability unverified | Medium | P2 |

### Category B: Production Readiness Gaps (Milestone 4)

| Gap ID | Description | Impact | Effort | Priority |
|--------|-------------|--------|--------|----------|
| **B1** | No agent health check endpoints | Cannot monitor agent system in production | Medium | P1 |
| **B2** | No circuit breakers for LLM/API calls | No graceful degradation on failures | Medium | P1 |
| **B3** | No workflow persistence/checkpoint-resume | Long workflows lost on restart | High | P2 |
| **B4** | No Prometheus/OpenTelemetry metrics | Cannot optimize or alert | Medium | P2 |
| **B5** | Frontend CI pipeline missing (only backend in workflows) | Frontend changes not gated | Medium | P1 |
| **B6** | No Docker containerization for frontend | Cannot deploy consistently | Medium | P2 |

### Category C: Feature Gaps (Post-Migration Enhancement)

Features listed in CLAUDE.md gap analysis that remain unimplemented.

| Gap ID | Description | Impact | Effort | Priority |
|--------|-------------|--------|--------|----------|
| **C1** | Batch processing queue UI | Cannot queue multiple operations | High | P2 |
| **C2** | PWA offline mode | No offline capability | High | P3 |
| **C3** | i18n internationalization | English only | High | P3 |
| **C4** | Real hardware integration (spectrophotometer, printer) | Cannot do real calibrations | Very High | P1 |
| **C5** | OpenAPI type generation not connected | Types manually maintained | Low | P2 |
| **C6** | Chat streaming (SSE/WebSocket) in UI | No progressive response rendering | Medium | P2 |

---

## Prioritized Sprint Plan

### Sprint 3: Testing Excellence (Milestone 3 Completion)

**Goal:** Close testing gaps A1-A3 to complete Milestone 3 (Review & Polish).

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Expand E2E tests with Playwright | A1 | testing-agent | 5+ workflow tests: calibration flow, curve editing, export, settings, chat. All 3 browsers pass. |
| Add axe-core accessibility audits | A2 | testing-agent | Every page component has axe-core test. 0 critical/serious violations. |
| Create visual regression baselines | A3 | testing-agent | Playwright snapshots for all 8 pages in light mode. CI integration for diff detection. |
| Add tablet.config.ts and client.ts tests | A5, A6 | testing-agent | 100% coverage on tablet config validation. Client interceptors tested in isolation. |

**Definition of Done:**
- `pnpm test:visual` creates and validates baselines
- `pnpm test:a11y` reports 0 critical violations
- E2E suite covers core user journeys
- All CI checks pass

---

### Sprint 4: Frontend CI & Production Pipeline (Milestone 4 Start)

**Goal:** Gate frontend changes with automated checks and prepare deployment.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Add frontend CI job to GitHub Actions | B5 | gap-remediation-agent | `pnpm check:all && pnpm test:run && pnpm build` runs on every PR. Matrix: Node 18/20. |
| Add agent health check endpoint | B1 | gap-remediation-agent | `GET /api/health` returns LLM connectivity, queue depth, uptime. Response <100ms. |
| Implement circuit breakers for LLM calls | B2 | gap-remediation-agent | 3 consecutive failures = OPEN state. 30s cooldown. Auto-recovery. Fallback to cached responses. |
| Connect OpenAPI type generation | C5 | gap-remediation-agent | `pnpm generate:types` produces valid TypeScript from FastAPI schema. Types used in API hooks. |

**Definition of Done:**
- PRs cannot merge without passing frontend checks
- `/api/health` returns structured status
- LLM failures degrade gracefully with user-visible status
- API types are generated, not hand-written

---

### Sprint 5: Chat Streaming & Real-Time Features

**Goal:** Enable real-time interaction patterns.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Implement SSE streaming for chat responses | C6 | ui-migration-agent | Tokens stream progressively to chat UI. Loading indicator during generation. Cancel button works. |
| Add WebSocket connection for processing status | - | ui-migration-agent | Real-time progress updates for image processing. Reconnection logic with backoff. |
| Test streaming reliability | A7 | testing-agent | Streaming interruption recovery tested. Network failure simulation. |

**Definition of Done:**
- Chat responses appear token-by-token
- Processing progress visible in real-time
- Network interruptions handled gracefully

---

### Sprint 6: Observability & Monitoring (Milestone 4 Completion)

**Goal:** Full production observability.

| Task | Gap IDs | Subagent | Acceptance Criteria |
|------|---------|----------|---------------------|
| Add structured metrics (OpenTelemetry) | B4 | gap-remediation-agent | Request latency, error rates, LLM token usage tracked. Dashboard-ready format. |
| Implement workflow persistence | B3 | gap-remediation-agent | Calibration workflow state persisted per step. Resume on page reload. Cleanup on completion. |
| Performance benchmarks in CI | A4 | testing-agent | Lighthouse CI scores tracked. Bundle size regression alerts. Curve preview <16ms measured. |
| Docker multi-stage build | B6 | gap-remediation-agent | `docker build` produces <200MB image. Health check in Dockerfile. GPU support via nvidia base. |

**Definition of Done:**
- Metrics exportable to Prometheus/Grafana
- Multi-step workflows survive page reloads
- CI fails on performance regressions
- Docker image builds and runs

---

### Sprint 7+: Feature Enhancement (Post-Milestone 4)

| Task | Gap IDs | Priority | Notes |
|------|---------|----------|-------|
| Batch processing queue UI | C1 | P2 | Queue panel in sidebar, progress indicators, cancel/retry |
| Real hardware integration | C4 | P1 | X-Rite i1 SDK, device discovery, CUPS printing |
| PWA offline mode | C2 | P3 | Service worker, IndexedDB cache, sync on reconnect |
| i18n internationalization | C3 | P3 | react-intl, message extraction, 2+ languages |

---

## Architecture Alignment: Universal-Dev-Agent Template

### Template Milestone → Project Mapping

| Template Milestone | Project Phase | Status | Remaining |
|-------------------|---------------|--------|-----------|
| **M1: Project Initialization** | Phase 1: Migration Infrastructure | COMPLETE | — |
| **M2: Core Implementation** | Phase 2-3: Component Migration | COMPLETE | — |
| **M3: Review and Polish** | Sprint 3 (Testing Excellence) | 75% | E2E, a11y, visual regression |
| **M4: Deployment Readiness** | Sprints 4-6 (CI, Observability, Docker) | 20% | CI, health checks, Docker |

### Template Subagent → Project Subagent Mapping

| Template Role | Project Agent | Current Use |
|---------------|---------------|-------------|
| **Planner** | migration-coordinator | Migration planning → Sprint planning |
| **SQE** | testing-agent | Test generation, equivalence verification |
| **Coder** | ui-migration-agent | Component implementation |
| **Reviewer** | gap-remediation-agent | Gap analysis, code hardening |
| **Orchestrator** | (Claude Code itself) | Multi-agent coordination |

### Template Principles Compliance

| Principle | Status | Evidence |
|-----------|--------|----------|
| **No hardcoding** | COMPLIANT | Config-driven values, env vars, Zustand middleware |
| **Backward compatible** | COMPLIANT | Gradio UI preserved, React additive, feature flags |
| **Reusable** | COMPLIANT | 14 Radix primitives, 6 UI components, 5 custom hooks, 7 store slices |
| **Observable** | PARTIAL | Structured logging done; metrics/health endpoints pending (Sprint 4-6) |
| **Test-first** | COMPLIANT | 726 tests, ~80% coverage, equivalence framework |
| **Self-healing** | PENDING | Circuit breakers not yet implemented (Sprint 4) |

---

## Execution Order

```
Current State (15/15 migrated, 726 tests)
    │
    ▼
Sprint 3: Testing Excellence ──────── Completes Milestone 3
    │
    ▼
Sprint 4: Frontend CI + Health ─┐
    │                           ├── Completes Milestone 4
Sprint 5: Chat Streaming        │
    │                           │
Sprint 6: Observability + Docker┘
    │
    ▼
Sprint 7+: Feature Enhancement ──── Post-milestone work
    ├── Batch processing
    ├── Hardware integration
    ├── PWA offline
    └── i18n
```

Sprints 4 and 5 can run in parallel (CI/health targets backend/infra; streaming targets frontend/API). Sprint 6 depends on Sprint 4's health endpoint foundation.

---

## Recommended Immediate Actions

1. **Start Sprint 3** — Testing Excellence is the highest-leverage next step:
   - Expands E2E coverage from 10% to 60%+
   - Adds accessibility audits (WCAG compliance)
   - Creates visual regression baselines
   - Closes Milestone 3

2. **Parallel: Sprint 4 CI setup** — Frontend CI can be added independently:
   - Add `frontend-ci` job to `.github/workflows/ci-cd.yml`
   - Gate PRs on typecheck + lint + test + build

3. **Update CLAUDE.md** — Reflect Phase 4 status:
   - Change "Current Phase" from Phase 3 to Phase 4
   - Update "Components Migrated" from 12/15 to 15/15
   - Update "Test Coverage" from ~75% to ~80%

---

## Success Metrics by Sprint

| Metric | Current | Sprint 3 | Sprint 4 | Sprint 6 | Sprint 7+ |
|--------|---------|----------|----------|----------|------------|
| E2E tests | 12 | 50+ | 50+ | 50+ | 60+ |
| A11y violations | Unknown | 0 critical | 0 critical | 0 critical | 0 critical |
| Visual baselines | 0 | 15+ | 15+ | 15+ | 20+ |
| CI gates frontend | No | No | Yes | Yes | Yes |
| Health endpoints | 0 | 0 | 1 | 3 | 3 |
| Circuit breakers | 0 | 0 | 1 | 3 | 3 |
| Docker image | No | No | No | Yes | Yes |
| Chat streaming | No | No | No | Yes | Yes |
| Bundle regression CI | No | No | No | Yes | Yes |

---

## File References

| Document | Path | Status |
|----------|------|--------|
| This document | `AGENTIC_NEXT_STEPS_V4.md` | Current |
| V3 Investigation | `AGENTIC_NEXT_STEPS_V3.md` | Superseded |
| V2 Investigation | `NEXT_STEPS_AGENTIC_DEVELOPMENT.md` | Superseded |
| V1 Investigation | `AGENTIC_NEXT_STEPS.md` | Superseded |
| Project Guide | `CLAUDE.md` | Needs Phase 4 update |
| Migration Progress | `docs/migration/progress.json` | Accurate (15/15) |
| Component Map | `docs/migration/component-map.json` | Complete |
| KB Ledger | `kb/ledger/ledger.jsonl` | 4 events logged |
| Dev-SQE State | `kb/sessions/dev-sqe.state.json` | Last: 2026-02-09 |

---

*Generated: 2026-02-15*
*Status: Investigation Complete — Ready for Sprint 3 execution*
