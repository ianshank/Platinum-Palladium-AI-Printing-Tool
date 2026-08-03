# Planning → DEV-SQE Handoff

**Date**: 2026-08-03 01:54:36  
**Session ID**: 1785722076-867973585  
**Branch**: claude/code-hygiene-modularity-9vjyci  
**Workflow**: wf_59d58118-81b (10 specialized agents, 3 phases)

---

## Executive Summary

Comprehensive code hygiene and modularity audit identified 40+ issues across frontend, backend, and shared utilities. Designed detailed 3-phase refactoring roadmap with 10 implementation tasks, 6 ADRs, and 12 quick wins. Multi-agent orchestration workflow launched and actively executing Phase 1 (Foundation).

**All tasks are ready for implementation via coordinated agent workflow.**

---

## Phase 1: Foundation Tasks (Executing Now)

### PHASE1-CONFIG: Split config.py into domain modules
**Priority**: CRITICAL  
**Files to Create/Modify**:
- `/src/ptpd_calibration/config/` (new directory)
  - `config/__init__.py` (compose, maintain get_settings())
  - `config/detection.py` (DetectionSettings, ExtractionSettings)
  - `config/curves.py` (CurveSettings)
  - `config/ml.py` (MLSettings, DeepLearningSettings)
  - `config/llm.py` (LLMSettings, AgentSettings)
  - `config/chemistry.py` (ChemistrySettings, process-specific)
  - `config/integration.py` (IntegrationSettings, VertexAISettings, APISettings)
  - `config/visualization.py` (VisualizationSettings)
  - `config/advanced.py` (AdvancedFeaturesSettings, NeuroSymbolicSettings, EducationSettings)
  - `config/performance.py` (PerformanceSettings, DataManagementSettings)
  - `config/calculations.py` (CalculationsSettings, WedgeAnalysisSettings, WorkflowSettings)
  - `config/qa.py` (QASettings)
- Delete: `/src/ptpd_calibration/config.py` (after migration complete)

**Acceptance Criteria**:
- [ ] All 27 Settings classes migrated to domain modules
- [ ] Backwards compatible: `get_settings()` interface unchanged
- [ ] Full type hints on all Settings classes
- [ ] Comprehensive docstrings on each domain
- [ ] Configuration loading tests pass
- [ ] Environment variable precedence working correctly
- [ ] All existing code imports still work

**Dependencies**: None (foundation task)

**Architectural Decisions**:
- **ADR-001**: Domain-specific config modules for maintainability and discoverability

**Agent Assignment**: Agent "backend:config-split" (Explore agent with specialized schema)

### PHASE1-ERRORS: Unify error handling with exception hierarchy
**Priority**: CRITICAL  
**Files to Create/Modify**:
- Create: `/src/ptpd_calibration/exceptions.py` (unified hierarchy)
  - Base: `PtPdException` with error_code, message, context, timestamp
  - Specific: `ValidationError`, `HardwareError`, `ProcessingError`, `ConfigError`, `APIError`, `DataError`, `NotFoundError`
- Modify: `/src/ptpd_calibration/ui/gradio_app.py` (replace 47 broad exception handlers)
- Consolidate: `/integrations/hardware/exceptions.py` → unified system
- Consolidate: `/ml/deep/exceptions.py` → unified system
- Update: `/api/server.py` (convert exceptions to HTTP responses with error_code)

**Acceptance Criteria**:
- [ ] Unified exception hierarchy with 7+ specific exception types
- [ ] All exceptions have: error_code, message, context, timestamp
- [ ] Replace top 15 critical broad exception handlers with specific catching
- [ ] All exceptions log structured context via logging system
- [ ] API error responses include error_code field
- [ ] Existing exception interfaces maintained where used
- [ ] Exception consolidation tests pass

**Dependencies**: None (parallel to Phase 1)

**Architectural Decisions**:
- **ADR-002**: Unified exception hierarchy with error_code field for consistency and debugging

**Agent Assignment**: Agent "backend:error-consolidation"

### PHASE1-LOGGING: Unify frontend/backend logging
**Priority**: HIGH  
**Files to Create/Modify**:
- Update: `/src/ptpd_calibration/core/logging.py` (unified backend logger)
- Create: `/src/ptpd_calibration/config/logging.py` (logging configuration domain)
- Merge: `/src/ptpd_calibration/agents/logging.py` → unified core/logging.py
- Update: `/frontend/src/lib/logger.ts` (match backend JSON format)
- Consolidate all logging configuration

**Acceptance Criteria**:
- [ ] Single logging architecture in backend (core/logging.py)
- [ ] Structured JSON output: timestamp (ISO 8601), level, logger, message, context, trace_id
- [ ] Frontend logger matches backend format
- [ ] All log levels consistent (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- [ ] Configuration via logging config domain
- [ ] Child logger support for source prefixing
- [ ] Performance timing decorators working
- [ ] Logging tests pass

**Dependencies**: None (parallel to Phase 1)

**Architectural Decisions**:
- **ADR-003**: Structured JSON logging across frontend/backend for unified observability

**Agent Assignment**: Agent "shared:logging-unification"

### PHASE1-OPENAPI: Generate OpenAPI schema and TypeScript types
**Priority**: CRITICAL  
**Files to Create/Modify**:
- Verify: `/src/ptpd_calibration/api/server.py` (all endpoints have type hints)
- Generate: `/openapi.json` (from FastAPI app)
- Create: `/frontend/src/types/api.ts` (generated from schema)
- Add: `pnpm generate:api` script to package.json
- Update: API hooks to use generated types

**Acceptance Criteria**:
- [ ] All API endpoints have complete type hints
- [ ] OpenAPI schema exported and validated (OpenAPI 3.0.0)
- [ ] All enums included in schema
- [ ] Error responses include error_code field
- [ ] TypeScript type generation working
- [ ] Generated types include all endpoints, parameters, responses
- [ ] Eliminates manual enum duplication
- [ ] Type generation tests pass

**Dependencies**: None (enables Phase 2)

**Architectural Decisions**:
- **ADR-004**: OpenAPI schema as source of truth for type generation

**Agent Assignment**: Agent "shared:openapi-generation"

---

## Phase 2: Code Organization Tasks (Queued)

### PHASE2-GRADIO: Modularize Gradio UI (4,337 lines)
**Priority**: CRITICAL  
**Scope**: Split `/src/ptpd_calibration/ui/gradio_app.py` into component modules

**Agent Assignment**: Agent "backend:gradio-split"

### PHASE2-CALCS: Consolidate duplicate calculations
**Priority**: HIGH  
**Scope**: Merge duplicate functions (test_strip_exposure, uv_exposure, drying_time)

**Agent Assignment**: Agent "backend:deduplicate-calculations"

### PHASE2-TYPES: Add return type hints
**Priority**: HIGH  
**Scope**: Complete type hints for: server.py, curves/, detection/, imaging/, ai/, deep_learning/, neuro_symbolic/

**Agent Assignment**: Agent "backend:type-hints"

### PHASE2-FRONTEND: Frontend refactoring
**Priority**: HIGH  
**Scope**: Barrel exports (11 dirs), component splitting (CurveEditor 592L), API hook factories, styled components

**Agent Assignment**: Agent "frontend:organization"

---

## Phase 3: Validation & Testing Tasks (Queued)

### PHASE3-VALIDATION: Type safety and coverage validation
**Priority**: HIGH  
**Scope**: Type checking (pyright, pylint), test coverage (pytest, vitest), performance validation

**Agent Assignment**: Agent "validation:type-safety"

### PHASE3-DOCS: Documentation updates
**Priority**: MEDIUM  
**Scope**: Update CLAUDE.md, create architecture guides, migration guide for developers

**Agent Assignment**: Agent "validation:documentation"

---

## Critical Context

### Current Code State
- **Frontend**: React 18 + TypeScript, Zustand stores, Vitest tests, ~726 tests passing
- **Backend**: FastAPI, Python 3.10+, Pytest tests, ~104 tests in api/ passing
- **Issues**: 40+ code hygiene issues identified (type duplication, large files, duplication, configuration fragmentation)

### Key Constraints
- ✅ **Backwards Compatible**: All refactoring must maintain existing interfaces
- ✅ **No Hardcoded Values**: Configuration-driven throughout
- ✅ **No Breaking Changes**: Existing imports should continue working
- ✅ **Type Safety**: Full type hints required for all new/refactored code
- ✅ **Testing**: 80%+ coverage on new code, existing tests must pass

### Architecture Decisions (ADRs)
1. **ADR-001**: Config domain split → 11 modules
2. **ADR-002**: Unified exception hierarchy → single error system
3. **ADR-003**: Structured JSON logging → both frontend/backend
4. **ADR-004**: OpenAPI schema → TypeScript type generation
5. **ADR-005**: Modularize Gradio UI → components + handlers
6. **ADR-006**: Factory pattern for API hooks → reduce boilerplate

### Implementation Notes for DEV-SQE
1. Each Phase 1 task can run in parallel (they're independent)
2. Phase 2 depends on Phase 1 completion (type hints needed for some refactoring)
3. Phase 3 validates all changes from Phases 1-2
4. Use structured logging when refactoring to track progress
5. Run verification loop after each component change (CLAUDE.md guidelines)
6. Commit frequently with clear messages describing each logical change
7. All tests must pass before moving to next phase

### Files to Reference
- Plan: `/root/.claude/plans/please-scan-code-base-idempotent-origami.md`
- CLAUDE.md: Project guidelines, verification loop commands
- Ledger: `kb/ledger/ledger.jsonl` (append-only task tracking)
- Summary: `kb/summaries/planning.md` (rolling documentation)

### Quick Wins (Consider First)
Low-effort improvements that can be done in parallel:
1. Add barrel exports to 11 component directories (5-10 min each)
2. Replace `clamp01()` with `clamp(value, 0, 1)` (2 min)
3. Extract wizard styled components (10 min)
4. Extract curve generation logic in Step4Preview (10 min)
5. Fix top 10 broad exception handlers (30 min)

---

**Handoff Status**: COMPLETE  
**Tasks Ready**: 10 (all with clear acceptance criteria)  
**Workflow Running**: Yes (wf_59d58118-81b)  
**Next Phase**: DEV-SQE implementation  

---

**This document is immutable** — changes are tracked in kb/ledger/ledger.jsonl

Created: 2026-08-03T01:54:36.000Z  
Session: 1785722076-867973585
