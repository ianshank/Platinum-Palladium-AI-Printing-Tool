# Platinum-Palladium AI Printing Tool Migration

## Project Overview
Migration from Gradio Python UI to React 18 + TypeScript + Zustand for a digital negative creation tool used in platinum/palladium alternative photographic printing processes.

## Migration Status
- **Current Phase**: Phase 3 - Migration Completion & Hardening
- **Components Migrated**: 12/15
- **Test Coverage**: ~75% (Target: 80%)
- **Started**: 2026-01-24
- **Backend Refactoring**: Complete - Domain-driven modular architecture

---

## Refactored Backend Architecture (NEW)

The backend has been restructured to support a growing feature set with clear domain boundaries, unified configuration, structured logging, and consistent error handling.

### Core Principles
1. **Domain-Driven Organization**: Each feature lives in its own module (curves, detection, ml, agents, ai, etc.)
2. **Configuration-First**: Environment-based settings with pydantic validation
3. **Structured Logging**: JSON-formatted logs with context and tracing
4. **Unified Error Handling**: Custom exception hierarchy for predictable error codes
5. **Module Independence**: Minimal coupling between domains, optional imports for heavy features

### Configuration System (NEW)
- **Location**: `src/ptpd_calibration/config.py` & `config/` domain module
- **Pattern**: Pydantic BaseSettings with validation and defaults
- **Usage**: Environment variables with `PTPD_` prefix override defaults
- **Details**: See [docs/CONFIG_SYSTEM.md](docs/CONFIG_SYSTEM.md)

Example:
```python
from ptpd_calibration.config import get_settings
settings = get_settings()
# Access nested: settings.api.cors_origins, settings.detection.canny_threshold
```

### Logging Architecture (NEW)
- **Location**: `src/ptpd_calibration/core/logging.py`
- **Features**: JSON formatting, context variables, color console output
- **Usage**: `from ptpd_calibration.core.logging import get_logger, setup_logging`
- **Details**: See [docs/LOGGING.md](docs/LOGGING.md)

Example:
```python
logger = get_logger(__name__)
logger.info("Processing started", extra={"record_id": 123})
# Output (JSON): {"timestamp": "...", "level": "INFO", "context": {...}, ...}
```

### Error Handling (NEW)
- **Location**: Domain-specific exception modules (e.g., `detection/exceptions.py`)
- **Pattern**: Custom exception hierarchy with structured messages
- **Usage**: Catch domain-specific exceptions, FastAPI converts to HTTP error codes
- **Details**: See [docs/ERROR_HANDLING.md](docs/ERROR_HANDLING.md)

Example:
```python
from ptpd_calibration.detection.exceptions import TabletDetectionError
try:
    tablet = detector.detect(image)
except TabletDetectionError as e:
    logger.error("Tablet detection failed", extra={"error": str(e)})
```

### Domain Modules
| Module | Purpose | Status |
|--------|---------|--------|
| `curves/` | Curve generation, export, modification | Stable |
| `detection/` | Step tablet detection, density extraction | Stable |
| `ml/` | Machine learning models, predictions | Stable |
| `llm/` | LLM integration (Claude, OpenAI) | Stable |
| `agents/` | Multi-agent orchestration with tool registry | NEW |
| `ai/` | Platinum/Palladium AI analysis | NEW |
| `advanced/` | Advanced features (blending, style transfer) | NEW |
| `workflow/` | Recipe management and workflow automation | NEW |
| `monitoring/` | Performance monitoring and profiling | NEW |
| `calculations/` | Enhanced technical calculations | NEW |
| `data/` | Data management, export/import, sync | NEW |
| `integrations/` | Hardware (printer, spectrophotometer, weather) | NEW |
| `qa/` | Quality assurance and validation | NEW |
| `education/` | Tutorials, glossary, tips | NEW |

---

## Technology Stack

### Source Stack (Gradio)
- **Framework**: Gradio 4.44.0
- **Language**: Python 3.10+
- **State Management**: Gradio implicit state
- **Backend**: FastAPI (embedded in Gradio)
- **Image Processing**: OpenCV, Pillow, NumPy

### Target Stack (React)
- **Framework**: React 18 + TypeScript 5.x (strict mode)
- **Build Tool**: Vite 5.4
- **State Management**: Zustand 4.5 with middleware
- **API Layer**: TanStack Query + Axios
- **Styling**: Tailwind CSS + Radix UI primitives
- **Charts**: Plotly.js (curves), Recharts (histograms)
- **Image Canvas**: Custom Canvas API + react-zoom-pan-pinch
- **Testing**: Vitest + React Testing Library + Playwright

### Backend (Retained/Enhanced)
- **Framework**: FastAPI 0.115
- **Task Queue**: Celery + Redis (for heavy image processing)
- **Image Processing**: OpenCV, Pillow, NumPy (unchanged)
- **AI/ML**: scikit-learn, Anthropic Claude, OpenAI (calibration predictions)

---

## Essential Commands

### Verification Loop (RUN AFTER EVERY CHANGE)
```bash
# Quick verification (single file)
cd frontend && pnpm lint:fix -- src/${CHANGED_FILE}
cd frontend && pnpm typecheck
cd frontend && pnpm test -- --run src/${CHANGED_FILE_TEST}

# Full verification (before commit)
cd frontend && pnpm check:all          # typecheck + lint + format
cd frontend && pnpm test               # full test suite
cd frontend && pnpm build              # build verification
```

### Migration-Specific Commands
```bash
pnpm migrate:status     # Show migration progress dashboard
pnpm migrate:verify     # Run equivalence tests against legacy
pnpm migrate:rollback   # Rollback last migrated component
pnpm test:visual        # Visual regression tests
pnpm test:a11y          # Accessibility audit
```

### Backend Commands
```bash
uvicorn src.ptpd_calibration.api.server:app --reload  # Start FastAPI server
pytest tests/ -v                                        # Backend tests
python -m ptpd_calibration.cli                         # CLI tool
```

---

## Directory Structure - Current (Refactored)

```
/home/user/Platinum-Palladium-AI-Printing-Tool/
├── CLAUDE.md                      # This file - project guide
├── docs/                          # Developer documentation (NEW)
│   ├── ARCHITECTURE.md            # System architecture overview
│   ├── CONFIG_SYSTEM.md           # Configuration management guide
│   ├── ERROR_HANDLING.md          # Error handling patterns
│   ├── LOGGING.md                 # Logging architecture
│   ├── TESTING.md                 # Testing strategies
│   ├── API_TYPES.md               # OpenAPI schema & types
│   ├── MIGRATION_GUIDE.md         # Migration from old structure
│   ├── ai-usage-guide.md          # AI/LLM integration
│   ├── architecture.md            # Detailed C4 diagrams
│   └── migration/                 # Migration documentation
├── .claude/
│   ├── settings.json              # Hooks, permissions, model config
│   ├── agents/                    # Sub-agent definitions
│   └── commands/                  # Custom slash commands
├── frontend/                      # React application (NEW)
│   ├── src/
│   │   ├── components/           # Migrated React components
│   │   ├── stores/               # Zustand stores (slices)
│   │   ├── hooks/                # Custom React hooks
│   │   ├── api/                  # API client (generated types)
│   │   ├── utils/                # Shared utilities
│   │   ├── lib/                  # Core libraries
│   │   └── __tests__/            # Component tests
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
├── src/                           # Python backend (REFACTORED)
│   └── ptpd_calibration/
│       ├── api/                  # FastAPI server & routers
│       │   ├── server.py         # Main FastAPI app
│       │   ├── models.py         # Request/response models
│       │   ├── deep_learning.py  # DL training endpoints
│       │   └── mcts_router.py    # MCTS optimization endpoints
│       ├── config/               # Configuration domain module (NEW)
│       │   └── calculations.py   # Configuration calculations
│       ├── core/                 # Core infrastructure
│       │   ├── logging.py        # Structured logging (NEW)
│       │   ├── models.py         # Data models
│       │   ├── types.py          # Type definitions
│       │   └── ...
│       ├── ui/                   # Gradio UI (legacy, modularized)
│       │   ├── config/           # UI configuration
│       │   ├── handlers/         # Event handlers
│       │   ├── validators/       # Input validation
│       │   ├── tabs/             # Tab components
│       │   └── gradio_app.py     # Main Gradio app
│       ├── curves/               # Curve generation/modification
│       ├── detection/            # Step tablet detection
│       ├── analysis/             # Wedge analysis
│       ├── chemistry/            # Chemistry calculations
│       ├── imaging/              # Image processing
│       ├── llm/                  # LLM integration
│       ├── ml/                   # Machine learning
│       ├── agents/               # Multi-agent system (NEW)
│       ├── ai/                   # AI analysis (NEW)
│       ├── advanced/             # Advanced features (NEW)
│       ├── workflow/             # Recipe & workflow (NEW)
│       ├── monitoring/           # Performance monitoring (NEW)
│       ├── calculations/         # Enhanced calculations (NEW)
│       ├── data/                 # Data management (NEW)
│       ├── integrations/         # Hardware integrations (NEW)
│       ├── qa/                   # Quality assurance (NEW)
│       ├── education/            # Educational content (NEW)
│       └── config.py             # Settings management (REFACTORED)
├── tests/                        # Python tests (EXISTING)
├── kb/                           # Knowledge base (handoffs, ledger)
└── docs/                         # Main documentation
```

### Key Refactoring Changes
- **Domain-driven modules**: Config, AI, Advanced, Workflow, Monitoring moved to separate modules
- **Unified configuration**: `config/` domain with pydantic-settings
- **Structured logging**: `core/logging.py` with JSON formatting
- **Error handling**: Unified exception hierarchy (see docs/ERROR_HANDLING.md)
- **UI modularization**: Gradio UI split into config, handlers, validators, tabs
- **API organization**: Separate routers for deep learning and MCTS

---

## Code Style Guidelines

### TypeScript Standards
- Strict mode with `noImplicitAny`, `strictNullChecks`, `noUncheckedIndexedAccess`
- Explicit return types for all exported functions
- Use `interface` for object shapes, `type` for unions/intersections
- Prefer `const` assertions for literal types

### React Patterns
- Functional components with hooks exclusively
- Co-located tests: `ComponentName.tsx` + `ComponentName.test.tsx`
- Custom hooks for reusable logic: `use{Feature}.ts`
- Prop types via TypeScript interfaces (no PropTypes)

### Zustand Patterns
- One store per domain slice (image, calibration, profile, ui)
- Selectors defined outside components to prevent re-renders
- Use `immer` middleware for complex nested updates
- Persist middleware for user preferences only

### Import Order (auto-enforced by ESLint)
```typescript
// 1. React and framework imports
// 2. External libraries
// 3. Internal absolute imports (@/...)
// 4. Relative imports
// 5. Type-only imports
```

---

## Migration Rules (CRITICAL)

### NEVER Do
- Modify files in `src/ptpd_calibration/ui/` directory (legacy reference)
- Skip the verification loop to "save time"
- Hardcode test values to make tests pass
- Add features not present in the original Gradio app
- Commit legacy and migrated code changes together

### ALWAYS Do
- Verify functional equivalence before marking a component complete
- Write tests FIRST if legacy tests don't exist
- Use feature flags for gradual rollout
- Document behavioral differences immediately
- Run accessibility audit on every new component

### Component Migration Workflow
1. **Analyze**: Read legacy component, document all behaviors
2. **Map**: Create component mapping in `migration/component-map.json`
3. **Test Legacy**: Ensure tests exist for legacy behavior
4. **Implement**: Create React component following target patterns
5. **Test New**: Verify all tests pass on new component
6. **Integrate**: Update parent component imports
7. **Verify**: Run equivalence tests
8. **Document**: Update progress tracker

---

## Gradio → React Component Mapping

| Gradio Component | React Equivalent | Zustand Store Slice | Notes |
|------------------|------------------|---------------------|-------|
| `gr.Textbox` | `<Input type="text">` | `useFormStore` | Add `onKeyDown` for Enter submit |
| `gr.Number` | `<NumberInput>` + react-number-format | `useFormStore` | Validate NaN cases |
| `gr.Slider` | `<Slider>` from @radix-ui/react-slider | `useFormStore` | Min/max from props |
| `gr.Dropdown` | `<Select>` from @radix-ui/react-select | `useFormStore` | Support search filter |
| `gr.Image` | `<ImageUpload>` + react-dropzone | `useImageStore` | URL.createObjectURL for preview |
| `gr.File` | `<FileUpload>` + react-dropzone | `useFileStore` | Progress tracking with XHR |
| `gr.Plot` | `react-plotly.js` (curves) / `recharts` (histograms) | `useChartStore` | See visualization section |
| `gr.Dataframe` | `@tanstack/react-table` | `useDataStore` | Virtual scroll for large data |
| `gr.Tabs` | `<Tabs>` from @radix-ui/react-tabs | `useUIStore` | Persist active tab |
| `gr.Accordion` | `<Accordion>` from @radix-ui/react-accordion | `useUIStore` | Support multiple open |
| `gr.Row/Column` | Flexbox `<div>` with Tailwind | N/A | `flex flex-row/col gap-4` |
| `gr.Button` | `<Button>` with loading state | N/A | Disable during async |
| `gr.Markdown` | `react-markdown` | N/A | Sanitize HTML |
| `gr.Chatbot` | Custom `<ChatInterface>` | `useChatStore` | Streaming support |
| `gr.Gallery` | `<ImageGallery>` | `useGalleryStore` | Lazy load thumbnails |

---

## Tab Component Mapping

| Gradio Tab | React Component | Priority |
|------------|-----------------|----------|
| Dashboard | `<DashboardPage>` | High |
| Calibration Wizard | `<CalibrationWizard>` | Critical |
| Chemistry Calculator | `<ChemistryCalculator>` | High |
| AI Assistant | `<AIAssistant>` | High |
| Session Log | `<SessionLog>` | Medium |
| Curve Display | `<CurveDisplay>` | Critical |
| Curve Editor | `<CurveEditor>` | Critical |
| Step Tablet Analysis | `<StepTabletAnalysis>` | Critical |
| Settings | `<SettingsPage>` | Medium |

---

## Zustand Store Architecture

```typescript
// stores/index.ts - Central store composition
import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

// Slice pattern for large applications
import { createImageSlice, ImageSlice } from './slices/imageSlice';
import { createCalibrationSlice, CalibrationSlice } from './slices/calibrationSlice';
import { createCurveSlice, CurveSlice } from './slices/curveSlice';
import { createChemistrySlice, ChemistrySlice } from './slices/chemistrySlice';
import { createChatSlice, ChatSlice } from './slices/chatSlice';
import { createSessionSlice, SessionSlice } from './slices/sessionSlice';
import { createUISlice, UISlice } from './slices/uiSlice';

type StoreState = ImageSlice & CalibrationSlice & CurveSlice &
                  ChemistrySlice & ChatSlice & SessionSlice & UISlice;

export const useStore = create<StoreState>()(
  devtools(
    subscribeWithSelector(
      immer((...args) => ({
        ...createImageSlice(...args),
        ...createCalibrationSlice(...args),
        ...createCurveSlice(...args),
        ...createChemistrySlice(...args),
        ...createChatSlice(...args),
        ...createSessionSlice(...args),
        ...createUISlice(...args),
      }))
    ),
    { name: 'PtPdPrintingTool' }
  )
);

// Typed selectors (define outside components)
export const selectCurrentImage = (state: StoreState) => state.currentImage;
export const selectCurvePoints = (state: StoreState) => state.calibration.curvePoints;
export const selectActiveTab = (state: StoreState) => state.ui.activeTab;
export const selectIsProcessing = (state: StoreState) => state.ui.isProcessing;
```

---

## Backend API Integration

### FastAPI Endpoint Structure (Existing)
```
/api/
├── /health              GET    # Health check
├── /analyze             POST   # Analyze density measurements
├── /scan/upload         POST   # Upload step tablet scan
├── /curves
│   ├── /generate        POST   # Generate calibration curve
│   ├── /export          POST   # Export curve (QTR, Piezography, CSV, JSON)
│   ├── /upload-quad     POST   # Upload QuadTone RIP file
│   ├── /parse-quad      POST   # Parse quad format
│   ├── /modify          POST   # Modify curve
│   ├── /smooth          POST   # Apply smoothing
│   ├── /blend           POST   # Blend two curves
│   ├── /enhance         POST   # AI-powered curve enhancement
│   └── /{id}            GET    # Retrieve curve
├── /calibrations
│   ├── /                GET/POST  # List/create calibrations
│   └── /{id}            GET    # Get calibration details
├── /chat
│   ├── /                POST   # Chat with LLM assistant
│   ├── /recipe          POST   # Get recipe suggestions
│   └── /troubleshoot    POST   # Troubleshooting help
├── /statistics          GET    # Get statistics
└── /export
    ├── /negative        POST   # Export negative TIFF
    ├── /curve           POST   # Export curve file
    └── /profile         POST   # Export profile JSON
```

---

## Testing Requirements

### Per-Component Requirements
- **Unit Tests**: ≥80% coverage
- **Accessibility**: axe-core audit (0 critical/serious violations)
- **Visual Regression**: Playwright snapshots for key states
- **Integration**: Data flow through Zustand store

### Migration Verification Tests
```typescript
// migration/equivalence-tests/CurveEditor.equiv.test.ts
describe('CurveEditor Equivalence', () => {
  it('produces identical output curve for same input points', async () => {
    const legacyResult = await callLegacyEndpoint(testInputs);
    const newResult = await callNewEndpoint(testInputs);
    expect(newResult.curve).toMatchCurve(legacyResult.curve, { tolerance: 0.001 });
  });
});
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| First Contentful Paint | < 1.5s | Lighthouse |
| Time to Interactive | < 3s | Lighthouse |
| Curve preview update | < 16ms (60fps) | Performance API |
| Image upload progress | Real-time | Manual verification |
| Store update latency | < 5ms | React DevTools |
| Bundle size (gzipped) | < 500KB | `pnpm build` |

---

## Environment Variables

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENABLE_DEVTOOLS=true
VITE_LOG_LEVEL=debug
```

### Backend (.env)
```bash
PTPD_LLM_PROVIDER=anthropic
PTPD_ANTHROPIC_API_KEY=<your-key>
PTPD_OPENAI_API_KEY=<your-key>
PTPD_DATA_DIR=./data
PTPD_LOG_LEVEL=INFO
```

---

## Dynamic Configuration System

The migration uses a configuration-driven approach. All dynamic values are loaded from:

1. **Environment Variables**: Runtime configuration
2. **`migration/config.json`**: Migration-specific settings
3. **`frontend/src/config/index.ts`**: Frontend configuration

### Configuration Priority (highest to lowest)
1. Environment variables
2. Local config files
3. Default values in code

---

## Logging and Debugging

### Frontend Logging
- Use `@/lib/logger` for all logging
- Log levels: DEBUG, INFO, WARN, ERROR
- Structured logging with context
- Performance timing for critical paths

### Backend Logging
- Python `logging` module with structured output
- Request/response logging for API calls
- Performance metrics via `performance.py`

### Debug Mode
- Enable with `VITE_ENABLE_DEVTOOLS=true`
- Zustand DevTools integration
- React Query DevTools
- Network request logging

---

## Gap Analysis Checklist

### Calibration Workflow
- [ ] Step wedge generation (21/31/41 step)
- [ ] Manual density input interface
- [ ] Scanner-based measurement (automated patch sampling)
- [ ] Curve generation algorithm (ChartThrob-equivalent)
- [ ] Curve preview and editing

### AI Integration
- [ ] Model loading and inference pipeline
- [ ] Automatic calibration from scanned prints
- [ ] Print quality prediction
- [ ] Curve recommendation system
- [ ] RAG-powered chat assistant

### Data Persistence
- [ ] Profile CRUD operations
- [ ] Calibration data storage
- [ ] Session logging
- [ ] Export/import (JSON, .acv, .quad)

### Missing Features to Implement
- [ ] Batch processing queue
- [ ] Keyboard shortcuts (replicate Ctrl+1-5)
- [ ] Undo/redo stack
- [ ] Mobile responsive layout
- [ ] Offline mode (PWA)
- [ ] i18n support

---

## Knowledge Base Protocol

This project uses a structured knowledge base in `kb/` for session continuity
across a three-role workflow pipeline: Planning → DEV-SQE → Pre-PR.

### Directory Structure

```
kb/
├── handoffs/                  # Immutable cross-role handoff documents
├── ledger/
│   └── ledger.jsonl           # Append-only event log (shared across all roles)
├── sessions/
│   ├── planning.state.json    # Planning role session state
│   ├── dev-sqe.state.json     # DEV-SQE role session state
│   └── pre-pr.state.json      # Pre-PR role session state
└── summaries/
    ├── planning.md            # Rolling planning summary
    ├── dev-sqe.md             # Rolling dev/sqe summary
    ├── pre-pr.md              # Rolling pre-pr summary
    └── design-contract-readiness.md
```

### Workflow Roles

| Role | Purpose | Upstream | Downstream |
|------|---------|----------|------------|
| **Planning** | Task breakdown, arch decisions, acceptance criteria | — | DEV-SQE |
| **DEV-SQE** | Implementation, testing, quality engineering | Planning | Pre-PR |
| **Pre-PR** | Validation, PR preparation, documentation | DEV-SQE | (PR ready) |

### Handoff Flow
- Planning → DEV-SQE: `PLANNING-HANDOFF` event + handoff doc
- DEV-SQE → Pre-PR: `DEV-SQE-HANDOFF` event + handoff doc
- DEV-SQE → Planning: `DESIGN-GAP` event (feedback loop)
- Pre-PR → DEV-SQE: `QA-GAP` event (feedback loop)
- Pre-PR → Planning: `SCOPE-CHANGE` event (feedback loop)

### Handoff Skills
- Planning: `.claude/skills/planning-handoff/SKILL.md`
- DEV-SQE: `.claude/skills/dev-sqe-handoff/SKILL.md`
- Pre-PR: `.claude/skills/pre-pr-handoff/SKILL.md`

### SessionStart Hooks
- Dispatcher: `.claude/hooks/kb-start.sh` (routes via `$CLAUDE_ROLE` env var)
- Planning: `.claude/hooks/planning-start.sh`
- DEV-SQE: `.claude/hooks/dev-sqe-start.sh`
- Pre-PR: `.claude/hooks/pre-pr-start.sh`

### Configuration (Environment Variables)
- `CLAUDE_ROLE` — Which role to load context for (planning, dev-sqe, pre-pr, all). Default: all
- `KB_DIR` — Override KB directory path. Default: `$CLAUDE_PROJECT_DIR/kb`
- `KB_MAX_SUMMARY_BYTES` — Max bytes per summary. Default: 4000
- `KB_MAX_CROSS_ROLE_BYTES` — Max bytes for cross-role summaries. Default: 2000
- `KB_MAX_LEDGER_LINES` — Recent ledger lines to inject. Default: 15-20
- `KB_MAX_HANDOFFS` — Max handoff docs to load. Default: 3
- `KB_DEBUG` — Enable debug logging (true/false). Default: false

### Rules
1. NEVER overwrite or truncate `kb/ledger/ledger.jsonl` — only append
2. Handoff files in `kb/handoffs/` are immutable once created
3. Session state files (`kb/sessions/*.state.json`) are the ONLY mutable state per role
4. Summary files are append-at-top with a cap of 20 entries
5. All JSONL entries must be valid JSON — validate with `jq` after writing
6. In Agent Teams, designate ONE teammate as the sole ledger writer to avoid race conditions
7. When spawning Agent Teams, each teammate should read relevant KB summaries before starting
8. Every session that does meaningful work MUST run its role's handoff skill before stopping
9. Cross-role handoffs create BOTH a ledger event AND a handoff doc
