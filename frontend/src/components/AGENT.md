# Components Directory

## Purpose
React UI components for the Pt/Pd Printing Tool, organized by feature domain. Each subdirectory groups related components with co-located tests.

## Key Subdirectories
- `calibration/` — CalibrationWizard, ScanUpload (step tablet workflow)
- `curves/` — CurveEditor, CurveUpload (curve visualization and editing)
- `wizard/` — 5-step calibration wizard (Step1Upload → Step5Export)
- `chemistry/` — ChemistryCalculator (Pt/Pd/cyanotype/silver gelatin)
- `assistant/` — AI chat interface
- `mcts/` — Monte Carlo Tree Search calibration UI
- `dashboard/` — Dashboard widgets and statistics
- `ui/` — Shared primitives (Button, Input, Select, Slider, Tabs, etc.)
- `upload/` — FileUpload, ImageUpload (react-dropzone based)
- `preview/` — Image preview with zoom/pan
- `export/` — Export format selection and download
- `session/` — Session log viewer
- `settings/` — Application settings panel
- `Layout/` — App shell (Sidebar, Header, MainContent)

## Conventions
- **Co-located tests**: Every `Component.tsx` should have a `Component.test.tsx` beside it
- **Functional components only**: Use hooks, never class components
- **Props via interfaces**: Define `interface ComponentNameProps {}` above the component
- **Accessibility required**: Run axe-core audit (0 critical/serious violations). See CLAUDE.md "Testing Requirements"
- **Gradio mapping**: See CLAUDE.md "Gradio → React Component Mapping" table for migration reference

## Patterns
- Shared primitives live in `ui/` — import from `@/components/ui/`
- Domain components import store selectors from `@/stores` — define selectors outside components
- Use Radix UI for interactive primitives (Tabs, Select, Slider, Accordion)
- Tailwind CSS for styling — avoid inline styles except for dynamic Canvas dimensions

## Testing
```bash
pnpm test -- --run src/components/{subdirectory}/{Component}.test.tsx
pnpm test:a11y  # accessibility audit
```

## Pitfalls
- Do NOT put business logic in components — extract to hooks or store actions
- Do NOT create new shared primitives without checking `ui/` first
- Canvas-based components (CurveEditor, preview) need cleanup in `useEffect` return

## Related
- `../stores/` — State management (Zustand slices)
- `../hooks/` — Domain hooks that compose API + store logic
- `../pages/` — Page-level components that compose these components
