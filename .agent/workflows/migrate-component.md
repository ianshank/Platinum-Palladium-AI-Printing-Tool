---
description: Step-by-step migration of a Gradio component to React + TypeScript + Zustand
---

# Component Migration Workflow

Migrate a single Gradio UI component to React following the established patterns.

## Prerequisites

- Component name identified from `CLAUDE.md` tab mapping
- Legacy Gradio source located in `src/ptpd_calibration/ui/`
- Target store slice identified (or new slice needed)

## Steps

1. **Analyze Legacy Component**
   - Read the Gradio source in `src/ptpd_calibration/ui/gradio_app.py`
   - Document all props, state variables, callbacks, and edge cases
   - Identify backend API endpoints the component calls
   - Note any component dependencies

2. **Create Feature Branch**
   Use the `/create-feature-branch` workflow with component name.

3. **Design React Equivalent**
   - Map Gradio patterns to React hooks per `.claude/agents/ui-migration-agent.md`
   - Design Zustand store slice if needed (follow `stores/slices/` conventions)
   - Plan component composition and props interface
   - Use config-driven values — no hardcoded strings or numbers

4. **Implement Component**
   - Create `frontend/src/components/{ComponentName}/{ComponentName}.tsx`
   - Follow the component template from `ui-migration-agent.md`
   - Add `data-testid` attributes for testing
   - Use `cn()` from `@/lib/utils` for className composition
   - Use Radix UI primitives for accessible interactive elements
   - Include `logger.debug()` calls for development tracing

5. **Implement Store Slice** (if needed)
   - Create `frontend/src/stores/slices/{sliceName}Slice.ts`
   - Follow the slice template from `ui-migration-agent.md`
   - Export typed selectors for component consumption
   - Register in `frontend/src/stores/index.ts`

6. **Write Tests**
   - Create `{ComponentName}.test.tsx` co-located with the component
   - Follow patterns from `.claude/agents/testing-agent.md`
   - Cover: rendering, interactions, error states, loading states
   - Target ≥80% coverage

7. **Run Verification**
   Use the `/verify-all` workflow to confirm nothing is broken.

8. **Create PR**
   Use the `/create-feature-branch` workflow (commit + push + PR step).

## Quality Checklist

- [ ] TypeScript strict mode compliant
- [ ] All props typed with JSDoc
- [ ] Zustand integration with stable selectors
- [ ] Mobile-first Tailwind styling
- [ ] Accessibility attributes (aria-*, role, data-testid)
- [ ] Tests passing (≥80% coverage)
- [ ] No console warnings/errors
- [ ] No hardcoded values
