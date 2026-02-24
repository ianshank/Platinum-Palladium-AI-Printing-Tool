---
name: migrate-component
description: Migrate a Gradio component to React
arguments:
  - name: component
    description: The name of the component to migrate (e.g., CurveEditor, Slider)
    required: true
---

# Migrate Component: $ARGUMENTS

Migrate the Gradio component specified to React following best practices.

## Pre-Migration Checklist
1. Verify component exists in legacy codebase
2. Check for existing tests
3. Review dependencies

## Migration Steps

### Step 1: Analyze Legacy Component
Read the Gradio component from `src/ptpd_calibration/ui/` and document:
- All props and their types
- Event handlers and callbacks
- State management patterns
- Visual appearance and styling
- Edge cases and error handling

### Step 2: Check for Existing Tests
Look in `tests/unit/ui/` for existing tests to understand expected behavior.

### Step 3: Create Component Mapping
Add entry to `migration/component-map.json`:
```json
{
  "$ARGUMENTS": {
    "gradioSource": "src/ptpd_calibration/ui/...",
    "reactTarget": "frontend/src/components/$ARGUMENTS/",
    "status": "in-progress",
    "dependencies": [],
    "notes": ""
  }
}
```

### Step 4: Create React Component
Create files:
- `frontend/src/components/$ARGUMENTS/$ARGUMENTS.tsx`
- `frontend/src/components/$ARGUMENTS/$ARGUMENTS.test.tsx`
- `frontend/src/components/$ARGUMENTS/index.ts`
- `frontend/src/components/$ARGUMENTS/README.md` (if complex)

Follow patterns from CLAUDE.md and ui-migration-agent.

### Step 5: Create Store Slice (if needed)
If component has state, create:
- `frontend/src/stores/slices/{slice}Slice.ts`
- `frontend/src/stores/slices/{slice}Slice.test.ts`

### Step 6: Write Tests
- Unit tests with ≥80% coverage
- Accessibility tests with axe-core
- Integration tests with store

### Step 7: Update Progress Tracker
Update `migration/progress.json` with completion status.

### Step 8: Run Verification
```bash
cd frontend && pnpm typecheck
cd frontend && pnpm test -- --run
cd frontend && pnpm build
```

## Quality Requirements
- [ ] TypeScript strict mode compliant
- [ ] All tests passing
- [ ] ≥80% test coverage
- [ ] Accessibility audit passing
- [ ] No console warnings/errors
- [ ] Documentation updated

## Output
Provide a summary including:
- Files created/modified
- Test results
- Coverage report
- Any behavioral differences from legacy
- Recommendations for next steps
