---
name: verify-equivalence
description: Verify functional equivalence between legacy Gradio and React component
arguments:
  - name: component
    description: The name of the component to verify
    required: true
---

# Verify Equivalence: $ARGUMENTS

Verify that the migrated React component is functionally equivalent to the legacy Gradio component.

## Verification Steps

### Step 1: Locate Components
- Legacy: `src/ptpd_calibration/ui/` or `src/ptpd_calibration/ui/tabs/`
- React: `frontend/src/components/$ARGUMENTS/`

### Step 2: Run Unit Tests
```bash
cd frontend && pnpm test -- --run --coverage src/components/$ARGUMENTS
```

Expected: All tests pass with ≥80% coverage.

### Step 3: Run Equivalence Tests
```bash
cd frontend && pnpm test -- --run migration/equivalence-tests/$ARGUMENTS.equiv.test.ts
```

### Step 4: Accessibility Audit
```bash
cd frontend && pnpm test:a11y -- --component=$ARGUMENTS
```

Expected: 0 critical or serious violations.

### Step 5: Visual Comparison
Run Playwright visual regression:
```bash
cd frontend && pnpm test:visual -- --grep "$ARGUMENTS"
```

### Step 6: Behavioral Verification
Manually verify:
- [ ] All props work as expected
- [ ] Event handlers fire correctly
- [ ] State updates propagate properly
- [ ] Error states display correctly
- [ ] Loading states display correctly
- [ ] Disabled state works properly
- [ ] Keyboard navigation works
- [ ] Screen reader announces correctly

### Step 7: Performance Check
- [ ] Render time < 16ms (60fps)
- [ ] No unnecessary re-renders
- [ ] Memory stable under repeated use

## Report Format

```markdown
## Equivalence Report: $ARGUMENTS

### Test Results
| Category | Status | Details |
|----------|--------|---------|
| Unit Tests | ✅/❌ | X/Y passing |
| Coverage | ✅/❌ | XX% |
| Equivalence | ✅/❌ | X/Y passing |
| Accessibility | ✅/❌ | X violations |
| Visual | ✅/❌ | Pass/Fail |

### Behavioral Differences
- [List any differences from legacy]

### Performance
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Render time | Xms | <16ms | ✅/❌ |
| Re-renders | X | Minimal | ✅/❌ |

### Recommendations
- [Any suggested improvements]

### Overall Status: VERIFIED / NEEDS ATTENTION / BLOCKED
```

## Failure Actions
If verification fails:
1. Document the failure in `migration/progress.json`
2. Create issue in blockers list
3. Assign to appropriate agent for remediation
