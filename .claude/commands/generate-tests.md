---
name: generate-tests
description: Generate comprehensive tests for a component
arguments:
  - name: component
    description: The name of the component to generate tests for
    required: true
  - name: type
    description: Type of tests to generate (unit, integration, e2e, all)
    required: false
    default: all
---

# Generate Tests: $ARGUMENTS

Generate comprehensive test suite for the specified component.

## Test Generation Process

### Step 1: Analyze Component
Read the component at `frontend/src/components/$ARGUMENTS/` and identify:
- All props and their types
- Event handlers and callbacks
- State management (Zustand slices)
- Conditional rendering logic
- Error and loading states

### Step 2: Generate Unit Tests
Create `$ARGUMENTS.test.tsx` with:

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { axe, toHaveNoViolations } from 'jest-axe';
import { $ARGUMENTS } from './$ARGUMENTS';
import { TestStoreProvider, createTestStore } from '@/test-utils';

expect.extend(toHaveNoViolations);

describe('$ARGUMENTS', () => {
  // Setup
  let store: ReturnType<typeof createTestStore>;
  const user = userEvent.setup();

  beforeEach(() => {
    store = createTestStore();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // Rendering tests
  describe('Rendering', () => {
    it('renders with default props', () => {});
    it('renders with all props', () => {});
    it('applies custom className', () => {});
    it('forwards ref correctly', () => {});
  });

  // Interaction tests
  describe('Interactions', () => {
    it('handles click events', async () => {});
    it('handles keyboard events', async () => {});
    it('handles focus/blur', async () => {});
  });

  // State tests
  describe('State Management', () => {
    it('reads from store correctly', () => {});
    it('updates store on interaction', async () => {});
    it('responds to store changes', async () => {});
  });

  // Edge cases
  describe('Edge Cases', () => {
    it('handles empty data', () => {});
    it('handles null/undefined values', () => {});
    it('handles maximum values', () => {});
    it('handles rapid interactions', async () => {});
  });

  // Error handling
  describe('Error Handling', () => {
    it('displays error message', () => {});
    it('calls onError callback', () => {});
    it('recovers from error state', async () => {});
  });

  // Loading states
  describe('Loading States', () => {
    it('shows loading indicator', () => {});
    it('disables interactions while loading', () => {});
    it('hides loading when complete', async () => {});
  });

  // Accessibility
  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(
        <TestStoreProvider store={store}>
          <$ARGUMENTS />
        </TestStoreProvider>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('is keyboard navigable', async () => {});
    it('has correct ARIA attributes', () => {});
    it('announces changes to screen readers', async () => {});
  });
});
```

### Step 3: Generate Store Slice Tests
If component uses Zustand, create slice tests:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { createStore } from '@/stores';

describe('{SliceName}Slice', () => {
  // Initial state
  // Actions
  // Selectors
  // Edge cases
});
```

### Step 4: Generate Integration Tests
Create integration tests for data flow:

```typescript
describe('$ARGUMENTS Integration', () => {
  it('integrates with API correctly', async () => {});
  it('handles API errors gracefully', async () => {});
  it('updates related components', async () => {});
});
```

### Step 5: Generate E2E Tests (Playwright)
Create `e2e/$ARGUMENTS.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test.describe('$ARGUMENTS', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/path-to-component');
  });

  test('user workflow', async ({ page }) => {});
  test('visual regression', async ({ page }) => {});
  test('keyboard navigation', async ({ page }) => {});
  test('responsive layout', async ({ page }) => {});
});
```

### Step 6: Generate Equivalence Tests
Create `migration/equivalence-tests/$ARGUMENTS.equiv.test.ts`:

```typescript
describe('$ARGUMENTS Equivalence', () => {
  // Compare legacy and new behavior
});
```

## Coverage Requirements
- Statements: ≥80%
- Branches: ≥75%
- Functions: ≥80%
- Lines: ≥80%

## Output
- List of test files created
- Initial test results
- Coverage report
- Recommendations for additional tests
