# Test Utilities Directory

## Purpose
Shared test helpers that wrap React Testing Library with the application's providers (Router, QueryClient, Theme, Store). All component tests should use these utilities instead of raw RTL.

## Key Files
- `index.tsx` — `renderWithProviders()`, `createTestStore()`, `createTestQueryClient()`, provider wrappers
- `setup.ts` — Vitest global setup (DOM mocks, global test configuration)

## Key Exports
- `renderWithProviders(ui, options?)` — Renders component wrapped in Router + QueryClient + Theme + optional store
- `createTestStore(initialState?)` — Creates isolated Zustand store instance for testing
- `createTestQueryClient()` — QueryClient with retry disabled and gcTime=0 for deterministic tests
- Re-exports all of `@testing-library/react` and `userEvent`

## Conventions
- **ALWAYS** use `renderWithProviders` instead of bare `render` from RTL
- **ALWAYS** use `createTestStore` for store-dependent tests — never import the singleton `useStore`
- **ALWAYS** use `createTestQueryClient` — default QueryClient retries cause flaky tests
- Import as: `import { renderWithProviders, createTestStore, screen } from '@/test-utils'`

## Pitfalls
- Do NOT add component-specific mocks here — keep them in the test file or a `__mocks__/` dir
- Do NOT import the production `useStore` in tests — it leaks state between test cases

## Related
- `../stores/index.ts` — `createStore()` factory used by `createTestStore`
- `../components/` — All component tests use these utilities
- CLAUDE.md "Testing Requirements" — Coverage targets and testing standards
