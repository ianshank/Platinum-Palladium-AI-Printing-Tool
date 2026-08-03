# Hooks Directory

## Purpose
Custom React hooks for domain logic and UI orchestration. These compose API hooks (`api/hooks.ts`) with Zustand store state to provide high-level functionality to components.

## Key Files
- `index.ts` — Barrel exports with documentation table distinguishing `hooks/` from `api/hooks.ts`
- `useChat.ts` — Chat interface orchestration (message sending, history, streaming)
- `useDashboardData.ts` — Aggregates dashboard statistics from multiple API queries
- `useKeyboardShortcuts.ts` — Global keyboard shortcut registration (`ShortcutConfig` interface)
- `useUndoRedo.ts` — Generic undo/redo stack with configurable history limit
- `useMCTSCalibration.ts` — MCTS calibration workflow orchestration

## Conventions
- **Naming**: `use{Feature}.ts` — always prefixed with `use`
- **Dependency direction**: Hooks here may import from `api/hooks.ts` but NOT vice versa
- **Co-located tests**: Prefer placing `use{Feature}.test.ts` beside each hook; new hooks should follow this pattern
- **Typed returns**: Export explicit return type interfaces (e.g., `UseChatReturn`, `UseUndoRedoReturn`)

## Testing
```bash
pnpm test -- --run src/hooks/use{Feature}.test.ts
```
Use `renderHook` from `@testing-library/react` wrapped with providers from `@/test-utils`.

## Pitfalls
- Do NOT access store state directly in hooks — use typed selectors from `@/stores`
- Do NOT put API call logic here — that belongs in `api/hooks.ts` as TanStack Query hooks
- Keyboard shortcuts register globally — always clean up in effect return

## Related
- `../api/hooks.ts` — TanStack Query hooks for API operations (lower-level)
- `../stores/` — State accessed via selectors
- `../components/` — Consume these hooks
