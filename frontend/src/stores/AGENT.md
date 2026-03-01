# Stores Directory

## Purpose
Zustand state management using the slice pattern. A single composed store (`useStore`) with domain-specific slices for modularity.

## Key Files
- `index.ts` — Store composition, middleware stack, typed selectors, `createStore()` for tests
- `slices/` — 8 domain slices:
  - `uiSlice.ts` — Active tab, sidebar, theme, processing flag
  - `calibrationSlice.ts` — Calibration wizard state, step tracking, history
  - `curveSlice.ts` — Current curve, points, modification tracking
  - `chemistrySlice.ts` — Recipe, paper size, metal ratio
  - `chatSlice.ts` — Chat messages, loading state
  - `sessionSlice.ts` — Session records, statistics
  - `imageSlice.ts` — Current image, preview, upload progress
  - `mctsSlice.ts` — MCTS calibration state

## Conventions
- **Slice pattern**: Each slice is a `create___Slice(set, get, store)` function returning its interface
- **Middleware stack**: `devtools → subscribeWithSelector → persist → immer` (see `index.ts` lines 42-83)
- **Persistence**: Only UI preferences are persisted (activeTab, sidebarOpen, theme). Never persist runtime state like `isProcessing`
- **Selectors**: Define typed selectors in `index.ts` (e.g., `selectActiveTab`), import in components — never create inline selectors
- **Immer**: Use `set(state => { state.x.y = z })` for nested updates — immer handles immutability

## Adding a New Slice
1. Create `slices/newSlice.ts` with interface and factory function
2. Add to `StoreState` type in `index.ts`
3. Compose into `useStore` and `createStore`
4. Add typed selectors
5. Add co-located test `slices/newSlice.test.ts`

## Testing
```bash
pnpm test -- --run src/stores/slices/{slice}.test.ts
```
Use `createStore()` from `index.ts` for isolated test instances — never import the singleton `useStore` in tests.

## Pitfalls
- Do NOT persist runtime/transient state — only user preferences
- Do NOT use `useStore(state => state)` — always use specific selectors to avoid re-renders
- The `@ts-expect-error` on middleware composition is intentional — Zustand's generic inference breaks with 4+ middleware layers

## Related
- `../components/` — Consume store via selectors
- `../hooks/` — Compose store state with API hooks
- `../api/hooks.ts` — API mutations update store on success
