# API Directory

## Purpose
API client layer: Axios HTTP client, TanStack Query hooks for caching/loading states, and MCTS-specific API endpoints.

## Key Files
- `client.ts` — Singleton Axios instance, `apiRequest<T>()` helper, `api` namespace with all endpoints
- `hooks.ts` — TanStack Query hooks (`useGenerateCurve`, `useUploadScan`, etc.) with `queryKeys` factory
- `index.ts` — Barrel re-export
- `mcts.ts` — MCTS-specific API functions (simulate, train, search)
- `mctsHooks.ts` — TanStack Query hooks for MCTS operations

## Conventions
- **Query key factory**: All keys derive from `queryKeys` object in `hooks.ts` — use these for cache invalidation
- **Endpoint grouping**: `api.curves.*`, `api.scan.*`, `api.calibrations.*`, `api.chat.*`, `api.statistics.*`, `api.analyze.*`
- **File uploads**: Use `FormData` + `multipart/form-data` header. See `api.scan.upload()` and `api.curves.uploadQuad()` for patterns
- **Upload progress**: Pass `onUploadProgress` callback via Axios config (see `scan.upload`)
- **Response types**: All imported from `@/types/models` — keep in sync with backend Pydantic models
- **Error type**: `ApiError` interface in `client.ts` — interceptors handle logging automatically

## Adding a New Endpoint
1. Add TypeScript types in `@/types/models`
2. Add method to `api` namespace in `client.ts`
3. Create TanStack Query hook in `hooks.ts` (query for reads, mutation for writes)
4. Add query key to `queryKeys` factory
5. Invalidate relevant queries on mutation success

## Testing
```bash
pnpm test -- --run src/api/hooks.test.tsx
```
Tests use `createTestQueryClient()` from `@/test-utils` — no actual network calls.

## Pitfalls
- Do NOT call `apiClient` directly from components — always go through `api.*` namespace or hooks
- Do NOT create query keys as ad-hoc strings — use `queryKeys` factory
- Base URL is empty in dev mode (Vite proxy handles routing) — only set in production

## Related
- `../types/models.ts` — Response/request type definitions
- `../stores/` — Mutations often update store state on success
- `../hooks/` — Domain hooks compose API hooks with store logic
- Backend: `src/ptpd_calibration/api/server.py` — endpoint implementations
