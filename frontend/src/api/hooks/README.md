# API Hooks Architecture

This project has **two sets of hooks** for API operations. This is intentional.

## `api/hooks.ts` — UI-coupled hooks (full)

Feature-rich hooks with:

- Toast notifications on success/error
- Zustand state updates (processing flags, chat messages, upload progress)
- Cache invalidation via TanStack Query

**Use these when:** you want the full integrated UX (dashboards, standalone pages).

## `api/hooks/` — Headless hooks (slim)

Minimal hooks that only call the API:

- `useCurves.ts` — `useGenerateCurve`, `useExportCurve`
- `useScan.ts` — `useUploadScan`, `assessScanQuality`

**Use these when:** the component handles its own UX feedback (e.g., wizard steps that show inline loading/errors).

## Adding a new hook

1. If the hook needs to integrate with store and toasts → add to `api/hooks.ts`
2. If the hook is consumed by a self-contained component → add to `api/hooks/`
3. If both → add to `api/hooks.ts` and create a slim version in `api/hooks/`
