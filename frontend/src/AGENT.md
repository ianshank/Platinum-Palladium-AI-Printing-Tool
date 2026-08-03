# Frontend Source Root

## Purpose
React 18 + TypeScript application for the Pt/Pd Printing Tool. This is the root of the frontend source tree.

## Directory Layout
- `components/` — UI components organized by feature domain (14 subdirectories)
- `pages/` — Page-level components: DashboardPage, CalibrationPage, CurvesPage, ChemistryPage, AIAssistantPage, SessionLogPage, SettingsPage, MCTSPage
- `stores/` — Zustand state management (slice pattern, 8 domain slices)
- `hooks/` — Custom React hooks for domain logic
- `api/` — Axios client + TanStack Query hooks
- `types/` — Shared TypeScript interfaces (mirrors backend Pydantic models)
- `config/` — App configuration, feature flags, tablet config
- `lib/` — Core utilities: `logger.ts` (structured logging), `utils.ts` (shared helpers)
- `styles/` — Theme and global styles
- `test-utils/` — Test helpers (`renderWithProviders`, `createTestStore`)
- `__tests__/` — Cross-cutting integration tests

## Entry Points
- `main.tsx` — App bootstrap (React root, providers, router)
- `App.tsx` — Root component with tab-based navigation

## Build & Verification
```bash
pnpm typecheck         # TypeScript strict mode check
pnpm lint:fix          # ESLint with auto-fix
pnpm test              # Vitest test suite
pnpm build             # Vite production build
pnpm check:all         # All of the above
```

## Import Resolution
- `@/` maps to `src/` — use absolute imports: `import { useStore } from '@/stores'`
- Import order enforced by ESLint: React → external → internal (`@/`) → relative → type-only

## Conventions
See CLAUDE.md sections: "TypeScript Standards", "React Patterns", "Zustand Patterns", "Import Order"

## Pitfalls
- Do NOT use relative imports for cross-directory references — use `@/` prefix
- Do NOT import from `index.ts` barrel files within the same directory — import the specific file
- `styled.d.ts` and `vite-env.d.ts` are type augmentation files — do not delete

## Related
- `../../CLAUDE.md` — Project-wide conventions and migration rules
- `../vite.config.ts` — Build configuration, proxy settings, path aliases
- `../tailwind.config.ts` — Tailwind theme configuration
- Backend: `../../src/ptpd_calibration/api/` — API server these components call
