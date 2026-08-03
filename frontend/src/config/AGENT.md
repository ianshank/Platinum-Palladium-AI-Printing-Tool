# Config Directory

## Purpose
Centralized application configuration: environment-driven settings, feature flags, and domain-specific configuration (step tablet types, calibration defaults).

## Key Files
- `index.ts` — `AppConfig` interface and `config` singleton. Loads all values from `VITE_*` env vars with typed fallbacks
- `featureFlags.ts` — Feature flag system for gradual rollout (`featureFlags.test.ts`)
- `tablet.config.ts` — Step tablet configuration: patch counts, dimensions, densities for Stouffer 21/31/41 wedges
- `config.test.ts` — Config loading tests

## Conventions
- **No hardcoded values**: Everything is configurable via env vars or config files
- **`VITE_` prefix**: Frontend env vars must use `VITE_` prefix (Vite requirement)
- **Typed getters**: `getEnv()`, `getEnvBool()`, `getEnvNumber()` — parse and validate
- **Config priority**: Environment variables > local config > defaults in code

## Key Config Sections
- `config.api` — `baseUrl`, `wsUrl`, `timeout`, `retryAttempts`, `staleTime`, `gcTime`
- `config.features` — `devtools`, `mockApi`, `darkMode`, `offlineMode`
- `config.logging` — `level`, `enableConsole`, `enableRemote`
- `config.ui` — `defaultTab`, `animationDuration`, `debounceDelay`, `undoHistoryLimit`
- `config.calibration` — `defaultSteps`, `maxCurvePoints`, `smoothingDefault`

## Testing
```bash
pnpm test -- --run src/config/
```

## Pitfalls
- Do NOT access `import.meta.env` directly in components — always go through `config`
- Feature flags must have safe defaults (disabled) — never default a flag to enabled
- Tablet config values are domain-specific — verify against physical step tablet specs

## Related
- `../.env` / `../.env.development` — Environment variable files
- `../stores/index.ts` — Store reads config for devtools and persistence settings
- CLAUDE.md "Environment Variables" — Full env var reference
