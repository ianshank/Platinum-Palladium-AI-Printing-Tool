/**
 * Domain & UI Hooks Barrel
 *
 * Re-exports all hooks from the `hooks/` directory.
 * These are domain-level and UI hooks that compose API hooks and store state.
 *
 * ## Hook Organization
 *
 * | Directory       | Purpose                          | Examples                              |
 * |-----------------|----------------------------------|---------------------------------------|
 * | `hooks/`        | Domain logic, UI orchestration   | useChat, useDashboardData             |
 * | `api/hooks.ts`  | React Query wrappers for API     | useGenerateCurve, useUploadScan       |
 *
 * Domain hooks in `hooks/` may import from `api/hooks.ts` but not vice versa.
 */

export {
  useKeyboardShortcuts,
  useAppShortcuts,
  type ShortcutConfig,
} from './useKeyboardShortcuts';

export { useChat, type UseChatReturn } from './useChat';
export { useDashboardData } from './useDashboardData';
