#!/usr/bin/env bash
set -euo pipefail

# Configuration
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-${PROJECT_DIR:-.}}"
HOOKS_DIR="${CLAUDE_HOOKS_DIR:-$PROJECT_DIR/.claude/hooks}"
ROLE="${CLAUDE_ROLE:-all}"
DEBUG="${KB_DEBUG:-false}"

# Debug helper
log_debug() { [ "$DEBUG" = "true" ] && echo "[KB-DEBUG] $*" >&2 || true; }

# Run a role-specific hook if it exists
run_hook() {
  local role="$1"
  local hook="$HOOKS_DIR/${role}-start.sh"

  if [ -f "$hook" ] && [ -x "$hook" ]; then
    log_debug "Running $role hook: $hook"
    bash "$hook"
  elif [ -f "$hook" ]; then
    log_debug "Running $role hook (not executable): $hook"
    bash "$hook"
  else
    log_debug "Hook not found: $hook"
    echo "Warning: Hook script not found: $hook" >&2
  fi
}

# ============================================================================
# KB SESSION START DISPATCHER
# ============================================================================

log_debug "KB SessionStart dispatcher invoked (role: $ROLE)"

case "$ROLE" in
  planning)
    run_hook "planning"
    ;;
  dev-sqe)
    run_hook "dev-sqe"
    ;;
  pre-pr)
    run_hook "pre-pr"
    ;;
  all)
    log_debug "Running all role hooks (default)"
    echo "=== Knowledge Base Context (All Roles) ==="
    echo ""
    run_hook "planning"
    echo ""
    echo "---"
    echo ""
    run_hook "dev-sqe"
    echo ""
    echo "---"
    echo ""
    run_hook "pre-pr"
    ;;
  *)
    echo "Warning: Unknown CLAUDE_ROLE '$ROLE'. Valid values: planning, dev-sqe, pre-pr, all" >&2
    log_debug "Unknown role: $ROLE"
    exit 1
    ;;
esac

log_debug "KB SessionStart dispatcher complete"
