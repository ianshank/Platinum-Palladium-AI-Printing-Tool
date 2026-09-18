#!/usr/bin/env bash
# Knowledge-base SessionStart dispatcher (ADR-0015).
#
# Default (CLAUDE_ROLE unset): print a short, capped digest of the knowledge
# base with a staleness check, never the full summaries. Role context is opt-in
# via CLAUDE_ROLE=planning|dev-sqe|pre-pr and is capped too.
#
# Configuration (environment variables):
#   CLAUDE_PROJECT_DIR   repository root (default: .)
#   KB_DIR               knowledge-base directory (default: $CLAUDE_PROJECT_DIR/kb)
#   CLAUDE_ROLE          planning | dev-sqe | pre-pr | all (opt-in role context)
#   KB_MAX_INJECT_BYTES  hard cap on bytes printed (default 4096)
#   KB_MAX_LEDGER_LINES  ledger events shown in the digest (default 5)
#   KB_MAX_HANDOFFS      handoff paths listed in the digest (default 3)
#   KB_DEBUG             "true" for debug output on stderr
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-${PROJECT_DIR:-.}}"
HOOKS_DIR="${CLAUDE_HOOKS_DIR:-$PROJECT_DIR/.claude/hooks}"
KB="${KB_DIR:-$PROJECT_DIR/kb}"
ROLE="${CLAUDE_ROLE:-}"
MAX_BYTES="${KB_MAX_INJECT_BYTES:-4096}"
MAX_LEDGER_LINES="${KB_MAX_LEDGER_LINES:-5}"
MAX_HANDOFFS="${KB_MAX_HANDOFFS:-3}"
DEBUG="${KB_DEBUG:-false}"

log_debug() { [ "$DEBUG" = "true" ] && echo "[KB-DEBUG] $*" >&2 || true; }

# Extract a top-level string field from the last ledger line without requiring jq.
ledger_field() {
  local field="$1" line="$2"
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$line" | jq -r --arg f "$field" '.[$f] // empty' 2>/dev/null || true
  else
    printf '%s' "$line" | sed -n "s/.*\"$field\":\"\([^\"]*\)\".*/\1/p"
  fi
}

staleness_banner() {
  local ledger="$KB/ledger/ledger.jsonl" last sha
  [ -s "$ledger" ] || { echo "ledger: empty"; return; }
  # Use the most recent event that carries a git_commit (handoff events may not).
  sha=""
  while IFS= read -r last; do
    sha="$(ledger_field git_commit "$last")"
    [ -n "$sha" ] && break
  done < <(tail -n 20 "$ledger" | tac)
  if [ -z "$sha" ]; then
    echo "ledger: last event carries no git_commit (cannot check staleness)"
  elif git -C "$PROJECT_DIR" merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
    echo "ledger: last event at commit $sha is an ancestor of HEAD (fresh)"
  else
    echo "STALE: last ledger event at commit $sha is NOT an ancestor of HEAD; treat KB claims as historical"
  fi
}

compact_ledger() {
  local ledger="$KB/ledger/ledger.jsonl"
  [ -s "$ledger" ] || return 0
  tail -n "$MAX_LEDGER_LINES" "$ledger" | while IFS= read -r line; do
    [ -z "$line" ] && continue
    if command -v jq >/dev/null 2>&1; then
      printf '%s' "$line" | jq -r '[(.timestamp // "?"), (.event // "?"), ((.summary // "") | tostring | .[0:140])] | join(" | ")' 2>/dev/null || printf '%.160s\n' "$line"
    else
      printf '%.160s\n' "$line"
    fi
  done
}

digest() {
  echo "=== Knowledge Base digest (capped at ${MAX_BYTES} bytes; set CLAUDE_ROLE=planning|dev-sqe|pre-pr for role context) ==="
  staleness_banner
  echo ""
  echo "Last ${MAX_LEDGER_LINES} ledger events (timestamp | event | summary):"
  compact_ledger
  echo ""
  echo "Newest handoff documents:"
  if [ -d "$KB/handoffs" ]; then
    ls -1t "$KB/handoffs" 2>/dev/null | head -n "$MAX_HANDOFFS" | sed "s#^#  $KB/handoffs/#" || true
  fi
  echo ""
  echo "State files (read on demand, not injected):"
  for role in planning dev-sqe pre-pr; do
    [ -f "$KB/sessions/$role.state.json" ] && echo "  $KB/sessions/$role.state.json"
  done
  echo "Protocol: docs/agents/kb-protocol.md"
}

run_role_hook() {
  local role="$1"
  local hook="$HOOKS_DIR/${role}-start.sh"
  if [ -f "$hook" ]; then
    log_debug "Running $role hook: $hook"
    bash "$hook"
  else
    echo "Warning: hook script not found: $hook" >&2
  fi
}

emit() {
  case "$ROLE" in
    "") digest ;;
    planning|dev-sqe|pre-pr) run_role_hook "$ROLE" ;;
    all)
      echo "=== Knowledge Base Context (All Roles) ==="
      for r in planning dev-sqe pre-pr; do echo ""; run_role_hook "$r"; echo "---"; done
      ;;
    *)
      echo "Warning: unknown CLAUDE_ROLE '$ROLE' (planning, dev-sqe, pre-pr, all)" >&2
      digest
      ;;
  esac
}

log_debug "KB SessionStart (role='${ROLE:-none}', cap=${MAX_BYTES})"
emit | head -c "$MAX_BYTES"
echo ""
