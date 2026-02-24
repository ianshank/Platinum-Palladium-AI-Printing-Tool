#!/usr/bin/env bash
set -euo pipefail

# Configuration (override via environment)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-${PROJECT_DIR:-.}}"
KB="${KB_DIR:-$PROJECT_DIR/kb}"
MAX_SUMMARY_BYTES="${KB_MAX_SUMMARY_BYTES:-4000}"
MAX_CROSS_ROLE_BYTES="${KB_MAX_CROSS_ROLE_BYTES:-2000}"
MAX_LEDGER_LINES="${KB_MAX_LEDGER_LINES:-15}"
MAX_HANDOFFS="${KB_MAX_HANDOFFS:-3}"
DEBUG="${KB_DEBUG:-false}"

# Debug helper
log_debug() { [ "$DEBUG" = "true" ] && echo "[KB-DEBUG] $@" >&2 || true; }

# Safe read helper - outputs content from file up to N bytes, or nothing
safe_read() {
  local file="$1" max_bytes="${2:-$MAX_SUMMARY_BYTES}"
  if [ -f "$file" ] && [ -s "$file" ]; then
    head -c "$max_bytes" "$file"
    log_debug "Loaded $(wc -c < "$file" 2>/dev/null || echo 0) bytes from $file (capped at $max_bytes)"
  else
    log_debug "Skipped missing/empty file: $file"
  fi
}

# Safe read JSON state file (no byte limit for state)
safe_read_json() {
  local file="$1"
  if [ -f "$file" ] && [ -s "$file" ]; then
    cat "$file"
    log_debug "Loaded JSON state from $file"
  else
    log_debug "Skipped missing/empty JSON: $file"
  fi
}

# Filter ledger events by role/type
filter_ledger() {
  local ledger="$KB/ledger/ledger.jsonl" max_lines="${1:-$MAX_LEDGER_LINES}"
  shift
  local patterns=("$@")

  if [ -f "$ledger" ] && [ -s "$ledger" ]; then
    local grep_pattern=$(IFS='|'; echo "${patterns[*]}")
    grep -E "\"event\"\s*:\s*\"($grep_pattern)\"" "$ledger" 2>/dev/null | tail -n "$max_lines" || true
    log_debug "Filtered ledger for patterns: ${patterns[*]} (max $max_lines lines)"
  else
    log_debug "Skipped missing/empty ledger: $ledger"
  fi
}

# Get newest handoff docs TO this role
get_handoffs() {
  local pattern="$1" max="${2:-$MAX_HANDOFFS}"
  local handoff_dir="$KB/handoffs"
  local count=0

  if [ -d "$handoff_dir" ]; then
    while IFS= read -r file; do
      [ -z "$file" ] && continue
      [ "$count" -ge "$max" ] && break
      echo "--- Handoff: $(basename "$file") ---"
      head -c "$MAX_CROSS_ROLE_BYTES" "$file"
      echo ""
      count=$((count + 1))
    done < <(find "$handoff_dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | sort -r)
    log_debug "Loaded $count handoff docs matching: $pattern (max $max)"
  else
    log_debug "Handoff directory not found: $handoff_dir"
  fi
}

# ============================================================================
# PRE-PR ROLE CONTEXT INJECTION
# ============================================================================

echo "=== PRE-PR Role Knowledge Base Context ==="
echo ""

# Planning summary (requirements context)
if planning=$(safe_read "$KB/summaries/planning.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$planning" ] && echo "## Planning Requirements" && echo "$planning" && echo ""
fi

# DEV-SQE summary (upstream implementation)
if dev_summary=$(safe_read "$KB/summaries/dev-sqe.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$dev_summary" ] && echo "## DEV-SQE Implementation" && echo "$dev_summary" && echo ""
fi

# Own Pre-PR summary
if pr_summary=$(safe_read "$KB/summaries/pre-pr.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$pr_summary" ] && echo "## Pre-PR Summary" && echo "$pr_summary" && echo ""
fi

# DEV-SQE session state (full context from upstream)
if dev_state=$(safe_read_json "$KB/sessions/dev-sqe.state.json"); then
  [ -n "$dev_state" ] && echo "## DEV-SQE Session State" && echo '```json' && echo "$dev_state" && echo '```' && echo ""
fi

# Recent ledger events (filtered to relevant types)
echo "## Recent Pre-PR Events"
filter_ledger "$MAX_LEDGER_LINES" "PRE-PR" "DEV-SQE-HANDOFF" "DEV-SQE" "QA-GAP"
echo ""

# Own session state
if state=$(safe_read_json "$KB/sessions/pre-pr.state.json"); then
  [ -n "$state" ] && echo "## Pre-PR Session State" && echo '```json' && echo "$state" && echo '```' && echo ""
fi

# Pending handoff docs TO pre-pr
echo "## Pending Handoffs to Pre-PR"
get_handoffs "*_to_pre-pr.md" "$MAX_HANDOFFS"

log_debug "Pre-PR context injection complete"
