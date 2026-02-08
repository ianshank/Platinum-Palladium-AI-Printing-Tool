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
log_debug() { [ "$DEBUG" = "true" ] && echo "[KB-DEBUG] $*" >&2 || true; }

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

# Safe read JSON state file
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
# DEV-SQE ROLE CONTEXT INJECTION
# ============================================================================

echo "=== DEV-SQE Role Knowledge Base Context ==="
echo ""

# Planning summary (upstream requirements)
if planning=$(safe_read "$KB/summaries/planning.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$planning" ] && echo "## Planning Requirements" && echo "$planning" && echo ""
fi

# Design contract readiness (critical for implementation)
if readiness=$(safe_read "$KB/summaries/design-contract-readiness.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$readiness" ] && echo "## Design Contract Readiness" && echo "$readiness" && echo ""
fi

# Own DEV-SQE summary
if dev_summary=$(safe_read "$KB/summaries/dev-sqe.md" "$MAX_SUMMARY_BYTES"); then
  [ -n "$dev_summary" ] && echo "## DEV-SQE Summary" && echo "$dev_summary" && echo ""
fi

# Pre-PR summary (downstream feedback)
if pr_summary=$(safe_read "$KB/summaries/pre-pr.md" "$MAX_CROSS_ROLE_BYTES"); then
  [ -n "$pr_summary" ] && echo "## Pre-PR Feedback" && echo "$pr_summary" && echo ""
fi

# Recent ledger events (filtered to relevant types)
echo "## Recent DEV-SQE Events"
filter_ledger "$MAX_LEDGER_LINES" "DEV-SQE" "PLANNING-HANDOFF" "QA-GAP" "DESIGN-GAP"
echo ""

# Own session state
if state=$(safe_read_json "$KB/sessions/dev-sqe.state.json"); then
  [ -n "$state" ] && echo "## DEV-SQE Session State" && echo '```json' && echo "$state" && echo '```' && echo ""
fi

# Pending handoff docs TO dev-sqe
echo "## Pending Handoffs to DEV-SQE"
get_handoffs "*_to_dev-sqe.md" "$MAX_HANDOFFS"

log_debug "DEV-SQE context injection complete"
