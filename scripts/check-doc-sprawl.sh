#!/usr/bin/env bash
# Enforces the documentation placement rules from AGENTS.md (ADR-0008).
#
#   1. The repository root may contain only the canonical markdown set.
#   2. Status-report style files (roadmaps, summaries, next-steps, quick
#      references, per-directory AGENT.md) are not allowed anywhere except
#      the archive.
#
# Configuration (environment variables, all optional):
#   DOC_SPRAWL_ROOT_ALLOWLIST  space-separated root files allowed (default below)
#   DOC_SPRAWL_EXEMPT_DIRS     space-separated path prefixes exempt from rule 2
#   DOC_SPRAWL_DEBUG           "true" to print every decision
set -euo pipefail

ROOT_DIR="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
ALLOWLIST="${DOC_SPRAWL_ROOT_ALLOWLIST:-README.md CONTRIBUTING.md SECURITY.md CHANGELOG.md LICENSE AGENTS.md CLAUDE.md}"
EXEMPT_DIRS="${DOC_SPRAWL_EXEMPT_DIRS:-docs/archive docs/plans node_modules .venv .venv-uv .git frontend/node_modules}"
DEBUG="${DOC_SPRAWL_DEBUG:-false}"

log_debug() { [ "$DEBUG" = "true" ] && echo "[doc-sprawl] $*" >&2 || true; }

is_allowed_root() {
  local name="$1"
  for allowed in $ALLOWLIST; do
    [ "$name" = "$allowed" ] && return 0
  done
  return 1
}

is_exempt() {
  local rel="$1"
  for prefix in $EXEMPT_DIRS; do
    case "$rel" in
      "$prefix"/*|"$prefix") return 0 ;;
    esac
  done
  return 1
}

fail=0

# Rule 1: root markdown allowlist
while IFS= read -r -d '' f; do
  name="$(basename "$f")"
  if is_allowed_root "$name"; then
    log_debug "root ok: $name"
  else
    echo "::error file=$name::root-level markdown '$name' is not in the allowlist ($ALLOWLIST). Move it under docs/ (see AGENTS.md)."
    fail=1
  fi
done < <(find "$ROOT_DIR" -maxdepth 1 -type f -name '*.md' -print0)

# Rule 2: forbidden file-name patterns anywhere outside exempt dirs
FORBIDDEN_REGEX='(^|/)([A-Za-z0-9_-]*_SUMMARY\.md|[A-Za-z0-9_-]*NEXT_STEPS[A-Za-z0-9_-]*\.md|[A-Za-z0-9_-]*_IMPLEMENTATION[A-Za-z0-9_-]*\.md|INVESTIGATION[A-Za-z0-9_-]*\.md|QUICK_REFERENCE\.md|AGENT\.md)$'
while IFS= read -r -d '' f; do
  rel="${f#"$ROOT_DIR"/}"
  if is_exempt "$rel"; then
    log_debug "exempt: $rel"
    continue
  fi
  if [[ "$rel" =~ $FORBIDDEN_REGEX ]]; then
    echo "::error file=$rel::'$rel' matches a forbidden documentation pattern (status reports belong in PR descriptions, decisions in docs/adr/, plans in docs/plans/)."
    fail=1
  fi
done < <(find "$ROOT_DIR" -type f -name '*.md' -not -path '*/node_modules/*' -not -path '*/.git/*' -print0)

if [ "$fail" -eq 0 ]; then
  echo "doc-sprawl: ok"
fi
exit "$fail"
