#!/bin/bash
# verify-kb-protocol.sh - Comprehensive verification script for KB Protocol
# Usage: ./scripts/verify-kb-protocol.sh [PROJECT_DIR]

set -euo pipefail

# Configuration
PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CLAUDE_PROJECT_DIR="${PROJECT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Helper functions
log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAILED=$((FAILED + 1))
}

check_exists() {
    local path="$1"
    local description="$2"

    if [[ -e "${PROJECT_DIR}/${path}" ]]; then
        log_pass "${description} exists: ${path}"
        return 0
    else
        log_fail "${description} missing: ${path}"
        return 1
    fi
}

check_executable() {
    local path="$1"
    local description="$2"

    if [[ -x "${PROJECT_DIR}/${path}" ]]; then
        log_pass "${description} is executable: ${path}"
        return 0
    else
        log_fail "${description} not executable: ${path}"
        return 1
    fi
}

check_json_valid() {
    local path="$1"
    local description="$2"

    if [[ ! -f "${PROJECT_DIR}/${path}" ]]; then
        log_fail "${description} does not exist: ${path}"
        return 1
    fi

    if jq empty "${PROJECT_DIR}/${path}" 2>/dev/null; then
        log_pass "${description} is valid JSON: ${path}"
        return 0
    else
        log_fail "${description} contains invalid JSON: ${path}"
        return 1
    fi
}

check_jsonl_valid() {
    local path="$1"
    local description="$2"

    if [[ ! -f "${PROJECT_DIR}/${path}" ]]; then
        log_fail "${description} does not exist: ${path}"
        return 1
    fi

    # Empty file is valid
    if [[ ! -s "${PROJECT_DIR}/${path}" ]]; then
        log_pass "${description} is empty (valid): ${path}"
        return 0
    fi

    # Check each line is valid JSON
    local line_num=0
    while IFS= read -r line; do
        ((line_num++))
        if ! echo "${line}" | jq empty 2>/dev/null; then
            log_fail "${description} has invalid JSON at line ${line_num}: ${path}"
            return 1
        fi
    done < "${PROJECT_DIR}/${path}"

    log_pass "${description} is valid JSONL (${line_num} lines): ${path}"
    return 0
}

check_hook_dryrun() {
    local hook_path="$1"
    local description="$2"

    if [[ ! -x "${PROJECT_DIR}/${hook_path}" ]]; then
        log_fail "${description} not executable, skipping dry-run: ${hook_path}"
        return 1
    fi

    # Run hook with dry-run approach (redirect to /dev/null, check exit code)
    if (cd "${PROJECT_DIR}" && "${PROJECT_DIR}/${hook_path}" > /dev/null 2>&1); then
        log_pass "${description} dry-run successful: ${hook_path}"
        return 0
    else
        log_fail "${description} dry-run failed: ${hook_path}"
        return 1
    fi
}

check_settings_hook_config() {
    local hook_name="$1"
    local description="$2"

    if ! jq -e ".hooks.${hook_name}" "${PROJECT_DIR}/.claude/settings.json" > /dev/null 2>&1; then
        log_fail "${description} not configured in settings.json: ${hook_name}"
        return 1
    fi

    log_pass "${description} configured in settings.json: ${hook_name}"
    return 0
}

# Banner
echo "========================================"
echo "KB Protocol Verification"
echo "Project: ${PROJECT_DIR}"
echo "========================================"
echo

# 1. Directory Structure Tests
log_info "Checking directory structure..."
check_exists "kb" "KB root directory"
check_exists "kb/ledger" "Ledger directory"
check_exists "kb/sessions" "Sessions directory"
check_exists "kb/summaries" "Summaries directory"
check_exists "kb/handoffs" "Handoffs directory"
check_exists ".claude" "Claude config directory"
check_exists ".claude/hooks" "Hooks directory"
check_exists ".claude/skills" "Skills directory"
echo

# 2. Hook Script Tests
log_info "Checking hook scripts..."
HOOKS=(
    ".claude/hooks/kb-start.sh"
    ".claude/hooks/planning-start.sh"
    ".claude/hooks/dev-sqe-start.sh"
    ".claude/hooks/pre-pr-start.sh"
)

for hook in "${HOOKS[@]}"; do
    check_exists "${hook}" "Hook script"
    if [[ -e "${PROJECT_DIR}/${hook}" ]]; then
        check_executable "${hook}" "Hook script"
    fi
done
echo

# 3. Skill File Tests
log_info "Checking skill files..."
SKILLS=(
    ".claude/skills/planning-handoff/SKILL.md"
    ".claude/skills/dev-sqe-handoff/SKILL.md"
    ".claude/skills/pre-pr-handoff/SKILL.md"
)

for skill in "${SKILLS[@]}"; do
    check_exists "${skill}" "Skill file"
done
echo

# 4. JSON Validation Tests
log_info "Checking JSON files..."

# Check settings.json
check_json_valid ".claude/settings.json" "Settings file"

# Check session state files if they exist
if [[ -d "${PROJECT_DIR}/kb/sessions" ]]; then
    SESSION_COUNT=$(find "${PROJECT_DIR}/kb/sessions" -name "*.state.json" | wc -l)
    if [[ ${SESSION_COUNT} -gt 0 ]]; then
        log_info "Found ${SESSION_COUNT} session state files, validating..."
        find "${PROJECT_DIR}/kb/sessions" -name "*.state.json" | while read -r session_file; do
            relative_path="${session_file#${PROJECT_DIR}/}"
            check_json_valid "${relative_path}" "Session state"
        done
    else
        log_info "No session state files found (acceptable for new installation)"
    fi
fi
echo

# 5. Ledger JSONL Validation
log_info "Checking ledger..."
if [[ -f "${PROJECT_DIR}/kb/ledger/ledger.jsonl" ]]; then
    check_jsonl_valid "kb/ledger/ledger.jsonl" "Ledger file"
else
    log_info "Ledger file does not exist yet (acceptable for new installation)"
fi
echo

# 6. Settings.json Hook Configuration
log_info "Checking settings.json hook configurations..."
if [[ -f "${PROJECT_DIR}/.claude/settings.json" ]]; then
    check_settings_hook_config "SessionStart" "SessionStart hook"
    check_settings_hook_config "Stop" "Stop hook"
    # Optional hooks
    if jq -e '.hooks.PostToolUse' "${PROJECT_DIR}/.claude/settings.json" > /dev/null 2>&1; then
        log_pass "PostToolUse hook configured in settings.json (optional)"
    fi
    if jq -e '.hooks.PreCommit' "${PROJECT_DIR}/.claude/settings.json" > /dev/null 2>&1; then
        log_pass "PreCommit hook configured in settings.json (optional)"
    fi
else
    log_fail "settings.json not found"
fi
echo

# 7. Hook Dry-Run Tests
log_info "Running hook dry-runs..."
for hook in "${HOOKS[@]}"; do
    if [[ -x "${PROJECT_DIR}/${hook}" ]]; then
        check_hook_dryrun "${hook}" "Hook script"
    fi
done
echo

# 8. Additional Checks
log_info "Additional protocol checks..."

# Check handoff naming convention (if any exist)
if [[ -d "${PROJECT_DIR}/kb/handoffs" ]]; then
    HANDOFF_COUNT=$(find "${PROJECT_DIR}/kb/handoffs" -name "*.handoff.md" | wc -l)
    if [[ ${HANDOFF_COUNT} -gt 0 ]]; then
        # Verify naming: YYYYMMDD_HHMMSS_<from>-to-<to>.handoff.md
        NAMING_VALID=true
        find "${PROJECT_DIR}/kb/handoffs" -name "*.handoff.md" | while read -r handoff_file; do
            filename=$(basename "${handoff_file}")
            if [[ ! ${filename} =~ ^[0-9]{8}_[0-9]{6}_[a-z-]+-to-[a-z-]+\.handoff\.md$ ]]; then
                log_fail "Handoff file has incorrect naming convention: ${filename}"
                NAMING_VALID=false
            fi
        done
        if ${NAMING_VALID}; then
            log_pass "All ${HANDOFF_COUNT} handoff files follow naming convention"
        fi
    else
        log_info "No handoff files found (acceptable)"
    fi
fi

# Check summary cap enforcement (check if summaries/*.md files exist and count entries)
if [[ -d "${PROJECT_DIR}/kb/summaries" ]]; then
    SUMMARY_COUNT=$(find "${PROJECT_DIR}/kb/summaries" -name "*.md" | wc -l)
    if [[ ${SUMMARY_COUNT} -gt 0 ]]; then
        log_pass "Found ${SUMMARY_COUNT} summary files"
        # Note: Cannot validate 20-entry cap without parsing markdown, left to Python tests
    else
        log_info "No summary files found (acceptable for new installation)"
    fi
fi

echo

# Summary
echo "========================================"
echo "Verification Summary"
echo "========================================"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo

if [[ ${FAILED} -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review the output above.${NC}"
    exit 1
fi
