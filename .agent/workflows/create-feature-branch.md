---
description: Create a feature branch, implement changes, commit, push, and create a PR via GitKraken MCP
---

# Feature Branch Workflow

Manages the full git lifecycle for a feature using GitKraken MCP tools.

## Parameters

- `FEATURE_NAME`: Short kebab-case name (e.g., `migrate-curve-editor`)
- `BRANCH_PREFIX`: One of `feature/`, `fix/`, `chore/`, `test/` (default: `feature/`)
- `PR_TITLE`: Title for the pull request
- `PR_BODY`: Description of changes

## Steps

1. **Create Branch**
   Use `mcp_GitKraken_git_branch` with action `create` and branch name `{BRANCH_PREFIX}{FEATURE_NAME}`.

2. **Checkout Branch**
   Use `mcp_GitKraken_git_checkout` to switch to the new branch.

3. **Implement Changes**
   Make code changes following the relevant workflow or sub-agent pattern.

4. **Stage Changes**
   Use `mcp_GitKraken_git_add_or_commit` with action `add`.

5. **Commit Changes**
   Use `mcp_GitKraken_git_add_or_commit` with action `commit`.
   Follow conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `test:` for test additions
   - `chore:` for tooling/infrastructure
   - `docs:` for documentation

6. **Push to Remote**
   Use `mcp_GitKraken_git_push`.

7. **Create Pull Request**
   Use `mcp_GitKraken_pull_request_create` with:
   - `repository_organization`: `ianshank` (from workspace config)
   - `repository_name`: `Platinum-Palladium-AI-Printing-Tool`
   - `source_branch`: `{BRANCH_PREFIX}{FEATURE_NAME}`
   - `target_branch`: `main`
   - `provider`: `github`

## Notes

- Always run `/verify-all` before committing
- Use descriptive commit messages referencing gap IDs when applicable
- Set `is_draft: true` for work-in-progress PRs
