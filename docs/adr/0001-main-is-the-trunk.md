# ADR-0001: main is the trunk

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

GitHub's default branch was `claude/implement-chat-requirements-014YeEiyBSMJL91puKEgVcih`, 134 commits ahead of an unprotected `main` that had not moved since 2026-02-08. Every "PR to main" carried unrelated commits and no branch could enforce a gate.

## Decision

`main` becomes the default branch and is fast-forwarded to the trunk tip (`main..trunk` is empty, nothing is lost). Rulesets on `main` require a pull request, the `all-green` check, linear history, and forbid force-pushes and deletion; `v*` tags are protected. The long-lived `claude/*` integration branch is deleted afterwards and open PRs are retargeted or closed (see the plan §3.5).

## Consequences

Owner action in repository settings (cannot be done from a branch). Until it lands, PRs target the current default branch and CI still runs on them.
