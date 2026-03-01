# .claude Directory

## Purpose
Claude Code configuration: agent definitions, custom commands, skills, hooks, and permission settings for the AI-assisted development workflow.

## Key Files
- `settings.json` — Global settings: permissions, allowed tools, hook configuration, model preferences
- `settings.local.json` — Local overrides (not committed — developer-specific)

## Key Subdirectories
- `agents/` — 5 sub-agent definitions:
  - `migration-coordinator.md` — Orchestrates overall migration
  - `ui-migration-agent.md` — Frontend component migration
  - `testing-agent.md` — Test writing and verification
  - `gap-remediation-agent.md` — Feature gap analysis and implementation
  - `documentation-agent.md` — Documentation maintenance
- `commands/` — Custom slash commands:
  - `migrate-component.md` — `/migrate-component` workflow
  - `verify-equivalence.md` — `/verify-equivalence` testing
  - `generate-tests.md` — `/generate-tests` scaffolding
- `hooks/` — Shell hooks for session lifecycle:
  - `kb-start.sh` — KB context loader (routes via `$CLAUDE_ROLE`)
  - Role-specific hooks: `planning-start.sh`, `dev-sqe-start.sh`, `pre-pr-start.sh`
- `skills/` — Handoff skills for the three-role workflow:
  - `planning-handoff/SKILL.md`
  - `dev-sqe-handoff/SKILL.md`
  - `pre-pr-handoff/SKILL.md`

## Conventions
- Agent definitions use YAML frontmatter + markdown body
- Commands and skills follow Claude Code skill/command format
- Hooks are bash scripts — must be executable (`chmod +x`)
- `settings.local.json` overrides `settings.json` — never commit local settings

## Pitfalls
- Do NOT modify `settings.json` without understanding permission implications
- Hook scripts must handle missing env vars gracefully (e.g., `$CLAUDE_ROLE` not set)
- Agent definitions reference tools by name — tool names must match Claude Code tool registry

## Related
- `../CLAUDE.md` — Root project instructions (always loaded)
- `../.agent/` — Additional agent rules, skills, and workflows
- `../kb/` — Knowledge base consumed by hooks and skills
