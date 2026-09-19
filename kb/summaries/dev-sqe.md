# DEV-SQE Summary

Entries before 2026-09-18 were archived to `kb/archive/2026-02/dev-sqe.md` because they
described a February 2026 state that no longer held (ADR-0015). New entries are prepended
here by the `dev-sqe-handoff` skill and must carry the commit and a CI run URL for any
numeric claim.

## [2026-09-18 16:30:00] Session e274d4ad

**Branch**: claude/ptpd-validation-sdlc-plan-j2a2gc
**Commit**: 843566e
**PR**: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/pull/37
**CI**: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/actions/runs/35368643634

### Summary
Phase 1 defect work, then the first Phase 1 feature. With Phase 0's gates in
place, reading the science paths against the physics surfaced defects a green
CI had been hiding, each fixed at its cause with a test that fails on the old
code.

Density was under-reported by roughly a factor of two because the extractor
took the log of gamma-encoded scanner values; a print already at full black was
told to expose longer. A lookup table was shared between different curves
because the cache was keyed on a name that every generated curve carries.
Curve generation accepted input that silently produces a ruined negative. The
frontend and the API disagreed on four field names, fixed at the mechanism by
generating the client's types from the schema and failing CI on drift. Logging
was installed by accident at whatever import order happened, and the health
endpoint reported a hard-coded string.

The imaging pipeline did not survive a 16-bit scan at all. The decode guard
refused it outright, three stages clipped or converted it, export widened an
8-bit result and called it 16-bit, and the AI entry point carried its own copy
of the clipping conversion. Depth is now carried through decode, load, curve,
inversion, preview and export, reduced only where the requested output format
demands it and then by scaling. ADR-0016 records the decision.

An adversarial subagent review of that change found fifteen further findings,
six of them real defects the first pass missed, including that the whole
feature was unreachable for any scan above the decode limit. All are addressed
in a follow-up commit with tests; several of the original tests asserted the
image mode rather than its pixels and passed on a frame that had been clipped
to white.

`POST /api/export/negative` closes the largest product gap ADR-0004 lists as
blocking Gradio retirement: until now only the frozen UI could make a negative.
The endpoint reuses the existing upload guards, carries its own decode limits
because a negative is printed at full size, and deletes both temporary files.
The client call is typed from the generated schema.

### Tasks
Phase 1 defect items and the negative export endpoint are complete. Wiring the
React export panel to the new endpoint is the remaining half of that plan item.
TASK-001 and TASK-011 remain blocked on the owner actions in plan section 9.

### Quality
Lint, format, typecheck and the schema-drift checks are clean; `all-green`
passed on this head (see the CI link above). `dependency review` is red for a
repository setting the owner must enable and has been reported on the PR.

---

## [2026-09-18 13:11:57] Session e274d4ad

**Branch**: claude/ptpd-validation-sdlc-plan-j2a2gc
**Commit**: d60bc24
**PR**: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/pull/37
**CI**: https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/actions/runs/35348644644

### Summary
Implemented Phase 0 of the validation-first SDLC plan. One SHA-pinned CI
workflow with a single required check replaces three red ones; uv lockfile and
declared dependencies make the package importable from its own metadata; pytest
runs strict with warnings-as-errors; ruff, mypy and the documentation-placement
rule are at zero; immediate security fixes land with settings-backed limits;
three curve defects (spline knots, endpoint pinning, .quad round trip) are
fixed with regression tests; Hypothesis property suites are added; the
documentation is reduced to the canonical set with fifteen ADRs.

Warnings-as-errors exposed and this branch fixed: pyplot figure retention,
a deprecated Pillow argument, an unclosed log handle, correlation of constant
input, the mean of an empty list, and environment leakage between test modules.

### Tasks
Completed TASK-002..010 and TASK-012. TASK-001 and TASK-011 remain blocked on
owner actions (default branch, rulesets, stale PRs, deploy secret).

### Handoff
kb/handoffs/20260918-131157_dev-sqe_to_pre-pr.md

---
