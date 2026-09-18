# Roadmap

The single living roadmap. Detailed work items, acceptance criteria, and
sequencing live in [docs/plans/2026-09-validation-sdlc-plan.md](plans/2026-09-validation-sdlc-plan.md);
decisions live in [docs/adr/](adr/). Superseded roadmaps are under
[docs/archive/](archive/).

| Phase | Goal | Exit criterion |
| --- | --- | --- |
| 0 Establish truth | Protected trunk, one green CI run, importable package, honest README | One green `all-green` run on protected `main` |
| 1 Honest scientific core | Truthful API, provenance, reachable simulator objective on a log-exposure axis, seeds and goldens, property and metamorphic suites, image and parser hardening | Property, metamorphic, and golden suites green; no test can pass silently |
| 2 Decide MCTS | Pre-registered ablation against random, Sobol, TPE, GP-UCB, differential evolution, grid | ADR-0005 records the measured result and the code matches it |
| 3 Ship one product | API-first closure of the Gradio gaps, Gradio retirement, Docker Space from an attested image, auth and quotas, SBOM and provenance, `v1.0.0` | The public artifact is the FastAPI + React app deployed from a tag and it can export a digital negative |
| 4 Ground truth | Real-print holdout dataset with measurement envelopes, fitted physics constants, sim-to-real report, agent red-team gates | Recommendations carry provenance and an uncertainty band derived from measured prints |

Explicitly deferred: MCTS dashboard, Celery/Redis queue, PWA offline mode,
i18n, bounded-context package split before Phase 2, any exposure of the agent
framework before ADR-0010 is satisfied.
