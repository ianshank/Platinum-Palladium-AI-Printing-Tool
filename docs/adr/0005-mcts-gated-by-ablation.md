# ADR-0005: MCTS is gated by a pre-registered ablation

- **Status:** proposed
- **Date:** 2026-09-18
- **Source:** docs/plans/2026-09-validation-sdlc-plan.md §4

## Context

No production code called the tree search; the search endpoint reported 800 simulations while running one; the trainer did not self-play; the simulator objective was nearly flat and one search dimension had no effect. The search space is five continuous parameters with a cheap simulator, for which Bayesian optimization is the canonical tool.

## Decision

`mcts/`, `neuro_symbolic/`, and `agents/` stay behind an extra. After the simulator objective is made reachable (SCI-01/02) and seeded (SCI-05), a pre-registered ablation (`expert-science.md` §6: random, Sobol, TPE, GP-UCB, differential evolution, grid; equal simulation budgets; 30 seeds; Wilcoxon and Cliff's δ) decides. Deletion rule: the engine is removed unless it beats random search by 0.02 absolute (p < 0.01) at every budget and is not beaten by TPE or GP-UCB by 0.02 at budgets up to 400. The result is recorded here.

## Consequences

README wording ("AlphaZero", "self-play") is removed until real self-play exists. Deleting the engine changes no user-facing behaviour today.
