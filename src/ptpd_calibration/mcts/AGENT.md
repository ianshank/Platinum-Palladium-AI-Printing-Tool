# MCTS Directory

## Purpose
Monte Carlo Tree Search engine for intelligent calibration optimization. Uses simulation-based search to find optimal curve parameters given physical printing constraints.

## Key Files
- `engine.py` — Core MCTS engine: selection, expansion, simulation, backpropagation
- `tree.py` — Tree data structures (nodes, edges, statistics)
- `simulator.py` — Print process simulator for rollout evaluation
- `constraints.py` — Physical constraints (ink limits, paper gamut, chemistry bounds)
- `agents.py` — MCTS agent coordination for multi-objective optimization
- `networks.py` — Optional neural network policy/value heads (requires PyTorch)
- `quality.py` — Print quality evaluation metrics
- `training.py` — Network training loop
- `export.py` — Export optimized parameters
- `config.py` — MCTS hyperparameters (exploration constant, rollout depth, etc.)
- `types.py` — Type definitions for MCTS state, actions, results

## Conventions
- **Physics-grounded**: Constraints in `constraints.py` encode real printing physics — do not relax without domain expertise
- **Optional PyTorch**: `networks.py` and `training.py` use lazy imports — MCTS works without neural networks (pure UCT)
- **Reproducibility**: All random operations use seeded `np.random.Generator` for deterministic testing
- **Config-driven**: Hyperparameters in `config.py` — never hardcode search parameters in engine code

## Testing
```bash
pytest tests/unit/ -v -k "mcts"
pytest tests/integration/ -v -k "mcts"
```

## Pitfalls
- PyTorch is optional — guard all torch imports with try/except
- Search budget can be expensive — respect `max_iterations` and `time_limit` from config
- Simulator fidelity affects result quality — validate against real print data

## Related
- `../api/mcts_router.py` — API endpoints for MCTS operations
- `../core/models.py` — Shared data models
- Frontend: `frontend/src/components/mcts/` — MCTS UI components
- Frontend: `frontend/src/api/mcts.ts` — MCTS API client
