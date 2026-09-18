# Scientific Validation & ML Engineering Review — MCTS / Simulator / Curves / Chemistry

Repository: `/home/user/Platinum-Palladium-AI-Printing-Tool` (read-only review, 2026-09-18)
Reviewer role: Scientific Validation & ML Engineering

Legend used throughout:
- **[V]** verified by reading code at the cited line or by executing it in `.venv/bin/python` (numpy 2.4.6; torch NOT installed in `.venv`, so all numeric probes ran the NumPy path `simulate_with_numpy`, which `_numpy_characteristic_curve` documents as a line-for-line replica of `CharacteristicCurve.forward`).
- **[I]** inference from domain knowledge or from combining verified facts; not directly asserted by the code.

Numeric probe (all [V], seed `np.random.default_rng(0)`, default `PhysicsConstants` and `MCTSSettings`):

| Probe | Result |
|---|---|
| Reported `dmax` over the 64 corners of the 6-D parameter box | 0.518 – 0.885 |
| Reported `dmax` over 5000 uniform random parameter sets (min / median / max) | 0.518 / 0.526 / 0.878 |
| Pre-clamp `ProcessParameters.dmax` at coating=3.0, exposure=600 s (the box maximum) | 0.974 |
| Fraction of random parameter sets whose pre-clamp dmax < dmin+0.5 (i.e. hit the clamp at `simulator.py:353`) | **61.1 %** |
| Overall quality score over the whole box (min / median / max) | 0.407 / 0.437 / 0.479 |
| Component means ± sd over 5000 random sets: linearity / dmax / smoothness / cost | 0.513±0.062 / **0.0006±0.0009** / 0.910±0.0065 / 0.497±0.208 |
| max abs Δdensity between `ferric_oxalate_pct=15` and `=27` (all else equal) | **0.0** |
| Non-monotone density curves in 2000 random sets | 0 |
| `RuntimeWarning: divide by zero encountered in log` at `simulator.py:368` | raised whenever `toe_position` clips to 0 (humidity ≥ 80 or ≤ 20 %RH) |

---

## 2. Epistemic loop: what is the source of truth for the value target?

**Answer [V]: the value target is `QualityScorer.score(ExtendedProcessSimulator.simulate(params))`, and nothing in the repository ever moves a measured density into the simulator's constants, the scorer, the value/policy networks, or the tree search. The loop is fully closed on hand-written code.**

Evidence chain:

1. Engine value. `engine.py:332-338` — `_evaluate` builds `parameters` (decided or random rollout), calls `self.simulator.simulate(parameters)` then `self.scorer.score(sim_result, target_curve=target_curve)`; the returned float is what `leaf.backpropagate(value)` (`engine.py:124`) pushes up the tree.
2. Training value. `training.py:352-356` — `_generate_episode_data` samples `params = self._sample_random_parameters()` (uniform, `training.py:389-404`), simulates once, and uses `quality` as `value_target` for every decision level (`training.py:371-375`). **No `MCTSEngine` is instantiated anywhere in `training.py`** [V, grep]; the docstring "Generate training data from one MCTS search episode" (`training.py:334-337`) is inaccurate. The policy target is a Gaussian bump centred on the *randomly drawn* value's bin (`training.py:369`, `452-493`), independent of quality — so the policy head is trained to imitate a uniform random sampler [V].
3. Networks never used in search. `engine.py:259-260` "Uniform prior (no neural network yet)"; `engine.py:275-276` "With network: use policy output (Phase 4 integration point)". `grep DualNetwork|predict\( engine.py agents.py` → no hits [V]. `_get_temperature` (`engine.py:384-405`) has no callers in `src/` [V]. The system is plain UCT with uniform priors and random rollouts; "AlphaZero-style" describes an aspiration, not the code.
4. PhysicsConstants are never fitted. All eight construction sites use defaults: `simulator.py:78`, `engine.py:54`, `training.py:212`, `agents.py:48`, `agents.py:227`, `constraints.py:568`, `:792`, `:912` [V]. No function in `src/` takes `CalibrationRecord`/densities and returns `PhysicsConstants` (`grep fit_physics|calibrate_physics|from_records` → none) [V]. `config.py:315-319` says constants are "configurable ... based on empirical calibration data", but the only configuration mechanism is constructor kwargs.
5. Real-measurement ingress points and where they dead-end:
   - `POST /api/mcts/feedback` (`mcts_router.py:439-461`) accepts `measured_curve` + `quality_rating` and only `logger.info`s them; comment at `:448`: "In production, this would go to a database" [V].
   - `MCTSSearchRequest.target_curve` (`mcts_router.py:55`) is placed in `context` (`:220`) but `CalibrationCoordinatorSubagent._coordinate_search` (`agents.py:415-458`) reads only `target_aesthetics`, `fixed_parameters`, `uv_source`; `grep target_curve agents.py` → none. **The API silently drops the user's target curve** [V].
   - `POST /api/mcts/search` never runs the tree search. `grep MCTSEngine src/` outside `mcts/engine.py` and `mcts/__init__.py` → **no production callers** [V]. The route runs two heuristic subagents and exactly one `simulator.simulate` call (`agents.py:466`), yet returns `num_simulations=settings.num_simulations` (`mcts_router.py:247`, default 800) [V]. The 5,161-LOC engine is reachable only from tests.
   - `MCTSEngine.search(target_curve=...)` does use the target in the linearity term (`quality.py:112-131`) — but only in the unreachable engine path.
6. Reverse contamination path (simulated → "measured"). `export.py:154` `measured_densities = self._generate_density_measurements(result.predicted_curve)` and `export.py:172` stores it in `CalibrationRecord.measured_densities`. `CurvePredictor.train` (`ml/predictor.py:44-52`) selects `database.get_all_records()` filtered only on non-empty `measured_densities`, never on `tags` [V]. Today `to_calibration_record` has no production callers (only `tests/unit/mcts/test_export.py:179,224,235,246,465`) [V], so contamination is latent, not active — but one `database.add_record(exporter.to_calibration_record(...))` would train the sklearn predictor on simulator output labelled as measurement.
7. The only real-data ML path (`CalibrationDatabase` → `CurvePredictor`, `ml/predictor.py:44-78`; features at `:308-320`) is disjoint from MCTS. `CalibrationDatabase._calculate_similarity` (`ml/database.py:107-155`) uses paper/chemistry/metal/contrast/developer/log-exposure, not humidity/temperature/densities.

Conclusion: the peer review's "closed loop, no physical ground truth" finding is correct and understated — there is no ingress for ground truth at all, the "self-play" does not self-play, the API does not call the search, and the one user-supplied target is discarded.

---

## 3. Physics model assessment

All mappings live in `ExtendedProcessSimulator.compute_process_parameters` (`simulator.py:86-180`) and are consumed by `_numpy_characteristic_curve` (`simulator.py:325-375`) / `CharacteristicCurve.forward` (`process_sim.py:156-180`). Exposure axis is `np.linspace(0,1,num_steps)` (`simulator.py:289`; torch: `:242`).

| Input (unit per `config.py`) | Maps to | Formula (defaults) | Monotone | Bounded | Plausible? | Notes |
|---|---|---|---|---|---|---|
| `metal_ratio` (Pt fraction 0–1, `config.py:37-43`) | `gamma` | `m·1.6 + (1−m)·2.2` (`simulator.py:110-113`) | ↓ in Pt | [1.6, 2.2] | **Direction unverifiable [I]** | "gamma" here is a power-law exponent on *linear relative exposure 0–1* (`simulator.py:356`), not sensitometric γ = dD/dlogE. With linear-exposure abscissa a higher exponent bows the curve; whether Pd should bow more than Pt cannot be checked without data. Naming invites misreading. |
| `coating_weight` ("ml/sq-inch" 0.5–3.0, `config.py:44-50`) | `dmax` capacity | `min(0.3·cw, 2.8−0.08)` (`simulator.py:121-124`) → 0.15–0.90 | ↑ | yes | **Unit implausible [V]** | Chemistry module uses 0.465 drops/sq-in at 20 drops/ml = 0.023 ml/sq-in (`config.py:421-430`); cyanotype uses 0.015 ml/sq-in (`config.py:677`). MCTS unit is ~65× off. Ceiling 2.72 needs cw ≥ 9.07, unreachable. |
| `exposure_time` (seconds 30–600, `config.py:58-64`) | `dmax` realisation | `1 − exp(−t/120)` (`simulator.py:128`) | ↑ | (0,1) | Form plausible; **scale wrong [V]** | Combined `dmax = 0.08 + coating·exposure ≤ 0.974`. `target_dmax=2.0` (`config.py:176-181`) is unreachable → `_dmax_score ≤ 0.0066` everywhere → the 0.3-weight dmax objective is flat (probe: mean 0.0006). 61 % of the box is then clamped to `dmin+0.5` at `simulator.py:353`, where coating and exposure have **zero** effect. |
| `ferric_oxalate_pct` (% 15–27) | `contrast` | `clip(1 + 0.03·(fo−20), 0.5, 2.0)` (`simulator.py:140-144`) | ↑ | yes | **Dead parameter [V]** | `contrast` is not passed to `_CharacteristicCurve(...)` (`simulator.py:222-227`) nor to `_numpy_characteristic_curve` (`:292-299`). Probe Δ = 0.0. `test_ferric_oxalate_affects_contrast` (`test_simulator.py:124-140`) only checks the intermediate dataclass, so it passes. One of six search dimensions is pure noise for the optimizer. |
| `developer_temp` (°C 20–50) | `shoulder_position` | `clip(0.85 + 0.01·(T−25), 0.5, 1.0)` (`simulator.py:149-155`) | ↑ | yes | **Weak / direction unverifiable [I]** | Effective `shoulder_strength = shoulder·0.5` exactly (sigmoid∘logit, `simulator.py:362-363`). Quality Δ across 20→50 °C is 0.004 (probe). Real developer-temperature effects in Pt/Pd are on Dmax and tone [I]; `dev_temp_rate_slope` (`config.py:382-387`) is unused [V]. |
| `humidity` (%RH 30–80) | `toe_position` | `clip(0.15 − 0.005·|h−50|, 0, 0.5)` (`simulator.py:159-165`) | V-shaped | yes | **Doc/code mismatch + numeric hazard [V]** | `config.py:374` says "deviation *above* optimal"; code uses `abs()`. Toe clips to exactly 0 at |h−50| ≥ 30 → `np.log(0/…)` = −inf → `RuntimeWarning` (`simulator.py:368`); works only because `sigmoid(−inf)=0`. Symmetric effect is physically doubtful [I]: humidity is generally reported to shift speed/Dmax asymmetrically. |
| `paper_type`, `uv_source` | nothing | — | — | — | **Metadata only [V]** | `simulate()` receives only the `params` dict (`simulator.py:182-186`); `paper_dmin_base` is a constant 0.08 (`config.py:408-413`). Two different papers produce identical curves. |
| Constants with **no consumer** [V] | `exposure_dmax_rate` (`config.py:350-355`, "used in quality scoring tests" — only asserted at `tests/unit/test_mcts_core.py:173`), `dev_temp_rate_slope` (`:382`), `shoulder_compression_factor` (`:428`), `toe_expansion_factor` (`:440`) | The 0.5 / 0.3 strengths are hard-coded at `simulator.py:363,369` and `process_sim.py:170,174`. | | | | |

Abscissa inconsistency [V code / I domain]: the simulated "density curve" is sampled on *linear* relative exposure 0..1 (`simulator.py:289`), whereas a measured 21/31-step tablet is sampled on *log* exposure (≈0.15 D per step for a Stouffer 21-step). `QualityScorer._linearity_score(curve, target_curve)` compares them index-by-index (`quality.py:112-134`). Even a perfect physics model would then be compared against real step-wedge data on the wrong x-axis. This must be fixed before any sim-to-real metric is meaningful.

Unit cross-reference (sign/unit errors searched for):
- exposure_time: seconds in MCTS (`config.py:63`, half-life 120 s at `:362-367`), `CalibrationRecord` (`core/models.py:251`), `RecipeDataGenerator` uniform(60,600) (`data_generators.py:1077`). Consistent [V].
- humidity: %RH in MCTS (`config.py:77`), `CalibrationRecord` 0–100 (`models.py:253`), API "Humidity %" (`api/deep_learning.py:79`). Consistent [V]. No fraction form found.
- temperature: MCTS `developer_temp` °C (`config.py:65-71`); `CalibrationRecord.temperature` is *ambient* °C (default 21.0 at `models.py:273`, `api/deep_learning.py:80`, `ml/predictor.py:320`). `export.py:171` writes `temperature=developer_temp` — **semantic slot mixup, same unit** [V]; locked in by `tests/unit/mcts/test_export.py:192` (`assert record.temperature == 25.0`). Silver-gelatin module carries both `temperature_c` and `temperature_f` fields (`silver_gelatin_calculator.py:98,160`) — no cross-use with Pt/Pd found.
- coating_weight: ~65× unit mismatch vs chemistry (above) [V].
- density targets: `target_dmax` 2.0, `exposure_dmax_ceiling` 2.5, `coating_weight_dmax_ceiling` 2.8. For reflection Dmax of Pt/Pd on matte papers these are high [I; typical published values are ≈1.4–1.7 visual]; needs a measured anchor before being used as objectives.
- Sign errors: none found in the arithmetic itself; the negative `humidity_uniformity_slope` (`config.py:370-375`, `le=0.0`) combined with `abs()` yields the intended "reduce toe away from optimum" [V]. Direction of every mapping is unvalidated [I].

Test-suite reality check [V]: `tests/unit/mcts/test_simulator.py` asserts ordering/sign on intermediate `ProcessParameters` only (`:94-122`, `:124-140`, `:142-161`, `:181-194`), never on the resulting density curve; `test_extreme_exposure_time` (`:371-382`) checks `dmax ≤ ceiling+0.1` (trivially true at 0.9); `TestNumpyTorchEquivalence` skips without torch and uses a 0.01 tolerance (`:293`). `test_quality.py` never scores a simulator output against the 2.0 target with real ranges. `test_engine.py:492-493` asserts `quality_score > 0.1` for "convergence" — the floor of the objective is 0.407, so this cannot fail.

---

## 4. Metamorphic relations and property invariants

Tolerances are absolute unless stated. Hypothesis 6.168.0 is importable in `.venv` but not declared in `pyproject.toml`/`uv.lock` (grep → none) [V]; it must be added to dev deps.

### 4a. Simulator (`ExtendedProcessSimulator`) and scorer (`QualityScorer`)

| # | Relation | Expected direction / equality | Function under test | Tolerance | Status today |
|---|---|---|---|---|---|
| MR-S1 | Increase `exposure_time`, all else fixed | `ProcessParameters.dmax` non-decreasing; `SimulationResult.dmax` non-decreasing | `compute_process_parameters`, `simulate` | ≥ −1e-12 | holds (plateau under clamp) |
| MR-S2 | Increase `coating_weight` | same as S1 | same | ≥ −1e-12 | holds |
| MR-S3 | `metal_ratio` m | `gamma(m) == m·pt + (1−m)·pd`; `gamma(0.5)` = mean | `compute_process_parameters` | 1e-12 | holds |
| MR-S4 | Reflect humidity about optimum, h→100−h when optimum=50 | identical curves (documents current symmetric model; convert to xfail-strict when physics is revised) | `simulate` | 1e-12 | holds |
| MR-S5 | Change `ferric_oxalate_pct` 15→27 | curve **must change** (max Δ > 1e-3) | `simulate` | — | **fails** (Δ=0) → write as `@pytest.mark.xfail(strict=True)` so SCI-01 flips it |
| MR-S6 | Any params in `DEFAULT_PARAMETER_RANGES` (Hypothesis strategy) | `np.diff(density_curve) ≥ −1e-9` | `simulate` | 1e-9 | holds (0/2000) |
| MR-S7 | Endpoint consistency | `curve[0]==dmin`, `curve[-1]==dmax`, `density_range==dmax−dmin`, `dmin==clip(paper_dmin_base,0,0.5)` | `simulate` | 1e-9 | holds |
| MR-S8 | Refinement: `simulate(p,41).density_curve[::2] == simulate(p,21).density_curve` | equal (both are linspace samples of one function) | `simulate` | 1e-9 | expected to hold |
| MR-S9 | NumPy vs Torch path | `simulate_with_numpy == simulate_with_torch` | both | 1e-5 (float32) | untestable here (torch absent); current test uses 0.01 |
| MR-S10 | Default-fill invariance | `simulate({}) == simulate(defaults_from_DEFAULT_PARAMETER_RANGES)` | `simulate` | exact | should hold (`simulator.py:100-105` defaults equal `config.py` defaults) |
| MR-S11 | No warnings | `simulate` emits no `RuntimeWarning` for any params in range (`pytest.warns(None)`/`filterwarnings=error`) | `_numpy_characteristic_curve` | — | **fails** at toe=0 |
| MR-S12 | Affine invariance of shape scores | `_linearity_score(a·c+b) == _linearity_score(c)`, same for `_smoothness_score`, a>0 | `QualityScorer` | 1e-9 | expected to hold (range-normalised) |
| MR-S13 | Target identity | `_linearity_score(c, target_curve=c) == 1.0` | `_linearity_score` | 1e-12 | holds |
| MR-S14 | Dmax symmetry/maximum | `_dmax_score(t+d)==_dmax_score(t−d)`, `_dmax_score(t)==1` | `_dmax_score` | 1e-12 | holds |
| MR-S15 | Cost monotonicity | `_cost_score` non-increasing in `metal_ratio` and in `coating_weight`; equals 1.0 at (0, min_coating) | `_cost_score` | 1e-12 | holds |
| MR-S16 | Weight-scale invariance | scaling all four weights by k>0 leaves `score` unchanged | `score` | 1e-12 | holds (`quality.py:74-79`) |
| MR-S17 | Reachability (objective sanity) | over 2000 Sobol samples, `max(dmax) ≥ target_dmax` and `max(score) − min(score) ≥ 0.3` | `simulate`+`score` | — | **fails** (spread 0.07) → xfail-strict until SCI-01 |
| MR-E1 | Engine budget accounting | number of `simulator.simulate` calls during `search()` == `num_simulations + 1 + len(terminals scored in _extract_alternatives)` | `MCTSEngine.search` (wrap simulator) | exact | expected to hold |
| MR-E2 | Best dominates alternatives | `score(best_parameters) ≥ score(alt)` for every `alt` in `SearchResult.alternatives` | `_extract_result`/`_extract_alternatives` | 1e-12 | **likely to fail intermittently**: best follows greedy visit path (`engine.py:437-465`) while alternatives are quality-sorted with the top one dropped on the *assumption* it equals best (`engine.py:545-546`) |
| MR-E3 | Fixed-parameter invariance | `search(fixed={d:v}).best_parameters[d]==v` and every alternative has `[d]==v` | `search` | exact | partially tested (`test_engine.py:246-254`) |

### 4b. Curves

| Invariant | Function | Tolerance | Status [V] |
|---|---|---|---|
| Monotone enforcement: output non-decreasing; idempotent; `out[i] ≥ in[i]`; unchanged when input already monotone | `CurveGenerator._enforce_monotonicity` (`generator.py:302-309`), `AutoLinearizer._enforce_monotonicity` (`linearization.py:505-518`) | exact | untested as property |
| Resample idempotence: `resample(resample(c,n),n) == resample(c,n)` | `CurveModifier.resample` (`modifier.py:465-506`) | 1e-12 | holds (probe Δ=0.0) |
| Resample endpoint preservation: `y_new[0]==y_old[0]`, `y_new[-1]==y_old[-1]` when `x_old` spans [0,1] | same | 1e-12 | expected |
| Resample monotone preservation | same, `method in {"pchip","linear"}` | −1e-12 | holds; **`method="cubic"` violates** (probe min diff −0.0027) → property must exclude cubic or assert the violation |
| Output bounded [0,1], length == `num_points` | same | exact | holds |
| `.quad` round trip: `QTRExporter.export(c, p, format="quad")` → `QuadFileParser.parse(p).to_curve_data("K")` ≈ `resample(c,256,"linear")` | `export.py:103-151`, `parser.py:260-303` | **≤ 1/255 + 1e-9** | holds at 0.0039. Cause: exporter writes 16-bit (`export.py:141`), parser quantises to 8-bit (`parser.py:298`), so 16-bit precision is lost silently — a property test would document this; consider it a bug to fix (SCI-04). |
| `.quad` monotone & endpoint preservation under round trip (with `ink_limit=100`) | same | 1/255 | expected |
| Linearization residual: `LinearizationResult.residual_error` equals recomputed RMSE of applying `curve` to `measured_densities` vs `target_densities` | `AutoLinearizer.linearize` (`linearization.py:95-161`) | 1e-9 | untested |
| `generate_linearization_curve` output monotone, endpoints 0/1, length == settings | `generator.py:311-336` | 1e-9 | partially tested |

### 4c. Chemistry (`ChemistryCalculator`, `chemistry/calculator.py`)

| Invariant | Function | Tolerance | Status |
|---|---|---|---|
| Mass balance: `total_drops == fo + fo_contrast + pd + pt + na2`; `total_ml == total_drops/drops_per_ml`; each `_ml == _drops/drops_per_ml` | `calculate` (`calculator.py:224-363`) | 1e-9 | untested as property |
| Bostick-Sullivan rule: `pt+pd == fo + fo_contrast` (`calculator.py:210-212` docstring) | `calculate` | 1e-9 | untested |
| Ratio consistency: `pt/(pt+pd) == platinum_ratio`; `palladium_ratio == 1−platinum_ratio` | `calculate` | 1e-9 | partly |
| Area linearity: drops ∝ `(w−2m)(h−2m)`; `calculate(2w,2h)` with margin 0 gives 4× drops | `calculate` | 1e-9 | untested |
| `scale_recipe` linearity: every drop/ml/cost field == k×original; ratios, area, absorbency, method unchanged; `scale(scale(r,a),b) == scale(r,ab)`; `scale(r,1)` identity except `notes` | `scale_recipe` (`calculator.py:397-459`) | 1e-9 | only total/fo/pd at k∈{2,0.5} (`tests/unit/test_chemistry_calculator.py:288-297`) |

---

## 5. Seeded-determinism + golden-file test design

Every stochastic call site in the MCTS path [V, grep]:

| Site | Call | Injection |
|---|---|---|
| `engine.py:295` | `random.randint` (bin choice) | `self._rng.randint` |
| `engine.py:374` | `random.uniform` (rollout) | `self._rng.uniform` |
| `tree.py:228` | `random.choice` (all-unvisited) | `rng` parameter on `best_child` |
| `tree.py:234` | `random.choices` (temperature sampling) | same |
| `training.py:121` | `random.sample` (replay batch) | `ReplayBuffer(seed)` → own `random.Random` |
| `training.py:396,400` | `random.randint`/`uniform` (episode params) | `MCTSTrainer(seed)` |
| `networks.py:82,89` (`nn.Linear`), `:359-361` (`kaiming_normal_`, `zeros_`), `:85` (`Dropout` if >0) | torch global RNG | `torch.manual_seed(seed)` immediately before `DualNetwork(...)` (`training.py:217`) |
| `data_generators.py:128,138` | `np.random.default_rng(config.seed)` | already seeded — keep |
| `process_sim.py` | none | — |

Non-RNG nondeterminism to exclude from snapshots [V]: `CalibrationState.id` (`types.py:36`, uuid4), `SearchResult.id/timestamp` (`types.py:129-130`), `TrainingMetrics.timestamp` (`:181`), `search_time_seconds` (`engine.py:93,135`).

Design:
1. Add `seed: int | None = None` to `MCTSEngine.__init__` (`engine.py:38-44`); create `self._rng = random.Random(seed)`; thread it into `_generate_action`, `_rollout`, and `TreeNode.best_child(temperature, rng=...)`. Add `seed` to `MCTSSettings` (`config.py:82`) so it is env-overridable (`PTPD_MCTS_SEED`). Same for `ReplayBuffer` and `MCTSTrainer`. Keep the global-`random` fallback when `seed is None` for backward compatibility.
2. Golden fixtures `tests/golden/mcts/search_seed{0,1,2}_sims200_bins11.json` generated once with `num_simulations=200, action_bins=11, decision_order=all 6, c_puct=1.4`. Snapshot: `best_parameters` (round 9 dp), `quality_score` (9 dp), per-dimension **integer** visit counts from `root.get_visit_distribution()` walked along the greedy path (the normalised floats in `SearchResult.visit_distribution` are derivable), `root.visit_count`, total node count, top-5 `(bin_index, visit_count, mean_value 9dp)` at depth 1, and `alternatives` (9 dp). Comparison: ints exact, floats `abs=1e-9`. Also snapshot the simulator alone: `simulate(p,21)` for 5 fixed parameter sets (Hypothesis-independent regression of the physics).
3. Trainer golden: with `torch.manual_seed(0)`, snapshot `sha256(state_dict bytes)` after init and the first 3 `_train_step` losses (`abs=1e-6`, float32 CPU). Torch determinism: `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`, `torch.backends.cudnn.deterministic=True; benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` if CUDA. Record `torch.__version__` in the fixture and skip (not fail) on version mismatch, since default init kernels can change between releases.
4. Determinism assertions: run `search()` twice with the same seed → identical snapshot; run with different seeds → `best_parameters` differ in ≥1 dimension (guards against a seed that is ignored).
5. CI: run under `-W error::RuntimeWarning` for the MCTS package once SCI-01 removes the `log(0)`.

---

## 6. MCTS-vs-baselines ablation design

Precondition [V]: with the current simulator the objective spans 0.407–0.481 over the entire box, FO is dead and 61 % of the box is clamped. Any ablation run today measures noise. SCI-01/SCI-02 must land first, and the experiment should be re-run on real prints once SCI-10 exists.

- Objective: `f(p) = QualityScorer.score(simulate(p), target_curve)` on the 6-D box in `DEFAULT_PARAMETER_RANGES`, for 3 targets (ideal-linear, and two synthetic step-wedge targets on the corrected log-exposure axis).
- Budget: count of `simulator.simulate` calls, enforced by a counting wrapper (MR-E1). Budgets B ∈ {100, 400, 1600}. For MCTS, count `num_simulations + 1 + alternatives` and stop at exactly B.
- Baselines (equal B, same seed list): (i) uniform random search; (ii) Sobol/LHS quasi-random (scipy `qmc.Sobol`); (iii) Optuna TPE (`TPESampler(seed)`); (iv) GP-UCB via scikit-optimize `gp_minimize(acq_func="LCB")` or BoTorch if torch is installed; (v) scipy `differential_evolution` with `maxfun=B`; (vi) coordinate-wise grid at 21 bins per dim (equivalent to the tree's discretisation, to isolate the value of the tree from the value of the bins). None of optuna/skopt/botorch are installed in `.venv` [V]; add as an `[experiments]` optional extra, not a runtime dep.
- Reference optimum: `differential_evolution` with `maxfun=200 000`, best of 10 seeds; report simple regret `f* − f(best_found)` and normalised regret.
- Seeds: 30 per method × budget × target (Python/NumPy/Optuna seeds derived from one base seed).
- Metrics: median and IQR of best-found quality at B; regret; wall-clock per simulate call; fraction of runs within 0.01 of f*.
- Statistics: Wilcoxon signed-rank on per-seed best-found quality (MCTS vs each baseline, paired by seed), Holm-corrected across baselines; report Cliff's δ and 95 % bootstrap CI of the median difference. Pre-register α = 0.01.
- Decision rule (pre-registered): delete `mcts/engine.py`, `tree.py`, `training.py`, `networks.py` (keep `simulator.py`, `quality.py`, `constraints.py`, `export.py`) if at every B the MCTS median is not higher than **random search** by ≥ 0.02 absolute with p < 0.01, or if TPE/GP-UCB beat MCTS by ≥ 0.02 at B ≤ 400. Keep MCTS only if it beats TPE at B = 1600 with δ ≥ 0.3. Because the API never calls the engine (§2.5), deletion has zero user-facing behavioural change today [V].
- Effort: harness + counting wrapper 1 d; baselines + extras 1 d; runs (CPU, ~10⁶ simulate calls at ~50 µs each ≈ minutes) + analysis notebook 1 d; write-up 0.5 d → **≈ 3.5–4 days** after SCI-01/02. Re-run against real prints later: +2 d once SCI-10 data exist.

---

## 7. Real-print holdout dataset design

Schema (one row per print run; JSON Lines or Parquet; all fields with explicit units in the field name):

```
print_run_id            uuid
session_id              uuid            # multiple runs per darkroom session
paper_batch_id          str             # manufacturer lot, e.g. "ArchesPlatine-2026-03-L17"
paper_type              str             # enum-able; maps to PaperProfile
paper_humidity_conditioning_rh_pct  float|null
chemistry_lot_id        str             # FO/Pt/Pd bottle lot
sensitizer_fo1_drops, sensitizer_fo2_drops, sensitizer_pt_drops, sensitizer_pd_drops, sensitizer_na2_drops  float
sensitizer_total_ml     float           # derived: drops/drops_per_ml; record drops_per_ml used
coating_area_sq_in      float
coating_method          enum {brush, rod}
coating_weight_ml_per_sq_in  float      # derived; THIS is the unit MCTS must adopt (see §3)
dry_time_min            float
exposure_time_s         float
uv_source_id            str             # lamp model + age; irradiance_mw_cm2 if measured
developer_type          enum; developer_temp_c float; develop_time_s float
ambient_rh_pct          float; ambient_temp_c float
step_tablet_model       str             # e.g. "Stouffer T2115 21-step", with certified transmission densities file id
negative_media_id       str             # printer/ink/film for the digital negative
measurements[]          list of {step:int, rel_log_exposure:float, density:MeasurementEnvelope}
dmin_measurement, dmax_measurement      MeasurementEnvelope
replicate_index         int             # for repeatability
operator_id             str; notes str
```

MeasurementEnvelope (every density value carries it):
```
value                   float   # density
uncertainty_95          float   # half-width, same unit
method                  enum {densitometer_reflection_statusA, densitometer_reflection_visual, scanner_calibrated, spectro_Lab_to_density}
instrument_id           str     # e.g. "X-Rite 528 s/n …"
source_artifact_id      uuid    # scan file / instrument export hash
calibration_profile_id  uuid    # densitometer zero/cal-plaque record or scanner→density profile (reference tablet + date)
measured_at             datetime; n_readings int (averaged)
```
Existing hooks [V]: `MeasurementUnit` enum (`core/types.py:107-114`: VISUAL_DENSITY, STATUS_A, STATUS_M, LAB, RGB) and `DensityMeasurement.unit` (`core/models.py:51-58`) — extend `DensityMeasurement` with the envelope rather than inventing a parallel model; `CalibrationRecord` (`models.py:229-274`) lacks paper batch, chemistry lot, coating weight, developer temperature, instrument, and uncertainty.

Minimum sample size [I, standard power reasoning]: to estimate sim-to-real RMSE(D) with a 95 % CI half-width of 0.05 D when between-run sd of per-run RMSE is ≈0.10 D, n ≈ (1.96·0.10/0.05)² ≈ 16 runs. Batch effects dominate in Pt/Pd, so the *effective* n is the number of paper batches; recommend **≥ 40 print runs over ≥ 4 paper batches, ≥ 2 chemistry lots, 3 exposure levels (−1, 0, +1 stop), 2 metal ratios, each cell replicated ×2** (replicates give the repeatability floor the simulator cannot be expected to beat). First 12 runs (one batch, replicates) are enough to reject the current model on dmax alone.

Split policy: group split by `paper_batch_id` (leave-one-batch-out CV for reporting) and never split the 21 patches of one run across sets; hold out one entire chemistry lot as the final test; no random patch-level splits. Store split assignment as a column, frozen by hash.

Metrics: per-run RMSE and max|Δ| of density on the common log-exposure axis after SCI-02; dmax error; rank correlation of simulated vs measured quality across runs (this is what decides whether the optimizer's objective means anything); calibrated uncertainty check (fraction of measured points inside the simulator's stated ±band once SCI-11 fits constants with uncertainties).

---

## 8. NewType / physical-units assessment

Where a mixup is possible today (3 concrete sites, all [V]):

1. `mcts/export.py:171` — `temperature=developer_temp` writes the *developer bath* temperature (20–50 °C, `config.py:65-71`) into `CalibrationRecord.temperature` (`core/models.py:254`), whose semantics everywhere else are *ambient* (default 21.0 at `models.py:273`, `api/deep_learning.py:80`, consumed as an ML feature at `ml/predictor.py:320`). Same unit, different quantity; pydantic's `le=50.0` cannot catch it. `tests/unit/mcts/test_export.py:192` asserts the wrong behaviour. Distinct `AmbientCelsius` and `DeveloperCelsius` NewTypes on those two fields would make this a mypy error; the real fix is a separate `developer_temp_c` field on `CalibrationRecord`.
2. `core/models.py:263-274` `get_feature_vector` and `ml/predictor.py:315-320` `_record_to_features` build **positional** float lists `[ratio, has_agent, amount, log_seconds, %RH, °C]`. Swapping humidity and temperature, or feeding minutes instead of seconds, is invisible. A `NamedTuple` of NewTypes (`Seconds`, `RelativeHumidityPct`, `Celsius`) returned instead of `list[float]` gives both mypy and runtime field names.
3. `mcts/simulator.py:99-105` and `mcts/quality.py:221-222` read `dict[str, float]` bags with silent defaults: `{"humidity": 0.5}` (fraction) or `{"exposure_time": 3.0}` (minutes) are accepted and produce plausible-looking curves (toe→0, dmax→dmin). The same bag type is the engine's state (`mcts/types.py:37 decided_parameters: dict[str, float]`), `CalibrationAction.value: float` (`types.py:21`), and the constraint input `np.array([action.value])` (`constraints.py:81`). Related unit mismatch: `coating_weight` "ml/sq-inch" 0.5–3.0 (`config.py:44-50`, `quality.py:236-239`) vs chemistry's 0.023 ml/sq-in (`config.py:421-430`) — a `MlPerSquareInch` NewType at both boundaries would have forced someone to write the conversion.

Skeptical assessment: `typing.NewType` is erased at runtime and only helps where values flow through *typed scalar parameters*. Sites 1 and 2 qualify; site 3 does not — the `dict[str, float]` bag defeats NewType entirely. Recommended shape: (a) NewTypes `Density`, `Seconds`, `RelativeHumidityPct`, `AmbientCelsius`, `DeveloperCelsius`, `MlPerSquareInch` in `core/types.py`; (b) a pydantic `CalibrationParameters` model (fields typed with those NewTypes plus range validators, e.g. humidity must be ≥ 1.0 unless an explicit `fraction=True` conversion is used; exposure ≥ 5 s) replacing the dict at `simulate()`, `score()`, `CalibrationState`, and `CalibrationAction`; (c) `mypy --strict` gating on `mcts/`, `core/models.py`, `ml/predictor.py`, `chemistry/`. Effort is moderate; value is real only if (b) is done — NewTypes alone would catch site 1 and little else.

---

## Plan items

Effort: S ≤ 1 day, M = 2–4 days, L ≥ 5 days.

| ID | Title | Effort | Dependencies | Acceptance criteria |
|---|---|---|---|---|
| SCI-01 | Make the simulator's objective reachable and remove dead inputs | M | — | (a) `contrast` from FO% is consumed by both curve paths, MR-S5 passes; (b) rescale coating/exposure constants so `dmax` over the box spans at least [0.6, target_dmax+0.3] (MR-S17 passes: score spread ≥ 0.3); (c) clamp at `simulator.py:353` hit in < 5 % of Sobol samples; (d) `log(0)` removed (MR-S11 passes under `-W error`); (e) unused constants (`exposure_dmax_rate`, `dev_temp_rate_slope`, `shoulder_compression_factor`, `toe_expansion_factor`) either wired or deleted with tests updated; (f) `config.py:374` docstring matches `abs()` semantics or code changed. |
| SCI-02 | Put simulated curves on the step-tablet log-exposure axis | S | — | `simulate(params, num_steps, tablet=StepTablet.STOUFFER_21)` samples relative exposure `10^(−0.15·k)` (configurable step); `QualityScorer` compares target and simulated curves on the same axis; documented in `mcts/AGENT.md`; regression golden updated. |
| SCI-03 | Hypothesis metamorphic suite for simulator, scorer, engine | M | SCI-01 (for xfail flips), hypothesis added to `[dev]` | Tests for MR-S1…S17, MR-E1…E3 exist under `tests/property/mcts/`; MR-S5/S17 are `xfail(strict=True)` before SCI-01 and pass after; MR-E2 either passes or `_extract_alternatives` is fixed to derive alternatives from the same scored set as `best_parameters`. |
| SCI-04 | Property tests for curves and chemistry | S | hypothesis in `[dev]` | All §4b/§4c invariants implemented; `.quad` round-trip tolerance documented as 1/255 and a follow-up bug filed (or fixed) for the 16→8-bit quantisation at `parser.py:298`; `resample(method="cubic")` monotonicity violation either documented in the docstring or PCHIP made default for monotone inputs. |
| SCI-05 | Seed injection + golden-file determinism tests | M | — | `MCTSEngine`, `ReplayBuffer`, `MCTSTrainer` accept `seed`; `PTPD_MCTS_SEED` honoured; golden JSON fixtures for 3 seeds; same-seed runs identical (ints exact, floats 1e-9); different seeds differ; torch goldens skip on version mismatch; `mcts/AGENT.md` reproducibility note updated. |
| SCI-06 | Make the API truthful about what it runs | S | — | `/api/mcts/search` either invokes `MCTSEngine.search(target_curve=…, fixed_parameters=…)` or stops reporting `num_simulations`; `target_curve` is provably consumed (test with two different targets yields different results); `/feedback` persists to `CalibrationDatabase` with `provenance="measured"` or is removed. |
| SCI-07 | Real self-play: train on search statistics and use the network in search | L | SCI-05, SCI-01 | `_generate_episode_data` runs `MCTSEngine.search`; policy target = normalised root visit counts per depth; value target = search-result quality; `_expand` uses policy priors and `_evaluate` uses value head for non-terminals when a network is loaded; golden updated; ablation (SCI-09) includes "MCTS+network" arm. |
| SCI-08 | Provenance guard between simulated and measured records | S | — | `CalibrationRecord.provenance: Literal["measured","simulated"]` (default measured, export sets simulated); `CurvePredictor.train` and `CalibrationDatabase` queries exclude simulated unless asked; `developer_temp_c` added to `CalibrationRecord`, `export.py:171` writes it instead of `temperature`; `test_export.py:192` updated. |
| SCI-09 | MCTS-vs-baselines ablation | M (≈4 d) | SCI-01, SCI-02, SCI-05 | Pre-registered protocol in `experiments/ablation/README.md`; counting wrapper enforces equal budgets; 30 seeds × 3 budgets × 3 targets; Wilcoxon + Cliff's δ report committed; decision rule from §6 applied and recorded as an ADR (keep or delete engine). |
| SCI-10 | Real-print holdout dataset (schema, ingestion, metric) | L | SCI-02, SCI-08 | Schema from §7 implemented as pydantic models with `MeasurementEnvelope`; ingestion CLI validates units and envelope completeness; ≥ 12 initial runs (1 batch, replicates) collected; sim-to-real RMSE, dmax error, and rank-correlation report generated by `pytest -m realdata` (skipped when dataset absent). |
| SCI-11 | Fit `PhysicsConstants` from holdout with uncertainty | L | SCI-10, SCI-01 | `fit_physics_constants(records) -> (PhysicsConstants, covariance)` via least squares on log-exposure densities; leave-one-batch-out RMSE reported; fitted constants beat defaults on held-out batch; constants file versioned with dataset hash. |
| SCI-12 | Physical-unit types at boundaries | M | SCI-08 | NewTypes in `core/types.py`; `CalibrationParameters` pydantic model replaces `dict[str,float]` at `simulate`, `score`, `CalibrationState`, `CalibrationAction`; `coating_weight` expressed in the same unit as `ChemistrySettings` with a tested conversion; `mypy --strict` passes on `mcts/`, `core/models.py`, `ml/predictor.py`, `chemistry/`. |
