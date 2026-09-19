# Test Architecture & Quality Engineering Review

Repository: `/home/user/Platinum-Palladium-AI-Printing-Tool` (commit c9ef03a, 2026-03-01). Environment: Python 3.11.15 venv, pytest 9.1.1, hypothesis 6.168.0 (installed, unused), no torch / psutil / gradio / pytest-timeout / pytest-xdist; 4 cores. Node: vitest 2, happy-dom, Playwright browsers not installed. All runs used `-o addopts="" -p no:cacheprovider`.

Legend: **[M]** = measured in this session, **[E]** = estimated / heuristic.

Scratch artifacts (all under the scratchpad dir): `baseline.log`, `vitest.log`, `coverage.log`, `coverage.json`, `collect.txt`, `nodeids.txt`, `probe_props.py`, `probe_fail.py`.

---

## 1. Pass/fail baseline

### Backend (`tests/api tests/unit`, ignoring `tests/unit/test_monitoring_data.py` and `tests/unit/ui`) [M]

Command: `timeout 900 .venv/bin/python -m pytest tests/api tests/unit --maxfail=50 -q -o addopts="" -p no:cacheprovider --ignore=... -rfEs --durations=15` (`--timeout` and `-p no:randomly` dropped: neither plugin is installed).

| Result | Count |
|---|---|
| selected | 4,187 (4,231 collected in these dirs minus the 2 ignored locations) |
| passed | **4,036** |
| failed | **3** |
| skipped | **148** |
| warnings | 81 |
| wall time | 93.7 s (single process) |

Failures (one line each):

1. `tests/unit/test_neuro_symbolic.py:1067` `TestSymbolicRegressionIntegration::test_discover_and_evaluate_formula` — `assert -0.224 >= -1e-09`. Unseeded stochastic symbolic regression (5 generations, population 20; no `seed`/`random_state` anywhere in the file). Flaky by construction.
2. `tests/unit/test_session_logger.py:483` `TestSessionLogger::test_get_paper_statistics` — `assert 1 == 2`. **Real product bug**: `src/ptpd_calibration/session/logger.py:470` references an undefined name `paper_stats` (`paper_stats["avg_exposure"].append(...)`); the resulting `NameError` is swallowed by the blanket `except Exception` at `:471`, so only the first record of every session is counted.
3. `tests/unit/test_session_qa_workflow.py:348` `TestSessionLogger::test_get_paper_statistics` — `assert 1 == 3`. Same root cause as (2).

Skip reasons (148): 94 "PyTorch not available"; 13 module-level skip in `tests/api/test_ai_endpoints.py:17-20` ("Async chat endpoints cause TestClient deadlocks"); 15 `qrcode` missing; 11 `tests/unit/test_kb_protocol.py:446-563` skip when `kb/` handoff/summary files are absent (environment-dependent tests); the rest are PIL ImageCms, pyserial, pycups, pywin32, psutil, gradio, google-cloud imports.

Slowest tests: 5.5 s `test_integrations.py::test_complete_spectrophotometer_workflow`, 5.45 s `tests/api/test_scan_endpoints.py::test_upload_scan_different_tablet_types`, then 4.3 s x3 in `test_hardware_integrations.py` (simulated printer sleeps). No test exceeds 6 s, so a 120 s global timeout is safe.

### Frontend vitest (full run, `pnpm exec vitest run`) [M]

55 files, 831 tests: **826 passed, 5 failed in 4 files**, 10.2 s. Because `.github/workflows/ci.yml:41-42` runs `pnpm run test:coverage` with no `continue-on-error`, the frontend CI job is red today.

| Failing test | Reason (root cause) |
|---|---|
| `src/hooks/useKeyboardShortcuts.test.ts:309` "does not fire action when a select element is focused" | Hook only guards `HTMLInputElement`, `HTMLTextAreaElement`, `isContentEditable` (`src/hooks/useKeyboardShortcuts.ts:28-30`); no `HTMLSelectElement` guard. The test is right; the hook has a gap. |
| `src/components/Layout/Layout.test.tsx` "calls toggleSidebar on menu button click" | Test queries label "Open sidebar"; component uses `aria-label="Toggle sidebar"` (`Layout.tsx:152`) and "Close sidebar" (`:76`). Test drift. |
| `src/components/curves/CurveEditor.test.tsx` "saves curve via API and calls onSave callback" | Test expects `adjustment_type: 'none'`; component hard-codes `'brightness'` (`CurveEditor.tsx:295`). Test/implementation drift; needs a product decision on the save payload. |
| `src/stores/slices/uiSlice.test.ts:16` "has correct initial values" | `sidebarOpen` default is `false` (`uiSlice.ts:53`), test expects `true` (default changed 2026-02-22). Stale test. |
| `src/stores/slices/uiSlice.test.ts:34` "toggleSidebar toggles sidebar state" | Same as above. |

---

## 2. Scientific-core coverage (branch) [M]

Command: `pytest tests/unit ... --cov=src/ptpd_calibration/curves --cov=.../chemistry --cov=.../mcts --cov-branch` (99 s). Per-package numbers computed from `coverage.json`:

| Package | Line | Branch | Notes |
|---|---|---|---|
| `chemistry/` | **94.8 %** (456/481) | **85.7 %** (132/154) | `calculator.py` 99 % / 91 % |
| `curves/` | **89.9 %** (1474/1639) | **71.6 %** (365/510) | `parser.py` 79 % / 60 %; `modifier.py` 92 % / 63 %; `generator.py` 87 % / 70 %; `export.py` 90 % / 83 %; `linearization.py` 97 % / 84 % |
| `mcts/` | **78.2 %** (1196/1529) | **69.2 %** (220/318) | `networks.py` 0 %, `training.py` 29 % (torch-gated); `simulator.py` 78 % / **33 %** (torch path never executed); `quality.py` 97 % / 94 %; `constraints.py` 94 % / 84 %; `engine.py` 94 % / 76 % |
| combined | 85.7 % | 73.0 % | coverage "combined" figure 83.0 % |

Take-away: line coverage of the scientific core is healthy, but branch coverage of `modifier.py` (63 %) and `parser.py` (60 %) is where the property-based tests should land first; `simulator.py` branch coverage is dominated by the untested torch path.

---

## 3. Test-quality audit

### Repo-wide AST scan [E, heuristic]

5,025 `test_*` functions classified by their assertions (a test is "all-vacuous" when every `assert` is `x is not None`, `isinstance(...)`, `hasattr(...)`, `len(x) > 0`, `"key" in data`, `status_code == N`, `data["success"] is True`, or a bare name/attribute):

| Bucket | Count | Share |
|---|---|---|
| behavioral / other (at least one value-bearing assertion, `pytest.raises`, or mock `assert_called`) | 3,772 | 75.1 % |
| all assertions vacuous | 924 | **18.4 %** |
| numeric assertion with tolerance (`approx`, `allclose`, `abs(...) <`, `atol`) | 198 | **3.9 %** |
| no assertion at all | 131 | 2.6 % |

By directory: `api` 70 % all-vacuous; `e2e` 22 % no-assert + 19 % vacuous; `performance` 77 % vacuous; `sanity` 51 % vacuous; `unit` 16 % vacuous, 4 % tolerance-based; `visual` 100 % no-assert (10 tests).

**Estimate: roughly one test in five (18-21 %) is low-value, and only ~4 % of tests make a tolerance-bearing numerical claim — in a codebase whose core is numerical.**

### Sampled files [M, grep counts] with path:line examples

| File | tests / asserts | vacuous | tolerance | verdict |
|---|---|---|---|---|
| `tests/unit/test_mcts_core.py` | 80 / 237 | 25 flagged | ~10 | Many init smoke tests: `:447-448` (`simulator.physics is not None`), `:910-913` (engine attrs not None), `:928-929` (`len(result.best_parameters) > 0`), `:192-202` (`hasattr(physics, ...)`), `:1026,1036,1182` (`len(violations) > 0`). Zero mocks. The engine tests (`:915-935`) run 50 simulations and only check types/ranges — they cannot detect a broken search. |
| `tests/unit/mcts/test_simulator.py` | 34 / 65 | ~5 | ~28 | Best file in the sample: physics with tolerances (`:71,81,234-245`), ordering relations (`:106,122,139-140`). `:269-298` is a real torch-vs-numpy differential test but it is skipped wherever torch is absent (this venv and CI, see section 8). Tolerance 0.01 is too loose (see section 8). |
| `tests/unit/test_modifier.py` | 34 / 48 | 1 | 10 | Genuine behavioral: directional deltas (`:78,105-118`), `np.allclose` for blends (`:274-300`), monotonicity (`:333,346`). |
| `tests/unit/test_curves.py` | 23 / 61 | 3 | 13 | Mostly good (`:37-40, 102, 113, 254-257`); export tests are string-presence only (`:167-171, 183-185`); JSON/CSV round trip exists (`:225-242`) but **no `.quad` round trip**. |
| `tests/unit/test_chemistry_calculator.py` | 52 / 95 | 2 | 8 | Behavioral orderings (`:150-151, 189-203`), scale linearity (`:288-297`); a few `> 0` only (`:135-139, 238`). |
| `tests/api/test_curve_endpoints.py` | 19 / 39 | **17 (44 %)** | 0 | Pattern is `status_code == 200` + `data["success"] is True` (`:24-29, 89-92, 106-109, 197-200, 255-258`); only `:231-232` compares a value. No numeric verification of any curve returned by the API. |
| `tests/e2e/test_journey.py` | 1 / 1 | 0 | 0 | Entire test `@pytest.mark.skip` (`:17`), file also `importorskip("playwright")`. |
| `tests/e2e/test_user_journeys.py` | 19 / 111 | 33 (30 %) | 0 | `:57-58, 69, 81-83, 171-172, 181-182, 211` are `is not None`/`len > 0`. |
| `tests/unit/mcts/test_quality.py` | 32 / 37 | 0 | 4 | Not vacuous but ~19 asserts are only `0.0 <= score <= 1.0` (`:145-146, 211, 321-342, 393-394, 421-519`); good ordering tests at `:114, 130, 239, 256`. |
| `tests/unit/mcts/test_constraints.py` | 47 / 121 | 7 | 3 | Mostly behavioral. |

Mock-heavy-without-behavior: `tests/api/test_ai_endpoints.py` (`:134 test_suggest_recipe_mocked`, `:192 test_troubleshoot_mocked`) — moot because the module is skipped. Snapshot-of-implementation: `tests/unit/test_chemistry_calculator.py:85-91` asserts exact drop counts (`12.0`) derived from the same settings the code uses — brittle but at least value-bearing.

### Findings from exploratory Hypothesis probes (scratch `probe_props.py`, 34 candidate invariants, 150 examples each, derandomized) [M]

29 PASS / 5 FAIL. The failures are real findings, not test bugs:

* **F1 — `CurveModifier.smooth(method=SPLINE)` crashes for curves with fewer than 10 points**: `modifier.py:349-351` uses `subsample = max(10, ...)` then `np.linspace(0, n-1, subsample).astype(int)` → duplicate knots → `PchipInterpolator: "x must be strictly increasing"`. Shrunk example: `output_values=[0.0, 0.0, 0.0]`, any strength.
* **F2 — `adjust_contrast` / `adjust_gamma` (and by the same mechanism all `adjust_*`) break monotonicity when the curve's endpoints are not (0, 1)**: `preserve_endpoints` pins `outputs[0]`/`outputs[-1]` back to the *original* values after transforming the interior (`modifier.py:141-143, 168-170`). Shrunk examples: `[0.0, 0.75, 0.75]` with contrast `+1.0` → `[0.0, 1.0, 0.75]`; `[0.0, 0.5, 0.5]` with gamma `0.5` → `[0.0, 0.707, 0.5]`. Property holds when the curve is anchored 0→1.
* **F3 — `save_curve(curve, "x.quad")` followed by `load_curve("x.quad")` always fails** with `ValueError: No curve data found`: the QTR exporter writes the list format (`export.py:103-147`), but `_load_text_curve` only parses `key=value` lines (`export.py:355-391`). The `QuadFileParser` *can* read it (`export → parser.parse` agreed within 2/255, PASS). The public round-trip contract is broken and untested.
* **F4 — `ParameterBoundsConstraint`: `loss_value == 0.0` does not imply `is_satisfied`** for sub-normal out-of-range values (`v = -2.2e-175` → loss underflows to 0.0, satisfied False). Harmless in practice; the invariant to test is `is_satisfied ⇔ in-range`, and `is_satisfied` must remain the pruning source of truth (`constraints.py:960-970`).
* F5 — Savitzky–Golay smoothing is not shape-preserving (`[0,0,0,0,1]` → overshoot). Expected; must not be asserted as an invariant.

Also observed: `mcts/simulator.py:368` emits `RuntimeWarning: divide by zero encountered in log` when `toe_position == 0` (clip lower bound).

---

## 4. The "4,400+ tests vs 104 + 726 + 9" discrepancy [M]

* `pytest --collect-only`: **4,687 node IDs** from 142 files (5 collection errors). Of those, **133 IDs come from 36 parametrized functions**; unique test functions = **4,590**. Parametrize inflation is therefore only +97 IDs (**2.1 %**) — this is *not* the source of the gap.
* AST count of `def test_`: 5,025. The ~435 uncollected definitions are: 137 in `tests/unit/test_monitoring_data.py` (psutil missing), 101 in `tests/unit/ui/*` (gradio missing), and the remainder are definitions pytest cannot see (duplicate names shadowed inside a class, `test_`-prefixed helpers outside `Test*` classes).
* "104 API passing" = the 117 `tests/api` IDs minus the 13 skipped chat tests — **exact match** with this run.
* "726 vitest passing" is stale: today 831 tests, 826 passing.
* "9/9 Playwright" = **9 `test()` calls in one spec file** (`frontend/e2e/app.spec.ts`), not 9 spec files; unverifiable here (no browsers installed).
* "84 % backend coverage" excludes `*/ui/*` and `api/server.py` (`pyproject.toml [tool.coverage.run] omit`), and the CI unit job ignores 6 test files (`ci-cd.yml:83-88`), runs e2e with `|| true` (`:106`), and `tests.yml` marks selenium/performance/visual/coverage-threshold steps `continue-on-error` (`:168, 199, 237, 291`).

So the numbers are *different populations*: 4,400+ is the whole pytest collection (of which 308 e2e — 44 selenium — and 94 torch tests never execute in CI), while 104/726/9 are the React-migration-relevant subsets. In this environment the honest backend figure is **4,187 selected → 4,036 pass / 148 skip / 3 fail**.

**Trustworthy headline metric**: a CI-generated table per suite — *collected / passed / skipped (grouped by reason) / failed*, plus *line and branch coverage per risk tier* — produced from the artifact of the latest `main` run, not typed into README. Raw test counts should never appear without the skip count next to them.

---

## 5. Property-based test plan (Hypothesis, already installed; add to `[project.optional-dependencies].dev`)

Shared strategies (`tests/property/strategies.py`):

```python
unit = st.floats(0, 1, allow_nan=False, allow_infinity=False)
@st.composite
def curves(draw, n_min=3, n_max=256, monotone=False, anchored=False):
    n = draw(st.integers(n_min, n_max)); ys = draw(st.lists(unit, min_size=n, max_size=n))
    if monotone: ys = sorted(ys)
    if anchored: ys[0], ys[-1] = 0.0, 1.0
    return CurveData(name="c", input_values=list(np.linspace(0, 1, n)), output_values=ys)
densities = st.lists(st.floats(0, 3), min_size=3, max_size=41).map(sorted).filter(lambda d: d[-1]-d[0] >= 0.02)
sim_params = st.fixed_dictionaries({"metal_ratio": st.floats(0,1), "coating_weight": st.floats(0.5,3), "ferric_oxalate_pct": st.floats(15,27), "exposure_time": st.floats(60,600), "developer_temp": st.floats(15,35), "humidity": st.floats(30,70)})
```
Profiles: `ci` = 200 examples, `deadline=None`, `derandomize=True`, `database=None` (deterministic in CI); `nightly` = 2,000 examples, randomized, DB kept as artifact. Shrinking: Hypothesis shrinks lists toward shorter/lower values, so counterexamples surface as 3-point curves like `[0.0, 0.0, 0.0]`, `[0.0, 0.5, 0.5]` (as observed) — ideal for bug reports.

### `curves/modifier.py`

| Function | Invariant (exact) | Strategy | Status / shrink expectation |
|---|---|---|---|
| all `adjust_*` | neutral amount is identity (`gamma=1`, others `0`) within 1e-12; outputs in [0,1]; length preserved; `notes` appended | `curves()` × `st.floats(-1,1)` | PASS |
| `adjust_*` with `preserve_endpoints=True` | monotone in ⇒ monotone out **only for anchored curves**; for un-anchored curves mark `xfail(strict=True)` citing F2 until endpoint pinning is redesigned (e.g. pin then re-run `enforce_monotonicity`, or interpolate the pin) | `curves(monotone=True, anchored=True/False)` | PASS anchored / FAIL un-anchored; shrinks to 3 points |
| `adjust_brightness` | amount>0 ⇒ pointwise ≥ input; amount<0 ⇒ ≤ ; `adjust_brightness(c, a)` then `(-a)` is *not* an inverse because of clipping — do not assert | `curves()` × `floats(0,1)` | new |
| `adjust_levels` | `black=0, white=1, mid=0.5` identity except pinned endpoints; output endpoints are exactly 0.0/1.0 (`modifier.py:204-206`) | `curves()` | new |
| `smooth` GAUSSIAN / MOVING_AVERAGE | length, bounds, endpoints preserved; monotone in ⇒ monotone out | `curves(monotone=True)` × `floats(0,1)` | PASS |
| `smooth` SAVGOL | bounds/endpoints/length only (no shape claim, F5) | same | PASS |
| `smooth` SPLINE | must not raise for any n ≥ 3 (F1 fix: `subsample = min(n, max(10, ...))`) | `curves(n_min=3, n_max=9)` | FAIL today; shrinks to n=3 |
| `enforce_monotonicity` | idempotent; non-decreasing; pointwise ≥ input; output[0]==input[0]; "decreasing" is the mirror image via `invert` | `curves()` | PASS |
| `blend` | AVERAGE/MULTIPLY/SCREEN/MIN/MAX commutative (1e-12); WEIGHTED w=0 → resampled a, w=1 → b; MULTIPLY ≤ min, SCREEN ≥ max; length = max(n1,n2); `blend(c,c,AVERAGE) == resample(c)` | `curves()` × `curves()` | PASS |
| `resample` | length = `num_points`; PCHIP & linear preserve monotonicity; knots reproduced exactly when new grid nests old grid `((n-1) % (m-1) == 0)`; `resample(resample(c,N),m)` ≈ `c` at knots | `curves(monotone=True)` × `integers(3,300)` | PASS; note "cubic" is not shape-preserving — assert bounds only |
| `invert` | involution (1e-12); `reverse` involution | `curves()` | PASS |

### `curves/generator.py`

* Bounded [0,1], monotone non-decreasing (setting `monotonicity_enforcement=True` default), length = `settings.num_output_points` (256), `input_values == linspace(0,1,256)`. PASS.
* Affine invariance: `generate(k·d + b) == generate(d)` for k>0 (normalization at `generator.py:126-131`). PASS (1e-6).
* Linear-density identity: `generate(linspace(0.1,2.0,n))` ≈ identity, max |y−x| < 0.02. PASS.
* Composition property (the actual purpose of the curve): with `measured_norm(x)` = PCHIP through the normalized densities, `measured_norm(curve(x)) ≈ target(x)` within 0.03 at the step positions, for all three `CurveType`s. New; this is the one that catches a wrong inverse.
* Validation: `< 2` steps and range `< 0.01` raise `ValueError` (`:123-131`); use `st.lists(max_size=1)` and near-constant lists.

### `curves/linearization.py`

* All 5 methods: output length = `output_points`, values in [0,1], `residual_error ≥ 0`, `max_deviation ≥ residual_error`; `preserve_endpoints` ⇒ `y[0]==0, y[-1]==1`. PASS (rms ≤ max verified).
* DIRECT_INVERSION on linear densities ≈ identity (0.02). PASS.
* `refine_curve(curve, target_densities_measured_perfectly)` returns a curve within 1e-6 of `curve` resampled (zero error ⇒ no correction; `linearization.py:191-201`). New.
* Monotone input ⇒ monotone output for DIRECT_INVERSION, ITERATIVE, HYBRID (they call `_enforce_monotonicity`); SPLINE/POLYNOMIAL: bounds only.

### `curves/parser.py` + `curves/export.py` (.quad round-trip)

* `QTRExporter.export(..., format="quad")` → `QuadFileParser.parse` → channel K normalized ≈ `np.interp(linspace(0,1,256), c)` within **2/255** (16-bit→8-bit truncation at `parser.py:298-302` costs up to 1/255 plus `int()` truncation). PASS. Tighten to 1/65535 once the parser stores 16-bit values (recommended: the 8-bit down-conversion silently destroys QTR precision).
* Ink limit: exporting with `ink_limit=L` scales the K channel by L/100 (max value ≤ `65535·L/100`). New.
* `save_curve(c, x.quad)` → `load_curve(x.quad)` ≈ c: **FAIL today (F3)**; fix `_load_text_curve` to delegate to `QuadFileParser` for `## QuadToneRIP` files, then assert 2/255.
* JSON round trip exact; CSV round trip 1e-6. PASS.
* Parser robustness: `parse_string` never raises on arbitrary text (`st.text()`), and on `[K]`-section format with `st.dictionaries(integers(0,255), integers(0,255))` reproduces exactly the written indices; channels with all zeros are `enabled=False` (`parser.py:381-383`).

### `chemistry/calculator.py`

* `total_drops == fo + fo_contrast + pd + pt + na2` (1e-6); **A+B == C rule**: `fo + fo_contrast == pd + pt` (1e-6); `total_ml · drops_per_ml == total_drops`; all components ≥ 0; `platinum_drops/metal_total == platinum_ratio`. PASS.
* `scale_recipe(r, k)`: every drop/ml field and `estimated_cost_usd` scale by exactly k; metadata unchanged; `k ≤ 0` raises. PASS.
* Monotonicity: `total_drops` non-decreasing in width and height; HIGH absorbency ≥ MEDIUM ≥ LOW; ROD ≤ BRUSH (settings multipliers). PASS for width.
* Margin: coating area = `max(0.5, w−2m)·max(0.5, h−2m)` — property `coating_area ≤ w·h`. New.
* Cost monotone non-decreasing in `platinum_ratio` when `platinum_cost_per_ml ≥ palladium_cost_per_ml`. New.
* Strategies: `floats(0.5, 40)` for dimensions, `floats(0,1)` ratios, `floats(0.1,10)` scale; shrinks toward `w=h=0.5`, ratio 0.

### `mcts/quality.py`

* `score()` and each sub-score in [0,1] for arbitrary `density_curve` (len ≥ 2, values in [0,4]) and parameters. PASS.
* `_dmax_score` symmetric about `target_dmax`, equals 1.0 at target, strictly decreasing in |deviation|. PASS (symmetry).
* Perfect linear ramp ⇒ `_linearity_score > 0.999` and `_smoothness_score > 0.999`; flat curve ⇒ linearity 0.0, smoothness 1.0 (`quality.py:138-141, 194-196`). PASS.
* Scale invariance: `_linearity_score(k·c + b) == _linearity_score(c)` (normalized by range) and same for smoothness. New.
* Weighted-sum sanity: with all weights equal to 0 except one, `score == that sub-score`. New.

### `mcts/constraints.py`

* `ParameterBoundsConstraint`: `is_satisfied ⇔ min ≤ v ≤ max`; `loss ≥ 0`; `is_satisfied ⇒ loss == 0` (one direction only, F4); violations non-empty iff unsatisfied. PASS (with the corrected direction).
* Soft constraints (FO, MetalRatio, Exposure, DevTemp, Humidity, Coating): `loss ≥ 0`, continuous at the sweet-spot edges (loss → 0 as v → edge), symmetric quadratic growth outside. FO PASS.
* `ActionPruner.prune_actions`: result ⊆ input, order preserved, keeps exactly the in-range actions; `score_action ∈ [0,1]`; every dimension's `default_value` scores 1.0. PASS.
* Wrong-arity input (`len(values) != 1`) ⇒ unsatisfied with the documented sentinel loss (1000 / 100). New.

### Six metamorphic relations for `mcts/simulator.py` (shared with the science expert; all verified on the NumPy path) [M]

| # | Relation | Status |
|---|---|---|
| MR1 | Monotone in exposure: `t2 > t1 ⇒ dmax(t2) ≥ dmax(t1)` (saturating exponential, `simulator.py:128`) | PASS |
| MR2 | Doubling sensitizer never decreases dmax: `coating_weight ↑ ⇒ dmax` non-decreasing (capped by `coating_weight_dmax_ceiling`). Note for the science expert: in the current model `ferric_oxalate_pct` affects only *contrast*, not dmax, so the FO% version of this relation is trivially true. | PASS |
| MR3 | Grid refinement: the n-step curve equals every other point of the (2n−1)-step curve (1e-9) | PASS |
| MR4 | Density curve non-decreasing in exposure step; all values in `[dmin, 4.0]`; `dmin`/`dmax` fields equal `min`/`max` of the curve | PASS |
| MR5 | `metal_ratio` changes only gamma, linearly between `pd_gamma_base` and `pt_gamma_base`; `dmin == paper_dmin_base` for all inputs | PASS |
| MR6 | Symmetry: `toe(h_opt + d) == toe(h_opt − d)`; `shoulder` monotone in `developer_temp` and clipped to [0.5, 1.0] | PASS |

---

## 6. Strict pytest configuration and migration path

Proposed `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "pytest-cov>=4.1", "pytest-timeout>=2.2",
       "hypothesis>=6.100", "ruff", "mypy", "pre-commit"]

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra --strict-markers --strict-config"      # coverage flags move to CI (today they run on every local invocation)
xfail_strict = true
timeout = 120
timeout_method = "thread"
markers = [
  "unit: fast, isolated", "integration: multi-module", "api: FastAPI endpoint tests",
  "e2e: end-to-end (no browser)", "selenium: needs Selenium WebDriver", "browser: needs Playwright",
  "functional: e2e without browser", "user_journey: multi-step scenario", "visual: visual regression",
  "performance: performance tests", "benchmark: benchmark tests", "slow: > 5 s", "deep: requires torch",
]
filterwarnings = [
  "error",
  # third-party, not actionable here; re-check on each starlette/anyio bump
  "ignore:The anyio.abc.BlockingPortal alias is deprecated:DeprecationWarning:starlette.testclient",
  # numpy internals invoked from our code; tracked fixes, delete the line when fixed
  "ignore:Mean of empty slice:RuntimeWarning",                            # ptpd_calibration.ai optimize_workflow (test_platinum_palladium_ai.py::test_high_success_rate_suggestions)
  "ignore:invalid value encountered in (scalar )?divide:RuntimeWarning",  # hardware_advanced comparison report
]
```

The ignore list is derived from the 81 warnings actually observed (`baseline.log`): 25 unknown-mark `slow`, 19 `unit`, 2 `deep` (fixed by `markers`, never ignored); 3 `PytestCollectionWarning` (fix: alias the imports at `tests/unit/test_subagents.py:39-43`, e.g. `from ...sqa import TestCase as SqaTestCase`); 2 `PydanticDeprecatedSince20` class-based `Config` at `src/ptpd_calibration/neuro_symbolic/symbolic_regression.py:304` and `curve_generator.py:44` plus `.dict()` at `integrations/spectrophotometer.py:592` (fix, not ignore — they fire at import so the blast radius is every test importing `neuro_symbolic`); 2 Pillow `'mode'` deprecations at `imaging/processor.py:615` and `:751` (**Pillow 13 removes it on 2026-10-15 — four weeks away; fix**); `RuntimeWarning: divide by zero in log` at `mcts/simulator.py:368` (guard `toe==0`/`shoulder==1` with `np.errstate` or clip to 1e-6); `overflow in power` at `neuro_symbolic/symbolic_regression.py:178` (wrap the candidate-formula evaluation in `np.errstate(over="ignore", invalid="ignore")` locally — a global ignore would hide real overflow elsewhere); the two remaining numpy warnings stay on the ignore list with owners until fixed.

Migration order — what breaks first when the flags flip:

1. `--strict-markers`: collection errors in every file using `e2e` (15 files), `slow` (5), `deep` (4), `integration` (3), `unit` (2), `user_journey` (1) — none are registered anywhere. Markers registered only in sub-`conftest.py` files (`api`, `visual`, `performance`, `benchmark`, `selenium`, `browser`, `functional`) also fail whenever a run does not load that conftest. Moving all to `pyproject` fixes both in one edit.
2. `filterwarnings = error`: (a) `tests/unit/test_subagents.py` fails at collection; (b) any test importing `ptpd_calibration.neuro_symbolic` errors at import (2 Pydantic warnings); (c) 5 tests fail at runtime (listed above). Fix (b) first.
3. `--strict-config`: clean today (all ini keys valid). Adding `timeout` requires `pytest-timeout` in the dev extra; without it `--strict-config` refuses the unknown key — so do TST-03 as one atomic change.
4. `xfail_strict`: zero `xfail`s exist, free. Future F1/F2 xfails must carry `strict=True` and a reason.
5. `timeout = 120`: slowest measured test 5.5 s; selenium journeys should carry `@pytest.mark.timeout(300)` explicitly.
6. Later: `pytest-randomly` to surface order dependence (the `kb` tests depend on repository files; the API `client` fixture is module-scoped with a shared upload dir).

---

## 7. Mutation testing scope and thresholds

* Tool: **mutmut 3.x** (in-process mutant execution, per-mutant test selection via coverage; `[tool.mutmut] paths_to_mutate`); cosmic-ray only if distribution across machines is needed.
* Scope (8 files): `curves/{modifier,generator,linearization,parser,export}.py`, `chemistry/calculator.py`, `mcts/quality.py`, `mcts/constraints.py` = 4,268 LOC, **1,446 statements** [M, from coverage.json] → **[E] 2,200–3,600 mutants** (1.5–2.5 per statement; ~2,600 "mutable" AST nodes counted).
* Test selection: the 11 scientific-core test files run **428 tests in 2.3 s** [M]; with the property suite target < 10 s. Estimated wall time [E]: mutmut 3 with 4 workers **10–25 min**; worst case (full subset per mutant, cold start) **2–3 h serial**. Run nightly on `main`; on PRs mutate only changed files in scope.
* Expected initial scores [E, from assertion-strength audit]: calculator 80–90 %, quality 75–85 %, constraints 65–80 %, modifier 60–75 %, generator/linearization 55–70 %, parser/export 45–60 %.
* Thresholds: commit the first run as `mutation-baseline.json`; gate = **no file drops > 2 points vs baseline** (ratchet). Targets once TST-05..08 land: **≥ 80 %** calculator & quality, **≥ 70 %** modifier & constraints, **≥ 60 %** generator, linearization, parser, export.
* Equivalent-mutant hygiene: `# pragma: no mutate` on logging f-strings, `notes` text, `1e-6`/`1e-10` epsilons and `np.clip` guard bounds; keep timeouts per mutant at 30 s.

---

## 8. Differential tests

### torch vs NumPy (`mcts/simulator.py:201-323`)

* Status: the differential test exists (`tests/unit/mcts/test_simulator.py:269-298`) but **never runs**: torch is absent here (94 skips) and in CI (`ci-cd.yml:64-66` installs `requirements.txt`; torch lives only in `requirements-dl.txt`). Its tolerance (0.01) would also hide a wrong `0.5`/`0.3` shoulder/toe coefficient.
* Design: (1) new `dl` CI job installing the CPU wheel (`torch --index-url https://download.pytorch.org/whl/cpu`); (2) Hypothesis `sim_params` × `num_steps ∈ {5, 21, 41, 101}`; (3) `atol = 1e-4` — torch uses float32 for `linspace`/sigmoid while NumPy is float64, so expected drift is O(1e-6) on densities ≤ 4.0; (4) explicit edge cases: `toe_position = 0.0` and `shoulder_position = 1.0` (logit → ±inf: `torch.sigmoid(-inf) = 0`, NumPy `1/(1+exp(inf)) = 0` with the `:368` warning; assert equal results *and* no warning after the guard), `exposure = 0` (clamp 1e-6), gamma at `pd_gamma_base`/`pt_gamma_base`; (5) compare `dmin/dmax/density_range/gamma` fields too; (6) `compute_process_parameters` is shared code, so the differential isolates `_numpy_characteristic_curve` vs `CharacteristicCurve.forward` (`ml/deep/process_sim.py:156-180`).
* Golden file (torch-independent): freeze 8 canonical parameter sets × 21 steps from the NumPy path, reviewed by the science expert, asserted at `1e-9` in every run — pins the physics even where torch is absent.

### MCTS vs exhaustive enumeration

* Constraint discovered: `MCTSSettings.action_bins` has `ge=5`, `num_simulations ge=50`, `max_actions_per_node ge=5` (`mcts/config.py:91-96, 148-153, 168`). A literal "3-bin" space needs `MCTSSettings.model_construct(...)` (bypasses validation) — recommended instead: **2 dimensions × 5 bins = 25 terminals**, which is still exhaustively enumerable in < 0.1 s.
* Setup: `MCTSSettings(decision_order=["exposure_time","coating_weight"], action_bins=5, num_simulations=200, max_actions_per_node=5)`; pass the other four dimensions through `fixed_parameters` (removed from `remaining_dimensions` at `engine.py:170-173`). Seed with `random.seed(k)` — the engine and tree use only the global `random` module (`engine.py:295, 374`, `tree.py:228, 234`); no NumPy RNG; `dirichlet_*` settings are unused by the engine.
* Oracle: for `i, j ∈ 0..4`, `value = min + (i/4)(max−min)` (same discretization as `engine.py:296-300`), `simulate_with_numpy` + `QualityScorer.score` → `best*` and the argmax set (ties are likely because dmax saturates in exposure; compare scores, not parameters).
* Assertions: (1) `result.quality_score ≥ best* − 1e-9` in ≥ 19 of 20 seeds (with 200 simulations on 25 leaves every leaf is reachable: root needs `n > c·k^0.5` visits to open its 5th child, i.e. n > 2); (2) `best_parameters` in the argmax set when unique; (3) `visit_distribution[dim]` has length `action_bins` and sums to 1; (4) determinism golden: seed 0 → byte-identical `best_parameters` and `visit_distribution` (SearchResult JSON committed); (5) regret non-increasing in `num_simulations` ∈ {50, 200, 800} averaged over seeds. Nightly variant: 3 dims × 5 bins = 125 terminals.

---

## 9. Frontend: equivalence tests, visual snapshots, a11y gating

* **Equivalence tests do not test equivalence.** `frontend/src/__tests__/equivalence/CurveEditor.equiv.test.ts` tests the comparison helper and its own fixtures (`:24-77`) and re-implements contrast/brightness/gamma math inline in TypeScript (`:105-148` region) — it never calls the React component, the API, or Gradio. `ScanAnalysis.equiv.test.ts` is a legitimate unit test of the pure TS function `assessScanQuality` (`@/api/hooks`) mislabeled as equivalence. `package.json:23` `migrate:verify` points at `migration/equivalence-tests`, which does not exist (dead script). With Gradio retired, "equivalence to Gradio" is not a meaningful oracle.
  Recommendation: **delete** `CurveEditor.equiv.test.ts`; **convert** `ScanAnalysis.equiv.test.ts` into `src/api/assessScanQuality.test.ts`; **replace** the concept with *contract goldens*: JSON fixtures generated by the Python `CurveModifier`/`assess` code (reviewed once), stored in a shared `contracts/` directory and asserted by both pytest and vitest (backend produces them, frontend must render/transform to the same numbers), plus `openapi-typescript`-generated types (`generate:types` script) type-checked in CI. Remove `migrate:verify`.
* **Playwright**: 1 spec (`frontend/e2e/app.spec.ts`, 9 tests) × 5 browser projects = 45 runs; no `toHaveScreenshot` call exists although `expect.toHaveScreenshot` is configured (`maxDiffPixels: 100, threshold: 0.2` — an absolute pixel budget is viewport-dependent). Policy: snapshots only on `chromium` desktop (1280×800) and `Mobile Chrome`, run inside the pinned `mcr.microsoft.com/playwright:v<version>` image so fonts/antialiasing are stable; `animations: 'disabled'`, `caret: 'hide'`, mask timestamps/plot canvases; `maxDiffPixelRatio: 0.01`; 6 states (dashboard, calibration wizard step 1 and 3, curve editor with the `typicalCalibration` fixture loaded, chemistry calculator, dark and light theme); baselines updated only via a PR labelled `update-snapshots` that runs `--update-snapshots`; Firefox/WebKit projects keep functional tests only.
* **axe gating**: `@axe-core/playwright` is a devDependency but unused; `jest-axe` is used in exactly one file (`src/components/ui/Button.test.tsx:147-161`), so the CLAUDE.md rule "accessibility audit on every new component" is met for 1 of 15 components; `test:a11y` greps for the word "Accessibility" and therefore runs only that file. Recommendation: route-level `AxeBuilder` in Playwright for the 5 routes, failing on `impact ∈ {serious, critical}` with a checked-in `a11y-allowlist.json` that may only shrink; component-level `jest-axe` in every migrated component test; report `moderate` findings as annotations only.

---

## Plan items

Effort: S ≤ 1 day, M 2–5 days, L > 1 week.

| ID | Title | Effort | Depends on | Acceptance criteria |
|---|---|---|---|---|
| TST-01 | Fix 3 backend failures: `session/logger.py:470` `paper_stats` NameError (+ narrow the `except`), seed or de-flake `test_neuro_symbolic.py:1067` | S | — | Baseline command: 0 failed; `get_paper_statistics` counts all records (new regression test with 3 records/2 papers). |
| TST-02 | Fix 5 vitest failures (add `HTMLSelectElement` guard in `useKeyboardShortcuts.ts:28`; update Layout/CurveEditor/uiSlice tests to current behaviour) | S | — | `vitest run` 831/831; `ci.yml` frontend job green. |
| TST-03 | Strict pytest config: register 13 markers in `pyproject`, add `pytest-timeout` + `hypothesis` to `dev`, `--strict-markers --strict-config`, `xfail_strict`, `timeout=120`, move `--cov` flags to CI | S | — | `pytest --collect-only -q` reports 0 warnings and 0 errors (excluding the 5 optional-dependency files); local `pytest tests/unit` runs without coverage overhead. |
| TST-04 | `filterwarnings = error` migration: fix Pydantic `class Config` ×2 and `.dict()`, Pillow `mode` ×2, `simulator.py:368` log(0), `test_subagents.py` import aliases, 3 numpy RuntimeWarnings | M | TST-03 | Baseline run shows 0 warnings; ignore list ≤ 3 entries, each with an owner comment. |
| TST-05 | Hypothesis property suite for `curves/modifier.py` (table in §5) incl. fixes for F1 (spline n<10) and F2 (endpoint pinning) or strict xfails | M | TST-03 | ≥ 12 properties × 200 examples green in `ci` profile; F1 fixed; F2 fixed or `xfail(strict=True)` with issue link. |
| TST-06 | Property suite for `generator.py` + `linearization.py` (incl. composition property) | S | TST-05 | 8 properties green; composition property tolerance ≤ 0.03. |
| TST-07 | `.quad` round-trip contract: fix `export._load_text_curve` to delegate to `QuadFileParser` (F3); keep 16-bit precision in parser | S | TST-05 | `save_curve → load_curve` for `.quad` within 1/65535 for all monotone curves; ink-limit property green. |
| TST-08 | Property suites for `chemistry/calculator.py`, `mcts/quality.py`, `mcts/constraints.py` (A+B==C rule, scale linearity, score bounds, `is_satisfied ⇔ in-range`) | S | TST-05 | 12 properties green; docstring on `ConstraintResult` states `is_satisfied` is authoritative (F4). |
| TST-09 | Simulator metamorphic suite (MR1–MR6) + NumPy golden file (8 parameter sets × 21 steps) signed off by the science expert | S | TST-05 | MR1–MR6 green at 200 examples; golden asserted at 1e-9; science expert approves the JSON. |
| TST-10 | torch-vs-NumPy differential in a `dl` CI job (CPU wheel), `atol=1e-4`, edge cases toe=0 / shoulder=1 / exposure=0 | M | TST-09 | Job runs on every PR touching `mcts/` or `ml/deep/`; differential no longer skipped; 0 warnings. |
| TST-11 | MCTS vs exhaustive oracle (2 dims × 5 bins) + seeded determinism golden + regret-vs-simulations check | M | TST-09 | Regret 0 in ≥ 19/20 seeds; seed-0 golden byte-identical; runtime < 60 s. |
| TST-12 | Mutation testing (mutmut 3) on the 8-file scope: nightly full run with ratchet, PR run on changed files | M | TST-05, TST-06, TST-07, TST-08 | `mutation-baseline.json` committed; nightly < 30 min; CI fails on > 2-point drop; targets §7 reached within two sprints. |
| TST-13 | Coverage policy: branch coverage on; per-package floors (curves/chemistry/quality/constraints ≥ 90 line / 80 branch, api ≥ 70, rest ≥ 60); `diff-cover --fail-under=85` on PRs; stop omitting `api/server.py` | M | TST-03 | Script reads `coverage json` and enforces tiers; PR check reports diff coverage; README omits no packages. |
| TST-14 | API contract hardening: schemathesis against the ASGI app (no server) with 5xx = failure; make `test_curve_endpoints.py` assert returned curve values against Python goldens; refactor the 13 skipped chat tests to `AsyncClient` | M | TST-01 | schemathesis job green with 0 5xx; ≥ 50 % of api tests assert payload values; 0 module-level skips in `tests/api`. |
| TST-15 | Retire "equivalence" tests: delete `CurveEditor.equiv.test.ts`, convert `ScanAnalysis.equiv.test.ts` to a unit test, delete `migrate:verify`; introduce shared `contracts/` goldens consumed by pytest and vitest | S | TST-07 | No test mentions Gradio; both suites assert the same golden JSON; `package.json` has no dead scripts. |
| TST-16 | Playwright visual policy (chromium + Mobile Chrome, 6 states, `maxDiffPixelRatio 0.01`, animations disabled, pinned image) and axe gating (5 routes serious/critical; jest-axe in all 15 components) | M | TST-02 | Snapshots stable across 5 consecutive CI runs; axe 0 serious/critical on 5 routes; 15/15 components have an a11y test; `test:a11y` no longer greps by name. |
| TST-17 | Honest headline metrics: CI-generated per-suite table (collected / passed / skipped-by-reason / failed, line+branch coverage) replaces README "4,400+ / 84 % / 726" | S | TST-13 | README numbers are produced by a script from the latest `main` artifact; skip counts always shown next to pass counts. |
| TST-18 | CI hygiene: remove `\|\| true` (`ci-cd.yml:106`) and `continue-on-error` on test steps (`tests.yml:168, 199, 237, 291`); replace the ignored-file list (`ci-cd.yml:83-88`) with `deep`/`slow` markers run in the `dl` job | M | TST-03, TST-10 | No test step can pass silently; `--ignore` list empty; torch-gated tests execute in `dl`. |
