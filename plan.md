# AlphaZero-Style Calibration Engine — Implementation Plan

## Executive Summary

Implement an AlphaZero-style Monte Carlo Tree Search (MCTS) calibration engine for alternative photographic printing. The engine treats calibration as a sequential decision problem (chemistry ratio → coating weight → exposure time → developer temperature → evaluate) and uses MCTS with learned value/policy networks to explore the high-dimensional parameter space efficiently.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AlphaZero Calibration Engine                      │
│                                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  MCTS    │  │ Value/Policy │  │ Physics Sim  │  │ Constraint │ │
│  │  Engine  │←→│  Networks    │←→│  (enhanced)  │←→│   Layer    │ │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│       │               │                 │                 │        │
│  ┌────┴─────────────────────────────────────────────────────┐      │
│  │              MangoMAS Multi-Agent Layer                   │      │
│  │  ┌───────────┐  ┌──────────────┐  ┌────────────────┐    │      │
│  │  │ Chemistry │  │   Exposure   │  │  Coordinator   │    │      │
│  │  │   Agent   │  │    Agent     │  │    Agent       │    │      │
│  │  └───────────┘  └──────────────┘  └────────────────┘    │      │
│  └──────────────────────────────────────────────────────────┘      │
│                           │                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │             CV Measurement Pipeline                       │      │
│  │  Camera Capture → Color Calibration → Density Extraction │      │
│  └──────────────────────────────────────────────────────────┘      │
│                           │                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │             Output Layer                                  │      │
│  │  ICC Profile │ Linearization Curve │ QTR/Piezography     │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration with Existing Codebase

The engine integrates deeply with existing modules:
- **`ml/deep/process_sim.py`** → Enhanced physics simulator (base: `ProcessParameters`, `CharacteristicCurve`)
- **`neuro_symbolic/constraints.py`** → Constraint layer (base: `SymbolicConstraint`, `ConstraintType`)
- **`agents/orchestrator.py`** → MangoMAS agents (base: `Workflow`, `WorkflowTask`)
- **`core/models.py`** → Domain models (base: `CalibrationRecord`, `CurveData`)
- **`core/types.py`** → Domain enums (`ChemistryType`, `ContrastAgent`, `DeveloperType`)
- **`config.py`** → Configuration system (`BaseSettings` with `PTPD_` env prefix)
- **`detection/`** → Step tablet detection for CV pipeline
- **`curves/export.py`** → ICC/QTR/Piezography output

---

## Phase 1: Core MCTS Engine & State Space (Backend)
**Priority: Critical | Dependencies: None**

### 1.1 Calibration State Space Models
**File: `src/ptpd_calibration/alphazero/state.py`**

Define the calibration state as a game-tree node. Each state encodes the current set of chosen parameters and the remaining decisions.

- `CalibrationState` — Pydantic model extending from `CalibrationRecord` patterns
  - `chemistry_ratio: float | None` (Pt:Pd ratio, 0.0–1.0)
  - `coating_weight: float | None` (ml/sq-inch, configurable range)
  - `ferric_oxalate_pct: float | None` (15–27% from constraints)
  - `exposure_time: float | None` (seconds, configurable range)
  - `developer_temp: float | None` (Celsius, configurable range)
  - `humidity: float | None` (0–100%)
  - `paper_type: str | None`
  - `uv_source: str | None`
  - `decision_level: int` (0–5, which parameter is being chosen)
  - `parent_state_id: UUID | None` (tree linkage)
  - `estimated_value: float` (from value network)
  - `visit_count: int` (MCTS statistics)
  - `prior_probability: float` (from policy network)

- `CalibrationAction` — Represents choosing a parameter value at a decision level
  - `level: int`
  - `parameter_name: str`
  - `value: float`
  - `discretization_index: int`

- `CalibrationStateSpace` — Manages the action space per decision level
  - Configurable discretization (number of bins per parameter)
  - Dynamic range narrowing (progressive refinement)
  - Integration with `ConstraintLayer` to prune invalid actions

### 1.2 MCTS Engine
**File: `src/ptpd_calibration/alphazero/mcts.py`**

Core MCTS algorithm with AlphaZero-style neural guidance.

- `MCTSNode` — Tree node with UCB1 statistics
  - `state: CalibrationState`
  - `children: dict[int, MCTSNode]` (action_index → child)
  - `visit_count: int`, `total_value: float`, `prior: float`
  - `is_terminal: bool`, `is_expanded: bool`

- `MCTSConfig` — Configurable via `BaseSettings` with `PTPD_MCTS_` env prefix
  - `num_simulations: int` (default 800, configurable)
  - `exploration_constant: float` (c_puct, default 1.4)
  - `temperature: float` (for action selection)
  - `temperature_threshold: int` (step after which temperature drops)
  - `dirichlet_alpha: float` (root noise for exploration)
  - `dirichlet_epsilon: float` (noise fraction)
  - `max_depth: int` (decision levels)
  - `virtual_loss: float` (for parallel MCTS)
  - `progressive_widening: bool` (enable/disable)

- `MCTSEngine` — Main search class
  - `search(root_state, num_simulations) → action_probs`
    - SELECT: Traverse tree using PUCT formula
    - EXPAND: Create child nodes for legal actions
    - EVALUATE: Call value network + physics simulator
    - BACKPROPAGATE: Update visit counts and values
  - `select_action(root, temperature) → action`
  - `get_action_probabilities(root) → dict[action, probability]`
  - Parallel search support via virtual loss
  - Integration point: calls `ValuePolicyNetwork.evaluate(state)` and `ConstraintLayer.get_legal_actions(state)`

### 1.3 Self-Play Training Loop
**File: `src/ptpd_calibration/alphazero/self_play.py`**

AlphaZero-style self-play data generation.

- `SelfPlayConfig` — Via `BaseSettings` with `PTPD_SELFPLAY_` env prefix
  - `num_episodes: int`
  - `num_simulations_per_move: int`
  - `temperature_schedule: list[tuple[int, float]]`
  - `max_replay_buffer_size: int`
  - `batch_size: int`

- `SelfPlayEpisode` — Records one complete calibration trajectory
  - `states: list[CalibrationState]`
  - `action_probs: list[dict[int, float]]`
  - `rewards: list[float]`
  - `final_value: float` (from simulator evaluation)

- `SelfPlayManager` — Orchestrates self-play episodes
  - `run_episode(mcts_engine, initial_state) → SelfPlayEpisode`
  - `generate_training_data(num_episodes) → ReplayBuffer`
  - Integration: Uses `PhysicsSimulator` to evaluate terminal states
  - Integration: Uses `MCTSEngine` for decision-making

- `ReplayBuffer` — Circular buffer for training data
  - `add(state, action_probs, value)`
  - `sample(batch_size) → batch`
  - `save/load` for persistence between sessions

### 1.4 Configuration Integration
**File: Extend `src/ptpd_calibration/config.py`**

Add new settings classes following existing `BaseSettings` pattern:
- `MCTSSettings(BaseSettings)` with `PTPD_MCTS_` prefix
- `AlphaZeroSettings(BaseSettings)` with `PTPD_ALPHAZERO_` prefix
- `SelfPlaySettings(BaseSettings)` with `PTPD_SELFPLAY_` prefix
- `CVMeasurementSettings(BaseSettings)` with `PTPD_CV_` prefix
- `MangoMASSettings(BaseSettings)` with `PTPD_MANGOMAS_` prefix
- Register in `AppSettings` alongside existing settings

**Tests for Phase 1:**
- `tests/unit/alphazero/test_state.py` — State creation, transitions, serialization
- `tests/unit/alphazero/test_mcts.py` — MCTS search correctness, UCB1, backpropagation
- `tests/unit/alphazero/test_self_play.py` — Episode generation, replay buffer
- `tests/unit/alphazero/test_config.py` — Settings validation, env var overrides

---

## Phase 2: Neural Value & Policy Networks (Backend)
**Priority: Critical | Dependencies: Phase 1**

### 2.1 Value Network
**File: `src/ptpd_calibration/alphazero/networks/value_network.py`**

Predicts print quality score from calibration parameters. Extends patterns from existing `ml/deep/models.py`.

- `ValueNetworkConfig` — Architecture configuration
  - `input_dim: int` (auto-calculated from state space)
  - `hidden_dims: list[int]` (default [256, 128, 64])
  - `dropout: float` (default 0.1)
  - `activation: str` ("relu", "gelu", "silu")
  - `use_batch_norm: bool`

- `ValueNetwork(nn.Module)` — PyTorch model
  - Input: Encoded calibration state vector
  - Output: Scalar value in [-1, 1] (quality score)
  - Architecture: MLP with skip connections
  - Feature encoding: Categorical → embedding, continuous → normalization
  - Integration: Uses `ProcessParameters` from `ml/deep/process_sim.py` for feature engineering
  - Graceful fallback when PyTorch unavailable (following existing `TORCH_AVAILABLE` pattern)

### 2.2 Policy Network
**File: `src/ptpd_calibration/alphazero/networks/policy_network.py`**

Predicts action probabilities at each decision level.

- `PolicyNetwork(nn.Module)` — PyTorch model
  - Input: Encoded calibration state vector
  - Output: Probability distribution over legal actions (per decision level)
  - Architecture: Shared trunk with level-specific heads
  - Masking: Invalid actions masked before softmax (uses constraint layer)

### 2.3 Combined Network
**File: `src/ptpd_calibration/alphazero/networks/dual_network.py`**

Single network with shared backbone and two heads (AlphaZero pattern).

- `DualNetwork(nn.Module)` — Combined value + policy network
  - Shared feature extractor backbone
  - Value head: FC → scalar
  - Policy head: FC → action probabilities (per level)
  - `evaluate(state) → (value, policy_dict)`
  - `train_step(batch) → loss_dict`
  - Loss: `L = (v - z)² - π·log(p) + c·||θ||²`
    - Value loss (MSE), Policy loss (cross-entropy), L2 regularization

### 2.4 State Encoder
**File: `src/ptpd_calibration/alphazero/networks/encoder.py`**

Encodes `CalibrationState` into neural network input vector.

- `StateEncoder` — Configurable feature engineering
  - Continuous features: Min-max or z-score normalization (configurable)
  - Categorical features: Learned embeddings for paper_type, chemistry_type, uv_source
  - One-hot encoding for decision level
  - Missing values: Learned mask embeddings
  - `encode(state) → tensor`
  - `get_input_dim() → int`

### 2.5 Training Pipeline
**File: `src/ptpd_calibration/alphazero/training.py`**

Training loop for the dual network.

- `TrainingConfig` — Via `BaseSettings`
  - `learning_rate: float`, `weight_decay: float`
  - `num_epochs: int`, `batch_size: int`
  - `checkpoint_interval: int`
  - `early_stopping_patience: int`
  - `lr_schedule: str` ("cosine", "step", "plateau")

- `Trainer` — Training orchestrator
  - `train(network, replay_buffer, num_epochs) → metrics`
  - Checkpoint saving/loading
  - TensorBoard/logging integration
  - Learning rate scheduling
  - Early stopping with validation set

**Tests for Phase 2:**
- `tests/unit/alphazero/test_value_network.py` — Forward pass, gradient flow, output range
- `tests/unit/alphazero/test_policy_network.py` — Action masking, probability distribution
- `tests/unit/alphazero/test_dual_network.py` — Combined training, loss computation
- `tests/unit/alphazero/test_encoder.py` — Feature encoding, dimensionality
- `tests/unit/alphazero/test_training.py` — Training loop, checkpointing, convergence

---

## Phase 3: Enhanced Physics Simulator (Backend)
**Priority: Critical | Dependencies: Phase 1**

### 3.1 Enhanced Process Simulator
**File: `src/ptpd_calibration/alphazero/simulator.py`**

Extends existing `ml/deep/process_sim.py` with full parameter-to-density modeling.

- `PrintingSimulator` — Full alt-process simulator
  - Extends `ProcessParameters` with additional fields:
    - `chemistry_ratio`, `coating_weight`, `ferric_oxalate_pct`
    - `developer_temp`, `humidity`, `paper_absorbency`
  - `simulate(state: CalibrationState) → SimulationResult`
    - Models UV transmission through digital negative
    - H&D characteristic curve with chemistry-dependent parameters
    - Temperature effects on development rate
    - Humidity effects on coating behavior
    - Paper absorbency effects on dmax
  - `evaluate_quality(result: SimulationResult) → float`
    - Multi-objective quality score:
      - Density range (dmax - dmin)
      - Tonal distribution (evenness across 21 steps)
      - Contrast target achievement
      - Color temperature target (warm/neutral/cool)
  - Differentiable (torch-compatible) for end-to-end training
  - Fallback NumPy implementation when PyTorch unavailable

- `SimulationResult` — Output from simulation
  - `density_curve: list[float]` (21/31/41 step densities)
  - `dmin, dmax, density_range: float`
  - `tonal_distribution: list[float]`
  - `color_temperature: float`
  - `quality_score: float`
  - `constraint_violations: list[ConstraintViolation]`

### 3.2 Chemistry Model
**File: `src/ptpd_calibration/alphazero/chemistry_model.py`**

Models the relationship between chemistry parameters and print characteristics.

- `ChemistryResponseModel`
  - Platinum vs palladium tone mapping (Pd → warmer, Pt → cooler)
  - Ferric oxalate concentration effects (15–27% valid range)
  - Contrast agent interactions (Na2, potassium chlorate, etc.)
  - Developer temperature response curves
  - Based on published Bostick-Sullivan data tables (parameterized, not hardcoded)

### 3.3 Paper Model
**File: `src/ptpd_calibration/alphazero/paper_model.py`**

Models paper-specific behavior using the existing knowledge graph.

- `PaperResponseModel`
  - Integration with `neuro_symbolic/knowledge_graph.py`
  - Water-drop spread test → absorbency coefficient
  - Sizing type effects on coating absorption
  - Paper weight effects on dimensional stability
  - Paper-specific dmax limits

**Tests for Phase 3:**
- `tests/unit/alphazero/test_simulator.py` — Physical plausibility, known configurations
- `tests/unit/alphazero/test_chemistry_model.py` — Known Pt/Pd behaviors
- `tests/unit/alphazero/test_paper_model.py` — Paper response curves

---

## Phase 4: Neuro-Symbolic Constraint Layer (Backend)
**Priority: High | Dependencies: Phase 1, Phase 3**

### 4.1 Photochemistry Constraint Registry
**File: `src/ptpd_calibration/alphazero/constraints.py`**

Extends existing `neuro_symbolic/constraints.py` with MCTS-specific constraints.

- `PhotochemistryConstraint(SymbolicConstraint)` — New constraint subclasses:
  - `FerricOxalateRangeConstraint` — FO% must be 15–27%
  - `PlatinumWarmthConstraint` — Pd produces warmer tones than Pt (soft constraint)
  - `ExposureMonotonicityConstraint` — More exposure → more density (to a point)
  - `DeveloperTemperatureConstraint` — Dev temp affects development rate within bounds
  - `CoatingWeightConstraint` — Min/max coating for paper type
  - `HumidityConstraint` — Humidity effects on coating (process-dependent)

- `ConstraintRegistry` — Central registry for all active constraints
  - `register(constraint: SymbolicConstraint)`
  - `get_legal_actions(state: CalibrationState, level: int) → list[CalibrationAction]`
  - `evaluate_constraints(state) → ConstraintResult`
  - `get_penalty(state) → float` (soft constraint penalty for value network)
  - Configurable constraint weights via settings

### 4.2 Domain Knowledge Rules
**File: `src/ptpd_calibration/alphazero/domain_rules.py`**

Encodes hard photochemistry rules as symbolic rules.

- `DomainRuleEngine`
  - Rules loaded from configuration (not hardcoded)
  - Example rules (expressed as constraint objects):
    - If `chemistry_type == ZIATYPE` then `developer == CITRIC_ACID`
    - If `ferric_oxalate_pct > 25` then `exposure_time` needs increase
    - If `humidity > 80` then `coating_weight` should decrease
  - Rules are composable and extensible
  - Integration with MCTS action pruning

**Tests for Phase 4:**
- `tests/unit/alphazero/test_constraints.py` — Each constraint type, edge cases
- `tests/unit/alphazero/test_domain_rules.py` — Rule engine, composability

---

## Phase 5: CV Measurement Pipeline (Backend)
**Priority: High | Dependencies: Phase 3**

### 5.1 Smartphone Camera Calibration
**File: `src/ptpd_calibration/alphazero/cv/camera_calibration.py`**

Calibrate smartphone camera for density measurement using a color reference target.

- `CameraCalibrationProfile` — Pydantic model
  - `color_matrix: list[list[float]]` (3x3 correction matrix)
  - `gamma_curve: list[float]` (linearization LUT)
  - `white_balance: tuple[float, float, float]`
  - `reference_target_id: str` (e.g., "X-Rite ColorChecker")
  - `calibration_date: datetime`
  - `device_model: str`
  - `quality_score: float`

- `CameraCalibrator`
  - `calibrate_from_reference(image, target_type) → CameraCalibrationProfile`
  - Uses least-squares fit between captured and reference colors
  - Support for X-Rite ColorChecker, SpyderCHECKR, IT8 targets
  - Integration with existing `detection/detector.py` for patch finding

### 5.2 Density Patch Measurement
**File: `src/ptpd_calibration/alphazero/cv/density_measurement.py`**

Measure density from step tablet photographs using calibrated camera.

- `DensityMeasurementPipeline`
  - `measure(image, calibration_profile, tablet_config) → list[DensityMeasurement]`
  - Pipeline steps:
    1. Apply camera calibration (color matrix, gamma correction)
    2. Detect step tablet using existing `detection/detector.py`
    3. Extract patches using existing `detection/extractor.py`
    4. Convert RGB → L*a*b* → density
    5. Apply paper-base subtraction
    6. Quality assessment (uniformity, clipping detection)
  - Integration with existing `ExtractionResult`, `PatchData`, `DensityMeasurement` models

### 5.3 Water-Drop Absorbency Test
**File: `src/ptpd_calibration/alphazero/cv/absorbency_test.py`**

Measure paper absorbency via water-drop spread test (CV-based).

- `AbsorbencyMeasurement`
  - `measure_spread(image_sequence, time_interval) → AbsorbencyResult`
  - Detects water drop boundary using edge detection
  - Tracks spread diameter over time
  - Calculates absorbency coefficient
  - Result feeds into `PaperResponseModel`

**Tests for Phase 5:**
- `tests/unit/alphazero/test_camera_calibration.py` — Color matrix, gamma correction
- `tests/unit/alphazero/test_density_measurement.py` — Density extraction accuracy
- `tests/unit/alphazero/test_absorbency.py` — Spread measurement, coefficient calculation

---

## Phase 6: MangoMAS Multi-Agent Architecture (Backend)
**Priority: High | Dependencies: Phase 1–4**

### 6.1 Chemistry Selection Agent
**File: `src/ptpd_calibration/alphazero/mas/chemistry_agent.py`**

Extends existing `agents/` framework for chemistry-specific decisions.

- `ChemistryAgent(BaseSubagent)` — Registered in subagent registry
  - Manages: `chemistry_ratio`, `ferric_oxalate_pct`, `contrast_agent`, `developer`
  - Uses MCTS for chemistry parameter sub-tree search
  - Consults `ChemistryResponseModel` for prior estimates
  - Reports to coordinator via `MessageBus`

### 6.2 Exposure Optimization Agent
**File: `src/ptpd_calibration/alphazero/mas/exposure_agent.py`**

- `ExposureAgent(BaseSubagent)` — Registered in subagent registry
  - Manages: `exposure_time`, `coating_weight`, `developer_temp`
  - Uses MCTS for exposure parameter sub-tree search
  - Consults `PrintingSimulator` for UV response modeling
  - Reports to coordinator via `MessageBus`

### 6.3 Coordinator Agent
**File: `src/ptpd_calibration/alphazero/mas/coordinator_agent.py`**

Resolves trade-offs between tonal range and contrast.

- `CoordinatorAgent(BaseSubagent)` — Top-level orchestrator
  - Receives proposals from Chemistry and Exposure agents
  - Evaluates composite quality score across objectives:
    - Tonal range breadth
    - Contrast target
    - Color temperature target
    - Material cost efficiency (palladium cost awareness)
  - Selects winning configuration via Pareto optimization
  - Manages the overall MCTS tree at the top level
  - Uses existing `Workflow` and `WorkflowTask` patterns from `orchestrator.py`

### 6.4 Agent Communication Protocol
**File: `src/ptpd_calibration/alphazero/mas/protocol.py`**

Define messages between agents, extending existing `communication.py`.

- `CalibrationProposal(AgentMessage)` — Chemistry/Exposure agent proposals
- `TradeoffResolution(AgentMessage)` — Coordinator decisions
- `ExperimentRequest(AgentMessage)` — Request for physical experiment
- `MeasurementResult(AgentMessage)` — CV measurement results

**Tests for Phase 6:**
- `tests/unit/alphazero/test_chemistry_agent.py` — Chemistry decisions
- `tests/unit/alphazero/test_exposure_agent.py` — Exposure optimization
- `tests/unit/alphazero/test_coordinator_agent.py` — Multi-objective resolution
- `tests/integration/alphazero/test_mas_workflow.py` — Full agent collaboration

---

## Phase 7: Output Generation (Backend)
**Priority: High | Dependencies: Phase 1–6**

### 7.1 ICC Profile Generator
**File: `src/ptpd_calibration/alphazero/output/icc_generator.py`**

Generate ICC-style color profiles from MCTS-optimized calibration.

- `ICCProfileGenerator`
  - `generate(calibration_state, simulation_result) → ICCProfile`
  - Create device-link profiles for specific paper/chemistry combinations
  - Integration with existing `integrations/icc_profiles.py`
  - Support for v2 and v4 ICC profile formats

### 7.2 Linearization Curve Generator
**File: `src/ptpd_calibration/alphazero/output/linearization.py`**

Generate linearization curves from optimal calibration.

- `LinearizationGenerator`
  - `generate(calibration_state, density_measurements) → CurveData`
  - Integration with existing `curves/linearization.py`
  - Output compatible with existing `CurveData` model
  - Export via existing `curves/export.py` (QTR, Piezography, CSV, JSON)

### 7.3 Calibration Report
**File: `src/ptpd_calibration/alphazero/output/report.py`**

Generate human-readable calibration report.

- `CalibrationReportGenerator`
  - `generate(search_result, optimal_state) → CalibrationReport`
  - Parameter recommendations with confidence intervals
  - Comparison to known-good calibrations
  - Cost estimate (material usage)
  - Visualization data for frontend

**Tests for Phase 7:**
- `tests/unit/alphazero/test_icc_generator.py` — Profile format, color accuracy
- `tests/unit/alphazero/test_linearization_output.py` — Curve generation, export
- `tests/unit/alphazero/test_report.py` — Report content, formatting

---

## Phase 8: API Endpoints (Backend)
**Priority: High | Dependencies: Phase 1–7**

### 8.1 AlphaZero API Router
**File: `src/ptpd_calibration/api/alphazero_router.py`**

FastAPI router for the AlphaZero engine, mounted on main app.

- `POST /api/alphazero/search` — Run MCTS search for optimal calibration
  - Request: paper_type, target_aesthetics, constraints, budget
  - Response: optimal_params, quality_score, alternatives, search_stats
  - Background task (Celery) for long-running searches

- `POST /api/alphazero/evaluate` — Evaluate a specific parameter set
  - Request: complete CalibrationState
  - Response: SimulationResult with quality breakdown

- `POST /api/alphazero/measure` — Upload step tablet photo for measurement
  - Request: image file + camera calibration profile
  - Response: density measurements, quality metrics

- `POST /api/alphazero/calibrate-camera` — Calibrate camera from reference target
  - Request: reference image + target type
  - Response: CameraCalibrationProfile

- `POST /api/alphazero/self-play` — Trigger self-play training episode
  - Request: num_episodes, configuration overrides
  - Response: training_job_id (async)

- `GET /api/alphazero/status/{job_id}` — Check async job status

- `POST /api/alphazero/export` — Export optimal calibration
  - Request: calibration_id, format (icc, qtr, piezography, csv, json)
  - Response: downloadable file

- `GET /api/alphazero/recommendations` — Get calibration recommendations
  - Request: paper_type, chemistry_type, goals
  - Response: top-N recommended parameter sets with rationale

- `POST /api/alphazero/feedback` — Submit real-world measurement feedback
  - Request: calibration_id + actual density measurements
  - Response: updated model, revised recommendations

**Tests for Phase 8:**
- `tests/api/test_alphazero_endpoints.py` — All endpoints, validation, error handling
- `tests/integration/alphazero/test_api_flow.py` — Complete workflow via API

---

## Phase 9: Frontend — Zustand Store & Hooks (Frontend)
**Priority: High | Dependencies: Phase 8 (API contracts)**

### 9.1 AlphaZero Store Slice
**File: `frontend/src/stores/slices/alphazeroSlice.ts`**

New Zustand store slice following existing slice patterns.

- State:
  - `searchState: 'idle' | 'searching' | 'complete' | 'error'`
  - `currentSearch: AlphaZeroSearchResult | null`
  - `searchHistory: AlphaZeroSearchResult[]`
  - `cameraCalibration: CameraCalibrationProfile | null`
  - `measurements: DensityMeasurement[]`
  - `recommendations: CalibrationRecommendation[]`
  - `trainingStatus: TrainingJobStatus | null`
  - `searchProgress: { iteration: number; total: number; bestScore: number }`

- Actions:
  - `startSearch(params)`, `setSearchResult(result)`, `clearSearch()`
  - `setCameraCalibration(profile)`, `addMeasurement(measurement)`
  - `setRecommendations(recs)`, `setTrainingStatus(status)`
  - `updateSearchProgress(progress)`

### 9.2 TypeScript Types
**File: `frontend/src/types/alphazero.ts`**

- `CalibrationState`, `CalibrationAction`, `SimulationResult`
- `AlphaZeroSearchRequest`, `AlphaZeroSearchResult`
- `CameraCalibrationProfile`, `DensityMeasurement`
- `CalibrationRecommendation`, `CalibrationReport`
- All types derived from API response shapes (no hardcoded values)

### 9.3 API Hooks
**File: `frontend/src/api/alphazero.ts`**

TanStack Query hooks following existing `api/hooks.ts` patterns.

- `useAlphaZeroSearch()` — Mutation for MCTS search
- `useEvaluateCalibration()` — Mutation for parameter evaluation
- `useMeasureDensity()` — Mutation for step tablet measurement
- `useCalibrateCamera()` — Mutation for camera calibration
- `useAlphaZeroRecommendations()` — Query for recommendations
- `useSearchStatus()` — Query with polling for async job status
- `useSubmitFeedback()` — Mutation for real-world feedback

### 9.4 Custom Hooks
**File: `frontend/src/hooks/useAlphaZero.ts`**

- `useAlphaZeroCalibration()` — Orchestrates the full calibration workflow
- `useSearchProgress()` — WebSocket-based progress tracking
- `useCameraCapture()` — Camera integration for mobile measurement

**Tests for Phase 9:**
- `frontend/src/stores/slices/__tests__/alphazeroSlice.test.ts`
- `frontend/src/api/__tests__/alphazero.test.ts`
- `frontend/src/hooks/__tests__/useAlphaZero.test.ts`

---

## Phase 10: Frontend — React Components (Frontend)
**Priority: High | Dependencies: Phase 9**

### 10.1 AlphaZero Calibration Page
**File: `frontend/src/pages/AlphaZeroPage.tsx`**

New page component, added to router alongside existing pages.

- Tab layout with sub-views:
  - Search Setup, Results, Recommendations, Camera Calibration, History

### 10.2 Search Configuration Panel
**File: `frontend/src/components/alphazero/SearchConfigPanel.tsx`**

- Paper type selector (dynamic from API, not hardcoded)
- Chemistry type selector
- Target aesthetics sliders (contrast, warmth, density range)
- Constraint overrides (advanced)
- Budget/material cost limits
- "Start Search" button with progress indicator

### 10.3 MCTS Visualization
**File: `frontend/src/components/alphazero/MCTSVisualization.tsx`**

- Interactive tree visualization using Plotly/D3
- Node coloring by visit count or value
- Click-to-expand subtrees
- Real-time updates during search via WebSocket
- Search statistics dashboard (nodes explored, best value, time)

### 10.4 Results Dashboard
**File: `frontend/src/components/alphazero/ResultsDashboard.tsx`**

- Optimal parameter display with confidence intervals
- Density curve preview (using existing `CurveEditor` patterns)
- Multi-objective Pareto front visualization
- Alternative configurations comparison table
- Export buttons (ICC, QTR, CSV, etc.)

### 10.5 Camera Measurement Interface
**File: `frontend/src/components/alphazero/CameraMeasurement.tsx`**

- Camera calibration wizard (reference target capture)
- Step tablet photo capture guide
- Live density readout overlay
- Quality indicators for captured measurements
- Integration with device camera API (mobile-first)

### 10.6 Recommendation Cards
**File: `frontend/src/components/alphazero/RecommendationCard.tsx`**

- Card-based display for recommended calibrations
- Material cost estimate
- Confidence score visualization
- Quick-apply to start new calibration

**Tests for Phase 10:**
- Co-located test files for each component (`.test.tsx`)
- Accessibility audits (`jest-axe`)
- Snapshot tests for key states
- Integration tests for data flow through store

---

## Phase 11: Integration & End-to-End Testing
**Priority: Critical | Dependencies: Phase 1–10**

### 11.1 End-to-End Workflow Tests
**File: `tests/e2e/alphazero/test_full_workflow.py`**

- Complete search → measure → refine → export workflow
- Multi-agent coordination under load
- Constraint satisfaction across all parameter combinations

### 11.2 Equivalence Tests
**File: `frontend/src/__tests__/equivalence/AlphaZero.equiv.test.ts`**

- API contract verification
- State management correctness
- Component rendering with real API data

### 11.3 Performance Tests
**File: `tests/performance/alphazero/test_mcts_performance.py`**

- MCTS search throughput (nodes/second)
- Neural network inference latency
- Memory usage under sustained search
- Parallel search scaling

### 11.4 Playwright E2E Tests
**File: `frontend/e2e/alphazero.spec.ts`**

- Full UI workflow automation
- Camera simulation for measurement flow
- Export verification

---

## Phase 12: Documentation & Deployment
**Priority: Medium | Dependencies: Phase 11**

### 12.1 API Documentation
- OpenAPI schema auto-generated from FastAPI
- Usage examples for each endpoint

### 12.2 Architecture Documentation
- System design document
- Data flow diagrams
- Agent interaction sequences

### 12.3 User Guide
- Step-by-step calibration workflow
- Camera setup instructions
- Interpreting results

---

## Implementation Order & Dependencies

```
Phase 1 (MCTS Core)          ←── No dependencies (START HERE)
    ├── Phase 2 (Networks)    ←── Needs Phase 1 state models
    ├── Phase 3 (Simulator)   ←── Needs Phase 1 state models
    │       └── Phase 5 (CV)  ←── Needs Phase 3 simulator
    └── Phase 4 (Constraints) ←── Needs Phase 1 + Phase 3
            └── Phase 6 (MAS) ←── Needs Phase 1–4
                    └── Phase 7 (Output) ←── Needs Phase 1–6
                            └── Phase 8 (API) ←── Needs Phase 1–7
                                    └── Phase 9 (Store) ←── Needs Phase 8
                                            └── Phase 10 (UI) ←── Needs Phase 9
                                                    └── Phase 11 (E2E) ←── Needs all
                                                            └── Phase 12 (Docs)
```

**Parallelizable work:**
- Phase 2 + Phase 3 can run in parallel (both depend only on Phase 1)
- Phase 4 + Phase 5 can run in parallel after Phase 3
- Phase 9 (frontend types/store) can start with API contract stubs before Phase 8 is complete

---

## Agent Assignment

| Phase | Primary Agent | Supporting Agents |
|-------|--------------|-------------------|
| 1 | gap-remediation-agent | testing-agent |
| 2 | gap-remediation-agent | testing-agent |
| 3 | gap-remediation-agent | testing-agent |
| 4 | gap-remediation-agent | testing-agent |
| 5 | gap-remediation-agent | testing-agent |
| 6 | gap-remediation-agent | testing-agent |
| 7 | gap-remediation-agent | testing-agent |
| 8 | gap-remediation-agent | testing-agent |
| 9 | ui-migration-agent | testing-agent |
| 10 | ui-migration-agent | testing-agent |
| 11 | testing-agent | gap-remediation-agent |
| 12 | documentation-agent | — |

---

## Key Design Principles

1. **No hardcoded values** — All parameters via `BaseSettings` with env var overrides
2. **Backwards compatible** — New modules are additive; existing functionality untouched
3. **Reusable components** — State encoder, constraint registry, simulation result reusable across modules
4. **Dynamic configuration** — Parameter ranges, discretization, network architecture all configurable
5. **Graceful degradation** — PyTorch optional (NumPy fallback), agents optional (direct MCTS fallback)
6. **Existing patterns** — Follow `SymbolicConstraint` ABC, `BaseSubagent` registration, `BaseSettings` pattern
7. **Full test coverage** — Unit tests per module, integration tests per phase, E2E tests for workflows
8. **Structured logging** — Use existing `core/logging.py` and `agents/logging.py` patterns
9. **Incremental delivery** — Each phase is independently testable and provides value
10. **Type safety** — Pydantic models for all API contracts, TypeScript strict mode for frontend
