# Next Steps: Agentic Development Plan for Pt/Pd Calibration Studio

> **Document Version:** 2.0
> **Date:** February 2026
> **Purpose:** Comprehensive roadmap for continuing agentic AI development using the universal-dev-agent pattern
> **Branch:** `claude/investigate-agentic-next-steps-OD1D4`

---

## Executive Summary

This document outlines the next steps for the Platinum-Palladium AI Printing Tool based on a thorough investigation of the codebase state, gap analysis, and alignment with the universal-dev-agent development pattern.

### Current State Assessment

| Category | Status | Items |
|----------|--------|-------|
| **Source Files** | 139 Python files | Production-ready modules |
| **Test Coverage** | 128 test files | Unit, integration, API, E2E, performance, visual |
| **Deep Learning** | 10+ AI systems | Fully implemented with PyTorch |
| **Agentic System** | Complete | ReAct loop, planning, memory, 10+ tools |
| **Hardware Integration** | ⚠️ Simulated | Critical gap - no real device support |
| **CI/CD Pipeline** | Operational | Multi-platform, multi-Python version |

### Priority Matrix

| Priority | Area | Effort | Impact |
|----------|------|--------|--------|
| **P0** | Real Hardware Integration | High | Critical |
| **P1** | GPU Acceleration Infrastructure | Medium | High |
| **P1** | Stream-Based Image Processing | Medium | High |
| **P2** | Agent Tool Expansion | Low | Medium |
| **P2** | Federated Learning Activation | Medium | Medium |
| **P3** | External Integrations | High | Future |

---

## Project Context (Universal-Dev-Agent Format)

```
PROJECT_NAME: Pt/Pd Calibration Studio
DOMAIN: AI-powered alternative photography calibration system
GOAL: Build production-grade, multi-agent calibration platform with real hardware support
CONSTRAINTS:
  - Tech Stack: Python 3.10+, PyTorch, Gradio, FastAPI, Pydantic v2
  - Compatibility: Cross-platform (Windows/macOS/Linux)
  - Hardware: X-Rite i1, Epson inkjet printers
  - License: MIT Open Source
```

---

## Milestone 1: Foundation Verification ✅ (Complete)

**Goal:** Bootstrap repo with architecture using Planner subagent.

### Epic 1.1: Repo Setup ✅
| Story | Status | Deliverable |
|-------|--------|-------------|
| 1.1.1: C4 Architecture | ✅ Complete | `ARCHITECTURE.md` |
| 1.1.2: CLAUDE.md Guide | ✅ Complete | 10.3K comprehensive guide |
| 1.1.3: Git Configuration | ✅ Complete | Pre-commit hooks, CI/CD |

### Epic 1.2: Environment Config ✅
| Story | Status | Deliverable |
|-------|--------|-------------|
| 1.2.1: Pydantic Settings | ✅ Complete | `config.py` with pydantic-settings |
| 1.2.2: Environment Variables | ✅ Complete | ANTHROPIC_API_KEY, etc. |
| 1.2.3: Lazy Imports | ✅ Complete | Optional dependencies pattern |

### Epic 1.3: Agentic System Foundation ✅
| Story | Status | Deliverable |
|-------|--------|-------------|
| 1.3.1: CalibrationAgent | ✅ Complete | `agents/agent.py` - ReAct loop |
| 1.3.2: Tool Registry | ✅ Complete | `agents/tools.py` - 10+ tools |
| 1.3.3: Planning Engine | ✅ Complete | `agents/planning.py` - Task decomposition |
| 1.3.4: Memory System | ✅ Complete | `agents/memory.py` - Long-term + working memory |

---

## Milestone 2: Critical Gap Resolution 🔴 (Priority)

**Goal:** Address critical hardware integration gap to enable real-world usage.

### Epic 2.1: Real Spectrophotometer Integration (CRITICAL)

**Current State:** All spectrophotometer code in `integrations/hardware/simulated.py` generates synthetic data. Real X-Rite i1 driver exists at `integrations/hardware/xrite_i1pro.py` but is not integrated.

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 2.1.1: X-Rite i1 SDK Integration | P0 | Coder | USB/Serial communication working; White/black calibration functional |
| 2.1.2: Device Discovery | P0 | Coder | Auto-detect connected spectrophotometers; Support multiple device types |
| 2.1.3: Measurement Protocol | P0 | SQE | Real density readings match expected values ±0.02; Lab values validated |
| 2.1.4: Spectral Data Capture | P1 | Coder | Full 380-730nm spectrum capture; Export to CGATS format |

**Implementation Path:**
```python
# Target: src/ptpd_calibration/integrations/hardware/xrite_i1pro.py
class XRiteI1Pro:
    """Real X-Rite i1 spectrophotometer integration."""

    def connect(self, port: str) -> bool:
        """Connect via USB HID or serial protocol."""
        # Implement: i1Pro SDK wrapper
        # Support: i1 Pro 2, i1 Pro 3, i1Studio
        pass

    def calibrate(self) -> bool:
        """Perform white reference calibration."""
        # Implement: SDK calibration call
        pass

    def read_measurement(self) -> DensityMeasurement:
        """Read actual spectral/density data."""
        # Implement: SDK measurement API
        pass
```

### Epic 2.2: Real Printer Integration (HIGH)

**Current State:** `integrations/printer_drivers.py` and `integrations/hardware/cups_printer.py` contain simulated implementations.

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 2.2.1: CUPS Integration | P1 | Coder | Print via CUPS on macOS/Linux; Query printer status |
| 2.2.2: Windows Print API | P1 | Coder | Windows printing support; Consistent cross-platform API |
| 2.2.3: Epson SDK Integration | P2 | Coder | Direct Epson control; Advanced paper handling |
| 2.2.4: QTR Format Export | P1 | Coder | Generate valid .quad files; Compatible with QuadToneRIP |

### Epic 2.3: Hardware Abstraction Layer

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 2.3.1: Protocol Interfaces | P0 | Planner | Clean interfaces in `protocols.py`; No hardware-specific code leaks |
| 2.3.2: Device Factory | P0 | Coder | Automatic device type detection; Graceful fallback to simulation |
| 2.3.3: Error Handling | P0 | SQE | Comprehensive error codes; User-friendly messages |

---

## Milestone 3: Performance & Intelligence Enhancement 🟡 (In Progress)

**Goal:** Build MVP logic with Coder and SQE subagents.

### Epic 3.1: GPU Acceleration Infrastructure

**Current State:** PyTorch is optional dependency but GPU acceleration not fully utilized.

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 3.1.1: CUDA Detection | P1 | Coder | Auto-detect CUDA availability; Graceful CPU fallback |
| 3.1.2: MPS Support | P1 | Coder | Apple Silicon acceleration; Metal Performance Shaders |
| 3.1.3: Batch Processing | P1 | Coder | GPU-accelerated batch image processing; 10x speedup target |
| 3.1.4: Model Optimization | P2 | Coder | TensorRT/ONNX export for inference; Quantization support |

### Epic 3.2: Deep Learning Enhancement

**Current State:** 10+ deep learning modules exist in `src/ptpd_calibration/deep_learning/`:

| Module | Lines | Status | Enhancement Needed |
|--------|-------|--------|-------------------|
| `diffusion_enhance.py` | 750 | ✅ Complete | Add attention visualization |
| `neural_curve.py` | 850 | ✅ Complete | Expand training dataset |
| `detection.py` | 900 | ✅ Complete | Real-time performance tuning |
| `image_quality.py` | 1100 | ✅ Complete | Fine-tune on Pt/Pd dataset |
| `defect_detection.py` | 1000 | ✅ Complete | Expand defect categories |
| `recipe_recommendation.py` | 950 | ✅ Complete | Add user preference learning |
| `multimodal_assistant.py` | 1100 | ✅ Complete | Tool use integration |
| `federated_learning.py` | 1000 | ✅ Complete | Production activation |

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 3.2.1: Model Training Pipeline | P1 | Coder | Reproducible training scripts; Experiment tracking |
| 3.2.2: Uncertainty Quantification | P1 | Coder | Confidence intervals on predictions; Ensemble methods |
| 3.2.3: Active Learning Integration | P1 | Coder | Smart sampling suggestions; Minimal measurements needed |
| 3.2.4: Attention Visualization | P2 | Coder | Explainable AI for defect detection; User trust building |

### Epic 3.3: Stream-Based Processing

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 3.3.1: Tile-Based Image Processing | P1 | Coder | Handle >100MP images; Memory <4GB peak |
| 3.3.2: Streaming LLM Responses | P1 | Coder | Real-time token streaming; Progressive UI updates |
| 3.3.3: Async Pipeline | P1 | Coder | Non-blocking operations; Parallel measurement processing |

---

## Milestone 4: Agent Enhancement 🟢 (Continuous)

**Goal:** Quality gate with Reviewer; Iterate via Orchestrator.

### Epic 4.1: Agent Tool Expansion

**Current Tools in `agents/tools.py`:**
1. `analyze_densities` - Density quality analysis
2. `compare_calibrations` - Comparative analysis
3. `search_calibrations` - Database querying
4. `get_calibration` / `save_calibration` - CRUD
5. `predict_response` - ML density prediction
6. `generate_curve` - Curve generation
7. `suggest_parameters` - Parameter recommendation
8. `create_test_plan` - Test plan generation

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 4.1.1: Hardware Control Tools | P1 | Coder | `measure_density` - Real spectrophotometer; `print_target` - Real printer |
| 4.1.2: Image Analysis Tools | P2 | Coder | `analyze_print_quality` - ViT assessment; `detect_defects` - U-Net segmentation |
| 4.1.3: Workflow Automation Tools | P2 | Coder | `run_full_calibration` - End-to-end; `optimize_curve` - Iterative refinement |
| 4.1.4: Knowledge Tools | P3 | Coder | `search_knowledge_base` - RAG retrieval; `explain_concept` - Educational |

### Epic 4.2: Multi-Agent Orchestration

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 4.2.1: Subagent Definitions | P2 | Planner | Specialized agents: Calibration, QA, Troubleshooting |
| 4.2.2: Agent Communication | P2 | Coder | Inter-agent messaging; Shared context |
| 4.2.3: Workflow Graphs | P2 | Coder | LangGraph-style state machines; Resumable workflows |

### Epic 4.3: Code Review & Quality

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 4.3.1: Automated Review | P1 | Reviewer | Ruff linting passes; MyPy type checking passes |
| 4.3.2: Security Audit | P1 | SQE | No hardcoded credentials; Input validation on all APIs |
| 4.3.3: Performance Benchmarks | P1 | SQE | Baseline metrics established; Regression detection |

---

## Milestone 5: Testing Excellence 🧪

**Goal:** Comprehensive test coverage following test-first principles.

### Epic 5.1: Unit Test Expansion

**Current State:** 50+ unit test files in `tests/unit/`

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 5.1.1: Hardware Mock Tests | P0 | SQE | 100% coverage of protocol interfaces |
| 5.1.2: Deep Learning Unit Tests | P1 | SQE | Model forward/backward pass tests; Deterministic with seeds |
| 5.1.3: Agent Tool Tests | P1 | SQE | Each tool has ≥3 test cases |

### Epic 5.2: Integration Testing

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 5.2.1: Hardware Integration Tests | P1 | SQE | Real device tests (when available); Simulation fallback |
| 5.2.2: Agent Workflow Tests | P1 | SQE | Full calibration workflow passes; Error recovery works |
| 5.2.3: API Integration Tests | P1 | SQE | All FastAPI endpoints tested; Auth flows validated |

### Epic 5.3: Performance & Visual Testing

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 5.3.1: Benchmark Suite | P2 | SQE | Key operations benchmarked; CI regression alerts |
| 5.3.2: Visual Regression | P2 | SQE | Gradio UI screenshots; Pixel-level comparison |
| 5.3.3: Load Testing | P3 | SQE | 100 concurrent users; <2s response time |

---

## Milestone 6: Deployment & Ecosystem 🌐 (Future)

**Goal:** Productionize with full subagent verification.

### Epic 6.1: Production Deployment

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 6.1.1: Docker Containerization | P2 | Orchestrator | Multi-stage builds; GPU support via nvidia-docker |
| 6.1.2: Hugging Face Spaces | P2 | Orchestrator | Live demo deployment; Auto-update on push |
| 6.1.3: Health Checks | P2 | SQE | Liveness/readiness probes; Prometheus metrics |

### Epic 6.2: External Integrations

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 6.2.1: Lightroom Plugin | P3 | Coder | Native Lua plugin; Curve export integration |
| 6.2.2: Photoshop Extension | P3 | Coder | CEP extension; Direct curve application |
| 6.2.3: Mobile Companion | P4 | Coder | React Native app; Remote measurement viewing |

### Epic 6.3: Community Platform

| Story | Priority | Subagent | Acceptance Criteria |
|-------|----------|----------|---------------------|
| 6.3.1: Profile Sharing | P3 | Coder | Upload/download calibration profiles; Rating system |
| 6.3.2: Federated Learning Hub | P3 | Coder | Privacy-preserving model aggregation; Opt-in participation |
| 6.3.3: Knowledge Wiki | P4 | Coder | Community-contributed tutorials; Expert verification |

---

## Immediate Action Items

### This Sprint (Priority P0)

1. **Real Spectrophotometer Integration**
   - Research X-Rite i1 SDK options (i1Profiler SDK, ColorGATE SDK)
   - Implement USB HID communication layer
   - Create device discovery mechanism
   - Write integration tests with hardware mocks

2. **Hardware Abstraction Refinement**
   - Review `integrations/protocols.py` interfaces
   - Ensure clean separation between real and simulated devices
   - Add comprehensive error codes in `integrations/hardware/exceptions.py`

3. **Test Coverage for Hardware**
   - Create mock device tests for all hardware protocols
   - Add CI workflow for hardware simulation tests
   - Document hardware testing procedures

### Next Sprint (Priority P1)

1. **GPU Acceleration**
   - Implement CUDA/MPS detection in config
   - Add batch processing utilities
   - Benchmark against CPU baseline

2. **Agent Tool Expansion**
   - Add `measure_density` tool with real hardware support
   - Add `print_target` tool for test prints
   - Integration test agent with hardware tools

3. **Performance Optimization**
   - Profile image processing pipeline
   - Implement tile-based processing
   - Add memory monitoring

---

## File Structure for New Development

```
src/ptpd_calibration/
├── integrations/
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base classes
│   │   ├── constants.py         # Hardware constants ✅
│   │   ├── exceptions.py        # Hardware exceptions ✅
│   │   ├── simulated.py         # Simulation devices ✅
│   │   ├── xrite_i1pro.py       # X-Rite implementation ⚠️ (enhance)
│   │   ├── colormunki.py        # ColorMunki support 🆕
│   │   └── cups_printer.py      # CUPS printing ✅
│   ├── protocols.py             # Protocol interfaces ✅
│   └── ...
├── agents/
│   ├── __init__.py
│   ├── agent.py                 # Main CalibrationAgent ✅
│   ├── tools.py                 # Tool registry ✅ (expand)
│   ├── planning.py              # Task planning ✅
│   ├── memory.py                # Memory system ✅
│   └── subagents/               # Specialized subagents 🆕
│       ├── calibration_agent.py
│       ├── qa_agent.py
│       └── troubleshooting_agent.py
└── ...
```

---

## Success Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Real Hardware Support | 0% | 100% | 3 months |
| GPU Acceleration | Partial | Full | 2 months |
| Test Coverage | ~70% | 90% | 2 months |
| Agent Tools | 10 | 20 | 3 months |
| Response Time (LLM) | 2-5s | <1s | 2 months |
| Image Processing | 2-5s | <500ms | 2 months |

---

## Conclusion

The Platinum-Palladium AI Printing Tool has a robust foundation with comprehensive agentic capabilities, deep learning modules, and testing infrastructure. The critical next step is **real hardware integration** to enable actual calibration workflows.

Following the universal-dev-agent pattern, development should proceed through the milestones with:
- **Planner** for architecture decisions
- **SQE** for test-first development
- **Coder** for implementation
- **Reviewer** for quality gates
- **Orchestrator** for integration

The agentic system is ready to be enhanced with hardware control tools, making it a truly autonomous calibration assistant.

---

## References

- Gap Analysis: `ANALYSIS_GAPS_AND_AI_INTEGRATION.md`
- Architecture: `ARCHITECTURE.md`
- Deep Learning Summary: `DEEP_LEARNING_IMPLEMENTATION.md`
- Project Guide: `CLAUDE.md`
- Agents Module: `src/ptpd_calibration/agents/`
- Hardware Integration: `src/ptpd_calibration/integrations/hardware/`
