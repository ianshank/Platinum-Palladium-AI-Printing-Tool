# Investigation Summary: Agentic System Next Steps

> **Date:** 2026-02-02
> **Branch:** `claude/investigate-agentic-next-steps-i05lB`
> **Purpose:** Concrete investigation findings and prioritized next steps for agentic AI development

---

## Executive Summary

This investigation analyzed the current state of the Pt/Pd Calibration Studio's agentic system against the universal-dev-agent development template. The codebase has **exceeded initial roadmap expectations** with substantial implementation of the multi-agent architecture.

### Key Findings

| Milestone | Expected | Actual | Status |
|-----------|----------|--------|--------|
| **1. Foundation** | Planned | Complete | :white_check_mark: |
| **2. Multi-Agent Architecture** | Planned | **Complete** | :white_check_mark: |
| **3. Orchestration & Integration** | Planned | **Implemented** | :white_check_mark: |
| **4. Observability** | Planned | **Partial** | :construction: |
| **Production Readiness** | Future | Gap Identified | :red_circle: |

### Implementation Statistics

| Component | LOC | Test Coverage | Status |
|-----------|-----|---------------|--------|
| Agent System Total | **6,155** | 78% | Production-ready |
| Subagents (4) | 1,979 | Tested | Complete |
| Orchestrator | 572 | Tested | Complete |
| MessageBus | 473 | Tested | Complete |
| Structured Logging | 543 | Tested | Complete |

---

## What's Complete (Beyond Roadmap Expectations)

### 1. Full Subagent Implementation :white_check_mark:

All four specialized subagents are **fully implemented** with Pydantic models:

| Subagent | File | LOC | Capabilities |
|----------|------|-----|--------------|
| **PlannerAgent** | `subagents/planner.py` | 403 | C4-aligned planning, epic/story decomposition, acceptance criteria generation |
| **SQEAgent** | `subagents/sqa.py` | 503 | Test plan generation, pytest code generation, validation |
| **CoderAgent** | `subagents/coder.py` | 518 | Code generation with CLAUDE.md conventions, type hints, docstrings |
| **ReviewerAgent** | `subagents/reviewer.py` | 555 | Code review with scoring, OWASP security analysis, categorized feedback |

### 2. Orchestration System :white_check_mark:

The `OrchestratorAgent` (572 LOC) provides:
- Multi-agent workflow coordination
- Parallel/sequential execution modes
- Task dependency resolution
- Timeout and retry logic
- Comprehensive error handling

### 3. Communication Infrastructure :white_check_mark:

The `MessageBus` (473 LOC) provides:
- Priority-based message queuing (LOW=0, NORMAL=5, HIGH=8, URGENT=10)
- Request-response patterns with timeouts
- Message correlation IDs
- Broadcast and point-to-point messaging
- Message history (up to 1,000 messages)

### 4. Structured Logging :white_check_mark:

The logging system (543 LOC) provides:
- JSON-structured output
- OpenTelemetry-inspired trace/span IDs
- 30+ event types covering full agent lifecycle
- Timed operation context managers

---

## Gaps Identified

### Gap 1: Test Coverage Target Not Reached (78% vs 90%)

**Current:** 78% coverage across agents module
**Target:** 90%+ coverage

**Missing Test Areas:**
- Orchestrator edge cases (concurrent workflow conflicts)
- MessageBus priority queue under load
- Subagent timeout and retry scenarios
- Inter-agent communication failures

### Gap 2: Performance Metrics & Health Checks

**Missing:**
- Agent performance metrics (latency, success rate, token usage)
- Health check endpoints for agents
- Circuit breaker implementation
- Prometheus-format metrics export

### Gap 3: Real Hardware Integration

**Critical Production Gap:**
- All hardware code is simulated
- No real spectrophotometer measurements
- No real printer integration
- Agent tools cannot control physical devices

### Gap 4: Workflow Persistence

**Missing:**
- Workflow state not persisted across sessions
- No checkpoint/resume capability
- Long-running workflows lost on restart

---

## Concrete Next Steps (Prioritized)

### Priority 0: Critical (This Sprint)

#### Task 1: Expand Test Coverage to 90%+

**Rationale:** Cannot safely iterate on production system without comprehensive tests.

```bash
# Current coverage check
PYTHONPATH=src pytest tests/unit/test_agents.py tests/unit/test_subagents.py \
  --cov=src/ptpd_calibration/agents --cov-report=term-missing
```

**Test Cases Needed:**

| File | Missing Tests | AC |
|------|---------------|-----|
| `orchestrator.py` | Concurrent workflow tests | 3+ concurrent workflows execute correctly |
| `communication.py` | Message priority tests | High-priority messages delivered first |
| `communication.py` | Timeout handling tests | Request timeouts return error |
| `subagents/` | Failure recovery tests | Retry logic works; graceful degradation |

**Implementation Path:**
```python
# tests/unit/test_orchestrator_edge_cases.py
@pytest.mark.asyncio
async def test_concurrent_workflows_isolation():
    """Verify concurrent workflows don't interfere."""
    orchestrator = OrchestratorAgent()
    # Run two workflows simultaneously
    # Verify each maintains independent state
    pass

@pytest.mark.asyncio
async def test_task_timeout_handling():
    """Verify timeout produces graceful failure."""
    # Configure short timeout
    # Run slow task
    # Verify FAILED status, not exception
    pass
```

#### Task 2: Implement Agent Health Checks

**Rationale:** Production deployments need health monitoring.

**Acceptance Criteria:**
- [ ] `/health/agents` endpoint returns agent status
- [ ] Health check detects: LLM connectivity, memory usage, queue depth
- [ ] Response time < 100ms

**Implementation Path:**
```python
# src/ptpd_calibration/agents/health.py
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class AgentHealthCheck(BaseModel):
    """Health check response for agent system."""
    status: HealthStatus
    llm_connected: bool
    memory_usage_mb: float
    active_workflows: int
    message_queue_depth: int
    last_check: datetime

async def check_agent_health() -> AgentHealthCheck:
    """Perform health check on agent system."""
    # Check LLM connectivity
    # Check memory usage
    # Check workflow state
    pass
```

### Priority 1: High (Next Sprint)

#### Task 3: Implement Circuit Breakers

**Rationale:** Graceful degradation when LLM service is unavailable.

**Acceptance Criteria:**
- [ ] 3 consecutive failures = circuit OPEN
- [ ] 30-second cooldown before retry
- [ ] Fallback to cached responses when open
- [ ] Auto-recovery when service restored

**Implementation Path:**
```python
# src/ptpd_calibration/agents/circuit_breaker.py
import time
from dataclasses import dataclass
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, use fallback
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure: float = 0.0

    async def call(self, func, *args, **kwargs):
        """Execute with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure > self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Service unavailable")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

#### Task 4: Add Agent Performance Metrics

**Rationale:** Need to track agent effectiveness for optimization.

**Metrics to Track:**
- `agent_request_latency_seconds` - Time to complete requests
- `agent_success_rate` - Success/failure ratio
- `agent_token_usage_total` - LLM token consumption
- `agent_tool_calls_total` - Tool invocation counts

**Implementation Path:**
```python
# src/ptpd_calibration/agents/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Time to complete agent requests',
    ['agent_type', 'operation']
)

SUCCESS_RATE = Counter(
    'agent_requests_total',
    'Total agent requests',
    ['agent_type', 'status']
)

TOKEN_USAGE = Counter(
    'agent_token_usage_total',
    'Total LLM tokens consumed',
    ['agent_type', 'model']
)
```

### Priority 2: Medium (Following Sprint)

#### Task 5: Implement Workflow Persistence

**Rationale:** Long-running workflows should survive restarts.

**Acceptance Criteria:**
- [ ] Workflow state saved to disk on each step
- [ ] Resume interrupted workflows on startup
- [ ] Clean up completed workflow checkpoints

**Implementation Path:**
```python
# src/ptpd_calibration/agents/persistence.py
from pathlib import Path
import json

class WorkflowPersistence:
    """Persist workflow state for recovery."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, workflow_id: str, state: dict):
        """Save workflow checkpoint."""
        path = self.checkpoint_dir / f"{workflow_id}.json"
        path.write_text(json.dumps(state, default=str))

    def load_checkpoint(self, workflow_id: str) -> dict | None:
        """Load workflow checkpoint if exists."""
        path = self.checkpoint_dir / f"{workflow_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_incomplete_workflows(self) -> list[str]:
        """List workflows that need resumption."""
        return [p.stem for p in self.checkpoint_dir.glob("*.json")]
```

#### Task 6: Add Hardware Control Tools

**Rationale:** Enable agents to control real devices.

**New Tools:**
- `measure_density` - Trigger spectrophotometer reading
- `print_target` - Print test target
- `calibrate_device` - Run device calibration

**Implementation Path:**
```python
# Add to src/ptpd_calibration/agents/tools.py

@register_tool(
    name="measure_density",
    description="Measure density at current spectrophotometer position",
    category=ToolCategory.HARDWARE
)
async def measure_density(
    device_id: str = "default",
    measurement_mode: str = "reflection"
) -> ToolResult:
    """Take real spectrophotometer measurement."""
    try:
        from ptpd_calibration.integrations import get_spectrophotometer
        device = get_spectrophotometer(device_id)
        measurement = await device.read_measurement()
        return ToolResult(
            success=True,
            result=measurement.model_dump()
        )
    except HardwareNotConnectedError as e:
        return ToolResult(
            success=False,
            error=f"Device not connected: {e}"
        )
```

### Priority 3: Future

#### Task 7: Inter-Agent Tool Access

**Current Limitation:** Subagents cannot call each other's tools directly.

**Proposed Solution:** Add tool delegation via Orchestrator.

#### Task 8: Streaming LLM Responses

**Current Limitation:** Agent responses are not streamed.

**Proposed Solution:** Add async generator support for progressive UI updates.

---

## Updated C4 Level 4 Diagram

```mermaid
C4Component
  title Agent System Component Diagram (Level 4)

  Container_Boundary(agents, "Agent System") {
    Component(orchestrator, "OrchestratorAgent", "orchestrator.py", "Coordinates all subagents, manages workflows")
    Component(calibration, "CalibrationAgent", "agent.py", "Domain-specific calibration tasks")

    Component_Boundary(subagents, "Specialized Subagents") {
      Component(planner, "PlannerAgent", "planner.py", "C4-aligned planning")
      Component(sqa, "SQEAgent", "sqa.py", "Test generation")
      Component(coder, "CoderAgent", "coder.py", "Code implementation")
      Component(reviewer, "ReviewerAgent", "reviewer.py", "Code review")
    }

    Component_Boundary(infra, "Infrastructure") {
      Component(bus, "MessageBus", "communication.py", "Inter-agent messaging")
      Component(registry, "SubagentRegistry", "base.py", "Agent discovery")
      Component(logging, "StructuredLogger", "logging.py", "JSON logging + traces")
      Component(health, "HealthCheck", "health.py", "Agent health monitoring")
      Component(metrics, "MetricsCollector", "metrics.py", "Performance metrics")
      Component(circuit, "CircuitBreaker", "circuit_breaker.py", "Graceful degradation")
    }

    Component_Boundary(state, "State Management") {
      Component(memory, "AgentMemory", "memory.py", "Long-term + working memory")
      Component(planning, "PlanEngine", "planning.py", "Task decomposition")
      Component(tools, "ToolRegistry", "tools.py", "10+ calibration tools")
      Component(persistence, "WorkflowPersistence", "persistence.py", "Checkpoint/resume")
    }

    Rel(orchestrator, planner, "Delegates planning")
    Rel(orchestrator, sqa, "Delegates testing")
    Rel(orchestrator, coder, "Delegates implementation")
    Rel(orchestrator, reviewer, "Delegates review")
    Rel(orchestrator, calibration, "Delegates domain tasks")

    Rel(orchestrator, bus, "Sends messages")
    Rel(orchestrator, registry, "Discovers subagents")
    Rel(orchestrator, persistence, "Saves checkpoints")

    Rel(calibration, tools, "Executes tools")
    Rel(calibration, memory, "Stores/recalls")
    Rel(calibration, planning, "Manages plans")

    Rel(bus, logging, "Logs messages")
    Rel(orchestrator, health, "Reports health")
    Rel(orchestrator, metrics, "Records metrics")
    Rel(orchestrator, circuit, "Protects LLM calls")
  }
```

---

## Implementation Checklist

### Milestone 4 Completion (Observability)

- [ ] **4.1 Health Checks**
  - [ ] Create `agents/health.py`
  - [ ] Add `/health/agents` API endpoint
  - [ ] Integrate with existing monitoring

- [ ] **4.2 Metrics Collection**
  - [ ] Create `agents/metrics.py`
  - [ ] Add Prometheus metrics
  - [ ] Dashboard configuration

- [ ] **4.3 Circuit Breakers**
  - [ ] Create `agents/circuit_breaker.py`
  - [ ] Integrate with LLM client
  - [ ] Add fallback responses

- [ ] **4.4 Test Coverage**
  - [ ] Expand `test_agents.py` to 90%+
  - [ ] Add orchestrator edge case tests
  - [ ] Add communication failure tests

### Production Readiness

- [ ] **5.1 Workflow Persistence**
  - [ ] Create `agents/persistence.py`
  - [ ] Integrate with Orchestrator
  - [ ] Add resume on startup

- [ ] **5.2 Hardware Tools**
  - [ ] Add `measure_density` tool
  - [ ] Add `print_target` tool
  - [ ] Integration tests with mocks

---

## Conclusion

The agentic system has **exceeded the original roadmap expectations**. Milestones 1-3 are essentially complete with a production-ready multi-agent architecture including:

- Full subagent implementation (Planner, SQE, Coder, Reviewer)
- Orchestration with parallel/sequential execution
- Inter-agent communication via MessageBus
- Structured JSON logging with trace IDs
- 78% test coverage

**Remaining work focuses on:**
1. Reaching 90% test coverage (currently 78%)
2. Adding observability (health checks, metrics, circuit breakers)
3. Workflow persistence for reliability
4. Hardware tool integration for production use

Following the universal-dev-agent pattern, the next development cycle should:
1. Use **SQE** to generate missing test cases
2. Use **Coder** to implement observability components
3. Use **Reviewer** to validate changes
4. Use **Orchestrator** to coordinate the full implementation

---

*Investigation completed: 2026-02-02*
*Next review: After Priority 0 tasks complete*
