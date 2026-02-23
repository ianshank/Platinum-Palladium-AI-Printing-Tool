# Pt/Pd Calibration Studio - C4 Architecture

This document outlines the architecture of the Pt/Pd Calibration Studio application using the C4 model.

## Level 1: System Context

This diagram shows the high-level context, including the users and external systems interacting with your application.

```mermaid
C4Context
  title System Context Diagram for Pt/Pd Calibration Studio

  Person(user, "Photographer/Print-Maker", "A user who wants to calibrate their printing process.")
  System(app, "Pt/Pd Calibration Studio", "A React + FastAPI web application for generating, editing, and exporting platinum/palladium printing calibration curves with AI assistance.")

  System_Ext(llm, "External LLM Service", "Provides AI-powered analysis and chat capabilities (e.g., OpenAI, Anthropic).")
  System_Ext(fs, "File System", "Stores user-provided scans, exported curves, and machine learning data.")

  Rel(user, app, "Uses")
  Rel(app, fs, "Reads/Writes", "Curves, Scans, ML Data (JSON)")
  Rel(app, llm, "Makes API calls for analysis & chat", "HTTPS/JSON")
```

## Level 2: Containers

This diagram zooms into the system to show the main containers. In this case, the application is a single monolithic process.

```mermaid
C4Container
  title Container Diagram for Pt/Pd Calibration Studio

  Person(user, "Photographer/Print-Maker", "A user who wants to calibrate their printing process.")

  System_Boundary(c1, "Pt/Pd Calibration Studio") {
    Container(frontend, "React Frontend", "React 18, TypeScript, Zustand, TanStack Query, Vite", "SPA providing curve editing, AI assistant, calibration wizard, and MCTS dashboard.")
    Container(api, "FastAPI Backend", "Python 3.10+, FastAPI, Pydantic", "REST API for curves, scans, calibrations, chat, and MCTS endpoints.")
    Container(mcts, "MCTS Engine", "PyTorch, NumPy, scikit-learn", "AlphaZero-style calibration optimizer with physics simulation.")
    ContainerDb(fs, "File Storage", "JSON files", "Curve data (data/curves/*.json), scans, exports, ML checkpoints.")
  }

  System_Ext(llm, "External LLM Service", "Provides AI-powered analysis and chat capabilities (Anthropic Claude, OpenAI).")

  Rel(user, frontend, "Interacts with", "HTTPS/Browser")
  Rel(frontend, api, "REST calls", "JSON/HTTPS")
  Rel(api, mcts, "Delegates search", "Python")
  Rel(api, fs, "Reads/Writes", "JSON")
  Rel(api, llm, "Sends prompts, receives completions", "JSON/HTTPS")
```

## Level 3: Components

This diagram breaks down the "Web Application" container into its major internal components and shows how they interact.

```mermaid
C4Component
  title Component Diagram for Pt/Pd Calibration Studio Web App

  Container_Boundary(c1, "FastAPI Backend") {
    Component(api_server, "API Server", "api/server.py", "FastAPI app with routes: /api/curves, /api/scan, /api/calibrations, /api/chat, /api/mcts")
    Component(analyzer, "Scan Analyzer", "detection/", "Reads step tablet scans and extracts density data from patches.")
    Component(curves, "Curve Tools", "curves/", "Parsing, generating, modifying, smoothing, blending, and exporting calibration curves.")
    Component(enhancer, "AI Curve Enhancer", "ai_enhance.py", "Hybrid rule-based & LLM system for intelligent curve improvement with configurable goals.")
    Component(assistant, "LLM Assistant", "llm/assistant.py", "RAG-powered chat, recipe suggestions, and troubleshooting via Anthropic/OpenAI.")
    Component(llm_client, "LLM Client", "llm/client.py", "Low-level client for LLM API calls with provider abstraction.")
    Component(persistence, "Curve Persistence", "api/server.py (_store_curve/_get_curve)", "Write-through cache: runtime dict + data/curves/*.json files for durability.")

    Rel(api_server, analyzer, "Uses", "Scan analysis")
    Rel(api_server, curves, "Uses", "Curve CRUD + export")
    Rel(api_server, enhancer, "Uses", "AI enhance endpoint")
    Rel(api_server, assistant, "Uses", "Chat endpoints")
    Rel(api_server, persistence, "Uses", "Store/retrieve curves by ID")

    Rel(enhancer, llm_client, "Uses")
    Rel(assistant, llm_client, "Uses")
  }

  System_Ext(llm_ext, "External LLM Service", "LLM API (Anthropic/OpenAI)")
  Rel(llm_client, llm_ext, "Makes API calls to", "HTTPS")
```

## Level 4: Agent System (Code)

This diagram details the Agent System component, showing the multi-agent architecture with specialized subagents, orchestration, and infrastructure.

```mermaid
C4Component
  title Agent System Component Diagram (Level 4)

  Container_Boundary(agents, "Agent System (src/ptpd_calibration/agents/)") {
    Component(orchestrator, "OrchestratorAgent", "orchestrator.py (572 LOC)", "Coordinates all subagents, manages workflows with parallel/sequential execution")
    Component(calibration, "CalibrationAgent", "agent.py (383 LOC)", "ReAct-style reasoning loop for domain-specific calibration tasks")

    Component_Boundary(subagents, "Specialized Subagents (subagents/)") {
      Component(planner, "PlannerAgent", "planner.py (403 LOC)", "C4-aligned planning, epic/story decomposition, acceptance criteria")
      Component(sqa, "SQEAgent", "sqa.py (503 LOC)", "Test plan generation, pytest code generation, validation")
      Component(coder, "CoderAgent", "coder.py (518 LOC)", "Code generation following CLAUDE.md conventions")
      Component(reviewer, "ReviewerAgent", "reviewer.py (555 LOC)", "Code review with scoring, OWASP security analysis")
    }

    Component_Boundary(infra, "Infrastructure") {
      Component(bus, "MessageBus", "communication.py (473 LOC)", "Priority-based inter-agent messaging")
      Component(registry, "SubagentRegistry", "subagents/base.py (528 LOC)", "Dynamic subagent discovery and registration")
      Component(logging, "StructuredLogger", "logging.py (543 LOC)", "JSON logging with OpenTelemetry-inspired trace IDs")
    }

    Component_Boundary(state, "State Management") {
      Component(memory, "AgentMemory", "memory.py (315 LOC)", "Tiered memory: long-term (1K items) + working memory")
      Component(planning, "PlanEngine", "planning.py (273 LOC)", "Task decomposition with adaptation tracking")
      Component(tools, "ToolRegistry", "tools.py (627 LOC)", "10+ calibration tools with Anthropic format conversion")
    }

    Rel(orchestrator, planner, "Delegates planning")
    Rel(orchestrator, sqa, "Delegates testing")
    Rel(orchestrator, coder, "Delegates implementation")
    Rel(orchestrator, reviewer, "Delegates review")
    Rel(orchestrator, calibration, "Delegates domain tasks")
    Rel(orchestrator, bus, "Sends messages")
    Rel(orchestrator, registry, "Discovers subagents")

    Rel(calibration, tools, "Executes tools")
    Rel(calibration, memory, "Stores/recalls context")
    Rel(calibration, planning, "Manages task plans")

    Rel(bus, logging, "Logs all messages")
  }
```

### Agent System Summary

| Component             | Lines of Code | Purpose                           |
| --------------------- | ------------- | --------------------------------- |
| **OrchestratorAgent** | 572           | Multi-agent workflow coordination |
| **CalibrationAgent**  | 383           | ReAct reasoning with tools        |
| **PlannerAgent**      | 403           | C4-aligned planning               |
| **SQEAgent**          | 503           | Test generation                   |
| **CoderAgent**        | 518           | Code implementation               |
| **ReviewerAgent**     | 555           | Code review                       |
| **MessageBus**        | 473           | Inter-agent communication         |
| **BaseSubagent**      | 528           | Subagent infrastructure           |
| **StructuredLogger**  | 543           | JSON logging                      |
| **AgentMemory**       | 315           | Tiered memory system              |
| **PlanEngine**        | 273           | Task planning                     |
| **ToolRegistry**      | 627           | Tool management                   |
| **Total**             | **6,155**     | Production-ready agent system     |

---

## Level 4: MCTS Calibration Engine (Code)

This diagram details the AlphaZero-style Monte Carlo Tree Search engine for automated calibration optimization.

```mermaid
C4Component
  title MCTS Calibration Engine Component Diagram (Level 4)

  Container_Boundary(mcts, "MCTS Engine (src/ptpd_calibration/mcts/)") {
    Component(engine, "MCTSEngine", "engine.py (552 LOC)", "Core MCTS search with UCB1 exploration, parallel leaf evaluation, configurable search budget")
    Component(tree, "TreeNode", "tree.py (242 LOC)", "Search tree nodes with visit counts, value estimates, prior probabilities, PUCT selection")

    Component_Boundary(nn, "Neural Networks (requires PyTorch)") {
      Component(dual, "DualNetwork", "networks.py (459 LOC)", "Policy + value head CNN for state evaluation and action priors")
      Component(trainer, "MCTSTrainer", "training.py (563 LOC)", "Expert Iteration self-play training with ReplayBuffer and prioritized sampling")
    }

    Component_Boundary(domain, "Domain Model") {
      Component(sim, "ExtendedProcessSimulator", "simulator.py (377 LOC)", "Physics-based Pt/Pd model: sensitizer diffusion, UV exposure curves, humidity effects")
      Component(quality, "QualityScorer", "quality.py (247 LOC)", "Multi-metric scoring: Dmax, tonal range, linearity, smoothness, overall quality grade")
      Component(constraints, "ConstraintChecker + ActionPruner", "constraints.py (1056 LOC)", "Photochemistry-safe parameter bounds, action space pruning, validation")
    }

    Component_Boundary(io, "I/O and Config") {
      Component(config, "MCTSSettings", "config.py (459 LOC)", "Pydantic settings: parameter ranges, physics constants, search hyperparameters")
      Component(types, "CalibrationState/Action", "types.py (186 LOC)", "Immutable state representation, action encoding, search results")
      Component(export, "MCTSResultExporter", "export.py (491 LOC)", "Export to JSON, CSV, QTR-compatible .quad format")
    }

    Component_Boundary(agents_mcts, "Domain Subagents") {
      Component(chem_agent, "ChemistrySubagent", "agents.py", "Validates chemistry parameters against safety bounds")
      Component(expo_agent, "ExposureSubagent", "agents.py", "Optimizes UV exposure time and intensity")
      Component(coord_agent, "CalibrationCoordinatorSubagent", "agents.py (487 LOC total)", "Coordinates multi-step calibration optimization")
    }

    Rel(engine, tree, "Expands/selects nodes")
    Rel(engine, dual, "Evaluates leaf states")
    Rel(engine, sim, "Simulates actions")
    Rel(engine, quality, "Scores simulation results")
    Rel(engine, constraints, "Prunes illegal actions")

    Rel(trainer, engine, "Runs self-play episodes")
    Rel(trainer, dual, "Trains network weights")

    Rel(coord_agent, engine, "Runs MCTS search")
    Rel(coord_agent, chem_agent, "Delegates chemistry checks")
    Rel(coord_agent, expo_agent, "Delegates exposure optimization")
    Rel(coord_agent, export, "Exports optimal parameters")
  }
```

### MCTS Engine Summary

| Component                    | Lines of Code | Purpose                              |
| ---------------------------- | ------------- | ------------------------------------ |
| **MCTSEngine**               | 552           | Core search algorithm with UCB1/PUCT |
| **TreeNode**                 | 242           | Search tree with statistics          |
| **DualNetwork**              | 459           | Policy + value neural network        |
| **MCTSTrainer**              | 563           | Self-play training pipeline          |
| **ExtendedProcessSimulator** | 377           | Physics-based process model          |
| **QualityScorer**            | 247           | Multi-metric quality evaluation      |
| **ConstraintChecker**        | 1,056         | Photochemistry validation            |
| **MCTSSettings**             | 459           | Configuration and constants          |
| **Types**                    | 186           | State/action data structures         |
| **MCTSResultExporter**       | 491           | Multi-format export                  |
| **Domain Subagents**         | 487           | Chemistry/exposure/coordination      |
| **Total**                    | **5,181**     | Full MCTS calibration system         |

---

## Cross-Cutting: Frontend Architecture

The React frontend connects to the FastAPI backend and provides the user interface.

```mermaid
C4Component
  title Frontend Architecture (Level 3)

  Container_Boundary(fe, "React Frontend (frontend/src/)") {
    Component_Boundary(page_layer, "Pages (pages/)") {
      Component(curves_page, "CurvesPage", "CurvesPage.tsx", "3-tab Radix UI structure: Upload .quad → Edit Curve (AI Enhance) → Export (QTR/CSV/JSON)")
      Component(ai_page, "AIAssistantPage", "AIAssistantPage.tsx", "Chat interface with context panel (paper/ratio/calibrations) + recipe/troubleshoot quick actions")
      Component(other_pages, "Other Pages", "Dashboard, Calibration, Chemistry, MCTS, Settings, SessionLog", "Domain-specific route-level pages")
    }

    Component_Boundary(comp_layer, "UI Components (components/)") {
      Component(curve_upload, "CurveUpload", "curves/CurveUpload.tsx", "Drag-and-drop .quad upload with paste mode, calls onLoadCurve(data, id, name)")
      Component(curve_editor, "CurveEditor", "curves/CurveEditor.tsx", "Recharts curve visualization, adjustments, AI Enhance with 7 goals, undo/redo")
      Component(export_panel, "ExportPanel", "export/ExportPanel.tsx", "Format selector (QTR/Piezography/CSV/JSON) + download via useExportCurve")
      Component(ai_assistant, "AIAssistant", "assistant/AIAssistant.tsx", "Streaming chat with context panel, useChat hook, recipe/troubleshoot mutations")
      Component(other_comps, "Other Components", "calibration/, chemistry/, preview/, ui/", "Reusable domain components")
    }

    Component(stores, "Zustand Store", "stores/", "Slices: chemistry, calibration, chat, curve, image, ui, mcts — immer + devtools middleware")
    Component(hooks, "TanStack Query Hooks", "api/hooks.ts", "useEnhanceCurve, useExportCurve, useRecipeSuggestion, useTroubleshootRequest, useChat, etc.")
    Component(client, "API Client", "api/client.ts", "Axios instance: api.curves.export → POST /{curveId}/export?format=, typed endpoints")

    Rel(curves_page, curve_upload, "Renders in Upload tab")
    Rel(curves_page, curve_editor, "Renders in Edit tab")
    Rel(curves_page, export_panel, "Renders in Export tab")
    Rel(ai_page, ai_assistant, "Renders")
    Rel(curve_editor, hooks, "useEnhanceCurve")
    Rel(ai_assistant, hooks, "useRecipeSuggestion, useTroubleshootRequest")
    Rel(ai_assistant, stores, "chemistry.paperSize, chemistry.metalRatio, calibration.history")
    Rel(export_panel, hooks, "useExportCurve")
    Rel(hooks, client, "HTTP requests")
  }

  System_Ext(api, "FastAPI Backend", "/api/* endpoints")
  Rel(client, api, "REST calls", "JSON/HTTPS")
```
