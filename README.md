---
title: Pt/Pd Calibration Studio
emoji: 📷
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - photography
  - calibration
  - platinum
  - palladium
  - alternative-process
  - digital-negative
  - curve-editor
short_description: AI-powered calibration for platinum/palladium printing
---

# Platinum/Palladium Calibration Studio

An AI-powered calibration system for platinum/palladium alternative photographic printing. Combines traditional densitometry with Monte Carlo Tree Search optimization and multi-agent AI assistance.

## Architecture

The application has three main subsystems:

| Layer                | Stack                                            | Purpose                                                           |
| -------------------- | ------------------------------------------------ | ----------------------------------------------------------------- |
| **Frontend**         | React 18 + TypeScript + Zustand + TanStack Query | Interactive UI with curve editing, file upload, chat              |
| **Backend API**      | FastAPI + Pydantic                               | REST endpoints for curves, scans, calibrations, chat, MCTS        |
| **ML / MCTS Engine** | PyTorch + NumPy + scikit-learn                   | Physics simulation, neural network-guided search, data generation |

See [docs/architecture.md](docs/architecture.md) for full C4 diagrams.

## Features

### Core Calibration

- **Step Tablet Reading**: Automated detection and density extraction from scanned step tablets
- **Curve Generation**: Create linearization curves for digital negatives
- **Multi-Format Export**: Export to QuadTone RIP (.quad), Piezography, CSV, and JSON
- **Quad File Upload**: Drag-and-drop .quad file import with channel selection and preview

### AlphaZero MCTS Calibration Engine

- **Monte Carlo Tree Search**: UCB1/PUCT-guided exploration of printing parameter space
- **Physics Simulator**: Models sensitizer diffusion, UV exposure curves, humidity effects
- **Neural Network**: Dual policy+value network trained via Expert Iteration (self-play)
- **Quality Scorer**: Multi-metric evaluation (Dmax, tonal range, linearity, smoothness)
- **Constraint System**: Photochemistry-aware bounds ensure safe parameter recommendations
- **Result Export**: Optimized parameters exportable to JSON, CSV, and QTR formats

### Image Processing

- **Image Preview**: Preview curve effects on images before processing
- **Digital Negative Creation**: Create inverted negatives with curves applied
- **Histogram Analysis**: Zone-based tonal distribution analysis

### Printing Tools

- **Chemistry Calculator**: Calculate coating solutions based on Bostick-Sullivan formulas
- **Exposure Calculator**: UV exposure calculations with test strip generator
- **Zone System**: Ansel Adams zone analysis with development recommendations
- **Soft Proofing**: Preview prints on different paper types

### AI Assistance

- **Multi-Agent System**: Orchestrator coordinating planner, coder, reviewer, and SQE agents
- **Natural Language Chat**: RAG-powered Q&A about Pt/Pd printing
- **Recipe Suggestions**: Get customized coating recipes
- **Troubleshooting**: Diagnose common problems with AI guidance

## Quick Start

### Backend

```bash
pip install -e ".[dev]"
uvicorn src.ptpd_calibration.api.server:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
# Backend
pytest tests/unit/ tests/sanity/ --timeout=15 -q

# Frontend
cd frontend && npm test
```

## Project Structure

```text
├── frontend/                  React 18 + TypeScript + Vite
│   ├── src/
│   │   ├── api/              API client, TanStack Query hooks
│   │   ├── components/       UI components (curves, chemistry, chat)
│   │   ├── stores/           Zustand slices (curve, image, ui, mcts)
│   │   ├── pages/            Route-level page components
│   │   └── types/            TypeScript type definitions
│   └── vite.config.ts
├── src/ptpd_calibration/      Python backend
│   ├── api/                  FastAPI REST endpoints
│   ├── mcts/                 AlphaZero MCTS engine (5,181 LOC)
│   ├── agents/               Multi-agent system (6,155 LOC)
│   ├── deep_learning/        Data generators, training pipelines
│   ├── curves/               Curve generation/modification
│   ├── detection/            Step tablet detection
│   ├── chemistry/            Chemistry calculations
│   ├── llm/                  LLM integration (Anthropic, OpenAI)
│   └── ml/                   scikit-learn predictor + database
├── tests/                    pytest test suite (4,400+ tests)
├── docs/architecture.md      C4 architecture diagrams
└── CHANGELOG.md              Release history
```

## Next Steps

- [ ] MCTS API endpoints (search, training status, result retrieval)
- [ ] Frontend MCTS dashboard with live search visualization
- [ ] Batch processing queue with Celery + Redis
- [ ] Visual regression tests with Playwright
- [ ] PWA offline mode
- [ ] i18n support

## Requirements

- Python 3.10+
- Node.js 18+
- PyTorch (optional, for MCTS neural network features)
- A step tablet scan (Stouffer 21/31/41 step or similar)

## Links

- [GitHub Repository](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool)
- [Architecture Docs](docs/architecture.md)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/issues)

## Created By

Ian Cruickshank

## License

MIT License - see [LICENSE](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool/blob/main/LICENSE) for details.
