# Platinum-Palladium AI Printing Tool - Copilot Instructions

## Project Context
This is a scientific calibration and AI printing tool for alternative photography processes (Platinum/Palladium).
It combines a FastAPI backend with a Gradio frontend and a Deep Learning module.

## Coding Standards

### Python
- **Type Hints**: Use strict type hints for all function arguments and return values.
- **Pydantic**: Use Pydantic v2 models for all data schemas (API requests, responses, internal data structures).
- **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.
- **Error Handling**: Use explicit error handling. Avoid bare `try-except`.
- **Testing**: Use `pytest`. Place unit tests in `tests/unit` and integration tests in `tests/integration`.

### Frontend (Gradio)
- **Theme**: Use the custom `ProLabTheme` defined in `src/ptpd_calibration/ui/theme.py`.
- **Components**: Use standard Gradio components but organized in `gr.Blocks`.
- **State**: Manage state carefully using `gr.State` components.

### Deep Learning (`src/ptpd_calibration/deep_learning`)
- **Configuration**: Use `pydantic-settings` for all model configurations.
- **Models**: Use PyTorch (`torch.nn.Module`).
- **Validation**: Ensure all AI outputs are validated against Pydantic models in `models.py`.
- **Dependencies**: Use lazy imports inside functions for heavy libraries (`torch`, `diffusers`) to keep startup time fast.

## Project Structure
- `src/ptpd_calibration/`: Main package source
- `src/ptpd_calibration/deep_learning/`: AI/ML module
- `src/ptpd_calibration/ui/`: Gradio UI components
- `tests/`: Test suite
- `examples/`: Usage validation scripts

## Git Conventions
- Commit messages should be descriptive (e.g., "feat: Add neural curve training pipeline").
