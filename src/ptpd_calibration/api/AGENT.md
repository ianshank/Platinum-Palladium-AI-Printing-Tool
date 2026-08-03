# Backend API Directory

## Purpose
FastAPI server providing REST endpoints for the Pt/Pd calibration system. Consumed by the React frontend.

## Key Files
- `server.py` — `create_app()` factory: CORS config, router registration, all endpoint definitions
- `deep_learning.py` — Deep learning router (conditionally loaded if PyTorch available)
- `mcts_router.py` — MCTS calibration endpoints
- `__init__.py` — Package exports

## Endpoint Groups
All endpoints are prefixed with `/api/`:
- `/api/health` — Health check (GET)
- `/api/analyze` — Density analysis (POST)
- `/api/scan/upload` — Step tablet scan upload (POST, multipart)
- `/api/curves/*` — Curve CRUD, generation, modification, export, quad upload/parse, enhance, smooth, blend
- `/api/calibrations` — Calibration records (GET list, POST create, GET by id)
- `/api/chat/*` — LLM chat, recipe suggestions, troubleshooting
- `/api/statistics` — System statistics (GET)
- `/api/export/*` — Negative TIFF, curve file, profile JSON exports

See CLAUDE.md "Backend API Integration" for full endpoint table.

## Conventions
- **Factory pattern**: `create_app()` returns configured FastAPI instance — lazy imports inside
- **Pydantic models**: Request/response bodies use models from `core/models.py` and `core/types.py`
- **Error handling**: Raise `HTTPException` with appropriate status codes
- **File uploads**: `UploadFile` + `Form` parameters for multipart endpoints
- **CORS**: Origins configured via settings; `localhost:3000` added automatically in dev/reload mode

## Testing
```bash
pytest tests/api/ -v
pytest tests/integration/ -v  # API integration tests
```

## Pitfalls
- Do NOT add endpoints without corresponding frontend types in `../../../frontend/src/types/models.ts`
- Deep learning router import is wrapped in try/except — PyTorch is optional
- Upload directory defaults to temp dir if not configured

## Related
- `../core/` — Pydantic models and types shared across all modules
- `../curves/` — Curve generation logic called by endpoints
- `../detection/` — Step tablet reader used by scan upload
- Frontend: `../../../frontend/src/api/client.ts` — TypeScript API client must mirror these endpoints
