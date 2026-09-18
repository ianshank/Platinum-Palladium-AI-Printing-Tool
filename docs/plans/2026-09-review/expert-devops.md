# DevOps / CI / Supply-Chain Review — Platinum-Palladium-AI-Printing-Tool

Date: 2026-09-18 · Reviewer role: DevOps/CI/Supply-chain · Scope: read-only analysis of
`/home/user/Platinum-Palladium-AI-Printing-Tool` (branch `claude/ptpd-validation-sdlc-plan-j2a2gc`, HEAD c9ef03a).

Companion files in this directory:

- `ci-proposed.yml` — the single consolidated workflow (SHA-pinned, least-privilege, gated).
- `Dockerfile.proposed` — multi-stage image for a Hugging Face **Docker** Space.
- `coverage-unit.txt` / `coverage-unit.xml` — full unit-suite coverage run used for the floors in §6
  (note: the XML stopped mid-write after `deep_learning`; per-file numbers in the .txt are complete).

Verification legend: **[V]** verified in this session (command/API/file cited), **[U]** unverified / from memory, **[F]** already-verified fact supplied by the coordinator.

New facts established this session (not in the supplied list):

- **The GitHub default branch is `claude/implement-chat-requirements-014YeEiyBSMJL91puKEgVcih`, not `main`**, and `main` is **unprotected** (`protected: false`). [V — `GET /repos/...` and `/branches/main`]
- Run 30861720378 (push to `claude/code-hygiene-modularity-9vjyci`, 2026-08-03) reports **"Deploy to Hugging Face Spaces: success" (8 s) while "Run Tests (unit)" failed**. The Space was last modified 29 Nov 2025 [V — HF API], so the job reported success without deploying (it `exit 0`s when `HF_USERNAME`/`HF_TOKEN` are unset, ci-cd.yml:177-185). Job logs are no longer retrievable (404). The deploy path is therefore a *false green* today and a *feature-branch-overwrites-prod* path the moment the variables are set.
- `.husky/pre-push` is **UTF-16LE with CRLF** (`file` output); its content is `npm test` and the root `package.json` has no `test` script → the hook cannot run correctly. [V]
- `[project.scripts] ptpd = "ptpd_calibration.cli:main"` (pyproject.toml:84) points at a module that **does not exist** (`find src -name 'cli*.py'` → only `llm/client.py`). [V]
- Hard (top-level) imports of packages not declared in `pyproject.toml`: `dotenv` (config.py:14), `psutil` (monitoring/performance.py:21), `yaml` (workflow/recipe_manager.py:20). `python-dotenv` only works because `pydantic-settings` drags it in. [V]
- `uv lock --dry-run --python 3.12` resolves the current `pyproject.toml` cleanly (exit 0) — the uv migration has no resolver blocker. [V] `uv 0.8.17` has **no `uv audit`** subcommand; `uv export` supports `requirements.txt` and `pylock.toml`; `uv lock --check` exists. [V]
- Ruff baseline: `ruff check .` (tests.yml scope) = **129** errors / **51** unformatted files; `ruff check src/ tests/` = 86 / 34; `ruff check --select S src/` = **65** (34×S311, 10×S608, 7×S101, 5×S112, 4×S110, 2×S104, S108, S301, S314). [V]
- Mypy errors by package (721 total): deep_learning 351, api 97, ui 96, neuro_symbolic 40, qa 36, agents 32, imaging 31, ml 28, monitoring 23, education 19, integrations 17, curves 15, detection 14, llm 13, advanced 10, gcp 8, data 6, zones 5, mcts 5, analysis 4, session 2, ai 2, batch 1. **Zero errors**: core, chemistry, calculations, exposure, proofing, papers, workflow, vertex, config.py. [V]
- Full unit suite (3 940 passed / 3 failed / 127 skipped, 110 s, torch absent): whole-package coverage **64 %** (branch-inclusive, `ui/` omitted) — ci.yml:79 demands 70 and would fail even with a clean collection. [V]
- Per PR push from a `claude/*` head branch today: **33 job slots** across 4 workflow runs (CI 2, Tests 17, CI/CD 7 on `pull_request`, CI/CD 7 again on `push`); measured 1 684 runner-seconds while failing fast. [V — `/actions/runs/{id}/jobs`]

---

## 1. CI defect table

Effort: S < ½ day, M = 1–3 days. "Fix" refers to the consolidated workflow unless stated.

| # | Location | Defect | Evidence | Fix |
|---|----------|--------|----------|-----|
| D01 | `.github/workflows/ci.yml:25-33` | `actions/setup-node` with `cache: pnpm` runs **before** `pnpm/action-setup` → "Unable to locate executable file: pnpm" | Frontend job failed in 11 s (run 30861723266) [V] | Install pnpm first (`ci-proposed.yml` frontend job); read version from `packageManager` |
| D02 | `ci.yml:70`, `ci-cd.yml:36/65/131`, `tests.yml:30/63` | Three different dependency sets: `.[dev,all]`, `requirements.txt` (+ad-hoc `pip install pytest … psutil`), `.[dev]`/`.[all,dev,test]` — none locked | requirements.txt has matplotlib/tifffile absent from pyproject [F]; fastapi absent from requirements.txt so ci-cd never runs `tests/api` | `uv sync --frozen` from `uv.lock` everywhere (§3) |
| D03 | `tests.yml:63,95,125,161,194,231,263` | Extra `test` does not exist (pyproject.toml:48-81) | pip only warns, so pytest-timeout/pytest-benchmark/selenium are never installed → `--timeout` (tests.yml:167) and `--benchmark-json` (:198) are unknown args | Add `dependency-groups` (§3) and use `--group dev` |
| D04 | `ci.yml:59` (3.12), `ci-cd.yml:15` (3.11), `tests.yml:23,49` (3.10-3.12), `pyproject.toml:11,174` (>=3.10 / mypy 3.10) | Inconsistent interpreter versions; no `.python-version`; `.gitignore:86` **ignores** `.python-version` | [V] | `.python-version = 3.12`, un-ignore it, `requires-python = ">=3.11,<3.14"` (3.10 EOL 2026-10) [U: product decision] |
| D05 | `ci-cd.yml:22` + `:40` + `:43` | Lint job is `continue-on-error` **and** every step ends in `\|\| true` — doubly masked; shows green with 86 ruff errors | run 30861723100 "Lint & Type Check: success" [V] | One blocking ruff/mypy step set (backend job) |
| D06 | `ci-cd.yml:102-106` | e2e pytest `\|\| true` | "Run Tests (e2e): success" while unit fails [V] | Remove; e2e is Playwright in the new e2e job, advisory only in phase 1 via job-level `continue-on-error` (transparent in the gate) |
| D07 | `ci-cd.yml:114-118` | `sanity-check` has `needs: test` + `if: always()` → runs/passes after unit failure; `:140-149` imports the **legacy Gradio app** as the deploy gate | run 30861720378: unit failure, sanity success, deploy success [V] | Deploy gate = `all-green` job |
| D08 | `ci-cd.yml:151-159, 191` | Deploy runs on push to **any `claude/*` branch** and `git push … --force` to the Space `main` | [V] | Deploy only on `v*` tags after gate, behind `environment: huggingface` with required reviewer |
| D09 | `ci-cd.yml:177-185` | Missing `HF_TOKEN`/`HF_USERNAME` → `exit 0` = green | 8-second "success" with no Space change [V] | `: "${HF_TOKEN:?}"` – fail loudly |
| D10 | `ci-cd.yml:188` | Token embedded in the remote URL (`https://user:token@…`) → persisted in `.git/config`, printable via `git remote -v` | [V] | HTTP `extraheader` auth, `persist-credentials: false` |
| D11 | `ci-cd.yml:4-11` | `push` **and** `pull_request` on `claude/*` → every PR commit runs the workflow twice | runs 30861723100 (PR) + 30861720378 (push) same SHA [V] | Trigger: `push: [main, tags]`, `pull_request: [main]` only |
| D12 | `ci-cd.yml:82-88` | Six `--ignore` flags quarantine tests silently (no issue link, no marker) | [V] | Register `xfail(strict)`/skip markers with reason + issue; `--strict-markers` |
| D13 | `tests.yml:5-7` | Triggers on `develop`, which does not exist | `git branch -a` [V] | Remove |
| D14 | `tests.yml:33,36` | `ruff check .` / `ruff format --check .` over the entire repo incl. `hf_check/`, `scripts/`, `examples/` → 129 errors / 51 files vs 86/34 for src+tests | [V] | Scope to `src/ tests/ app.py scripts/`; untrack `hf_check/` (§5) |
| D15 | `tests.yml:44-49, 67` | 3 OS × 3 Python unit matrix with `-x`; macOS billed 10×, Windows 2× (private-repo rates); no OS-specific code except guarded win32 imports (`integrations/hardware/win32_printer.py:62-64`) | 9 jobs, 41–190 s each while failing at collection [V] | Single `ubuntu/3.12` on PR; optional weekly 3.11/3.13 matrix |
| D16 | `tests.yml:168,199,237,291` | `continue-on-error` on e2e / performance / visual / coverage threshold → four always-green jobs | [V] | Drop selenium/visual/perf from PR CI; keep perf as scheduled advisory |
| D17 | `tests.yml:69-76,101-106,131-136,282-286` | `codecov/codecov-action@v4` without `CODECOV_TOKEN`, `fail_ci_if_error: false` | no `CODECOV_TOKEN` reference anywhere in `.github/` [V] | Either add token + `codecov.yml` (§6) or use diff-cover (no SaaS) |
| D18 | `tests.yml:129,167,198,236`; `pyproject.toml:104-107` | Markers `api`, `selenium`, `performance`, `visual`, `e2e`, `slow`, `unit`, `integration`, `deep`, `user_journey`, `browser` are **unregistered** | no `markers =` in pyproject [V] | Register (§6) and run with `--strict-markers --strict-config` |
| D19 | `tests.yml:293-309` | Gate job depends on `lint`, which is always red (D14) → gate is useless for branch protection | PR #36: all runs failed [F] | New `all-green` job with explicit required set |
| D20 | `pyproject.toml:107` | `addopts = "-v --cov=… --cov-report=term-missing"` forces coverage into every pytest invocation (slower, noisy, doubles `--cov` in tests.yml:267-274) | [V] | `addopts = "-ra --strict-markers --strict-config"`; coverage flags only in CI |
| D21 | all three workflows | No `permissions:` block → `GITHUB_TOKEN` gets the repo default (write on repos created before 2023-02) [U: check Settings → Actions → Workflow permissions] | [V absence] | `permissions: {}` at top; per-job grants |
| D22 | all three workflows | Actions pinned by mutable tag (`@v4`, `@v5`, `@v1`, `@v2`) incl. third-party `browser-actions/setup-chrome`, `nanasess/setup-chromedriver`, `codecov/codecov-action` | [V] | Full-SHA pins + Dependabot `github-actions` ecosystem |
| D23 | `ci-cd.yml`, `tests.yml` | No `concurrency` → superseded commits keep running; no `timeout-minutes` anywhere | [V] | Workflow-level concurrency; per-job timeouts |
| D24 | `ci.yml:64-67`, others | `setup-python` `cache: pip` with no `cache-dependency-path` — key derived from `requirements*.txt`/pyproject [U: default glob], so the ad-hoc `pip install` lines are never cached | — | `astral-sh/setup-uv` `enable-cache` keyed on `uv.lock` |
| D25 | `ci.yml:44-50` | Uploads `htmlcov/` which is never produced (`--cov-report=term-missing` only) | [V] | Produce `xml` + `html` explicitly |
| D26 | `ci.yml:79` | `--cov-fail-under=70` vs measured 64 % → fails even when tests pass | [V] | Floors from measured baseline (§6) |
| D27 | GitHub settings | Default branch is a `claude/*` branch; `main` unprotected; no rulesets; no tag protection | [V] | §5 branch protection |
| D28 | `.husky/pre-push` | UTF-16LE + CRLF, runs `npm test` which doesn't exist at root | [V] | Rewrite as ASCII or delete (CI covers it) |
| D29 | root `package.json:19-22` | lint-staged calls `ruff` without declaring where it comes from; `pre-commit` is in the dev extra (pyproject.toml:80) but no `.pre-commit-config.yaml` exists | [V] | `.pre-commit-config.yaml` (§5) |
| D30 | `pyproject.toml:84` | `ptpd` console script targets missing `ptpd_calibration.cli` | [V] | Remove or implement (owner: DEV) |

---

## 2. Consolidated workflow (`ci-proposed.yml`)

Full YAML: `scratchpad/ci-proposed.yml` (≈ 520 lines). Structure:

```
changes ─┬─ backend ───┐
         └─ frontend ──┴─ e2e ──┐
security ───────────────────────┤
dependency-review (PR) ─────────┼─ gate "all-green" ─ sbom ─ release (tags) ─ deploy-hf (tags | dispatch rollback)
codeql (py, js-ts) ─────────────┘
scorecard (schedule | push main)
```

Key properties

- **Triggers**: `push: main + v* tags`, `pull_request: main`, `merge_group`, weekly `schedule`, `workflow_dispatch(deploy_ref)`. Feature/claude branches are not built on push → halves today's job count immediately (D11).
- **`permissions: {}`** at workflow level; each job requests the minimum (`contents: read` almost everywhere; `security-events: write` only for CodeQL/Scorecard; `packages/attestations/id-token: write` only in `sbom`/`release`).
- **Concurrency** group per PR number / ref; `cancel-in-progress` only for PRs (never cancel a main/tag build).
- **SHA pins**: every `uses:` is `owner/action@<40-hex> # vX.Y.Z`. SHAs were resolved on 2026-09-18 with `git ls-remote --tags` (peeled commits) — they are real, not invented — but **TODO before merge: re-verify with `pinact run` / `frizbee`** and let Dependabot maintain them. Versions used: checkout v7.0.1, setup-uv v10.1.0, pnpm/action-setup v6.1.0, setup-node v7.0.0, upload/download-artifact v7.0.1/v8.0.1, dorny/paths-filter v4.0.3, gitleaks-action v3.0.0, pypa/gh-action-pip-audit v1.1.0, dependency-review-action v5.0.0, codeql-action v4.38.0, scorecard-action v2.4.4, anchore/sbom-action v0.24.2, attest-build-provenance v4.2.2, attest-sbom v4.1.0, docker/* (buildx v4.4.1, login v4.6.0, metadata v6.2.0, build-push v7.4.0), step-security/harden-runner v2.21.1. [U: input names for gitleaks-action v3 and setup-uv v10 assumed unchanged from v2/v6 — check on first run.]
- **Backend job**: `uv python install` (reads `.python-version`) → `uv lock --check` → `uv sync --frozen --group dev --extra api --extra ml --extra llm --extra plot --extra ui` → `ruff check` → `ruff format --check` → `mypy` (allowlist via `[tool.mypy] files`) → `pytest --strict-markers --strict-config -m "not selenium and not visual and not performance and not browser" --cov --cov-branch --cov-report=xml` → per-package `coverage report --include … --fail-under` floors → `diff-cover --fail-under=90` on PRs.
- **Frontend job**: pnpm before node (D01), `--frozen-lockfile`, `typecheck`, `lint`, `format:check`, `test:coverage` (vitest thresholds 80/75 already live in `frontend/vitest.config.ts:39-44` [V]), `build` with `VITE_API_URL=/`, uploads `dist/`.
- **E2E job**: starts `uvicorn ptpd_calibration.api.server:create_app --factory` on :8000 (`create_app()` is a zero-arg factory, server.py:15 [V]), waits on `/api/health` (server.py:198 [V]), then Playwright `--project=chromium` (the config's `webServer` runs `pnpm dev` on :3000 and proxies `/api` → 127.0.0.1:8000, `frontend/vite.config.ts:28-38` [V]). Five browser projects in `playwright.config.ts:25-46` are reduced to chromium in CI (phase 2: add webkit on schedule).
- **Security job**: gitleaks (full history), `ruff --select S` (advisory `--exit-zero` in phase 1), `uv export --frozen … --format requirements.txt` → `pip-audit --no-deps` on the lock, `pnpm audit --prod --audit-level=high`. `dependency-review` on PRs (`fail-on-severity: high`). CodeQL python + javascript-typescript. Scorecard weekly + on main with `publish_results: true`.
- **Gate job `all-green`**: the single required status check. Reads `toJSON(needs)`; phase-1 required = `changes backend frontend security` (skipped-by-path-filter counts as OK only because `changes` itself must succeed); e2e / dependency-review / codeql are printed as advisory.
- **sbom**: `uv build` + `uv export --format pylock.toml` + syft SPDX SBOM (covers `uv.lock` **and** `pnpm-lock.yaml`), attestations (`attest-build-provenance`, `attest-sbom`) on tags.
- **release** (tags only, `environment: release`): buildx image → `ghcr.io/ianshank/ptpd-calibration:{version,major.minor,sha}` with `provenance: mode=max`, `sbom: true`, registry attestation, then `gh release create --verify-tag` with wheel/sdist/pylock/SBOM.
- **deploy-hf**: tags after release, or `workflow_dispatch` with `deploy_ref` (rollback). Token via `http.<remote>.extraheader` (never in URL), `--force-with-lease` against the fetched Space `main`, hard-fails on missing secrets (D09), then polls `GET /api/spaces/{u}/{s}` for `runtime.stage == RUNNING` (advisory). [U: `runtime.stage` field name from memory of the HF API — verify.]

### Ratchet plan — blocking vs advisory

| Check | Phase 1 (merge week) | Phase 2 (+30–60 days) | Phase 3 (+90 days) |
|-------|----------------------|------------------------|--------------------|
| ruff check / format (src, tests, app.py, scripts) | **blocking** — requires OPS-05 to zero the 86/34 baseline first (or `--extend-exclude` the 5 worst files with an issue) | blocking | add `S`, `PTH`, `RUF` to `select` |
| mypy | **blocking on allowlist** (9 zero-error packages, §3) | add ≤10-error packages (batch, ai, session, analysis, mcts, zones, data, gcp, advanced) | curves, detection, llm, integrations, education; consider `ty` once stable [U] |
| pytest (unit+integration+api) | **blocking**; fix 5 collection errors first (psutil/gradio deps) | blocking | add weekly 3.11/3.13 matrix |
| Coverage floors | **blocking** at baseline−5 (mcts 70, chemistry 85, curves 80, whole 60) | mcts 80, chemistry 90, curves 85, whole 70 | 90/90/85/75 |
| diff-cover 90 % | **blocking** (optionally `continue-on-error` for the first two weeks) | blocking | blocking |
| Frontend tsc/eslint/prettier/vitest/build | **blocking** (fix 5 failing vitest tests, OPS-06); eslint keeps `--max-warnings 100` | `--max-warnings 0` | — |
| e2e Playwright | **advisory** (`continue-on-error: true` at job level) | blocking, in gate | + webkit weekly |
| gitleaks | **blocking** | blocking | — |
| ruff S | advisory (`--exit-zero`, 65 findings) | blocking with per-file-ignores | — |
| pip-audit / pnpm audit | advisory | blocking with dated `ignore-vulns` allowlist | — |
| dependency-review | runs, not in gate | in gate, `fail-on-severity: high` | add licence policy |
| CodeQL | runs, not in gate | in gate (security-and-quality) | — |
| harden-runner | `egress-policy: audit` | `block` with allowlist from audit logs | — |
| Scorecard | weekly, published | fix findings ≥ 7/10 target | badge |

---

## 3. uv migration design

### Proposed `pyproject.toml` layout

```toml
[project]
name = "ptpd-calibration"
version = "1.0.0"
requires-python = ">=3.11,<3.14"        # 3.10 EOL Oct-2026 [U: decision]; classifiers 3.13 already claimed (line 35)
dependencies = [
  "numpy>=1.26,<3",
  "scipy>=1.11",
  "pillow>=10.0",
  "pydantic>=2.5,<3",
  "pydantic-settings>=2.1,<3",
  "python-dotenv>=1.0",       # config.py:14 imports it directly
  "psutil>=5.9",              # monitoring/performance.py:21 (top-level); also agents/health.py:277, core/debug.py:375
  "pyyaml>=6.0",              # workflow/recipe_manager.py:20 (top-level); data/export_import.py:78,383
  "tifffile>=2024.1.0",       # imaging/processor.py:18 – 16-bit TIFF is core to digital negatives
]

[project.optional-dependencies]
plot     = ["matplotlib>=3.7"]                       # curves/visualization.py:12, imaging/histogram.py:382, ui/*
ml       = ["scikit-learn>=1.3"]
llm      = ["anthropic>=0.39", "openai>=1.0", "httpx>=0.27"]
api      = ["fastapi>=0.109", "uvicorn[standard]>=0.27", "python-multipart>=0.0.9"]
gcp      = ["google-cloud-storage>=2.14", "google-cloud-aiplatform>=1.38", "google-auth>=2.27"]
ui       = ["gradio>=4.44,<5", "huggingface_hub>=0.25,<1.0", "gradio_client", "typer>=0.12", "rich>=13",
            "ptpd-calibration[plot]"]                # legacy Gradio UI (ui/manage_rag.py:18 uses typer)
hardware = ["pyserial>=3.5", "zeroconf>=0.130",
            "pycups>=2.0; sys_platform == 'linux'", "pywin32>=306; sys_platform == 'win32'"]
qr       = ["qrcode[pil]>=7.4", "pyzbar>=0.1.9"]     # advanced/features.py:21,990
dl       = ["torch>=2.1", "torchvision>=0.16", "timm>=0.9", "ultralytics>=8", "segment-anything>=1.0",
            "diffusers>=0.35", "lpips>=0.1.4", "opencv-python-headless>=4.8", "scikit-image>=0.22",
            "controlnet-aux>=0.0.7", "open-clip-torch>=2.24"]    # see notes on `clip`
server   = ["ptpd-calibration[api,ml,llm,plot]"]     # exactly what the Docker Space installs
all      = ["ptpd-calibration[ml,llm,api,ui,gcp,plot,qr,hardware]"]   # deliberately excludes dl

[dependency-groups]                                   # PEP 735 – never published, always dev-only
dev  = ["pytest>=8", "pytest-asyncio>=0.23", "pytest-cov>=5", "pytest-timeout>=2.3", "pytest-benchmark>=4",
        "ruff==0.15.8", "mypy>=1.14", "pre-commit>=4", "diff-cover>=9", "pip-audit>=2.7",
        "types-PyYAML", "types-psutil", "types-requests"]
e2e  = ["selenium>=4.20"]                            # tests/e2e/selenium, run outside PR CI

[tool.uv]
required-version = ">=0.8,<0.9"
default-groups = ["dev"]
# Torch from the CPU index on Linux (CI + Docker) – avoids ~2 GB CUDA wheels; macOS/Windows use PyPI.
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
[tool.uv.sources]
torch       = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra --strict-markers --strict-config"
xfail_strict = true
markers = [
  "unit", "integration", "api", "e2e", "selenium", "browser", "visual", "performance",
  "slow", "deep: requires torch", "user_journey",
]
filterwarnings = ["error::pytest.PytestUnknownMarkWarning"]
```

Notes and drift found [V]:

- `requirements.txt:21,24` (matplotlib, tifffile) are missing from pyproject; conversely pyproject lacks python-dotenv/psutil/pyyaml which are hard imports. The Gradio Space currently works only because `requirements.txt` happens to be the file it installs.
- `requirements-dl.txt` lists 11 packages with **zero import sites** in `src/` (transformers, accelerate, safetensors, peft, einops, albumentations, tensorboard, wandb, tqdm, plus pytest-benchmark/pytest-timeout which are test deps) → drop them from the `dl` extra.
- `import clip` (deep_learning/image_quality.py:98,661) refers to OpenAI CLIP, which is not the PyPI package named `clip`; use `open-clip-torch` (API differs) or a git source `clip @ git+https://github.com/openai/CLIP` — **DEV decision**. `controlnet_aux` (diffusion_enhance.py:278) → `controlnet-aux`; `skimage` → `scikit-image`; `cv2` → `opencv-python-headless` for servers (no libGL).
- `boto3` (data/cloud_sync.py:257) and `vertexai` (vertex/*) are guarded; put `boto3` in a `s3` extra or drop.
- `.gitignore:86` ignores `.python-version` — delete that line.

### Lock, export, retirement of requirements files

```bash
echo 3.12 > .python-version
uv lock                                   # writes uv.lock (universal: all extras/groups/platforms) – COMMIT IT
uv sync --frozen                          # dev env = dev group + no extras; add --extra ... as needed
uv export --frozen --all-extras --no-dev --format pylock.toml -o pylock.toml   # PEP 751, CI artifact + SBOM input
# transitional, only while the Gradio Space still installs requirements.txt:
uv export --frozen --no-dev --no-emit-project --no-hashes \
  --extra ui --extra ml --extra llm --extra plot --extra api --format requirements.txt -o requirements.txt
```

- CI runs `uv lock --check` (fails when pyproject changed without relocking) and, during the transition, `git diff --exit-code requirements.txt` after regenerating it.
- `requirements-dl.txt` → deleted immediately (replaced by `--extra dl`). `requirements.txt` → **generated** until the Docker Space is live (OPS-14), then deleted (OPS-17). Docs/README install instructions change to `uv sync --extra server` / `uv pip install "ptpd-calibration[server]"`.
- Dependabot supports the `uv` ecosystem, so `uv.lock` bumps arrive as PRs (§5). [U: ecosystem key `"uv"` — verify against current Dependabot docs.]
- The HF Space Dockerfile installs `uv sync --frozen --no-dev --extra server` (see §4) — no torch, no gradio, no gcp.

### mypy allowlist (`[tool.mypy]`)

```toml
[tool.mypy]
python_version = "3.12"
files = [                              # PHASE-1 allowlist = packages with 0 errors today
  "src/ptpd_calibration/__init__.py", "src/ptpd_calibration/config.py",
  "src/ptpd_calibration/core", "src/ptpd_calibration/chemistry", "src/ptpd_calibration/calculations",
  "src/ptpd_calibration/exposure", "src/ptpd_calibration/proofing", "src/ptpd_calibration/papers",
  "src/ptpd_calibration/workflow", "src/ptpd_calibration/vertex",
]
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
ignore_missing_imports = true          # replaces the CLI flag; tighten per-module later
exclude = ["tests"]
```

---

## 4. Hugging Face Docker Space deploy

Facts from the HF docs (fetched via the HF connector, `hub/spaces-sdks-docker.md`) [V]:

- README YAML: `sdk: docker`; default exposed port is **7860**, override with `app_port:`.
- The container runs as **UID 1000** → `RUN useradd -m -u 1000 user`, `USER user`, `COPY --chown=user`.
- **Variables** are build-args at build time and env vars at runtime; **Secrets** are exposed at build time only via `RUN --mount=type=secret,id=NAME,mode=0444,required=true` (read from `/run/secrets/NAME`) and at runtime as env vars.
- Disk is ephemeral; `/data` persists only with the paid Storage add-on and only at runtime. GPU is not available at build time.
- [U] Build-time limits (file size >10 MB requires LFS; build timeout) — from memory, not in the fetched page.

### README.md YAML header changes

```yaml
---
title: Pt/Pd Calibration Studio
emoji: 📷
colorFrom: yellow
colorTo: gray
sdk: docker            # was: sdk: gradio
app_port: 7860         # replaces: sdk_version: "4.44.0" and app_file: app.py  (README.md:6-8)
pinned: false
license: mit
tags: [...unchanged...]
short_description: AI-powered calibration for platinum/palladium printing
---
```

Also `docs/huggingface-deployment.md:1-20` duplicates the same header and must change in lockstep (or be deleted; the Space's own README is `HUGGINGFACE_README.md` in the Space tree [V]).

### Dockerfile

See `Dockerfile.proposed`. Stages: `node:22-bookworm-slim` (pnpm via corepack, `pnpm build` with `VITE_API_URL=/`) → `ghcr.io/astral-sh/uv:0.8.17-python3.12-bookworm-slim` (`uv sync --frozen --no-dev --no-install-project --extra server`, then project) → `python:3.12-slim-bookworm` runtime copying `.venv`, `src`, `static`; UID 1000; `EXPOSE 7860`; `HEALTHCHECK` on `/api/health`; `CMD uvicorn ptpd_calibration.api.server:create_app --factory --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips=*`.

Required code changes (owner: DEV, not made here):

1. `src/ptpd_calibration/api/server.py` — mount the SPA: add `static_dir: Path | None` to `APISettings` (config.py:355-364, env `PTPD_API_STATIC_DIR`), then after all `/api` routes `app.mount("/assets", StaticFiles(directory=static/assets))` and a catch-all `GET /{path:path}` → `FileResponse(static/index.html)` so react-router routes (`/calibration`, `/curves`, …) deep-link. Today the API has no static serving at all [V — only `FileResponse` for exports, server.py:353,373].
2. Frontend base URL: `getEnv()` (frontend/src/config/index.ts:53-57) treats `""` as unset and falls back to `http://127.0.0.1:8000` in production builds (`client.ts:48`). Building with `VITE_API_URL=/` gives same-origin requests with no code change (axios joins `/` + `/api/x`). Longer term make the production default `""`.
3. CORS: `cors_origins=["*"]` with `allow_credentials=True` (config.py:367-368) is rejected by browsers; same-origin serving makes it moot but the default should be tightened.
4. Secrets on the Space: `PTPD_ANTHROPIC_API_KEY`, `PTPD_OPENAI_API_KEY` (CLAUDE.md env section) as HF **Secrets**; `PTPD_LOG_LEVEL`, `PTPD_LLM_PROVIDER` as **Variables**. `PTPD_DATA_DIR=/data` only persists with Storage.

### Two ways to build the Space (recommend B)

- **A. Build from source in the Space**: push the git tree (Dockerfile at root) → HF builds. Simple, but builds on HF's builder, and the image that ran CI is not the image that serves.
- **B. Thin Dockerfile in the Space**: CI builds/pushes `ghcr.io/ianshank/ptpd-calibration:<ver>` (attested) and the Space Dockerfile is `FROM ghcr.io/ianshank/ptpd-calibration@sha256:<digest>` (+ `USER user`, `EXPOSE 7860`). The exact tested/attested image serves; **rollback = change one digest line** (re-run `deploy-hf` with `deploy_ref=<previous tag>`). Requires the GHCR package to be **public** (MIT project — fine). [U: HF Docker builder pulling from public GHCR — expected to work like any public registry; verify with a first push.]

### Deploy policy and rollback

- Deploy **only** from `v*` tags after `all-green` + `release` (or `workflow_dispatch(deploy_ref)`), inside `environment: huggingface` with a required reviewer → manual approval before production changes.
- `main` on the Space is written only by CI with `--force-with-lease` against the fetched remote head → concurrent/stale pushes fail instead of clobbering.
- Rollback runbook: (1) `gh workflow run ci.yml -f deploy_ref=v1.2.2` (previous good tag) → pushes that tree/digest; (2) if only runtime is wedged: `huggingface_hub.HfApi().restart_space(repo_id, factory_reboot=True)`; (3) emergency: `git push --force <remote> v1.2.2:main` with the same header auth from a maintainer machine. Tags are protected by the ruleset in §5 so a rollback target can never disappear.
- Retire `app.py` + `.gradio/` + the Gradio sanity gate (`ci-cd.yml:140-149`) once cutover is verified; `tests/sanity/test_deployment_sanity.py:200-254` (imports gradio) becomes `@pytest.mark.legacy_ui` / skipped when gradio is absent.

---

## 5. Repo hygiene runbook

### Untrack ignored artifacts (no history rewrite)

```bash
git rm -r --cached node_modules .gradio hf_check           # 268 + 1 + 1 files [V]
printf 'npx lint-staged\n' > .husky/pre-commit              # already ASCII; keep
git rm .husky/pre-push                                      # UTF-16 "npm test" – CI owns tests (or rewrite as ASCII: printf 'cd frontend && pnpm typecheck\n')
# .gitignore edits: delete line 86 (.python-version); add: .gradio/  .coverage  reports/  *.sarif  pylock.toml?(no – commit it)
git add .gitignore .husky
git commit -m "chore(repo): untrack node_modules/.gradio/hf_check, fix pre-push hook encoding, un-ignore .python-version"
```

`.gradio/certificate.pem` is a public certificate, not a key (`BEGIN CERTIFICATE`, no `PRIVATE`) [V]; no secret-pattern hits in tracked files (`git grep` for sk-/AKIA/hf_/ghp_/PRIVATE KEY/xox) [V]. History rewriting is unnecessary.

### `.pre-commit-config.yaml` (replaces lint-staged for Python; keep lint-staged for TS or move fully)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0            # TODO: pin to current tag
    hooks:
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: mixed-line-ending
        args: ["--fix=lf"]
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: detect-private-key
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.8
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
  - repo: https://github.com/astral-sh/uv-pre-commit
    rev: 0.8.17
    hooks:
      - id: uv-lock          # keeps uv.lock in sync with pyproject.toml
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.24.0           # TODO: pin
    hooks:
      - id: gitleaks
  - repo: https://github.com/rhysd/actionlint
    rev: v1.7.7
    hooks:
      - id: actionlint
  - repo: https://github.com/woodruffw/zizmor-pre-commit
    rev: v1.5.2
    hooks:
      - id: zizmor           # workflow security linter (template injection, unpinned actions, credentials persistence)
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.3.3
    hooks:
      - id: prettier
        files: ^frontend/.*\.(ts|tsx|css|json|md)$
        additional_dependencies: ["prettier@3.3.3", "prettier-plugin-tailwindcss@0.6.x"]
```

Install: `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`; add `pre-commit run --all-files` as a **phase-2** CI step. Root `package.json`/`package-lock.json` (npm) coexisting with `frontend/pnpm-lock.yaml` (pnpm) is a second package manager for four dev tools — after pre-commit lands, delete root `package.json`, `package-lock.json`, `.husky/`.

### CODEOWNERS (`.github/CODEOWNERS`)

```
*                                   @ianshank
/.github/                           @ianshank
/pyproject.toml /uv.lock /Dockerfile @ianshank
/src/ptpd_calibration/chemistry/    @ianshank
/src/ptpd_calibration/curves/       @ianshank
/src/ptpd_calibration/mcts/         @ianshank
/frontend/                          @ianshank
```

Single-maintainer today; the value is that rulesets can require code-owner review for workflow/dependency/Dockerfile changes.

### SECURITY.md (outline)

Supported versions (latest `v*` tag + `main`); report via GitHub **private vulnerability reporting** (enable in Settings → Security); acknowledgement ≤ 3 business days, fix target 30 days for high; scope: API keys are only read from `PTPD_*` env vars/HF Secrets, never committed; dependency policy: pip-audit/pnpm audit/dependency-review gates; disclosure credit.

### CONTRIBUTING.md (outline)

`uv sync --extra api --extra ml --extra llm --extra plot`, `cd frontend && pnpm install`; verification loop from CLAUDE.md; `pre-commit install`; branch naming; Conventional Commits; PR must pass `all-green`; how to run e2e locally (`uvicorn … --factory` + `pnpm exec playwright test`); how to cut a release (`git tag v1.2.3 && git push --tags` → release + deploy with approval).

### Branch protection / rulesets (via `gh api` or UI)

1. Set default branch to `main`: `gh api -X PATCH repos/ianshank/Platinum-Palladium-AI-Printing-Tool -f default_branch=main`.
2. Ruleset "main" (`gh api -X POST repos/…/rulesets`): target `refs/heads/main`; rules: `pull_request` (1 approval, dismiss stale, require code-owner review, require conversation resolution), `required_status_checks` with `all-green` (strict / up-to-date), `required_linear_history`, `non_fast_forward` (no force-push), `deletion`, optionally `required_signatures`. Bypass: none (admins included).
3. Ruleset "tags": target `refs/tags/v*`; rules: `creation` restricted to maintainers, `update`, `deletion` blocked.
4. Settings → Actions: workflow permissions **Read repository contents**; "Allow GitHub Actions to create and approve pull requests" off; environments `release` and `huggingface` with required reviewer = @ianshank; secrets `HF_TOKEN` (write-scoped to the Space) and variables `HF_USERNAME`, `HF_SPACE_NAME` moved to the `huggingface` environment.

### Dependabot (`.github/dependabot.yml`)

```yaml
version: 2
updates:
  - package-ecosystem: github-actions
    directory: /
    schedule: { interval: weekly, day: monday }
    groups: { actions: { patterns: ["*"] } }
  - package-ecosystem: uv                 # [U] verify ecosystem key; fallback: "pip"
    directory: /
    schedule: { interval: weekly }
    groups:
      python-minor: { update-types: ["minor", "patch"] }
    ignore:
      - dependency-name: "torch*"         # bump manually (CPU index)
  - package-ecosystem: npm
    directory: /frontend
    schedule: { interval: weekly }
    groups:
      frontend-minor: { update-types: ["minor", "patch"] }
  - package-ecosystem: docker
    directory: /
    schedule: { interval: weekly }
```

Renovate alternative (`renovate.json`): `{"extends": ["config:best-practices", "helpers:pinGitHubActionDigests", ":pinDevDependencies"], "lockFileMaintenance": {"enabled": true, "schedule": ["before 6am on monday"]}}` — best-practices already pins action digests and enables `uv` lock support. Pick one; Renovate if digest-pinning automation matters more than zero setup.

---

## 6. Coverage gating

### Baselines (full unit suite, 2026-09-18, torch absent, `ui/` and `api/server.py` omitted per pyproject:158-164) [V]

| Package | Line | Branch-inclusive ("combined") | Weakest files |
|---------|------|-------------------------------|---------------|
| chemistry | 94.8 % | **92.6 %** | silver_gelatin_calculator.py 87 % |
| curves | 89.9 % | **85.6 %** | parser.py 74 %, ai_enhance.py 80 % |
| mcts | 78.2 % (line, from per-file table; XML truncated before `mcts`) | ≈ 75 % [U] | networks.py 0 % (needs torch), training.py 29 %, simulator.py 76 % |
| core | 92.7 % | 91.0 % | — |
| api | 40.3 % | 39.3 % | (unit run only; `tests/api` adds more) |
| whole (`coverage report` TOTAL) | — | **64 %** | — |

### Config

```toml
[tool.coverage.run]
branch = true
source = ["src/ptpd_calibration"]
relative_files = true                  # required for diff-cover/codecov path matching
omit = ["*/ui/*", "*/api/server.py"]   # keep until the Gradio UI is deleted; api/server.py should come OFF the omit list once e2e is blocking

[tool.coverage.report]
fail_under = 60                        # PHASE-1 global floor (ci.yml asks 70 today and fails); PHASE-2 = 70
skip_covered = true
show_missing = true
exclude_also = ["if TYPE_CHECKING:", "raise NotImplementedError", "if __name__ == .__main__.:", "@overload"]

[tool.coverage.xml]
output = "coverage.xml"

[tool.diff_cover]
compare_branch = "origin/main"
fail_under = 90
include_untracked = true
```

Per-package floors cannot be expressed in `[tool.coverage.report]` (it has one global `fail_under`), so the backend job runs four `coverage report --include=<pkg> --fail-under=N` steps over the same `.coverage` data (see `ci-proposed.yml`, "Coverage floor" steps): **mcts 70, chemistry 85, curves 80, whole 60** in phase 1 → **80 / 90 / 85 / 70** in phase 2 → **90 / 90 / 85 / 75** target. Raising mcts past ~80 needs either a torch-enabled job for `networks.py` (140 statements, 0 %) or `# pragma: no cover` on the torch-only class bodies plus real tests for `training.py`.

Diff coverage: `uv run diff-cover coverage.xml --compare-branch=origin/${{ github.base_ref }} --fail-under=90 --markdown-report reports/diff-cover.md` on PRs (checkout with `fetch-depth: 0`). No SaaS dependency. If Codecov is preferred instead (needs `CODECOV_TOKEN`), equivalent `codecov.yml`:

```yaml
coverage:
  status:
    project: { default: { target: auto, threshold: 0.5% } }
    patch:   { default: { target: 90%, threshold: 0% } }
component_management:
  individual_components:
    - component_id: curves    ; paths: ["src/ptpd_calibration/curves/**"]    ; statuses: [{type: project, target: 80%}]
    - component_id: chemistry ; paths: ["src/ptpd_calibration/chemistry/**"] ; statuses: [{type: project, target: 85%}]
    - component_id: mcts      ; paths: ["src/ptpd_calibration/mcts/**"]      ; statuses: [{type: project, target: 70%}]
```

Frontend: `frontend/vitest.config.ts:39-44` already enforces statements 80 / branches 75 (+functions/lines) [V]; the frontend job's `pnpm test:coverage` therefore already gates. Add `--changed` diff-gating in phase 2 if needed.

---

## 7. CI cost — today vs proposed

Measured from the last PR #36 iteration (commit 2026-08-03 23:17 UTC, four runs) [V]:

| Workflow | Event | Jobs | Executed | Runner-seconds (failing fast) |
|----------|-------|------|----------|-------------------------------|
| CI (ci.yml) | pull_request | 2 | 2 | 65 |
| Tests (tests.yml) | pull_request | 17 | 11 (6 skipped after unit failure) | 815 (incl. 3 Windows ×2 and 3 macOS ×10 billing multipliers) |
| CI/CD (ci-cd.yml) | pull_request | 7 | 5 | 393 |
| CI/CD (ci-cd.yml) | push (same SHA, `claude/*`) | 7 | 7 | 411 |
| **Total per PR push** | | **33 job slots** | 25 | **1 684 s ≈ 28 runner-min while broken** |

Healthy-state estimate for the same three workflows (all jobs reaching their tests): ≈ 110–120 runner-minutes per PR push; billed-equivalent ≈ 250 min because of the macOS/Windows matrix (free on a public repo, but the 5-concurrent-macOS limit and queueing still slow the wall clock to ~20 min). [U: estimates]

Proposed (`ci-proposed.yml`):

| Event | Jobs | Est. runner-min | Wall-clock |
|-------|------|-----------------|-----------|
| pull_request | 9 (changes, backend, frontend, e2e, security, dependency-review, codeql×2, gate) — 7 without CodeQL | ≈ 28–32 (backend 6, frontend 4, e2e 6, security 3, codeql 2×5, rest <1) | ≈ 12 min (e2e serial after backend+frontend) |
| push main | 11 (+ sbom, scorecard) | ≈ 36 | ≈ 15 min |
| tag `v*` | 13 (+ release, deploy-hf) | ≈ 50 | ≈ 25 min incl. approval |
| weekly schedule | 2 (security, scorecard) | ≈ 5 | — |

Net: **33 → 9 job slots per PR (−73 %)**, no duplicate runs, no macOS/Windows minutes, and one required check instead of zero usable ones. Path filtering (`changes` job) skips backend or frontend on docs-only / single-side PRs.

---

## Plan items

| ID | Title | Effort | Depends on | Acceptance criteria |
|----|-------|--------|------------|---------------------|
| OPS-01 | Fix GitHub default branch → `main`; create `main` and `v*` rulesets; set workflow permissions to read-only; create `release`/`huggingface` environments with required reviewer | S | — | `GET /repos/...` shows `default_branch: main`; `GET /branches/main` `protected: true`; a direct push to main is rejected; env approvals visible in Settings |
| OPS-02 | Untrack `node_modules/`, `.gradio/`, `hf_check/`; fix/remove UTF-16 `.husky/pre-push`; un-ignore `.python-version`; add `.gradio/`, `.coverage`, `reports/` to `.gitignore` | S | — | `git ls-files -i -c --exclude-standard` is empty; `file .husky/*` reports ASCII or files deleted |
| OPS-03 | uv migration: restructure `[project]`/extras/`[dependency-groups]`, add python-dotenv/psutil/pyyaml/tifffile to deps, torch CPU index, `.python-version=3.12`, commit `uv.lock`; delete `requirements-dl.txt`; generate `requirements.txt` transitionally | M | OPS-02 | `uv lock --check` passes; `uv sync --frozen --extra server` installs without torch/gradio; `python -c "import ptpd_calibration.monitoring, ptpd_calibration.workflow"` works in a fresh env; `uv export --format pylock.toml` succeeds |
| OPS-04 | Register pytest markers, `--strict-markers --strict-config`, `xfail_strict`; fix the 5 collection errors (psutil dep; gradio-dependent `tests/unit/ui/*` and `tests/sanity` gated by `importorskip`/marker) | S | OPS-03 | `pytest --collect-only -q` → 0 errors; `-W error::PytestUnknownMarkWarning` clean |
| OPS-05 | Lint/format baseline to zero: 86 ruff errors (incl. F821 real bug `session/logger.py:470`), 34 unformatted files; scope = `src tests app.py scripts` | M | — | `ruff check src tests app.py scripts` and `ruff format --check …` exit 0 |
| OPS-06 | Frontend CI prerequisites: `"packageManager": "pnpm@10.x"` in `frontend/package.json`, `frontend/.nvmrc`, fix 5 failing vitest tests, build with `VITE_API_URL=/` | S/M | — | `pnpm install --frozen-lockfile && pnpm check:all && pnpm test:coverage && pnpm build` exit 0 locally and in CI |
| OPS-07 | mypy allowlist (`[tool.mypy] files=` with the 9 zero-error packages) and blocking CI step | S | OPS-03 | `uv run mypy` exit 0 on the allowlist; list documented in CONTRIBUTING with the phase-2 expansion order |
| OPS-08 | Replace `ci.yml`, `ci-cd.yml`, `tests.yml` with the consolidated workflow (phase-1 settings) | M | OPS-03..07 | Exactly one workflow file; `all-green` passes on a no-op PR; `actionlint` + `zizmor` clean; per-PR jobs ≤ 9 |
| OPS-09 | Supply-chain pinning: verify all SHAs (`pinact`), add `.github/dependabot.yml` (actions, uv, npm, docker) or `renovate.json` | S | OPS-08 | `zizmor` reports no `unpinned-uses`; first Dependabot PRs open within a week |
| OPS-10 | Security jobs live: gitleaks (history), ruff S (advisory), pip-audit on exported lock, pnpm audit, dependency-review on PRs, CodeQL, weekly Scorecard with published results | S | OPS-08 | Security tab shows CodeQL + Scorecard results; Scorecard badge ≥ 5 initially; gitleaks passes on full history |
| OPS-11 | Coverage gating: `[tool.coverage.*]` config, per-package floors (70/85/80/60), diff-cover 90 % on PRs | S | OPS-04, OPS-08 | A PR that adds an untested function fails `diff-cover`; floors pass on `main` |
| OPS-12 | SBOM + attestations: `uv build`, pylock export, syft SPDX, `attest-build-provenance` + `attest-sbom` on tags | S | OPS-08 | `gh attestation verify dist/*.whl -R ianshank/…` succeeds for a test tag |
| OPS-13 | Serve the SPA from FastAPI (`PTPD_API_STATIC_DIR`, `/assets` mount, SPA fallback) + `Dockerfile` + `.dockerignore`; local smoke test | M | OPS-03, OPS-06 | `docker build . && docker run -p 7860:7860` → `/` returns the React app, `/api/health` 200, deep link `/curves` 200; image < 1.5 GB [U target] |
| OPS-14 | HF Docker Space cutover: README header (`sdk: docker`, `app_port: 7860`), Space secrets/variables, `deploy-hf` from tags with approval, thin `FROM ghcr.io/...@sha256` Dockerfile in the Space, rollback drill | M | OPS-13, OPS-08, OPS-01 | Space `runtime.stage == RUNNING` after a tag deploy; rollback via `workflow_dispatch(deploy_ref)` restores the previous tag in < 10 min; no push from non-tag refs possible |
| OPS-15 | Governance docs: `CODEOWNERS`, `SECURITY.md`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`; remove root npm/husky/lint-staged after pre-commit works | S | — | `pre-commit run --all-files` clean; private vulnerability reporting enabled; PR template references `all-green` |
| OPS-16 | Phase-2 ratchet: e2e/dependency-review/CodeQL into the gate; pip-audit, pnpm audit, ruff S blocking; eslint `--max-warnings 0`; harden-runner `block`; floors 80/90/85/70; mypy +9 packages | M | OPS-08..11, +30–60 days | Gate required set updated; all advisory `continue-on-error`/`--exit-zero` markers removed from the workflow |
| OPS-17 | Retire legacy deploy artefacts: delete `requirements.txt`, `app.py`, `.gradio/`, Gradio sanity import gate; move `tests/sanity` gradio tests behind `legacy_ui` marker | S | OPS-14 | No `requirements*.txt` in repo; `uv sync --extra server` is the only documented install path |
| OPS-18 | Fix broken console script `ptpd = ptpd_calibration.cli:main` (module missing) — implement or drop | S | — | `uv run ptpd --help` works or the entry point is removed and docs updated |

Owner boundaries: OPS-05, OPS-13 (server.py static serving), OPS-18 change files under `src/` and belong to DEV-SQE; everything else is repo/CI configuration.
