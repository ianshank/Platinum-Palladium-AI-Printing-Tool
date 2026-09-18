# Security policy

## Supported versions

The latest `v*` tag and the `main` branch.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository (Security tab,
"Report a vulnerability"). Do not open a public issue for a security problem.
Reports are acknowledged within three business days; fixes for high-severity
findings target thirty days.

## Scope and controls

- API keys are read only from `PTPD_LLM_*` environment variables (or Hugging
  Face Space secrets). They are never committed and never echoed by the API,
  and neither the React client nor the API accepts one from a browser. The
  frozen legacy Gradio UI (`src/ptpd_calibration/ui/`, retirement tracked by
  ADR-0004) is the exception: it still offers a key field that sets a
  process-local runtime key, so run it only where every operator is trusted.
- File uploads are size-capped (`PTPD_API_MAX_UPLOAD_SIZE_MB`,
  `PTPD_API_MAX_REQUEST_BODY_MB`), extension-allowlisted, stored under
  server-generated names, and never written to client-supplied paths.
- Request payload lists and strings are bounded (`PTPD_API_MAX_LIST_LENGTH`).
- CORS never combines `*` with credentials.
- Model artifacts are loaded with `weights_only=True` or safetensors, with one
  documented exception: the sklearn predictor unpickles, and only after the file
  resolves inside `PTPD_ARTIFACTS_ALLOWED_DIRS` and its SHA-256 matches the
  recorded manifest. A file failing either check is never unpickled.
- Dependencies are locked (`uv.lock`, `pnpm-lock.yaml`) and audited in CI
  (`pip-audit`, `pnpm audit`, dependency review, CodeQL, gitleaks, OpenSSF
  Scorecard). Releases carry an SPDX SBOM and build provenance attestations.

## Disclosure

Fixes are released before details are published; reporters are credited in the
release notes unless they ask otherwise.
