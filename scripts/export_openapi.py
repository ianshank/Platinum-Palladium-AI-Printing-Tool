#!/usr/bin/env python3
"""Write the FastAPI application's OpenAPI schema to a file.

The frontend's ``generate:types`` script consumes this file. It pointed at a
path that had never existed, so the generated client was never produced and
``client.ts`` was written by hand. Four request-field mismatches reached the
default branch that way, each of which made a user-facing call fail.

Used two ways:

* ``python scripts/export_openapi.py <path>`` writes the schema;
* ``python scripts/export_openapi.py <path> --check`` exits non-zero when the
  file on disk differs from the application, which is what CI runs so the
  committed schema cannot drift from the routes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "src" / "ptpd_calibration" / "api" / "openapi.json"
# Two spaces and a trailing newline keep the file diffable and pre-commit clean.
INDENT = 2


def build_schema() -> dict:
    """Return the OpenAPI schema of the application CI and the frontend target."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ptpd_calibration.api.server import create_app

    return create_app().openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, indent=INDENT, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the file on disk differs from the application's schema",
    )
    args = parser.parse_args(argv)

    rendered = render(build_schema())

    if args.check:
        if not args.output.is_file():
            print(f"{args.output} is missing; run: python {Path(__file__).name} {args.output}")
            return 1
        if args.output.read_text(encoding="utf-8") != rendered:
            print(
                f"{args.output} is out of date with the API routes.\n"
                f"Regenerate it with: python {Path(__file__).name} {args.output}"
            )
            return 1
        print(f"{args.output}: up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
