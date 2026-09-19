"""The request bodies the React client sends must be the ones the API accepts.

Four mismatches reached the default branch because each side was tested against
its own idea of the contract: the frontend mocked its client and the backend
asserted its own models, and nothing compared them. Every call the user's first
calibration depends on failed.

These tests read the payload shapes from the generated OpenAPI schema, which is
the same file the TypeScript client is generated from, so a route change that
the client does not follow fails here as well as in the frontend build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ptpd_calibration.api.server import create_app  # noqa: E402

pytestmark = pytest.mark.api

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "ptpd_calibration" / "api" / "openapi.json"
)
DENSITIES = [0.08, 0.22, 0.41, 0.63, 0.88, 1.10, 1.28, 1.41, 1.48, 1.52]


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    if not SCHEMA_PATH.is_file():
        pytest.fail(
            f"{SCHEMA_PATH} is missing. Regenerate it with "
            "`python scripts/export_openapi.py`; the frontend's type "
            "generation reads the same file."
        )
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


class TestCommittedSchemaMatchesTheApp:
    def test_schema_is_current(self, schema: dict[str, Any]) -> None:
        """The committed schema is what the frontend generates its types from."""
        from scripts.export_openapi import build_schema

        assert schema == json.loads(json.dumps(build_schema(), sort_keys=True))

    def test_curve_request_names_densities_not_measurements(self, schema: dict[str, Any]) -> None:
        properties = schema["components"]["schemas"]["CurveRequest"]["properties"]

        assert "densities" in properties
        assert "measurements" not in properties
        assert "curve_type" in properties
        assert "type" not in properties


class TestGenerateEndpointContract:
    def test_the_payload_the_client_sends_is_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/api/curves/generate",
            json={"densities": DENSITIES, "curve_type": "linear", "name": "Contract"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["success"] is True

    def test_a_generated_curve_can_be_fetched_and_exported(self, client: TestClient) -> None:
        """The identifier the endpoint returns must refer to a stored curve."""
        curve_id = client.post(
            "/api/curves/generate",
            json={"densities": DENSITIES, "curve_type": "linear", "name": "Roundtrip"},
        ).json()["curve_id"]

        assert client.get(f"/api/curves/{curve_id}").status_code == 200
        assert client.post(f"/api/curves/{curve_id}/export?format=json").status_code == 200

    def test_the_full_curve_is_returned_not_a_sample(self, client: TestClient) -> None:
        """The UI plots what this returns, so a truncated response is a wrong curve."""
        body = client.post(
            "/api/curves/generate",
            json={"densities": DENSITIES, "curve_type": "linear", "name": "Full"},
        ).json()

        assert len(body["output_values"]) == body["num_points"]
        assert len(body["input_values"]) == body["num_points"]

    @pytest.mark.parametrize("curve_type", ["monotonic", "cubic", "linearization"])
    def test_curve_types_the_ui_used_to_offer_are_refused_clearly(
        self, client: TestClient, curve_type: str
    ) -> None:
        """These were offered by the UI and are not members of the server enum."""
        response = client.post(
            "/api/curves/generate",
            json={"densities": DENSITIES, "curve_type": curve_type, "name": "Bad"},
        )

        assert response.status_code in (400, 422)

    def test_a_reversed_wedge_is_accepted_and_oriented(self, client: TestClient) -> None:
        """A wedge read from the wrong end is a correct measurement, backwards."""
        forward = client.post(
            "/api/curves/generate",
            json={"densities": DENSITIES, "curve_type": "linear", "name": "Forward"},
        ).json()
        reversed_body = client.post(
            "/api/curves/generate",
            json={
                "densities": list(reversed(DENSITIES)),
                "curve_type": "linear",
                "name": "Reversed",
            },
        ).json()

        assert reversed_body["output_values"] == pytest.approx(forward["output_values"])

    def test_a_wedge_that_rises_and_falls_is_refused(self, client: TestClient) -> None:
        """No single inverse exists, so the API must say so rather than guess."""
        hump = [0.1, 0.4, 0.8, 1.2, 1.5, 1.2, 0.8, 0.4, 0.2, 0.1]

        response = client.post(
            "/api/curves/generate",
            json={"densities": hump, "curve_type": "linear", "name": "Hump"},
        )

        assert response.status_code == 422
        assert "monotonic" in response.json()["detail"]


class TestParseQuadContract:
    def test_parse_quad_takes_form_fields(self, client: TestClient) -> None:
        """The client sent JSON; the endpoint declares Form fields."""
        # A flat channel parses but is inactive, which the endpoint rejects, so
        # use a real ramp.
        ramp = "\n".join(str(round(i * 65535 / 255)) for i in range(256))
        content = f"## QuadToneRIP K\n# K Curve\n{ramp}"

        as_form = client.post(
            "/api/curves/parse-quad",
            data={"content": content, "name": "Form", "channel": "K"},
        )
        as_json = client.post(
            "/api/curves/parse-quad",
            json={"content": content, "name": "Json", "channel": "K"},
        )

        assert as_form.status_code == 200, as_form.text
        assert as_json.status_code == 422
