"""Quick sanity check for the generate_curve endpoint fixes."""
from ptpd_calibration.api.server import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)

DENSITIES = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7]

tests = [
    ("measurements key (wizard)", {"measurements": DENSITIES, "name": "Wizard", "curve_type": "linear"}, 200),
    ("monotonic curve_type (unknown enum)", {"measurements": DENSITIES, "name": "Mono", "curve_type": "monotonic"}, 200),
    ("densities key (legacy)", {"densities": DENSITIES, "name": "Legacy", "curve_type": "linear"}, 200),
    ("empty body yields 422", {}, 422),
]

all_passed = True
for name, payload, expected_status in tests:
    r = client.post("/api/curves/generate", json=payload)
    status_ok = r.status_code == expected_status
    extra = ""
    if r.status_code == 200:
        d = r.json()
        arrays_match = len(d.get("input_values", [])) == d.get("num_points", -1)
        extra = "num_points={} arrays_match={}".format(d.get("num_points"), arrays_match)
        if not arrays_match:
            status_ok = False
    else:
        detail = r.json().get("detail", "")
        extra = str(detail)[:80] if isinstance(detail, str) else str(detail)[:80]
    symbol = "PASS" if status_ok else "FAIL"
    print("{} [{}] {}: {}".format(symbol, r.status_code, name, extra))
    if not status_ok:
        all_passed = False

print()
print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
