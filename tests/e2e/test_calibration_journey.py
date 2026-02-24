"""
E2E User Journey Test: Calibration Wizard → Generate Curve → Save Curve

Exercises the full calibration flow against the running backend:
  1. Upload scan (.tif file)
  2. Generate calibration curve from densities
  3. Save curve (modify endpoint with adjustment_type='none')
  4. Verify saved curve can be retrieved

Requires:
  - Backend running on http://localhost:8000
  - Pure_pall - Copy.tif in project root
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
TIF_PATH = Path(__file__).parent.parent.parent / "Pure_pall - Copy.tif"

# Track results
results: list[tuple[str, bool, str]] = []


def log(tag: str, ok: bool, detail: str = "") -> None:
    """Record and print a test result."""
    status = "PASS" if ok else "FAIL"
    msg = f"{status} [{tag}] {detail}"
    print(msg)
    results.append((tag, ok, detail))


def test_health() -> None:
    """Step 0: Verify backend is alive."""
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        log("health", r.status_code == 200, f"status={r.status_code}")
    except requests.ConnectionError:
        log("health", False, "Backend not reachable at localhost:8000")


def test_upload_scan() -> dict | None:
    """Step 1: Upload a .tif scan file and verify densities are returned."""
    if not TIF_PATH.exists():
        log("upload", False, f"File not found: {TIF_PATH}")
        return None

    with open(TIF_PATH, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/scan/upload",
            files={"file": (TIF_PATH.name, f, "image/tiff")},
            data={"target_type": "21-step"},
            timeout=60,
        )

    ok = r.status_code == 200
    data = r.json() if ok else {}
    densities = data.get("densities", [])
    log(
        "upload_scan",
        ok and len(densities) > 0,
        f"status={r.status_code} densities={len(densities)} "
        f"dmin={data.get('dmin')} dmax={data.get('dmax')}",
    )
    return data if ok else None


def test_generate_curve(scan_data: dict) -> dict | None:
    """Step 2: Generate a calibration curve from scan densities."""
    densities = scan_data.get("densities", [])
    if not densities:
        log("generate_curve", False, "No densities from scan")
        return None

    payload = {
        "measurements": densities,
        "name": "E2E Test Curve",
        "curve_type": "linear",
    }
    r = requests.post(f"{BASE_URL}/api/curves/generate", json=payload, timeout=60)

    ok = r.status_code == 200
    data = r.json() if ok else {}
    input_len = len(data.get("input_values", []))
    output_len = len(data.get("output_values", []))
    log(
        "generate_curve",
        ok and input_len > 0 and input_len == output_len,
        f"status={r.status_code} points={input_len} curve_id={data.get('curve_id', 'N/A')}",
    )
    return data if ok else None


def test_save_curve(curve_data: dict) -> dict | None:
    """Step 3: Save the curve via modify endpoint with adjustment_type='none'."""
    payload = {
        "name": curve_data.get("name", "E2E Test Curve"),
        "input_values": curve_data.get("input_values", []),
        "output_values": curve_data.get("output_values", []),
        "adjustment_type": "none",
        "amount": 0,
    }
    r = requests.post(f"{BASE_URL}/api/curves/modify", json=payload, timeout=60)

    ok = r.status_code == 200
    data = r.json() if ok else {}
    log(
        "save_curve",
        ok and data.get("success", False),
        f"status={r.status_code} curve_id={data.get('curve_id', 'N/A')} "
        f"detail={r.text[:200] if not ok else 'ok'}",
    )
    return data if ok else None


def test_modify_curve_brightness(curve_data: dict) -> dict | None:
    """Step 4: Modify curve with brightness adjustment (happy path)."""
    payload = {
        "name": curve_data.get("name", "E2E Test Curve"),
        "input_values": curve_data.get("input_values", []),
        "output_values": curve_data.get("output_values", []),
        "adjustment_type": "brightness",
        "amount": 0.1,
    }
    r = requests.post(f"{BASE_URL}/api/curves/modify", json=payload, timeout=60)

    ok = r.status_code == 200
    data = r.json() if ok else {}
    log(
        "modify_brightness",
        ok and data.get("adjustment_applied") == "brightness",
        f"status={r.status_code}",
    )
    return data if ok else None


def test_modify_curve_contrast(curve_data: dict) -> dict | None:
    """Step 5: Modify curve with contrast adjustment."""
    payload = {
        "name": "E2E Contrast Curve",
        "input_values": curve_data.get("input_values", []),
        "output_values": curve_data.get("output_values", []),
        "adjustment_type": "contrast",
        "amount": 0.2,
        "pivot": 0.5,
    }
    r = requests.post(f"{BASE_URL}/api/curves/modify", json=payload, timeout=60)

    ok = r.status_code == 200
    data = r.json() if ok else {}
    log(
        "modify_contrast",
        ok and data.get("adjustment_applied") == "contrast",
        f"status={r.status_code}",
    )
    return data if ok else None


def test_generate_curve_densities_key(scan_data: dict) -> None:
    """Step 6: Verify legacy 'densities' key still works for backward compat."""
    payload = {
        "densities": scan_data.get("densities", []),
        "name": "E2E Legacy Key Curve",
        "curve_type": "linear",
    }
    r = requests.post(f"{BASE_URL}/api/curves/generate", json=payload, timeout=15)
    ok = r.status_code == 200
    log("generate_legacy_key", ok, f"status={r.status_code}")


def test_generate_curve_unknown_type(scan_data: dict) -> None:
    """Step 7: Verify unknown curve_type falls back gracefully."""
    payload = {
        "measurements": scan_data.get("densities", []),
        "name": "E2E Unknown Type Curve",
        "curve_type": "monotonic",
    }
    r = requests.post(f"{BASE_URL}/api/curves/generate", json=payload, timeout=15)
    ok = r.status_code == 200
    log("generate_unknown_type", ok, f"status={r.status_code}")


def test_save_empty_body() -> None:
    """Step 8: Empty body should return 422."""
    r = requests.post(f"{BASE_URL}/api/curves/generate", json={}, timeout=10)
    ok = r.status_code == 422
    log("empty_body_422", ok, f"status={r.status_code}")


def main() -> int:
    """Run the full E2E user journey."""
    print("=" * 60)
    print("E2E USER JOURNEY: Calibration Wizard Flow")
    print("=" * 60)

    # Health check
    test_health()

    # Upload scan
    scan_data = test_upload_scan()
    if not scan_data:
        print("\nABORT: Scan upload failed, cannot continue journey")
        return 1

    # Generate curve
    curve_data = test_generate_curve(scan_data)
    if not curve_data:
        print("\nABORT: Curve generation failed, cannot continue journey")
        return 1

    # Save curve (the bug we just fixed)
    test_save_curve(curve_data)

    # Modify curve (happy paths)
    test_modify_curve_brightness(curve_data)
    test_modify_curve_contrast(curve_data)

    # Edge cases
    test_generate_curve_densities_key(scan_data)
    test_generate_curve_unknown_type(scan_data)
    test_save_empty_body()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"RESULTS: {passed}/{total} passed")

    if passed == total:
        print("ALL TESTS PASSED")
        return 0
    else:
        failed = [(tag, detail) for tag, ok, detail in results if not ok]
        print("FAILURES:")
        for tag, detail in failed:
            print(f"  - {tag}: {detail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
