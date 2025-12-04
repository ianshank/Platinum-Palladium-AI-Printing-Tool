"""
Locust Load Testing Configuration.

Run with: locust -f tests/performance/locustfile.py

Environment variables:
- PTPD_API_URL: Base URL for the API (default: http://localhost:8000)
"""

import json
import os
import random

try:
    from locust import HttpUser, between, task

    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    HttpUser = object

    def between(*args):
        pass

    def task(weight=1):
        def decorator(func):
            return func

        return decorator


class PTPDAPIUser(HttpUser):
    """Simulated user for PTPD API load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    host = os.environ.get("PTPD_API_URL", "http://localhost:8000")

    def on_start(self):
        """Called when a simulated user starts."""
        self.sample_densities = [0.1 + i * 0.1 for i in range(21)]
        self.curve_ids = []
        self.calibration_ids = []

    @task(10)
    def health_check(self):
        """Frequently check health endpoint."""
        self.client.get("/api/health")

    @task(5)
    def get_root(self):
        """Get root endpoint."""
        self.client.get("/")

    @task(3)
    def analyze_densities(self):
        """Analyze density measurements."""
        self.client.post(
            "/api/analyze",
            json={"densities": self.sample_densities},
        )

    @task(5)
    def generate_curve(self):
        """Generate a calibration curve."""
        response = self.client.post(
            "/api/curves/generate",
            json={
                "densities": self.sample_densities,
                "name": f"Load Test Curve {random.randint(1, 1000)}",
                "curve_type": random.choice(["linear", "spline"]),
            },
        )
        if response.status_code == 200:
            data = response.json()
            if "curve_id" in data:
                self.curve_ids.append(data["curve_id"])
                # Keep only last 10 curve IDs
                self.curve_ids = self.curve_ids[-10:]

    @task(3)
    def modify_curve(self):
        """Modify an existing curve."""
        input_values = [i / 255 for i in range(256)]
        output_values = [x**0.9 for x in input_values]

        self.client.post(
            "/api/curves/modify",
            json={
                "input_values": input_values,
                "output_values": output_values,
                "name": "Modified Curve",
                "adjustment_type": random.choice(
                    ["brightness", "contrast", "gamma"]
                ),
                "amount": random.uniform(-0.2, 0.2),
            },
        )

    @task(2)
    def smooth_curve(self):
        """Smooth a curve."""
        input_values = [i / 255 for i in range(256)]
        output_values = [x**0.9 for x in input_values]

        self.client.post(
            "/api/curves/smooth",
            json={
                "input_values": input_values,
                "output_values": output_values,
                "name": "Smoothed Curve",
                "method": random.choice(["gaussian", "savgol"]),
                "strength": random.uniform(0.3, 0.7),
            },
        )

    @task(2)
    def get_curve(self):
        """Get a stored curve."""
        if self.curve_ids:
            curve_id = random.choice(self.curve_ids)
            self.client.get(f"/api/curves/{curve_id}")

    @task(3)
    def list_calibrations(self):
        """List calibration records."""
        self.client.get("/api/calibrations")

    @task(2)
    def list_calibrations_filtered(self):
        """List calibrations with filter."""
        papers = ["Arches Platine", "Bergger COT320", "Hahnemuhle Platinum Rag"]
        self.client.get(
            "/api/calibrations",
            params={"paper_type": random.choice(papers), "limit": 20},
        )

    @task(2)
    def create_calibration(self):
        """Create a new calibration record."""
        response = self.client.post(
            "/api/calibrations",
            json={
                "paper_type": random.choice(
                    ["Arches Platine", "Bergger COT320", "Test Paper"]
                ),
                "exposure_time": random.uniform(120, 240),
                "metal_ratio": random.uniform(0.3, 0.7),
                "densities": self.sample_densities,
            },
        )
        if response.status_code == 200:
            data = response.json()
            if "id" in data:
                self.calibration_ids.append(data["id"])
                self.calibration_ids = self.calibration_ids[-10:]

    @task(2)
    def get_calibration(self):
        """Get a specific calibration."""
        if self.calibration_ids:
            cal_id = random.choice(self.calibration_ids)
            self.client.get(f"/api/calibrations/{cal_id}")

    @task(1)
    def get_statistics(self):
        """Get database statistics."""
        self.client.get("/api/statistics")


class PTPDHeavyUser(HttpUser):
    """Simulated heavy user for stress testing."""

    wait_time = between(0.5, 1.5)
    host = os.environ.get("PTPD_API_URL", "http://localhost:8000")

    @task(10)
    def rapid_health_checks(self):
        """Rapid health check requests."""
        self.client.get("/api/health")

    @task(5)
    def rapid_curve_generation(self):
        """Rapid curve generation."""
        densities = [0.1 + i * 0.1 for i in range(21)]
        self.client.post(
            "/api/curves/generate",
            json={
                "densities": densities,
                "name": "Stress Test Curve",
                "curve_type": "linear",
            },
        )

    @task(3)
    def rapid_analysis(self):
        """Rapid density analysis."""
        densities = [random.uniform(0.1, 2.0) for _ in range(21)]
        self.client.post("/api/analyze", json={"densities": densities})


if __name__ == "__main__":
    if LOCUST_AVAILABLE:
        print("Run Locust with: locust -f tests/performance/locustfile.py")
    else:
        print("Locust not installed. Install with: pip install locust")
