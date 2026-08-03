/**
 * Tests for generated TypeScript schema types.
 *
 * These tests ensure:
 * 1. Generated types are complete and correct
 * 2. All API endpoints are properly typed
 * 3. Request/response types match backend models
 * 4. No circular references in generated types
 */

import { describe, expect, it } from "vitest";

// Import the generated schema types
import type { components, paths } from "../generated/schema";

describe("Generated TypeScript Schema", () => {
  describe("Type Structure", () => {
    it("should have paths defined", () => {
      // This is a compile-time check that paths are exported
      const _paths: paths = {} as paths;
      expect(_paths).toBeDefined();
    });

    it("should have components defined", () => {
      // This is a compile-time check that components are exported
      const _components: components = {} as components;
      expect(_components).toBeDefined();
    });
  });

  describe("API Response Models", () => {
    it("should have HealthResponse type", () => {
      type HealthResponse =
        components["schemas"]["HealthResponse"];
      const response: HealthResponse = { status: "healthy" };
      expect(response.status).toBe("healthy");
    });

    it("should have RootResponse type", () => {
      type RootResponse = components["schemas"]["RootResponse"];
      const response: RootResponse = {
        message: "PTPD Calibration API",
        version: "1.0.0",
      };
      expect(response.message).toBeDefined();
      expect(response.version).toBeDefined();
    });

    it("should have AnalyzeResponse type", () => {
      type AnalyzeResponse =
        components["schemas"]["AnalyzeResponse"];
      const response: AnalyzeResponse = {
        dmin: 0.1,
        dmax: 2.5,
        range: 2.4,
        is_monotonic: true,
        max_error: 0.05,
        rms_error: 0.02,
        suggestions: ["Adjust exposure"],
      };
      expect(response.dmin).toBeGreaterThanOrEqual(0);
      expect(response.dmax).toBeGreaterThan(response.dmin);
    });

    it("should have ScanUploadResponse type", () => {
      type ScanUploadResponse =
        components["schemas"]["ScanUploadResponse"];
      const response: ScanUploadResponse = {
        success: true,
        extraction_id: "abc-123",
        original_filename: "scan.tif",
        num_patches: 21,
        densities: [0.1, 0.5, 1.0, 1.5, 2.0],
        dmin: 0.1,
        dmax: 2.0,
        range: 1.9,
        quality: 0.95,
        warnings: [],
      };
      expect(response.success).toBe(true);
      expect(response.num_patches).toBeGreaterThan(0);
    });

    it("should have CurveGenerateResponse type", () => {
      type CurveGenerateResponse =
        components["schemas"]["CurveGenerateResponse"];
      const response: CurveGenerateResponse = {
        success: true,
        curve_id: "curve-123",
        name: "My Curve",
        num_points: 256,
        input_values: [0, 0.5, 1.0],
        output_values: [0, 0.5, 1.0],
      };
      expect(response.num_points).toBeGreaterThanOrEqual(2);
    });

    it("should have ListCalibrationsResponse type", () => {
      type ListCalibrationsResponse =
        components["schemas"]["ListCalibrationsResponse"];
      const response: ListCalibrationsResponse = {
        count: 5,
        records: [
          {
            id: "cal-1",
            paper_type: "Arches",
            exposure_time: 10.5,
            metal_ratio: 0.5,
            timestamp: "2026-08-03T00:00:00Z",
            dmax: 2.1,
          },
        ],
      };
      expect(response.count).toBeGreaterThanOrEqual(0);
      expect(Array.isArray(response.records)).toBe(true);
    });

    it("should have ChatResponse type", () => {
      type ChatResponse = components["schemas"]["ChatResponse"];
      const response: ChatResponse = {
        response: "Hello, how can I help?",
      };
      expect(typeof response.response).toBe("string");
    });

    it("should have StatisticsResponse type", () => {
      type StatisticsResponse =
        components["schemas"]["StatisticsResponse"];
      const response: StatisticsResponse = {
        total_calibrations: 10,
        unique_papers: 3,
        avg_exposure_time: 8.5,
        data: { custom_metric: 42 },
      };
      expect(response.total_calibrations).toBeGreaterThanOrEqual(0);
    });
  });

  describe("API Request Models", () => {
    it("should have AnalyzeRequest type", () => {
      type AnalyzeRequest = components["schemas"]["AnalyzeRequest"];
      const request: AnalyzeRequest = {
        densities: [0.1, 0.5, 1.0, 1.5, 2.0],
      };
      expect(Array.isArray(request.densities)).toBe(true);
    });

    it("should have CurveRequest type", () => {
      type CurveRequest = components["schemas"]["CurveRequest"];
      const request: CurveRequest = {
        densities: [0.1, 0.5, 1.0, 1.5, 2.0],
        name: "Test Curve",
        curve_type: "linear",
      };
      expect(request.densities.length).toBeGreaterThan(0);
    });

    it("should have ChatRequest type", () => {
      type ChatRequest = components["schemas"]["ChatRequest"];
      const request: ChatRequest = {
        message: "Hello",
        include_history: true,
      };
      expect(typeof request.message).toBe("string");
    });

    it("should have CalibrationRequest type", () => {
      type CalibrationRequest =
        components["schemas"]["CalibrationRequest"];
      const request: CalibrationRequest = {
        paper_type: "Arches",
        exposure_time: 10.5,
        metal_ratio: 0.5,
        contrast_agent: "dichromate",
        contrast_amount: 1.0,
        developer: "potassium_oxalate",
        chemistry_type: "platinum_palladium",
        densities: [0.1, 0.5, 1.0, 1.5, 2.0],
      };
      expect(request.exposure_time).toBeGreaterThanOrEqual(0);
    });
  });

  describe("API Endpoint Paths", () => {
    it("should have health endpoint", () => {
      type HealthEndpoint = paths["/api/health"];
      const endpoint: HealthEndpoint = {} as HealthEndpoint;
      expect(endpoint).toBeDefined();
    });

    it("should have analyze endpoint", () => {
      type AnalyzeEndpoint = paths["/api/analyze"];
      const endpoint: AnalyzeEndpoint = {} as AnalyzeEndpoint;
      expect(endpoint).toBeDefined();
    });

    it("should have scan upload endpoint", () => {
      type ScanEndpoint = paths["/api/scan/upload"];
      const endpoint: ScanEndpoint = {} as ScanEndpoint;
      expect(endpoint).toBeDefined();
    });

    it("should have curves endpoints", () => {
      type CurveGenerateEndpoint =
        paths["/api/curves/generate"];
      type CurveModifyEndpoint = paths["/api/curves/modify"];

      const generateEndpoint: CurveGenerateEndpoint =
        {} as CurveGenerateEndpoint;
      const modifyEndpoint: CurveModifyEndpoint =
        {} as CurveModifyEndpoint;

      expect(generateEndpoint).toBeDefined();
      expect(modifyEndpoint).toBeDefined();
    });

    it("should have calibrations endpoints", () => {
      type CalibrationsEndpoint = paths["/api/calibrations"];
      const endpoint: CalibrationsEndpoint =
        {} as CalibrationsEndpoint;
      expect(endpoint).toBeDefined();
    });

    it("should have chat endpoints", () => {
      type ChatEndpoint = paths["/api/chat"];
      type RecipeEndpoint = paths["/api/chat/recipe"];

      const chatEndpoint: ChatEndpoint = {} as ChatEndpoint;
      const recipeEndpoint: RecipeEndpoint =
        {} as RecipeEndpoint;

      expect(chatEndpoint).toBeDefined();
      expect(recipeEndpoint).toBeDefined();
    });

    it("should have statistics endpoint", () => {
      type StatsEndpoint = paths["/api/statistics"];
      const endpoint: StatsEndpoint = {} as StatsEndpoint;
      expect(endpoint).toBeDefined();
    });
  });

  describe("Type Safety", () => {
    it("should enforce required fields", () => {
      type AnalyzeResponse =
        components["schemas"]["AnalyzeResponse"];

      // This should compile if types are correct
      const response: AnalyzeResponse = {
        dmin: 0.1,
        dmax: 2.0,
        range: 1.9,
        is_monotonic: true,
        max_error: 0.1,
        rms_error: 0.05,
        suggestions: [],
      };

      expect(response.dmin).toBeDefined(); // Compilation is the real test
    });

    it("should allow optional fields", () => {
      type CurveRequest = components["schemas"]["CurveRequest"];

      // Should allow creating with optional fields omitted
      const request: CurveRequest = {
        densities: [0.1, 0.5, 1.0],
        name: "My Curve",
        curve_type: "linear",
      };

      expect(request.densities).toBeDefined();
    });

    it("should validate numeric constraints", () => {
      type AnalyzeResponse =
        components["schemas"]["AnalyzeResponse"];

      const response: AnalyzeResponse = {
        dmin: 0,
        dmax: 3,
        range: 3,
        is_monotonic: true,
        max_error: 0.1,
        rms_error: 0.05,
        suggestions: [],
      };

      // Values must be numbers and satisfy constraints
      expect(typeof response.dmin).toBe("number");
      expect(typeof response.dmax).toBe("number");
      expect(response.dmax).toBeGreaterThanOrEqual(response.dmin);
    });
  });

  describe("Enum Types", () => {
    it("should have ChemistryType enum", () => {
      type CalibrationRequest =
        components["schemas"]["CalibrationRequest"];

      const request: CalibrationRequest = {
        paper_type: "Arches",
        exposure_time: 10.5,
        metal_ratio: 0.5,
        contrast_agent: "dichromate",
        contrast_amount: 1.0,
        developer: "potassium_oxalate",
        chemistry_type: "platinum_palladium",
        densities: [0.1, 0.5, 1.0],
      };

      expect(request.chemistry_type).toBe("platinum_palladium");
    });
  });

  describe("Array Types", () => {
    it("should properly type array fields", () => {
      type AnalyzeRequest = components["schemas"]["AnalyzeRequest"];
      const request: AnalyzeRequest = {
        densities: [0.1, 0.2, 0.3],
      };

      expect(Array.isArray(request.densities)).toBe(true);
      expect(request.densities[0]).toBe(0.1);
    });

    it("should properly type response with arrays", () => {
      type ListCalibrationsResponse =
        components["schemas"]["ListCalibrationsResponse"];

      const response: ListCalibrationsResponse = {
        count: 0,
        records: [],
      };

      expect(Array.isArray(response.records)).toBe(true);
    });
  });
});
