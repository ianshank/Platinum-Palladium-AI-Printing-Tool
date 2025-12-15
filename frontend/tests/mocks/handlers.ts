/**
 * MSW (Mock Service Worker) handlers for API mocking in tests.
 * Provides realistic API responses for frontend testing.
 */

import { http, HttpResponse, delay } from 'msw';
import { env } from '@/config/env';

const API_BASE = env.VITE_API_BASE_URL;

// Mock data factories
const createMockCurveData = (overrides = {}) => ({
  id: 'curve-1',
  name: 'Test Curve',
  type: 'contrast',
  input_values: Array.from({ length: 21 }, (_, i) => i * 5),
  output_values: Array.from({ length: 21 }, (_, i) => Math.round(i * 4.76)),
  created_at: new Date().toISOString(),
  modified_at: new Date().toISOString(),
  metadata: {
    paper_type: 'Arches Platine',
    chemistry: 'Pt/Pd Classic',
    notes: 'Test curve',
  },
  ...overrides,
});

const createMockChemistryRecipe = (overrides = {}) => ({
  id: 'recipe-1',
  name: 'Standard Pt/Pd Mix',
  platinum_ml: 10,
  palladium_ml: 10,
  ferric_oxalate_ml: 20,
  contrast_agent_ml: 0,
  contrast_agent_type: 'none',
  total_volume_ml: 40,
  metal_ratio: 0.5,
  coverage_area_sq_in: 80,
  notes: 'Standard balanced mix',
  created_at: new Date().toISOString(),
  ...overrides,
});

const createMockSession = (overrides = {}) => ({
  id: 'session-1',
  name: 'Test Session',
  date: new Date().toISOString(),
  prints: [],
  notes: '',
  ...overrides,
});

// API Handlers
export const handlers = [
  // Health check
  http.get(`${API_BASE}/health`, async () => {
    await delay(100);
    return HttpResponse.json({ status: 'healthy', version: '1.0.0' });
  }),

  // Curves endpoints
  http.get(`${API_BASE}/api/v1/curves`, async () => {
    await delay(150);
    return HttpResponse.json({
      curves: [
        createMockCurveData({ id: 'curve-1', name: 'Contrast Curve 1' }),
        createMockCurveData({ id: 'curve-2', name: 'Contrast Curve 2', type: 'linearization' }),
      ],
      total: 2,
    });
  }),

  http.get(`${API_BASE}/api/v1/curves/:id`, async ({ params }) => {
    await delay(100);
    const { id } = params;
    return HttpResponse.json(createMockCurveData({ id: id as string }));
  }),

  http.post(`${API_BASE}/api/v1/curves/generate`, async ({ request }) => {
    await delay(500);
    const body = await request.json();
    return HttpResponse.json(createMockCurveData(body as object));
  }),

  http.post(`${API_BASE}/api/v1/curves/:id/modify`, async ({ params, request }) => {
    await delay(300);
    const { id } = params;
    const body = await request.json();
    return HttpResponse.json(
      createMockCurveData({
        id: id as string,
        ...(body as object),
        modified_at: new Date().toISOString(),
      })
    );
  }),

  http.post(`${API_BASE}/api/v1/curves/:id/smooth`, async ({ params }) => {
    await delay(200);
    const { id } = params;
    return HttpResponse.json(
      createMockCurveData({
        id: id as string,
        name: 'Smoothed Curve',
      })
    );
  }),

  http.post(`${API_BASE}/api/v1/curves/blend`, async () => {
    await delay(300);
    return HttpResponse.json(createMockCurveData({ name: 'Blended Curve' }));
  }),

  // Calibration endpoints
  http.get(`${API_BASE}/api/v1/calibrations`, async () => {
    await delay(150);
    return HttpResponse.json({
      calibrations: [],
      total: 0,
    });
  }),

  http.post(`${API_BASE}/api/v1/calibrations`, async ({ request }) => {
    await delay(400);
    const body = await request.json();
    return HttpResponse.json({
      id: 'cal-1',
      ...(body as object),
      created_at: new Date().toISOString(),
      status: 'completed',
    });
  }),

  // Scan endpoints
  http.post(`${API_BASE}/api/v1/scan/upload`, async () => {
    await delay(800);
    return HttpResponse.json({
      id: 'scan-1',
      filename: 'test-scan.tiff',
      quality_score: 0.85,
      measurements: Array.from({ length: 21 }, (_, i) => ({
        step: i,
        density: i * 0.15,
        lab: { l: 100 - i * 4.5, a: 0, b: 0 },
      })),
      issues: [],
    });
  }),

  http.post(`${API_BASE}/api/v1/scan/analyze`, async () => {
    await delay(600);
    return HttpResponse.json({
      analysis: {
        dmax: 2.1,
        dmin: 0.05,
        contrast_range: 2.05,
        quality_assessment: 'good',
        recommendations: ['Consider longer exposure for deeper blacks'],
      },
    });
  }),

  // Chemistry endpoints
  http.get(`${API_BASE}/api/v1/chemistry/recipes`, async () => {
    await delay(100);
    return HttpResponse.json({
      recipes: [createMockChemistryRecipe()],
      total: 1,
    });
  }),

  http.post(`${API_BASE}/api/v1/chemistry/calculate`, async ({ request }) => {
    await delay(200);
    const body = (await request.json()) as {
      print_area_sq_in: number;
      metal_ratio?: number;
    };
    const area = body.print_area_sq_in || 100;
    const ratio = body.metal_ratio || 0.5;
    const totalMetal = area * 0.25;
    return HttpResponse.json(
      createMockChemistryRecipe({
        platinum_ml: totalMetal * ratio,
        palladium_ml: totalMetal * (1 - ratio),
        coverage_area_sq_in: area,
        metal_ratio: ratio,
      })
    );
  }),

  // Chat endpoints
  http.post(`${API_BASE}/api/v1/chat`, async ({ request }) => {
    await delay(1000);
    const body = (await request.json()) as { message: string };
    return HttpResponse.json({
      id: 'msg-1',
      role: 'assistant',
      content: `This is a mock response to: "${body.message}"`,
      created_at: new Date().toISOString(),
    });
  }),

  http.get(`${API_BASE}/api/v1/chat/history`, async () => {
    await delay(100);
    return HttpResponse.json({
      messages: [],
      total: 0,
    });
  }),

  // Sessions endpoints
  http.get(`${API_BASE}/api/v1/sessions`, async () => {
    await delay(150);
    return HttpResponse.json({
      sessions: [createMockSession()],
      total: 1,
    });
  }),

  http.post(`${API_BASE}/api/v1/sessions`, async ({ request }) => {
    await delay(200);
    const body = await request.json();
    return HttpResponse.json(createMockSession(body as object));
  }),

  // Dashboard metrics
  http.get(`${API_BASE}/api/v1/dashboard/metrics`, async () => {
    await delay(150);
    return HttpResponse.json({
      total_prints: 42,
      total_sessions: 8,
      active_curves: 5,
      recent_activity: [],
    });
  }),
];

// Error handlers for testing error states
export const errorHandlers = {
  networkError: http.get(`${API_BASE}/*`, () => {
    return HttpResponse.error();
  }),

  serverError: http.get(`${API_BASE}/*`, () => {
    return HttpResponse.json(
      { error: 'Internal Server Error', message: 'Something went wrong' },
      { status: 500 }
    );
  }),

  unauthorized: http.get(`${API_BASE}/*`, () => {
    return HttpResponse.json(
      { error: 'Unauthorized', message: 'Authentication required' },
      { status: 401 }
    );
  }),

  notFound: http.get(`${API_BASE}/api/v1/curves/:id`, () => {
    return HttpResponse.json(
      { error: 'Not Found', message: 'Resource not found' },
      { status: 404 }
    );
  }),

  validationError: http.post(`${API_BASE}/*`, () => {
    return HttpResponse.json(
      {
        error: 'Validation Error',
        message: 'Invalid input',
        details: [{ field: 'name', message: 'Name is required' }],
      },
      { status: 422 }
    );
  }),
};
