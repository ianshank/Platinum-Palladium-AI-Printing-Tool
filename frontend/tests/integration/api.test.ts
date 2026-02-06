/**
 * API integration tests.
 * Tests API client and hooks with mocked backend.
 */

import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { server } from '@tests/mocks/server';
import { handlers, errorHandlers } from '@tests/mocks/handlers';
import { ReactNode } from 'react';

// Start mock server
beforeAll(() => {
  server.listen({ onUnhandledRequest: 'warn' });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});

// Create wrapper with QueryClient
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
};

describe('API Integration', () => {
  describe('Health Check', () => {
    it('returns healthy status', async () => {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();

      expect(data.status).toBe('healthy');
      expect(data.version).toBeDefined();
    });
  });

  describe('Curves API', () => {
    it('fetches curves list', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves');
      const data = await response.json();

      expect(data.curves).toBeDefined();
      expect(Array.isArray(data.curves)).toBe(true);
      expect(data.total).toBeGreaterThanOrEqual(0);
    });

    it('fetches single curve by id', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves/curve-1');
      const data = await response.json();

      expect(data.id).toBe('curve-1');
      expect(data.input_values).toBeDefined();
      expect(data.output_values).toBeDefined();
    });

    it('generates a new curve', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'New Curve',
          type: 'contrast',
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.id).toBeDefined();
      expect(data.input_values).toBeDefined();
    });

    it('modifies an existing curve', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves/curve-1/modify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          adjustments: { contrast: 1.2 },
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.modified_at).toBeDefined();
    });

    it('smooths a curve', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves/curve-1/smooth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strength: 0.5 }),
      });

      expect(response.ok).toBe(true);
    });

    it('blends multiple curves', async () => {
      const response = await fetch('http://localhost:8000/api/v1/curves/blend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          curve_ids: ['curve-1', 'curve-2'],
          weights: [0.5, 0.5],
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.name).toBe('Blended Curve');
    });
  });

  describe('Scan API', () => {
    it('uploads scan image', async () => {
      const formData = new FormData();
      formData.append('file', new Blob(['test']), 'test.tiff');

      const response = await fetch('http://localhost:8000/api/v1/scan/upload', {
        method: 'POST',
        body: formData,
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.quality_score).toBeDefined();
      expect(data.measurements).toBeDefined();
    });

    it('analyzes scan', async () => {
      const response = await fetch('http://localhost:8000/api/v1/scan/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_id: 'scan-1' }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.analysis).toBeDefined();
      expect(data.analysis.dmax).toBeDefined();
    });
  });

  describe('Chemistry API', () => {
    it('fetches recipes list', async () => {
      const response = await fetch('http://localhost:8000/api/v1/chemistry/recipes');
      const data = await response.json();

      expect(data.recipes).toBeDefined();
      expect(Array.isArray(data.recipes)).toBe(true);
    });

    it('calculates chemistry', async () => {
      const response = await fetch('http://localhost:8000/api/v1/chemistry/calculate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          print_area_sq_in: 80,
          metal_ratio: 0.5,
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.platinum_ml).toBeDefined();
      expect(data.palladium_ml).toBeDefined();
      expect(data.ferric_oxalate_ml).toBeDefined();
    });
  });

  describe('Chat API', () => {
    it('sends chat message', async () => {
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: 'How do I improve contrast?',
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.role).toBe('assistant');
      expect(data.content).toBeDefined();
    });

    it('fetches chat history', async () => {
      const response = await fetch('http://localhost:8000/api/v1/chat/history');
      const data = await response.json();

      expect(data.messages).toBeDefined();
      expect(Array.isArray(data.messages)).toBe(true);
    });
  });

  describe('Sessions API', () => {
    it('fetches sessions list', async () => {
      const response = await fetch('http://localhost:8000/api/v1/sessions');
      const data = await response.json();

      expect(data.sessions).toBeDefined();
      expect(Array.isArray(data.sessions)).toBe(true);
    });

    it('creates a new session', async () => {
      const response = await fetch('http://localhost:8000/api/v1/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'New Print Session',
          date: new Date().toISOString(),
        }),
      });

      expect(response.ok).toBe(true);
      const data = await response.json();
      expect(data.id).toBeDefined();
      expect(data.name).toBe('New Print Session');
    });
  });

  describe('Dashboard API', () => {
    it('fetches dashboard metrics', async () => {
      const response = await fetch('http://localhost:8000/api/v1/dashboard/metrics');
      const data = await response.json();

      expect(data.total_prints).toBeDefined();
      expect(data.total_sessions).toBeDefined();
      expect(data.active_curves).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('handles 404 errors', async () => {
      server.use(errorHandlers.notFound);

      const response = await fetch('http://localhost:8000/api/v1/curves/nonexistent');
      expect(response.status).toBe(404);
    });

    it('handles 500 errors', async () => {
      server.use(errorHandlers.serverError);

      const response = await fetch('http://localhost:8000/api/v1/curves');
      expect(response.status).toBe(500);
    });

    it('handles validation errors', async () => {
      server.use(errorHandlers.validationError);

      const response = await fetch('http://localhost:8000/api/v1/curves/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });

      expect(response.status).toBe(422);
      const data = await response.json();
      expect(data.details).toBeDefined();
    });
  });
});
