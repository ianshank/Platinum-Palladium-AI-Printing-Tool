/**
 * API endpoint configuration.
 * Centralizes all API routes to avoid hardcoded strings throughout the app.
 */

import { env } from './env';

export const apiConfig = {
  baseUrl: env.VITE_API_BASE_URL,
  timeout: env.VITE_API_TIMEOUT,

  endpoints: {
    // Health
    health: '/api/health',

    // Curves
    curves: {
      generate: '/api/curves/generate',
      export: '/api/curves/export',
      modify: '/api/curves/modify',
      smooth: '/api/curves/smooth',
      blend: '/api/curves/blend',
      enhance: '/api/curves/enhance',
      uploadQuad: '/api/curves/upload-quad',
      parseQuad: '/api/curves/parse-quad',
      byId: (id: string) => `/api/curves/${id}`,
      enforceMonotonicity: (id: string) =>
        `/api/curves/${id}/enforce-monotonicity`,
    },

    // Scan
    scan: {
      upload: '/api/scan/upload',
    },

    // Analyze
    analyze: '/api/analyze',

    // Calibrations
    calibrations: {
      list: '/api/calibrations',
      create: '/api/calibrations',
      byId: (id: string) => `/api/calibrations/${id}`,
    },

    // Chemistry
    chemistry: {
      calculate: '/api/chemistry/calculate',
      presets: '/api/chemistry/presets',
    },

    // Chat
    chat: {
      message: '/api/chat',
      recipe: '/api/chat/recipe',
      troubleshoot: '/api/chat/troubleshoot',
    },

    // Statistics
    statistics: '/api/statistics',

    // Deep Learning (if enabled)
    deepLearning: {
      train: '/api/dl/train',
      predict: '/api/dl/predict',
      status: (taskId: string) => `/api/dl/status/${taskId}`,
    },
  },
} as const;

export type ApiEndpoints = typeof apiConfig.endpoints;
