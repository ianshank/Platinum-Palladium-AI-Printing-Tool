/**
 * TanStack Query hooks for scan operations.
 */

import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../client';
import { apiConfig } from '@/config/api.config';
import type { ScanUploadResponse, AnalyzeRequest, AnalyzeResponse } from '@/types';

/**
 * Upload and process a step tablet scan
 */
export const useUploadScan = () => {
  return useMutation({
    mutationFn: async ({
      file,
      tabletType,
    }: {
      file: File;
      tabletType: string;
    }): Promise<ScanUploadResponse> => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('tablet_type', tabletType);

      const { data } = await apiClient.post<ScanUploadResponse>(
        apiConfig.endpoints.scan.upload,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          // Longer timeout for image processing
          timeout: 60000,
        }
      );
      return data;
    },
  });
};

/**
 * Analyze density measurements
 */
export const useAnalyzeDensities = () => {
  return useMutation({
    mutationFn: async (densities: number[]): Promise<AnalyzeResponse> => {
      const request: AnalyzeRequest = { densities };
      const { data } = await apiClient.post<AnalyzeResponse>(
        apiConfig.endpoints.analyze,
        request
      );
      return data;
    },
  });
};

/**
 * Result type for scan quality assessment
 */
export interface ScanQualityAssessment {
  overall: 'excellent' | 'good' | 'acceptable' | 'poor';
  score: number;
  issues: {
    type: 'warning' | 'error';
    message: string;
    suggestion?: string;
  }[];
  recommendations: string[];
}

/**
 * Assess scan quality from upload response
 */
export const assessScanQuality = (
  response: ScanUploadResponse
): ScanQualityAssessment => {
  const issues: ScanQualityAssessment['issues'] = [];
  const recommendations: string[] = [];
  let score = 100;

  // Check density range
  const range = response.range;
  if (range < 1.5) {
    issues.push({
      type: 'error',
      message: `Density range is low (${range.toFixed(2)})`,
      suggestion: 'Increase exposure time or development time',
    });
    score -= 25;
  } else if (range < 1.8) {
    issues.push({
      type: 'warning',
      message: `Density range is slightly low (${range.toFixed(2)})`,
      suggestion: 'Consider adjusting exposure or development',
    });
    score -= 10;
  }

  // Check Dmin
  if (response.dmin > 0.15) {
    issues.push({
      type: 'warning',
      message: `Dmin is elevated (${response.dmin.toFixed(2)})`,
      suggestion: 'This may indicate fog or stale chemistry',
    });
    score -= 10;
  }

  // Check Dmax
  if (response.dmax < 1.8) {
    issues.push({
      type: 'warning',
      message: `Dmax is low (${response.dmax.toFixed(2)})`,
      suggestion: 'Increase exposure or consider higher metal ratio',
    });
    score -= 10;
  }

  // Check quality score
  if (response.quality < 0.7) {
    issues.push({
      type: 'warning',
      message: 'Patch uniformity is below optimal',
      suggestion: 'Ensure even coating and consistent development',
    });
    score -= 15;
  }

  // Add warnings from the response
  response.warnings.forEach((warning) => {
    issues.push({
      type: 'warning',
      message: warning,
    });
    score -= 5;
  });

  // Determine overall quality
  let overall: ScanQualityAssessment['overall'];
  if (score >= 90) {
    overall = 'excellent';
    recommendations.push('Excellent calibration quality. Proceed with curve generation.');
  } else if (score >= 75) {
    overall = 'good';
    recommendations.push('Good quality calibration. Minor improvements possible.');
  } else if (score >= 50) {
    overall = 'acceptable';
    recommendations.push('Acceptable quality. Consider addressing issues for better results.');
  } else {
    overall = 'poor';
    recommendations.push('Poor quality. Address issues before generating curves.');
  }

  return {
    overall,
    score: Math.max(0, score),
    issues,
    recommendations,
  };
};
