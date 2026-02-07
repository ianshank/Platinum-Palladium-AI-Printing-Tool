import { useMutation } from '@tanstack/react-query';
import { api } from '../client';

export function useUploadScan() {
    return useMutation({
        mutationFn: (data: { file: File; tabletType: string }) =>
            api.scan.upload(data.file, data.tabletType),
    });
}

/** Input shape for scan quality assessment (matches Step2Analyze usage) */
export interface ScanAnalysisInput {
    densities: number[];
    dmax: number;
    dmin: number;
    range: number;
    num_patches: number;
}

export interface QualityIssue {
    type: 'warning' | 'error';
    message: string;
    suggestion?: string;
}

export interface QualityAssessment {
    quality: string;
    score: number;
    overall: 'excellent' | 'good' | 'acceptable' | 'poor';
    issues: QualityIssue[];
}

/**
 * Assess scan quality from extracted density data.
 * Returns quality grade, score (0–100), and any issues found.
 */
export const assessScanQuality = (input: ScanAnalysisInput): QualityAssessment => {
    const issues: QualityIssue[] = [];
    let score = 100;

    // Check density range
    if (input.range < 0.5) {
        issues.push({ type: 'warning', message: 'Low density range', suggestion: 'Increase exposure time or check chemistry' });
        score -= 20;
    } else if (input.range < 1.0) {
        issues.push({ type: 'warning', message: 'Moderate density range', suggestion: 'Consider adjusting exposure' });
        score -= 10;
    }

    // Check patch count
    if (input.num_patches < 11) {
        issues.push({ type: 'error', message: `Only ${input.num_patches} patches detected`, suggestion: 'Rescan with better alignment' });
        score -= 30;
    }

    // Check Dmax
    if (input.dmax < 0.8) {
        issues.push({ type: 'warning', message: 'Low Dmax — print may lack shadow detail', suggestion: 'Check chemistry concentration' });
        score -= 15;
    }

    const clampedScore = Math.max(0, Math.min(100, score));
    const overall = clampedScore >= 90 ? 'excellent' : clampedScore >= 70 ? 'good' : clampedScore >= 50 ? 'acceptable' : 'poor';

    return {
        quality: overall,
        score: clampedScore,
        overall,
        issues,
    };
};
