/**
 * Config Module Tests
 *
 * Covers:
 * - config object default values
 * - isDev / isProd / isTest helpers
 * - Config structure validation
 */

import { describe, it, expect } from 'vitest';
import { config, isDev, isProd, isTest } from './index';
import type { AppConfig } from './index';

describe('config', () => {
    describe('config object structure', () => {
        it('has all top-level sections', () => {
            expect(config).toHaveProperty('app');
            expect(config).toHaveProperty('api');
            expect(config).toHaveProperty('features');
            expect(config).toHaveProperty('logging');
            expect(config).toHaveProperty('ui');
            expect(config).toHaveProperty('calibration');
        });

        it('satisfies AppConfig type', () => {
            // Type assertion — will fail at compile time if shape is wrong
            const _cfg: AppConfig = config;
            expect(_cfg).toBeDefined();
        });
    });

    describe('app section defaults', () => {
        it('has a name', () => {
            expect(typeof config.app.name).toBe('string');
            expect(config.app.name.length).toBeGreaterThan(0);
        });

        it('has a valid version', () => {
            expect(config.app.version).toMatch(/^\d+\.\d+\.\d+/);
        });

        it('environment is test in vitest context', () => {
            expect(config.app.environment).toBe('test');
        });
    });

    describe('api section defaults', () => {
        it('has a baseUrl', () => {
            expect(config.api.baseUrl).toContain('http');
        });

        it('timeout is a positive number', () => {
            expect(config.api.timeout).toBeGreaterThan(0);
        });

        it('retryAttempts is a non-negative number', () => {
            expect(config.api.retryAttempts).toBeGreaterThanOrEqual(0);
        });

        it('staleTime and gcTime are positive', () => {
            expect(config.api.staleTime).toBeGreaterThan(0);
            expect(config.api.gcTime).toBeGreaterThan(0);
        });
    });

    describe('ui section defaults', () => {
        it('has positive animation duration', () => {
            expect(config.ui.animationDuration).toBeGreaterThan(0);
        });

        it('has positive toast duration', () => {
            expect(config.ui.toastDuration).toBeGreaterThan(0);
        });

        it('has a default tab', () => {
            expect(config.ui.defaultTab).toBe('dashboard');
        });
    });

    describe('calibration section defaults', () => {
        it('defaultSteps is a positive integer', () => {
            expect(config.calibration.defaultSteps).toBeGreaterThan(0);
            expect(Number.isInteger(config.calibration.defaultSteps)).toBe(true);
        });

        it('maxCurvePoints is positive', () => {
            expect(config.calibration.maxCurvePoints).toBeGreaterThan(0);
        });

        it('smoothingDefault is between 0 and 1', () => {
            expect(config.calibration.smoothingDefault).toBeGreaterThanOrEqual(0);
            expect(config.calibration.smoothingDefault).toBeLessThanOrEqual(1);
        });
    });

    describe('environment helpers', () => {
        it('isTest is true in vitest', () => {
            expect(isTest).toBe(true);
        });

        it('isDev is false in test env', () => {
            expect(isDev).toBe(false);
        });

        it('isProd is false in test env', () => {
            expect(isProd).toBe(false);
        });
    });
});
