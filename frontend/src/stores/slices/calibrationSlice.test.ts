import { describe, it, expect, beforeEach } from 'vitest';
import { createStore } from '@/stores';

describe('calibrationSlice', () => {
    let store: ReturnType<typeof createStore>;

    beforeEach(() => {
        store = createStore();
    });

    describe('Initial State', () => {
        it('has correct initial values', () => {
            const state = store.getState();
            expect(state.calibration.current).toBeNull();
            expect(state.calibration.currentStep).toBe(0);
            expect(state.calibration.totalSteps).toBe(5);
            expect(state.calibration.history).toEqual([]);
            expect(state.calibration.isAnalyzing).toBe(false);
            expect(state.calibration.analysisProgress).toBe(0);
            expect(state.calibration.error).toBeNull();
        });
    });

    describe('Wizard Step Navigation', () => {
        it('nextStep increments currentStep', () => {
            store.getState().calibration.nextStep();
            expect(store.getState().calibration.currentStep).toBe(1);
        });

        it('nextStep does not exceed totalSteps - 1', () => {
            for (let i = 0; i < 10; i++) {
                store.getState().calibration.nextStep();
            }
            expect(store.getState().calibration.currentStep).toBe(4);
        });

        it('previousStep decrements currentStep', () => {
            store.getState().calibration.nextStep();
            store.getState().calibration.nextStep();
            store.getState().calibration.previousStep();
            expect(store.getState().calibration.currentStep).toBe(1);
        });

        it('previousStep does not go below 0', () => {
            store.getState().calibration.previousStep();
            expect(store.getState().calibration.currentStep).toBe(0);
        });

        it('setCurrentStep sets step directly', () => {
            store.getState().calibration.setCurrentStep(3);
            expect(store.getState().calibration.currentStep).toBe(3);
        });
    });

    describe('Calibration Lifecycle', () => {
        it('startCalibration creates a new calibration', () => {
            store.getState().calibration.startCalibration('21-step');
            const state = store.getState();

            expect(state.calibration.current).not.toBeNull();
            expect(state.calibration.current?.tabletType).toBe('21-step');
            expect(state.calibration.currentStep).toBe(0);
            expect(state.calibration.error).toBeNull();
        });

        it('setMeasurements stores measurement data', () => {
            store.getState().calibration.startCalibration('21-step');
            const measurements = [
                { step: 1, targetDensity: 0.1, measuredDensity: 0.12 },
                { step: 2, targetDensity: 0.2, measuredDensity: 0.22 },
            ];

            store.getState().calibration.setMeasurements(measurements);
            expect(store.getState().calibration.current?.measurements).toEqual(measurements);
        });

        it('addMeasurement appends to existing measurements', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.addMeasurement({
                step: 1,
                targetDensity: 0.1,
                measuredDensity: 0.12,
            });

            expect(store.getState().calibration.current?.measurements).toHaveLength(1);
        });

        it('updateMeasurement modifies a specific step', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.addMeasurement({
                step: 1,
                targetDensity: 0.1,
                measuredDensity: 0.12,
            });

            store.getState().calibration.updateMeasurement(1, {
                step: 1,
                targetDensity: 0.1,
                measuredDensity: 0.15,
            });

            expect(store.getState().calibration.current?.measurements[0]?.measuredDensity).toBe(0.15);
        });
    });

    describe('Analysis State', () => {
        it('setAnalyzing toggles analysis flag', () => {
            store.getState().calibration.setAnalyzing(true);
            expect(store.getState().calibration.isAnalyzing).toBe(true);

            store.getState().calibration.setAnalyzing(false);
            expect(store.getState().calibration.isAnalyzing).toBe(false);
        });

        it('setAnalysisProgress updates progress', () => {
            store.getState().calibration.setAnalysisProgress(50);
            expect(store.getState().calibration.analysisProgress).toBe(50);
        });

        it('setError sets error string', () => {
            store.getState().calibration.setError('Something went wrong');
            expect(store.getState().calibration.error).toBe('Something went wrong');
        });

        it('setError with null clears error', () => {
            store.getState().calibration.setError('Error');
            store.getState().calibration.setError(null);
            expect(store.getState().calibration.error).toBeNull();
        });
    });

    describe('Calibration History', () => {
        it('saveCalibration adds current to history', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.saveCalibration('My Calibration', 'Test notes');

            expect(store.getState().calibration.history).toHaveLength(1);
            expect(store.getState().calibration.history[0]?.name).toBe('My Calibration');
        });

        it('loadCalibration restores from history', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.saveCalibration('Saved Cal');

            const savedId = store.getState().calibration.history[0]?.id;
            store.getState().calibration.clearCurrent();

            expect(store.getState().calibration.current).toBeNull();
            if (savedId) {
                store.getState().calibration.loadCalibration(savedId);
                expect(store.getState().calibration.current?.name).toBe('Saved Cal');
            }
        });

        it('deleteCalibration removes from history', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.saveCalibration('To Delete');

            const savedId = store.getState().calibration.history[0]?.id;
            if (savedId) {
                store.getState().calibration.deleteCalibration(savedId);
                expect(store.getState().calibration.history).toHaveLength(0);
            }
        });
    });

    describe('Reset Operations', () => {
        it('clearCurrent clears current without affecting history', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.saveCalibration('Keep This');
            store.getState().calibration.clearCurrent();

            expect(store.getState().calibration.current).toBeNull();
            expect(store.getState().calibration.history).toHaveLength(1);
        });

        it('resetCalibration resets all state', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.nextStep();
            store.getState().calibration.nextStep();
            store.getState().calibration.setError('some error');

            store.getState().calibration.resetCalibration();

            expect(store.getState().calibration.current).toBeNull();
            expect(store.getState().calibration.currentStep).toBe(0);
            expect(store.getState().calibration.error).toBeNull();
        });

        it('updateMetadata merges metadata into current calibration', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.updateMetadata({ dMax: 1.8, dMin: 0.05 });

            expect(store.getState().calibration.current?.metadata).toEqual({
                dMax: 1.8,
                dMin: 0.05,
            });
        });

        it('updateMetadata merges with existing metadata', () => {
            store.getState().calibration.startCalibration('21-step');
            store.getState().calibration.updateMetadata({ dMax: 1.8 });
            store.getState().calibration.updateMetadata({ dMin: 0.05 });

            expect(store.getState().calibration.current?.metadata).toEqual({
                dMax: 1.8,
                dMin: 0.05,
            });
        });
    });
});
