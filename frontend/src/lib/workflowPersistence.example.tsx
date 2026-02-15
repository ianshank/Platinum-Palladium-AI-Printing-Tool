/**
 * Example usage of workflow persistence in a React component
 *
 * This shows how to integrate workflow checkpoint/resume functionality
 * into the calibration wizard or any multi-step workflow.
 */

import { useEffect } from 'react';
import { useStore } from '@/stores';
import { cleanupExpiredCheckpoints } from '@/lib/workflowPersistence';

/**
 * Example: CalibrationWizard component with checkpoint persistence
 */
export function CalibrationWizardExample(): JSX.Element {
  const currentStep = useStore((state) => state.calibration.currentStep);
  const current = useStore((state) => state.calibration.current);
  const nextStep = useStore((state) => state.calibration.nextStep);
  const saveWorkflowCheckpoint = useStore((state) => state.calibration.saveWorkflowCheckpoint);
  const loadWorkflowCheckpoint = useStore((state) => state.calibration.loadWorkflowCheckpoint);
  const clearWorkflowCheckpoint = useStore((state) => state.calibration.clearWorkflowCheckpoint);

  // Load checkpoint on mount
  useEffect(() => {
    const checkpoint = loadWorkflowCheckpoint();
    if (checkpoint) {
      // Optional: Show a notification to the user that we're resuming
      console.log(`Resuming calibration from step ${checkpoint.step + 1}`);
    }

    // Clean up any expired checkpoints
    cleanupExpiredCheckpoints();
  }, [loadWorkflowCheckpoint]);

  // Save checkpoint whenever the step or calibration data changes
  useEffect(() => {
    if (current) {
      saveWorkflowCheckpoint();
    }
  }, [currentStep, current, saveWorkflowCheckpoint]);

  const handleComplete = (): void => {
    // Clear checkpoint when workflow is completed
    clearWorkflowCheckpoint();
    console.log('Calibration complete!');
  };

  const handleCancel = (): void => {
    // Optionally clear checkpoint on cancel
    clearWorkflowCheckpoint();
    console.log('Calibration cancelled');
  };

  return (
    <div>
      <h1>Calibration Wizard - Step {currentStep + 1}</h1>

      {/* Wizard steps go here */}

      <button onClick={() => nextStep()}>Next Step</button>
      <button onClick={handleComplete}>Complete</button>
      <button onClick={handleCancel}>Cancel</button>
    </div>
  );
}

/**
 * Example: Manual checkpoint management
 *
 * For more fine-grained control, you can use the checkpoint functions directly
 * instead of relying on the Zustand store actions.
 */
import {
  clearCheckpoint,
  loadCheckpoint,
  saveCheckpoint,
} from '@/lib/workflowPersistence';
import type { CalibrationWorkflowCheckpoint } from '@/stores/slices/calibrationSlice';

export function manualCheckpointExample(): void {
  // Save a checkpoint
  const workflowData: CalibrationWorkflowCheckpoint = {
    current: {
      id: 'cal-123',
      name: 'Test Calibration',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      tabletType: '21-step',
      measurements: [],
    },
    currentStep: 2,
    measurements: [],
    metadata: {
      dmin: 0.05,
      dmax: 2.5,
    },
  };

  saveCheckpoint('calibration-wizard', 2, workflowData);

  // Load a checkpoint
  const checkpoint = loadCheckpoint<CalibrationWorkflowCheckpoint>('calibration-wizard');
  if (checkpoint) {
    console.log(`Loaded checkpoint from step ${checkpoint.step}`);
    console.log('Data:', checkpoint.data);
  }

  // Clear a checkpoint
  clearCheckpoint('calibration-wizard');
}

/**
 * Example: Custom TTL (Time-To-Live)
 *
 * By default, checkpoints expire after 24 hours.
 * You can customize this by passing a TTL value in milliseconds.
 */
export function customTTLExample(): void {
  const data = { step: 1, value: 'test' };

  // Save with 1-hour TTL
  const oneHour = 60 * 60 * 1000;
  saveCheckpoint('short-workflow', 1, data, oneHour);

  // Save with 7-day TTL
  const sevenDays = 7 * 24 * 60 * 60 * 1000;
  saveCheckpoint('long-workflow', 1, data, sevenDays);
}

/**
 * Example: Periodic cleanup
 *
 * You might want to periodically clean up expired checkpoints
 * to keep localStorage tidy.
 */
export function periodicCleanupExample(): void {
  // Run cleanup on app initialization
  cleanupExpiredCheckpoints();

  // Or run periodically (e.g., every hour)
  setInterval(() => {
    cleanupExpiredCheckpoints();
  }, 60 * 60 * 1000); // 1 hour
}
