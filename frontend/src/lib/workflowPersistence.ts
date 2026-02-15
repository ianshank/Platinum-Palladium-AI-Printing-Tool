/**
 * Workflow persistence utility
 *
 * Provides localStorage-based persistence for multi-step workflows
 * with automatic cleanup on completion.
 *
 * Uses config-driven storage keys and TTL values.
 */

import { logger } from '@/lib/logger';

const STORAGE_PREFIX = 'ptpd-workflow';
const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

export interface WorkflowCheckpoint<T> {
  id: string;
  step: number;
  data: T;
  savedAt: string; // ISO timestamp
  expiresAt: string; // ISO timestamp
}

/**
 * Save a workflow checkpoint to localStorage
 */
export function saveCheckpoint<T>(
  workflowId: string,
  step: number,
  data: T,
  ttlMs: number = DEFAULT_TTL_MS
): void {
  const key = `${STORAGE_PREFIX}:${workflowId}`;
  const now = new Date();

  const checkpoint: WorkflowCheckpoint<T> = {
    id: workflowId,
    step,
    data,
    savedAt: now.toISOString(),
    expiresAt: new Date(now.getTime() + ttlMs).toISOString(),
  };

  try {
    localStorage.setItem(key, JSON.stringify(checkpoint));
    logger.debug('Workflow checkpoint saved', { workflowId, step });
  } catch (err) {
    logger.warn('Failed to save workflow checkpoint', {
      workflowId,
      error: String(err),
    });
  }
}

/**
 * Load a workflow checkpoint from localStorage
 * Returns null if expired or not found
 */
export function loadCheckpoint<T>(
  workflowId: string
): WorkflowCheckpoint<T> | null {
  const key = `${STORAGE_PREFIX}:${workflowId}`;

  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;

    const checkpoint = JSON.parse(raw) as WorkflowCheckpoint<T>;

    // Check expiration
    if (new Date(checkpoint.expiresAt) < new Date()) {
      logger.debug('Workflow checkpoint expired', { workflowId });
      localStorage.removeItem(key);
      return null;
    }

    logger.debug('Workflow checkpoint loaded', {
      workflowId,
      step: checkpoint.step,
    });
    return checkpoint;
  } catch (err) {
    logger.warn('Failed to load workflow checkpoint', {
      workflowId,
      error: String(err),
    });
    return null;
  }
}

/**
 * Clear a workflow checkpoint
 */
export function clearCheckpoint(workflowId: string): void {
  const key = `${STORAGE_PREFIX}:${workflowId}`;
  localStorage.removeItem(key);
  logger.debug('Workflow checkpoint cleared', { workflowId });
}

/**
 * Clear all expired workflow checkpoints
 */
export function cleanupExpiredCheckpoints(): void {
  const now = new Date();
  const keysToRemove: string[] = [];

  // First pass: identify keys to remove
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key?.startsWith(STORAGE_PREFIX)) continue;

    try {
      const raw = localStorage.getItem(key);
      if (!raw) continue;

      const checkpoint = JSON.parse(raw) as WorkflowCheckpoint<unknown>;
      if (new Date(checkpoint.expiresAt) < now) {
        keysToRemove.push(key);
      }
    } catch {
      // Mark corrupted entries for removal
      if (key) keysToRemove.push(key);
    }
  }

  // Second pass: remove identified keys
  keysToRemove.forEach((key) => localStorage.removeItem(key));

  if (keysToRemove.length > 0) {
    logger.debug('Cleaned up expired workflow checkpoints', {
      count: keysToRemove.length,
    });
  }
}
