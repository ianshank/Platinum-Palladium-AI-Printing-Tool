/**
 * Undo/Redo controls for curve editing
 * Provides buttons to navigate through edit history
 */

import React from 'react';
import { Button } from '@/components/ui/Button';
import { Redo2, RefreshCw, Undo2 } from 'lucide-react';
import { logger } from '@/lib/logger';

export interface UndoRedoControlsProps {
  /** Whether undo action is available */
  canUndo: boolean;
  /** Whether redo action is available */
  canRedo: boolean;
  /** Callback when undo is clicked */
  onUndo: () => void;
  /** Callback when redo is clicked */
  onRedo: () => void;
  /** Callback when reset is clicked */
  onReset: () => void;
  /** Optional CSS class */
  className?: string;
}

/**
 * Undo/Redo control buttons for curve editing
 *
 * @example
 * ```tsx
 * <UndoRedoControls
 *   canUndo={canUndo}
 *   canRedo={canRedo}
 *   onUndo={handleUndo}
 *   onRedo={handleRedo}
 *   onReset={handleReset}
 * />
 * ```
 */
export const UndoRedoControls = React.forwardRef<
  HTMLDivElement,
  UndoRedoControlsProps
>(({ canUndo, canRedo, onUndo, onRedo, onReset, className }, ref) => {
  const handleUndoClick = () => {
    logger.debug('UndoRedoControls: undo clicked');
    onUndo();
  };

  const handleRedoClick = () => {
    logger.debug('UndoRedoControls: redo clicked');
    onRedo();
  };

  const handleResetClick = () => {
    logger.info('UndoRedoControls: reset clicked');
    onReset();
  };

  return (
    <div ref={ref} className={className}>
      <div className="flex gap-2">
        <Button
          variant="outline"
          onClick={handleUndoClick}
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
          aria-disabled={!canUndo}
        >
          <Undo2 className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          onClick={handleRedoClick}
          disabled={!canRedo}
          title="Redo (Ctrl+Y / Ctrl+Shift+Z)"
          aria-label="Redo"
          aria-disabled={!canRedo}
        >
          <Redo2 className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          onClick={handleResetClick}
          title="Reset curve to initial state"
          aria-label="Reset"
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Reset
        </Button>
      </div>
    </div>
  );
});

UndoRedoControls.displayName = 'UndoRedoControls';
