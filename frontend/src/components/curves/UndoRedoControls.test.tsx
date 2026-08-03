import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { UndoRedoControls } from './UndoRedoControls';

describe('UndoRedoControls', () => {
  it('renders undo, redo, and reset buttons', () => {
    const handlers = {
      onUndo: vi.fn(),
      onRedo: vi.fn(),
      onReset: vi.fn(),
    };

    render(
      <UndoRedoControls
        canUndo={true}
        canRedo={true}
        {...handlers}
      />
    );

    expect(screen.getByRole('button', { name: /undo/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /redo/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument();
  });

  it('disables undo button when canUndo is false', () => {
    render(
      <UndoRedoControls
        canUndo={false}
        canRedo={true}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        onReset={vi.fn()}
      />
    );

    const undoButton = screen.getByRole('button', { name: /undo/i });
    expect(undoButton).toBeDisabled();
  });

  it('disables redo button when canRedo is false', () => {
    render(
      <UndoRedoControls
        canUndo={true}
        canRedo={false}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        onReset={vi.fn()}
      />
    );

    const redoButton = screen.getByRole('button', { name: /redo/i });
    expect(redoButton).toBeDisabled();
  });

  it('calls onUndo when undo button is clicked', () => {
    const onUndo = vi.fn();
    render(
      <UndoRedoControls
        canUndo={true}
        canRedo={true}
        onUndo={onUndo}
        onRedo={vi.fn()}
        onReset={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /undo/i }));
    expect(onUndo).toHaveBeenCalledOnce();
  });

  it('calls onRedo when redo button is clicked', () => {
    const onRedo = vi.fn();
    render(
      <UndoRedoControls
        canUndo={true}
        canRedo={true}
        onUndo={vi.fn()}
        onRedo={onRedo}
        onReset={vi.fn()}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /redo/i }));
    expect(onRedo).toHaveBeenCalledOnce();
  });

  it('calls onReset when reset button is clicked', () => {
    const onReset = vi.fn();
    render(
      <UndoRedoControls
        canUndo={true}
        canRedo={true}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        onReset={onReset}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /reset/i }));
    expect(onReset).toHaveBeenCalledOnce();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <UndoRedoControls
        ref={ref}
        canUndo={true}
        canRedo={true}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        onReset={vi.fn()}
      />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('applies custom className', () => {
    const { container } = render(
      <UndoRedoControls
        canUndo={true}
        canRedo={true}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        onReset={vi.fn()}
        className="custom-class"
      />
    );

    const div = container.firstChild as HTMLElement;
    expect(div).toHaveClass('custom-class');
  });
});
