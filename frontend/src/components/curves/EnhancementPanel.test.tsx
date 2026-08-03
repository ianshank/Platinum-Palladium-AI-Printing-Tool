import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { type EnhancementGoal, EnhancementPanel } from './EnhancementPanel';

describe('EnhancementPanel', () => {
  const defaultProps = {
    enhancementGoal: 'linearization' as EnhancementGoal,
    onGoalChange: vi.fn(),
    isEnhancing: false,
    onEnhance: vi.fn(),
    onDismissResult: vi.fn(),
    showResult: false,
  };

  it('renders enhancement goal select', () => {
    render(<EnhancementPanel {...defaultProps} />);

    expect(screen.getByLabelText(/enhancement goal/i)).toBeInTheDocument();
  });

  it('renders AI Enhance button', () => {
    render(<EnhancementPanel {...defaultProps} />);

    expect(screen.getByTestId('ai-enhance-btn')).toBeInTheDocument();
  });

  it('calls onGoalChange when goal select changes', () => {
    const onGoalChange = vi.fn();
    render(
      <EnhancementPanel
        {...defaultProps}
        onGoalChange={onGoalChange}
      />
    );

    const select = screen.getByLabelText(/enhancement goal/i);
    fireEvent.change(select, { target: { value: 'maximize_range' } });

    expect(onGoalChange).toHaveBeenCalledWith('maximize_range');
  });

  it('calls onEnhance when enhance button is clicked', () => {
    const onEnhance = vi.fn();
    render(
      <EnhancementPanel
        {...defaultProps}
        onEnhance={onEnhance}
      />
    );

    fireEvent.click(screen.getByTestId('ai-enhance-btn'));

    expect(onEnhance).toHaveBeenCalledOnce();
  });

  it('disables enhance button when isEnhancing is true', () => {
    render(
      <EnhancementPanel
        {...defaultProps}
        isEnhancing={true}
      />
    );

    expect(screen.getByTestId('ai-enhance-btn')).toBeDisabled();
  });

  it('displays enhancement result when showResult is true', () => {
    const enhanceResult = {
      success: true,
      curve_id: 'test-id',
      name: 'Enhanced Curve',
      goal: 'linearization',
      confidence: 0.95,
      analysis: 'Curve looks good',
      changes_made: ['Adjusted highlights', 'Enhanced shadows'],
      input_values: [0, 128, 255],
      output_values: [0, 140, 255],
    };

    render(
      <EnhancementPanel
        {...defaultProps}
        showResult={true}
        enhanceResult={enhanceResult}
      />
    );

    expect(screen.getByTestId('enhance-result')).toBeInTheDocument();
    expect(screen.getByText('Confidence: 95%')).toBeInTheDocument();
    expect(screen.getByText('Curve looks good')).toBeInTheDocument();
  });

  it('displays changes made in enhancement result', () => {
    const enhanceResult = {
      success: true,
      curve_id: 'test-id',
      name: 'Enhanced Curve',
      goal: 'linearization',
      confidence: 0.95,
      analysis: 'Analysis text',
      changes_made: ['Change 1', 'Change 2'],
      input_values: [0, 128, 255],
      output_values: [0, 140, 255],
    };

    render(
      <EnhancementPanel
        {...defaultProps}
        showResult={true}
        enhanceResult={enhanceResult}
      />
    );

    expect(screen.getByText('Change 1')).toBeInTheDocument();
    expect(screen.getByText('Change 2')).toBeInTheDocument();
  });

  it('calls onDismissResult when dismiss button is clicked', () => {
    const enhanceResult = {
      success: true,
      curve_id: 'test-id',
      name: 'Enhanced Curve',
      goal: 'linearization',
      confidence: 0.95,
      analysis: 'Analysis',
      changes_made: [],
      input_values: [],
      output_values: [],
    };

    const onDismissResult = vi.fn();
    render(
      <EnhancementPanel
        {...defaultProps}
        showResult={true}
        enhanceResult={enhanceResult}
        onDismissResult={onDismissResult}
      />
    );

    fireEvent.click(screen.getByTestId('enhance-dismiss-btn'));

    expect(onDismissResult).toHaveBeenCalledOnce();
  });

  it('does not display result when showResult is false', () => {
    render(<EnhancementPanel {...defaultProps} showResult={false} />);

    expect(screen.queryByTestId('enhance-result')).not.toBeInTheDocument();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <EnhancementPanel
        {...defaultProps}
        ref={ref}
      />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('applies custom className', () => {
    const { container } = render(
      <EnhancementPanel
        {...defaultProps}
        className="custom-class"
      />
    );

    const div = container.firstChild as HTMLElement;
    expect(div).toHaveClass('custom-class');
  });

  it('includes all enhancement goal options', () => {
    render(<EnhancementPanel {...defaultProps} />);

    const select = screen.getByLabelText(/enhancement goal/i);
    const options = Array.from(select.options).map(opt => opt.value);

    expect(options).toContain('linearization');
    expect(options).toContain('maximize_range');
    expect(options).toContain('smooth_gradation');
    expect(options).toContain('highlight_detail');
    expect(options).toContain('shadow_detail');
    expect(options).toContain('neutral_midtones');
    expect(options).toContain('print_stability');
  });
});
