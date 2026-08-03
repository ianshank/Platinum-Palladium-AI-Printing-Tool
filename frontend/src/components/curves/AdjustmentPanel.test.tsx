import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AdjustmentPanel, type AdjustmentType } from './AdjustmentPanel';

describe('AdjustmentPanel', () => {
  const defaultProps = {
    adjustmentType: 'contrast' as AdjustmentType,
    onAdjustmentTypeChange: vi.fn(),
    amount: 25,
    onAmountChange: vi.fn(),
    isApplying: false,
    onApply: vi.fn(),
  };

  it('renders adjustment type select', () => {
    render(<AdjustmentPanel {...defaultProps} />);

    expect(screen.getByLabelText(/adjustment type/i)).toBeInTheDocument();
  });

  it('renders amount slider', () => {
    render(<AdjustmentPanel {...defaultProps} />);

    expect(screen.getByLabelText(/adjustment amount/i)).toBeInTheDocument();
  });

  it('renders apply button', () => {
    render(<AdjustmentPanel {...defaultProps} />);

    expect(screen.getByRole('button', { name: /apply adjustment/i })).toBeInTheDocument();
  });

  it('displays current amount value', () => {
    render(<AdjustmentPanel {...defaultProps} amount={50} />);

    expect(screen.getByText('50')).toBeInTheDocument();
  });

  it('calls onAdjustmentTypeChange when select changes', () => {
    const onAdjustmentTypeChange = vi.fn();
    render(
      <AdjustmentPanel
        {...defaultProps}
        onAdjustmentTypeChange={onAdjustmentTypeChange}
      />
    );

    const select = screen.getByLabelText(/adjustment type/i);
    fireEvent.change(select, { target: { value: 'brightness' } });

    expect(onAdjustmentTypeChange).toHaveBeenCalledWith('brightness');
  });

  it('calls onApply when apply button is clicked', () => {
    const onApply = vi.fn();
    render(
      <AdjustmentPanel
        {...defaultProps}
        onApply={onApply}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /apply adjustment/i }));

    expect(onApply).toHaveBeenCalledOnce();
  });

  it('disables apply button when isApplying is true', () => {
    render(
      <AdjustmentPanel
        {...defaultProps}
        isApplying={true}
      />
    );

    expect(screen.getByRole('button', { name: /apply adjustment/i })).toBeDisabled();
  });

  it('displays error message when provided', () => {
    render(
      <AdjustmentPanel
        {...defaultProps}
        error="Something went wrong"
      />
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('does not display error message by default', () => {
    render(<AdjustmentPanel {...defaultProps} />);

    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <AdjustmentPanel
        {...defaultProps}
        ref={ref}
      />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('applies custom className', () => {
    const { container } = render(
      <AdjustmentPanel
        {...defaultProps}
        className="custom-class"
      />
    );

    const div = container.firstChild as HTMLElement;
    expect(div).toHaveClass('custom-class');
  });

  it('includes adjustment type options', () => {
    render(<AdjustmentPanel {...defaultProps} />);

    const select = screen.getByLabelText(/adjustment type/i);
    const options = Array.from(select.options).map(opt => opt.value);

    expect(options).toContain('contrast');
    expect(options).toContain('brightness');
    expect(options).toContain('gamma');
    expect(options).toContain('sigmoid');
  });
});
