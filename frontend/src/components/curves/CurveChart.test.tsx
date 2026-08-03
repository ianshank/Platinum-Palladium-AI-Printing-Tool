import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { CurveChart } from './CurveChart';

describe('CurveChart', () => {
  const defaultProps = {
    inputValues: [0, 64, 128, 192, 255],
    outputValues: [0, 70, 128, 185, 255],
    curveName: 'Test Curve',
    maxValue: 255,
  };

  it('renders chart container with correct aria-label', () => {
    render(<CurveChart {...defaultProps} />);

    const chartContainer = screen.getByRole('img', {
      name: /curve chart for test curve/i,
    });
    expect(chartContainer).toBeInTheDocument();
  });

  it('renders with correct height when specified', () => {
    const { container } = render(
      <CurveChart {...defaultProps} height={500} />
    );

    const chartDiv = container.firstChild as HTMLElement;
    expect(chartDiv).toHaveStyle('height: 500px');
  });

  it('calls onChartClick when chart is clicked', () => {
    const onChartClick = vi.fn();
    render(
      <CurveChart {...defaultProps} onChartClick={onChartClick} />
    );

    const chartContainer = screen.getByRole('img');
    fireEvent.click(chartContainer);

    expect(onChartClick).toHaveBeenCalled();
  });

  it('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <CurveChart {...defaultProps} ref={ref} />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });

  it('applies custom className', () => {
    const { container } = render(
      <CurveChart
        {...defaultProps}
        className="custom-chart-class"
      />
    );

    const chartDiv = container.firstChild as HTMLElement;
    expect(chartDiv).toHaveClass('custom-chart-class');
  });

  it('has crosshair cursor for interactive feedback', () => {
    const { container } = render(
      <CurveChart {...defaultProps} />
    );

    const chartDiv = container.firstChild as HTMLElement;
    expect(chartDiv).toHaveClass('cursor-crosshair');
  });

  it('renders default height when not specified', () => {
    const { container } = render(
      <CurveChart {...defaultProps} />
    );

    const chartDiv = container.firstChild as HTMLElement;
    expect(chartDiv).toHaveClass('h-[400px]');
  });
});
