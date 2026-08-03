import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import {
  ChartContainer,
  MetricLabel,
  MetricValue,
  StepFooter,
  WizardContainer,
} from './styles';

describe('Wizard Styled Components', () => {
  describe('WizardContainer', () => {
    it('renders children', () => {
      render(
        <WizardContainer>
          <p>Test content</p>
        </WizardContainer>
      );

      expect(screen.getByText('Test content')).toBeInTheDocument();
    });

    it('renders title and description', () => {
      render(
        <WizardContainer
          title="Step 1"
          description="Upload your image"
        >
          <p>Content</p>
        </WizardContainer>
      );

      expect(screen.getByText('Step 1')).toBeInTheDocument();
      expect(screen.getByText('Upload your image')).toBeInTheDocument();
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();
      render(
        <WizardContainer ref={ref}>
          <p>Content</p>
        </WizardContainer>
      );

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });

    it('applies custom className', () => {
      const { container } = render(
        <WizardContainer className="custom-class">
          <p>Content</p>
        </WizardContainer>
      );

      const div = container.firstChild as HTMLElement;
      expect(div).toHaveClass('custom-class');
    });
  });

  describe('MetricLabel', () => {
    it('renders label text', () => {
      render(<MetricLabel label="Exposure Time" />);

      expect(screen.getByText('Exposure Time')).toBeInTheDocument();
    });

    it('displays required indicator when required', () => {
      render(<MetricLabel label="Field" required />);

      const requiredIndicator = screen.getByText('*');
      expect(requiredIndicator).toBeInTheDocument();
      expect(requiredIndicator).toHaveClass('text-destructive');
    });

    it('does not show required indicator when not required', () => {
      const { container } = render(<MetricLabel label="Field" required={false} />);

      expect(container.textContent).not.toContain('*');
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();
      render(<MetricLabel label="Test" ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });

  describe('MetricValue', () => {
    it('renders numeric value', () => {
      render(<MetricValue value={42.5} />);

      expect(screen.getByText('42.5')).toBeInTheDocument();
    });

    it('displays unit when provided', () => {
      render(<MetricValue value={42} unit="seconds" />);

      expect(screen.getByText('seconds')).toBeInTheDocument();
    });

    it('applies correct level class for normal', () => {
      const { container } = render(
        <MetricValue value={42} level="normal" />
      );

      const div = container.firstChild as HTMLElement;
      expect(div).toHaveClass('text-foreground');
    });

    it('applies correct level class for highlight', () => {
      const { container } = render(
        <MetricValue value={42} level="highlight" />
      );

      const div = container.firstChild as HTMLElement;
      expect(div).toHaveClass('text-primary');
      expect(div).toHaveClass('font-semibold');
    });

    it('applies correct level class for warning', () => {
      const { container } = render(
        <MetricValue value={42} level="warning" />
      );

      const div = container.firstChild as HTMLElement;
      expect(div).toHaveClass('text-yellow-600');
      expect(div).toHaveClass('font-semibold');
    });

    it('applies correct level class for error', () => {
      const { container } = render(
        <MetricValue value={42} level="error" />
      );

      const div = container.firstChild as HTMLElement;
      expect(div).toHaveClass('text-destructive');
      expect(div).toHaveClass('font-semibold');
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();
      render(<MetricValue value={42} ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });

  describe('ChartContainer', () => {
    it('renders children', () => {
      render(
        <ChartContainer>
          <p>Chart content</p>
        </ChartContainer>
      );

      expect(screen.getByText('Chart content')).toBeInTheDocument();
    });

    it('renders title when provided', () => {
      render(
        <ChartContainer title="Test Chart">
          <p>Content</p>
        </ChartContainer>
      );

      expect(screen.getByText('Test Chart')).toBeInTheDocument();
    });

    it('applies custom height', () => {
      const { container } = render(
        <ChartContainer height={500}>
          <p>Content</p>
        </ChartContainer>
      );

      const inner = container.querySelector('[style*="height"]') as HTMLElement;
      expect(inner).toHaveStyle('height: 500px');
    });

    it('applies default height when not specified', () => {
      const { container } = render(
        <ChartContainer>
          <p>Content</p>
        </ChartContainer>
      );

      const inner = container.querySelector('[style*="height"]') as HTMLElement;
      expect(inner).toHaveStyle('height: 300px');
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();
      render(
        <ChartContainer ref={ref}>
          <p>Content</p>
        </ChartContainer>
      );

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });

  describe('StepFooter', () => {
    it('renders previous button when showPrevious is true', () => {
      render(<StepFooter showPrevious={true} showNext={false} />);

      expect(screen.getByRole('button', { name: /previous/i })).toBeInTheDocument();
    });

    it('renders next button when showNext is true', () => {
      render(<StepFooter showPrevious={false} showNext={true} />);

      expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
    });

    it('hides previous button when showPrevious is false', () => {
      render(<StepFooter showPrevious={false} showNext={true} />);

      expect(screen.queryByRole('button', { name: /previous/i })).not.toBeInTheDocument();
    });

    it('hides next button when showNext is false', () => {
      render(<StepFooter showPrevious={true} showNext={false} />);

      expect(screen.queryByRole('button', { name: /next/i })).not.toBeInTheDocument();
    });

    it('calls onPrevious when previous button is clicked', () => {
      const onPrevious = vi.fn();
      render(<StepFooter showPrevious={true} showNext={false} onPrevious={onPrevious} />);

      fireEvent.click(screen.getByRole('button', { name: /previous/i }));

      expect(onPrevious).toHaveBeenCalledOnce();
    });

    it('calls onNext when next button is clicked', () => {
      const onNext = vi.fn();
      render(<StepFooter showPrevious={false} showNext={true} onNext={onNext} />);

      fireEvent.click(screen.getByRole('button', { name: /next/i }));

      expect(onNext).toHaveBeenCalledOnce();
    });

    it('disables next button when nextDisabled is true', () => {
      render(<StepFooter showNext={true} nextDisabled={true} />);

      expect(screen.getByRole('button', { name: /next/i })).toBeDisabled();
    });

    it('uses custom button labels', () => {
      render(
        <StepFooter
          showPrevious={true}
          showNext={true}
          previousLabel="Back"
          nextLabel="Continue"
        />
      );

      expect(screen.getByRole('button', { name: 'Back' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Continue' })).toBeInTheDocument();
    });

    it('forwards ref correctly', () => {
      const ref = React.createRef<HTMLDivElement>();
      render(<StepFooter ref={ref} />);

      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });
});
