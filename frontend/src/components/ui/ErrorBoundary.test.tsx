import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary } from './ErrorBoundary';

// Suppress React error boundary console.error in tests
const originalConsoleError = console.error;
beforeEach(() => {
  console.error = vi.fn();
});

afterEach(() => {
  console.error = originalConsoleError;
});

// A component that throws on command
function ProblemChild({
  shouldThrow = true,
}: {
  shouldThrow?: boolean;
}): JSX.Element {
  if (shouldThrow) {
    throw new Error('Test error from ProblemChild');
  }
  return <div data-testid="child-content">All good</div>;
}

describe('ErrorBoundary', () => {
  describe('Normal rendering', () => {
    it('renders children when no error occurs', () => {
      render(
        <ErrorBoundary>
          <ProblemChild shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('child-content')).toBeInTheDocument();
      expect(screen.getByText('All good')).toBeInTheDocument();
    });

    it('does not show fallback when no error', () => {
      render(
        <ErrorBoundary>
          <ProblemChild shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(
        screen.queryByTestId('error-boundary-fallback')
      ).not.toBeInTheDocument();
    });
  });

  describe('Error handling', () => {
    it('shows default fallback UI when a child throws', () => {
      render(
        <ErrorBoundary>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
      expect(
        screen.getByText(/An unexpected error occurred/)
      ).toBeInTheDocument();
    });

    it('displays error message in details section', () => {
      render(
        <ErrorBoundary>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByText('Show error details')).toBeInTheDocument();
      expect(
        screen.getByText(/Test error from ProblemChild/)
      ).toBeInTheDocument();
    });

    it('shows retry and go-home buttons', () => {
      render(
        <ErrorBoundary>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('error-boundary-retry')).toBeInTheDocument();
      expect(screen.getByTestId('error-boundary-home')).toBeInTheDocument();
      expect(screen.getByText('Try Again')).toBeInTheDocument();
      expect(screen.getByText('Go Home')).toBeInTheDocument();
    });

    it('resets error state when retry is clicked', () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ProblemChild />
        </ErrorBoundary>
      );

      // Verify error state
      expect(screen.getByTestId('error-boundary-fallback')).toBeInTheDocument();

      // Re-render with non-throwing child, then click retry
      rerender(
        <ErrorBoundary>
          <ProblemChild shouldThrow={false} />
        </ErrorBoundary>
      );

      fireEvent.click(screen.getByTestId('error-boundary-retry'));

      // Should now show good content
      expect(
        screen.queryByTestId('error-boundary-fallback')
      ).not.toBeInTheDocument();
      expect(screen.getByTestId('child-content')).toBeInTheDocument();
    });

    it('calls onError callback when error is caught', () => {
      const onError = vi.fn();

      render(
        <ErrorBoundary onError={onError}>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(onError).toHaveBeenCalledOnce();
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: 'Test error from ProblemChild' }),
        expect.objectContaining({ componentStack: expect.any(String) })
      );
    });
  });

  describe('Custom fallback', () => {
    it('renders custom fallback when provided', () => {
      const customFallback = (error: Error, reset: () => void): JSX.Element => (
        <div data-testid="custom-fallback">
          <p>Custom: {error.message}</p>
          <button onClick={reset}>Custom Reset</button>
        </div>
      );

      render(
        <ErrorBoundary fallback={customFallback}>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();
      expect(
        screen.getByText(/Custom: Test error from ProblemChild/)
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId('error-boundary-fallback')
      ).not.toBeInTheDocument();
    });

    it('custom fallback reset function works', () => {
      const customFallback = (error: Error, reset: () => void): JSX.Element => (
        <div data-testid="custom-fallback">
          <p>{error.message}</p>
          <button data-testid="custom-reset" onClick={reset}>
            Reset
          </button>
        </div>
      );

      const { rerender } = render(
        <ErrorBoundary fallback={customFallback}>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByTestId('custom-fallback')).toBeInTheDocument();

      // Re-render with non-throwing, then reset
      rerender(
        <ErrorBoundary fallback={customFallback}>
          <ProblemChild shouldThrow={false} />
        </ErrorBoundary>
      );

      fireEvent.click(screen.getByTestId('custom-reset'));

      expect(screen.queryByTestId('custom-fallback')).not.toBeInTheDocument();
      expect(screen.getByTestId('child-content')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('fallback has role="alert"', () => {
      render(
        <ErrorBoundary>
          <ProblemChild />
        </ErrorBoundary>
      );

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
  });
});
