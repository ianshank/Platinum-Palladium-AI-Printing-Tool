import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { logger } from '@/lib/logger';
import { cn } from '@/lib/utils';

/**
 * Props for the ErrorBoundary component.
 */
interface ErrorBoundaryProps {
  /** Child components to render when no error. */
  children: ReactNode;
  /** Optional custom fallback UI. Receives the error and a reset function. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
  /** Optional handler called when an error is caught. */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Catches unhandled React render errors and displays a fallback UI.
 *
 * Wraps child components and prevents the entire app from crashing.
 * Provides "Retry" and "Go Home" recovery actions.
 *
 * @example
 * ```tsx
 * <ErrorBoundary>
 *   <App />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  static displayName = 'ErrorBoundary';

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    logger.error('ErrorBoundary: Uncaught error', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
    });
    this.props.onError?.(error, errorInfo);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  handleGoHome = (): void => {
    window.location.href = '/';
  };

  override render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, fallback } = this.props;

    if (hasError && error) {
      // Custom fallback
      if (fallback) {
        return fallback(error, this.handleReset);
      }

      // Default fallback UI
      return (
        <div
          className={cn(
            'flex min-h-[400px] flex-col items-center justify-center p-8',
            'text-center'
          )}
          role="alert"
          data-testid="error-boundary-fallback"
        >
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10">
            <AlertTriangle className="h-8 w-8 text-destructive" />
          </div>

          <h2 className="mb-2 text-xl font-semibold text-foreground">
            Something went wrong
          </h2>

          <p className="mb-6 max-w-md text-sm text-muted-foreground">
            An unexpected error occurred. You can try again or return to the
            dashboard.
          </p>

          {/* Error details (collapsed by default) */}
          <details className="mb-6 w-full max-w-lg text-left">
            <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              Show error details
            </summary>
            <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-muted p-3 text-xs text-muted-foreground">
              {error.message}
              {error.stack && `\n\n${error.stack}`}
            </pre>
          </details>

          {/* Recovery actions */}
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={this.handleGoHome}
              data-testid="error-boundary-home"
            >
              <Home className="mr-2 h-4 w-4" />
              Go Home
            </Button>
            <Button
              onClick={this.handleReset}
              data-testid="error-boundary-retry"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Try Again
            </Button>
          </div>
        </div>
      );
    }

    return children;
  }
}
