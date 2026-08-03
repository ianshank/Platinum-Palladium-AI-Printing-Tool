import React from 'react';
import { cn } from '@/lib/utils';

export interface WizardContainerProps {
  title?: string;
  description?: string;
  className?: string;
  children?: React.ReactNode;
}

export const WizardContainer = React.forwardRef<HTMLDivElement, WizardContainerProps>(
  ({ title, description, className, children }, ref) => (
    <div
      ref={ref}
      className={cn(
        'space-y-6 rounded-lg border bg-card p-6 text-card-foreground shadow-sm',
        className
      )}
    >
      {title && (
        <div className="space-y-2">
          <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
          {description && <p className="text-muted-foreground">{description}</p>}
        </div>
      )}
      <div className="space-y-4">{children}</div>
    </div>
  )
);
WizardContainer.displayName = 'WizardContainer';

export interface MetricLabelProps {
  label: string;
  required?: boolean;
  className?: string;
}

export const MetricLabel = React.forwardRef<HTMLDivElement, MetricLabelProps>(
  ({ label, required, className }, ref) => (
    <div ref={ref} className={cn('flex items-center justify-between text-sm', className)}>
      <label className="font-medium text-foreground">
        {label}
        {required && <span className="ml-1 text-destructive">*</span>}
      </label>
    </div>
  )
);
MetricLabel.displayName = 'MetricLabel';

export interface MetricValueProps {
  value: string | number;
  unit?: string;
  level?: 'normal' | 'highlight' | 'warning' | 'error';
  className?: string;
}

export const MetricValue = React.forwardRef<HTMLDivElement, MetricValueProps>(
  ({ value, unit, level = 'normal', className }, ref) => {
    const levelClass = {
      normal: 'text-foreground',
      highlight: 'text-primary font-semibold',
      warning: 'text-yellow-600 font-semibold',
      error: 'text-destructive font-semibold',
    };

    return (
      <div ref={ref} className={cn('text-sm font-mono', levelClass[level], className)}>
        <span className="text-lg">{value}</span>
        {unit && <span className="ml-1 text-muted-foreground">{unit}</span>}
      </div>
    );
  }
);
MetricValue.displayName = 'MetricValue';

export interface ChartContainerProps {
  title?: string;
  height?: number;
  className?: string;
  children?: React.ReactNode;
}

export const ChartContainer = React.forwardRef<HTMLDivElement, ChartContainerProps>(
  ({ title, height = 300, className, children }, ref) => (
    <div ref={ref} className={cn('space-y-3 rounded-lg border bg-muted/50 p-4', className)}>
      {title && <h3 className="text-sm font-semibold text-foreground">{title}</h3>}
      <div
        className="w-full rounded-md border bg-white/5"
        style={{ height: `${height}px` }}
      >
        {children}
      </div>
    </div>
  )
);
ChartContainer.displayName = 'ChartContainer';

export interface StepFooterProps {
  showPrevious?: boolean;
  showNext?: boolean;
  onPrevious?: () => void;
  onNext?: () => void;
  nextDisabled?: boolean;
  nextLabel?: string;
  previousLabel?: string;
  className?: string;
}

export const StepFooter = React.forwardRef<HTMLDivElement, StepFooterProps>(
  (
    {
      showPrevious = true,
      showNext = true,
      onPrevious,
      onNext,
      nextDisabled = false,
      nextLabel = 'Next',
      previousLabel = 'Previous',
      className,
    },
    ref
  ) => {
    return (
      <div
        ref={ref}
        className={cn('flex items-center justify-between border-t pt-6', className)}
      >
        <div>
          {showPrevious && (
            <button
              onClick={onPrevious}
              className="inline-flex h-10 items-center justify-center rounded-md border border-input bg-background px-4 py-2 text-sm font-medium ring-offset-background transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              type="button"
            >
              {previousLabel}
            </button>
          )}
        </div>
        <div>
          {showNext && (
            <button
              onClick={onNext}
              disabled={nextDisabled}
              className="inline-flex h-10 items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
              type="button"
            >
              {nextLabel}
            </button>
          )}
        </div>
      </div>
    );
  }
);
StepFooter.displayName = 'StepFooter';
