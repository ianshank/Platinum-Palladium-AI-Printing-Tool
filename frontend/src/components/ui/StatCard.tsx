/**
 * StatCard — Reusable metric display card
 *
 * Used on the Dashboard and potentially other summary views.
 * Supports icon, label, value, and optional subtitle.
 * Accessible with proper semantics.
 */

import { type FC, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface StatCardProps {
  /** Display label for the metric */
  label: string;
  /** Primary value to display */
  value: string | number;
  /** Optional subtitle or trend indicator */
  subtitle?: string;
  /** Optional icon element */
  icon?: ReactNode;
  /** Additional CSS class names */
  className?: string;
  /** Test identifier */
  'data-testid'?: string;
}

/**
 * A metric card showing a labeled value with optional icon and subtitle.
 */
export const StatCard: FC<StatCardProps> = ({
  label,
  value,
  subtitle,
  icon,
  className,
  'data-testid': testId,
}) => (
  <div
    className={cn(
      'rounded-lg border bg-card p-6 shadow-sm transition-shadow hover:shadow-md',
      className
    )}
    data-testid={testId ?? 'stat-card'}
    role="group"
    aria-label={`${label}: ${value}`}
  >
    <div className="flex items-center justify-between">
      <p className="text-sm font-medium text-muted-foreground">{label}</p>
      {icon && (
        <span className="text-muted-foreground" aria-hidden="true">
          {icon}
        </span>
      )}
    </div>
    <p className="mt-2 text-3xl font-bold tracking-tight">{value}</p>
    {subtitle && (
      <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
    )}
  </div>
);

StatCard.displayName = 'StatCard';
