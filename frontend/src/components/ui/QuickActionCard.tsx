/**
 * QuickActionCard — A clickable action card for common workflows.
 *
 * Reusable across Dashboard and other pages that offer shortcut actions.
 * Supports label, description, icon, and navigation via callback.
 */

import { type FC, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface QuickActionCardProps {
    /** Action title */
    title: string;
    /** Brief description of the action */
    description: string;
    /** Optional icon element */
    icon?: ReactNode;
    /** Click handler (typically navigates or opens a modal) */
    onClick: () => void;
    /** Whether the action is disabled */
    disabled?: boolean;
    /** Additional CSS class names */
    className?: string;
    /** Test identifier */
    'data-testid'?: string;
}

/**
 * A clickable card that represents a quick action / shortcut.
 */
export const QuickActionCard: FC<QuickActionCardProps> = ({
    title,
    description,
    icon,
    onClick,
    disabled = false,
    className,
    'data-testid': testId,
}) => (
    <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className={cn(
            'group w-full rounded-lg border bg-card p-4 text-left shadow-sm',
            'transition-all hover:border-primary hover:shadow-md',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            className
        )}
        data-testid={testId ?? 'quick-action-card'}
        aria-label={title}
    >
        <div className="flex items-start gap-3">
            {icon && (
                <span
                    className="mt-0.5 text-muted-foreground transition-colors group-hover:text-primary"
                    aria-hidden="true"
                >
                    {icon}
                </span>
            )}
            <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-foreground">{title}</h3>
                <p className="mt-1 text-xs text-muted-foreground">{description}</p>
            </div>
        </div>
    </button>
);

QuickActionCard.displayName = 'QuickActionCard';
