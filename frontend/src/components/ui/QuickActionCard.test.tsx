/**
 * QuickActionCard Tests
 *
 * Covers rendering, click handling, accessibility, and disabled state.
 */

import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { QuickActionCard } from './QuickActionCard';

describe('QuickActionCard', () => {
    const defaultProps = {
        title: 'New Calibration',
        description: 'Start a new calibration workflow.',
        onClick: vi.fn(),
    };

    describe('Rendering', () => {
        it('renders title and description', () => {
            render(<QuickActionCard {...defaultProps} />);

            expect(screen.getByText('New Calibration')).toBeInTheDocument();
            expect(
                screen.getByText('Start a new calibration workflow.')
            ).toBeInTheDocument();
        });

        it('renders icon when provided', () => {
            render(
                <QuickActionCard
                    {...defaultProps}
                    icon={<span data-testid="action-icon">➕</span>}
                />
            );
            expect(screen.getByTestId('action-icon')).toBeInTheDocument();
        });
    });

    describe('Interactions', () => {
        it('calls onClick when clicked', () => {
            const onClick = vi.fn();
            render(<QuickActionCard {...defaultProps} onClick={onClick} />);

            fireEvent.click(screen.getByRole('button'));
            expect(onClick).toHaveBeenCalledTimes(1);
        });

        it('does not call onClick when disabled', () => {
            const onClick = vi.fn();
            render(
                <QuickActionCard {...defaultProps} onClick={onClick} disabled />
            );

            fireEvent.click(screen.getByRole('button'));
            expect(onClick).not.toHaveBeenCalled();
        });
    });

    describe('Disabled State', () => {
        it('applies disabled attribute', () => {
            render(<QuickActionCard {...defaultProps} disabled />);
            expect(screen.getByRole('button')).toBeDisabled();
        });

        it('applies disabled styling', () => {
            render(<QuickActionCard {...defaultProps} disabled />);
            expect(screen.getByRole('button')).toHaveClass('disabled:opacity-50');
        });
    });

    describe('Accessibility', () => {
        it('has aria-label matching title', () => {
            render(<QuickActionCard {...defaultProps} />);
            expect(screen.getByRole('button')).toHaveAttribute(
                'aria-label',
                'New Calibration'
            );
        });

        it('renders as a button element', () => {
            render(<QuickActionCard {...defaultProps} />);
            expect(screen.getByRole('button')).toBeInTheDocument();
        });

        it('has type=button to prevent form submission', () => {
            render(<QuickActionCard {...defaultProps} />);
            expect(screen.getByRole('button')).toHaveAttribute('type', 'button');
        });

        it('hides icon from screen readers', () => {
            render(
                <QuickActionCard
                    {...defaultProps}
                    icon={<span>🎯</span>}
                />
            );
            const iconWrapper = screen.getByText('🎯').parentElement;
            expect(iconWrapper).toHaveAttribute('aria-hidden', 'true');
        });
    });

    describe('Customization', () => {
        it('applies custom className', () => {
            render(
                <QuickActionCard {...defaultProps} className="custom-class" />
            );
            expect(screen.getByTestId('quick-action-card')).toHaveClass(
                'custom-class'
            );
        });

        it('uses custom data-testid', () => {
            render(
                <QuickActionCard {...defaultProps} data-testid="my-action" />
            );
            expect(screen.getByTestId('my-action')).toBeInTheDocument();
        });
    });
});
