/**
 * StatCard Tests
 *
 * Covers rendering, accessibility, and prop variations.
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatCard } from './StatCard';

describe('StatCard', () => {
    describe('Rendering', () => {
        it('renders label and value', () => {
            render(<StatCard label="Total Records" value={42} />);

            expect(screen.getByText('Total Records')).toBeInTheDocument();
            expect(screen.getByText('42')).toBeInTheDocument();
        });

        it('renders string values', () => {
            render(<StatCard label="Range" value="10–50s" />);
            expect(screen.getByText('10–50s')).toBeInTheDocument();
        });

        it('renders subtitle when provided', () => {
            render(
                <StatCard label="Papers" value={3} subtitle="Hahnemühle, Arches, Bergger" />
            );
            expect(screen.getByText('Hahnemühle, Arches, Bergger')).toBeInTheDocument();
        });

        it('does not render subtitle when not provided', () => {
            const { container } = render(<StatCard label="Test" value={1} />);
            const subtitleElements = container.querySelectorAll('.text-xs');
            expect(subtitleElements.length).toBe(0);
        });

        it('renders icon when provided', () => {
            render(
                <StatCard
                    label="With Icon"
                    value={5}
                    icon={<span data-testid="test-icon">📊</span>}
                />
            );
            expect(screen.getByTestId('test-icon')).toBeInTheDocument();
        });
    });

    describe('Accessibility', () => {
        it('has aria-label combining label and value', () => {
            render(<StatCard label="Total" value={100} />);
            expect(screen.getByRole('group')).toHaveAttribute(
                'aria-label',
                'Total: 100'
            );
        });

        it('hides icon from screen readers', () => {
            render(
                <StatCard
                    label="Test"
                    value={1}
                    icon={<span>🔍</span>}
                />
            );
            const iconWrapper = screen.getByText('🔍').parentElement;
            expect(iconWrapper).toHaveAttribute('aria-hidden', 'true');
        });
    });

    describe('Customization', () => {
        it('applies custom className', () => {
            render(<StatCard label="Custom" value={0} className="bg-red-500" />);
            expect(screen.getByTestId('stat-card')).toHaveClass('bg-red-500');
        });

        it('uses custom data-testid when provided', () => {
            render(<StatCard label="Custom ID" value={7} data-testid="my-stat" />);
            expect(screen.getByTestId('my-stat')).toBeInTheDocument();
        });

        it('uses default data-testid when not provided', () => {
            render(<StatCard label="Default ID" value={0} />);
            expect(screen.getByTestId('stat-card')).toBeInTheDocument();
        });
    });
});
