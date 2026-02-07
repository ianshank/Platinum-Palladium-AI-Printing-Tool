/**
 * SessionLogPage Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SessionLogPage } from './SessionLogPage';

vi.mock('@/components/session/SessionLog', () => ({
    SessionLog: () => <div data-testid="mock-session-log">SessionLog</div>,
}));

describe('SessionLogPage', () => {
    it('renders the page container', () => {
        render(<SessionLogPage />);
        expect(screen.getByTestId('session-page')).toBeInTheDocument();
    });

    it('renders page heading', () => {
        render(<SessionLogPage />);
        expect(screen.getByText('Session Log')).toBeInTheDocument();
    });

    it('renders the SessionLog component', () => {
        render(<SessionLogPage />);
        expect(screen.getByTestId('mock-session-log')).toBeInTheDocument();
    });
});
