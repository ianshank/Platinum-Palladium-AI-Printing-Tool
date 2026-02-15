/**
 * SessionLogPage Tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { SessionLogPage } from './SessionLogPage';

expect.extend(toHaveNoViolations);

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

  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(<SessionLogPage />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
