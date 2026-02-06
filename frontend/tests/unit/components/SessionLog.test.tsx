/**
 * SessionLog component unit tests.
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@tests/utils/render';
import { SessionLog } from '@/pages/SessionLog';

describe('SessionLog', () => {
  it('renders the page title', () => {
    render(<SessionLog />);
    expect(screen.getByText('Session Log')).toBeInTheDocument();
  });

  it('renders the page subtitle', () => {
    render(<SessionLog />);
    expect(
      screen.getByText(/Track your print sessions and results over time/)
    ).toBeInTheDocument();
  });

  it('renders empty state when no sessions', () => {
    render(<SessionLog />);
    expect(screen.getByText('No Sessions Yet')).toBeInTheDocument();
  });

  it('renders empty state description', () => {
    render(<SessionLog />);
    expect(
      screen.getByText(/Start logging your print sessions to track your progress/)
    ).toBeInTheDocument();
  });

  it('renders session icon in empty state', () => {
    render(<SessionLog />);
    // Check for SVG element
    const svg = document.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });
});
