/**
 * Dashboard component unit tests.
 */

import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@tests/utils/render';
import { Dashboard } from '@/pages/Dashboard';

// Mock zustand store
vi.mock('@/store', () => ({
  useUIStore: vi.fn(() => ({
    addToast: vi.fn(),
  })),
}));

describe('Dashboard', () => {
  it('renders the page title', () => {
    render(<Dashboard />);
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('renders the welcome message', () => {
    render(<Dashboard />);
    expect(
      screen.getByText(/Welcome to your platinum\/palladium calibration studio/)
    ).toBeInTheDocument();
  });

  it('renders all stat cards', () => {
    render(<Dashboard />);
    expect(screen.getByText('Total Prints')).toBeInTheDocument();
    expect(screen.getByText('Active Curves')).toBeInTheDocument();
    expect(screen.getByText('This Month')).toBeInTheDocument();
    expect(screen.getByText('Avg Rating')).toBeInTheDocument();
  });

  it('renders quick action buttons', () => {
    render(<Dashboard />);
    expect(screen.getByText('New Calibration')).toBeInTheDocument();
    expect(screen.getByText('Calculate Chemistry')).toBeInTheDocument();
    expect(screen.getByText('Ask AI Assistant')).toBeInTheDocument();
    expect(screen.getByText('Log Session')).toBeInTheDocument();
  });

  it('renders quick actions section title', () => {
    render(<Dashboard />);
    expect(screen.getByText('Quick Actions')).toBeInTheDocument();
  });

  it('renders getting started tips section', () => {
    render(<Dashboard />);
    expect(screen.getByText('Getting Started')).toBeInTheDocument();
  });

  it('renders all tip cards', () => {
    render(<Dashboard />);
    expect(screen.getByText('Start with Calibration')).toBeInTheDocument();
    expect(screen.getByText('Log Your Sessions')).toBeInTheDocument();
    expect(screen.getByText('Ask the AI')).toBeInTheDocument();
  });

  it('renders tip descriptions', () => {
    render(<Dashboard />);
    expect(
      screen.getByText(/Create a calibration curve to ensure accurate tonal reproduction/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Keep track of your print sessions to refine your process/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Get personalized recommendations and troubleshoot issues/)
    ).toBeInTheDocument();
  });

  it('renders navigation links with correct hrefs', () => {
    render(<Dashboard />);

    const calibrationLink = screen.getByRole('link', { name: /New Calibration/i });
    expect(calibrationLink).toHaveAttribute('href', '/calibration');

    const chemistryLink = screen.getByRole('link', { name: /Calculate Chemistry/i });
    expect(chemistryLink).toHaveAttribute('href', '/chemistry');

    const aiLink = screen.getByRole('link', { name: /Ask AI Assistant/i });
    expect(aiLink).toHaveAttribute('href', '/assistant');

    const sessionLink = screen.getByRole('link', { name: /Log Session/i });
    expect(sessionLink).toHaveAttribute('href', '/sessions');
  });
});
