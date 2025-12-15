/**
 * NotFound component unit tests.
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@tests/utils/render';
import { NotFound } from '@/pages/NotFound';

describe('NotFound', () => {
  it('renders 404 error code', () => {
    render(<NotFound />);
    expect(screen.getByText('404')).toBeInTheDocument();
  });

  it('renders page not found title', () => {
    render(<NotFound />);
    expect(screen.getByText('Page Not Found')).toBeInTheDocument();
  });

  it('renders description text', () => {
    render(<NotFound />);
    expect(
      screen.getByText(/The page you are looking for does not exist or has been moved/)
    ).toBeInTheDocument();
  });

  it('renders link to dashboard', () => {
    render(<NotFound />);
    const link = screen.getByRole('link', { name: /Return to Dashboard/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '/dashboard');
  });
});
