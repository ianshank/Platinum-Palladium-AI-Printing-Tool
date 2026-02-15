/**
 * SettingsPage Tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';
import { SettingsPage } from './SettingsPage';

expect.extend(toHaveNoViolations);

vi.mock('@/components/settings/Settings', () => ({
  Settings: () => <div data-testid="mock-settings">Settings</div>,
}));

describe('SettingsPage', () => {
  it('renders the page container', () => {
    render(<SettingsPage />);
    expect(screen.getByTestId('settings-page')).toBeInTheDocument();
  });

  it('renders page heading', () => {
    render(<SettingsPage />);
    expect(
      screen.getByRole('heading', { name: 'Settings' })
    ).toBeInTheDocument();
  });

  it('renders the Settings component', () => {
    render(<SettingsPage />);
    expect(screen.getByTestId('mock-settings')).toBeInTheDocument();
  });

  describe('Accessibility', () => {
    it('has no accessibility violations', async () => {
      const { container } = render(<SettingsPage />);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
