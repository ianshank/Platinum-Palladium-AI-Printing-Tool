/**
 * Settings component unit tests.
 */

import { describe, it, expect, vi } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { render } from '@tests/utils/render';
import { Settings } from '@/pages/Settings';

// Mock store values
const mockSetTheme = vi.fn();
const mockSetSidebarCollapsed = vi.fn();

vi.mock('@/store', () => ({
  useUIStore: vi.fn((selector) => {
    const state = {
      theme: 'dark',
      setTheme: mockSetTheme,
      sidebarCollapsed: false,
      setSidebarCollapsed: mockSetSidebarCollapsed,
    };
    return selector(state);
  }),
}));

describe('Settings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the page title', () => {
    render(<Settings />);
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders Appearance section', () => {
    render(<Settings />);
    expect(screen.getByText('Appearance')).toBeInTheDocument();
  });

  it('renders About section', () => {
    render(<Settings />);
    expect(screen.getByText('About')).toBeInTheDocument();
  });

  it('renders theme selector with current value', () => {
    render(<Settings />);
    const themeSelect = screen.getByRole('combobox');
    expect(themeSelect).toHaveValue('dark');
  });

  it('renders all theme options', () => {
    render(<Settings />);
    expect(screen.getByRole('option', { name: 'Dark' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Light' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'System' })).toBeInTheDocument();
  });

  it('calls setTheme when theme is changed', () => {
    render(<Settings />);
    const themeSelect = screen.getByRole('combobox');
    fireEvent.change(themeSelect, { target: { value: 'light' } });
    expect(mockSetTheme).toHaveBeenCalledWith('light');
  });

  it('renders compact sidebar toggle', () => {
    render(<Settings />);
    expect(screen.getByText('Compact Sidebar')).toBeInTheDocument();
    expect(screen.getByText('Show collapsed sidebar by default')).toBeInTheDocument();
  });

  it('renders toggle button with correct aria-label', () => {
    render(<Settings />);
    const toggle = screen.getByRole('button', { name: /Toggle compact sidebar/i });
    expect(toggle).toBeInTheDocument();
  });

  it('calls setSidebarCollapsed when toggle is clicked', () => {
    render(<Settings />);
    const toggle = screen.getByRole('button', { name: /Toggle compact sidebar/i });
    fireEvent.click(toggle);
    expect(mockSetSidebarCollapsed).toHaveBeenCalledWith(true);
  });

  it('renders about information', () => {
    render(<Settings />);
    expect(screen.getByText('PTPD Calibration Studio')).toBeInTheDocument();
    expect(
      screen.getByText(/AI-powered calibration system for platinum\/palladium printing/)
    ).toBeInTheDocument();
  });

  it('renders technology stack info', () => {
    render(<Settings />);
    expect(screen.getByText(/React, TypeScript, and FastAPI/)).toBeInTheDocument();
  });
});
