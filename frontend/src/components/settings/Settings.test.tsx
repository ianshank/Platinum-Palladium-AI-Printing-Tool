/**
 * Settings Component Tests
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { Settings } from './Settings';

vi.mock('@/lib/logger', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock('@/config', () => ({
  config: {
    api: { baseUrl: 'http://localhost:8000', staleTime: 30000 },
    calibration: { defaultSteps: 21 },
    features: { devtools: true },
    ui: { defaultTab: 'dashboard', toastDuration: 4000 },
  },
}));

const mockUIState = {
  theme: 'dark' as 'light' | 'dark',
  sidebarOpen: true,
  activeTab: 'dashboard',
  isProcessing: false,
  isInitialized: true,
  toasts: [],
  modals: {},
  setTheme: vi.fn(),
  toggleTheme: vi.fn(),
  setSidebarOpen: vi.fn(),
  toggleSidebar: vi.fn(),
  setActiveTab: vi.fn(),
  setProcessing: vi.fn(),
  initializeApp: vi.fn(),
  addToast: vi.fn(),
  removeToast: vi.fn(),
  clearToasts: vi.fn(),
  openModal: vi.fn(),
  closeModal: vi.fn(),
  toggleModal: vi.fn(),
  resetUI: vi.fn(),
};

vi.mock('@/stores', () => ({
  // eslint-disable-next-line @typescript-eslint/no-unsafe-return
  useStore: (selector: (s: any) => any) => selector({ ui: mockUIState }),
}));

describe('Settings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUIState.theme = 'dark';
    mockUIState.sidebarOpen = true;
  });

  describe('Rendering', () => {
    it('renders the settings container', () => {
      render(<Settings />);
      expect(screen.getByTestId('settings')).toBeInTheDocument();
    });

    it('shows appearance section', () => {
      render(<Settings />);
      expect(screen.getByText('Appearance')).toBeInTheDocument();
    });

    it('shows application info section', () => {
      render(<Settings />);
      expect(screen.getByText('Application')).toBeInTheDocument();
    });

    it('shows danger zone section', () => {
      render(<Settings />);
      expect(screen.getByText('Danger Zone')).toBeInTheDocument();
    });
  });

  describe('Theme', () => {
    it('renders light theme button', () => {
      render(<Settings />);
      expect(screen.getByTestId('theme-light-btn')).toBeInTheDocument();
    });

    it('renders dark theme button', () => {
      render(<Settings />);
      expect(screen.getByTestId('theme-dark-btn')).toBeInTheDocument();
    });

    it('dark button is pressed when theme is dark', () => {
      render(<Settings />);
      expect(screen.getByTestId('theme-dark-btn')).toHaveAttribute(
        'aria-pressed',
        'true'
      );
      expect(screen.getByTestId('theme-light-btn')).toHaveAttribute(
        'aria-pressed',
        'false'
      );
    });

    it('calls setTheme with light on light button click', () => {
      render(<Settings />);
      fireEvent.click(screen.getByTestId('theme-light-btn'));
      expect(mockUIState.setTheme).toHaveBeenCalledWith('light');
    });

    it('calls setTheme with dark on dark button click', () => {
      render(<Settings />);
      fireEvent.click(screen.getByTestId('theme-dark-btn'));
      expect(mockUIState.setTheme).toHaveBeenCalledWith('dark');
    });
  });

  describe('Sidebar', () => {
    it('renders sidebar toggle', () => {
      render(<Settings />);
      expect(screen.getByTestId('sidebar-toggle')).toBeInTheDocument();
    });

    it('sidebar toggle has switch role', () => {
      render(<Settings />);
      expect(screen.getByTestId('sidebar-toggle')).toHaveAttribute(
        'role',
        'switch'
      );
    });

    it('sidebar toggle reflects current state', () => {
      render(<Settings />);
      expect(screen.getByTestId('sidebar-toggle')).toHaveAttribute(
        'aria-checked',
        'true'
      );
    });

    it('calls setSidebarOpen on toggle click', () => {
      render(<Settings />);
      fireEvent.click(screen.getByTestId('sidebar-toggle'));
      expect(mockUIState.setSidebarOpen).toHaveBeenCalledWith(false);
    });
  });

  describe('Application Info', () => {
    it('shows API URL', () => {
      render(<Settings />);
      expect(screen.getByTestId('info-api-url')).toHaveTextContent(
        'http://localhost:8000'
      );
    });

    it('shows stale time', () => {
      render(<Settings />);
      expect(screen.getByTestId('info-stale-time')).toHaveTextContent('30s');
    });

    it('shows default steps', () => {
      render(<Settings />);
      expect(screen.getByTestId('info-default-steps')).toHaveTextContent('21');
    });

    it('shows devtools status', () => {
      render(<Settings />);
      expect(screen.getByTestId('info-devtools')).toHaveTextContent('Enabled');
    });
  });

  describe('Danger Zone', () => {
    it('renders reset UI button', () => {
      render(<Settings />);
      expect(screen.getByTestId('reset-ui-btn')).toBeInTheDocument();
    });

    it('calls resetUI on reset click', () => {
      render(<Settings />);
      fireEvent.click(screen.getByTestId('reset-ui-btn'));
      expect(mockUIState.resetUI).toHaveBeenCalledTimes(1);
    });
  });

  describe('Customization', () => {
    it('applies custom className', () => {
      render(<Settings className="my-settings" />);
      expect(screen.getByTestId('settings')).toHaveClass('my-settings');
    });
  });
});
