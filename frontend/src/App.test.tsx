import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock dependencies before importing App
vi.mock('@/stores', () => ({
  useStore: vi.fn(),
}));

vi.mock('@/components/Layout', () => ({
  Layout: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/ErrorBoundary', () => ({
  ErrorBoundary: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="error-boundary">{children}</div>
  ),
}));

vi.mock('@/components/ui/Toaster', () => ({
  Toaster: () => <div data-testid="toaster" />,
}));

vi.mock('@/pages/DashboardPage', () => ({
  DashboardPage: () => <div data-testid="dashboard-page">Dashboard</div>,
}));

vi.mock('@/pages/CalibrationPage', () => ({
  CalibrationPage: () => <div data-testid="calibration-page">Calibration</div>,
}));

vi.mock('@/pages/CurvesPage', () => ({
  CurvesPage: () => <div data-testid="curves-page">Curves</div>,
}));

vi.mock('@/pages/ChemistryPage', () => ({
  ChemistryPage: () => <div data-testid="chemistry-page">Chemistry</div>,
}));

vi.mock('@/pages/AIAssistantPage', () => ({
  AIAssistantPage: () => <div data-testid="ai-page">AI Assistant</div>,
}));

vi.mock('@/pages/SessionLogPage', () => ({
  SessionLogPage: () => <div data-testid="session-page">Session Log</div>,
}));

vi.mock('@/pages/SettingsPage', () => ({
  SettingsPage: () => <div data-testid="settings-page">Settings</div>,
}));

vi.mock('@/lib/logger', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { useStore } from '@/stores';
import { App } from './App';

const mockUseStore = vi.mocked(useStore);

describe('App', () => {
  const mockInitializeApp = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Loading state', () => {
    it('shows LoadingScreen when not initialized', () => {
      mockUseStore.mockImplementation((selector: any) =>
        selector({
          ui: {
            initializeApp: mockInitializeApp,
            isInitialized: false,
          },
        })
      );

      render(<App />);

      expect(screen.getByTestId('loading-screen')).toBeInTheDocument();
      expect(screen.getByText('Pt/Pd Calibration Studio')).toBeInTheDocument();
      expect(screen.getByText('Initializing workspace…')).toBeInTheDocument();
    });

    it('shows the Pt logo badge in loading screen', () => {
      mockUseStore.mockImplementation((selector: any) =>
        selector({
          ui: {
            initializeApp: mockInitializeApp,
            isInitialized: false,
          },
        })
      );

      render(<App />);

      expect(screen.getByText('Pt')).toBeInTheDocument();
    });

    it('calls initializeApp on mount', () => {
      mockUseStore.mockImplementation((selector: any) =>
        selector({
          ui: {
            initializeApp: mockInitializeApp,
            isInitialized: false,
          },
        })
      );

      render(<App />);

      expect(mockInitializeApp).toHaveBeenCalledOnce();
    });
  });

  describe('Initialized state', () => {
    beforeEach(() => {
      mockUseStore.mockImplementation((selector: any) =>
        selector({
          ui: {
            initializeApp: mockInitializeApp,
            isInitialized: true,
          },
        })
      );
    });

    it('renders the main layout when initialized', () => {
      render(<App />);

      expect(screen.queryByTestId('loading-screen')).not.toBeInTheDocument();
      expect(screen.getByTestId('error-boundary')).toBeInTheDocument();
      expect(screen.getByTestId('layout')).toBeInTheDocument();
    });

    it('renders the Toaster component', () => {
      render(<App />);

      expect(screen.getByTestId('toaster')).toBeInTheDocument();
    });

    it('wraps content in ErrorBoundary', () => {
      render(<App />);

      const errorBoundary = screen.getByTestId('error-boundary');
      const layout = screen.getByTestId('layout');
      expect(errorBoundary).toContainElement(layout);
    });
  });
});
