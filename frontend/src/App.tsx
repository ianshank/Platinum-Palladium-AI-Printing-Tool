import { type FC, useEffect } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

import { Layout } from '@/components/Layout';
import { useAppShortcuts } from '@/hooks';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';
import { Toaster } from '@/components/ui/Toaster';

import { DashboardPage } from '@/pages/DashboardPage';

import { CalibrationPage } from '@/pages/CalibrationPage';
import { CurvesPage } from '@/pages/CurvesPage';

import { ChemistryPage } from '@/pages/ChemistryPage';

import { AIAssistantPage } from '@/pages/AIAssistantPage';
import { MCTSPage } from '@/pages/MCTSPage';
import { SessionLogPage } from '@/pages/SessionLogPage';
import { SettingsPage } from '@/pages/SettingsPage';

/**
 * Branded loading screen shown during app initialization.
 */
const LoadingScreen: FC = () => (
  <div
    className="flex h-screen flex-col items-center justify-center bg-background"
    data-testid="loading-screen"
  >
    <div className="flex animate-fade-in flex-col items-center gap-6">
      {/* Logo badge */}
      <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary shadow-lg">
        <span className="text-2xl font-bold text-primary-foreground">Pt</span>
      </div>

      {/* Title */}
      <div className="text-center">
        <h1 className="text-lg font-semibold text-foreground">
          Pt/Pd Calibration Studio
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Initializing workspace…
        </p>
      </div>

      {/* Spinner */}
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent" />
    </div>
  </div>
);

LoadingScreen.displayName = 'LoadingScreen';

const ShortcutHandler: FC = () => {
  useAppShortcuts();
  return null;
};

/**
 * Main application component
 */
export const App: FC = () => {
  const initializeApp = useStore((state) => state.ui.initializeApp);
  const isInitialized = useStore((state) => state.ui.isInitialized);

  useEffect(() => {
    logger.debug('App: Initializing application');
    initializeApp();
  }, [initializeApp]);

  if (!isInitialized) {
    return <LoadingScreen />;
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ShortcutHandler />
        <Layout>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/calibration" element={<CalibrationPage />} />
            <Route path="/curves" element={<CurvesPage />} />
            <Route path="/chemistry" element={<ChemistryPage />} />
            <Route path="/assistant" element={<AIAssistantPage />} />
            <Route path="/mcts" element={<MCTSPage />} />
            <Route path="/session" element={<SessionLogPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Layout>
        <Toaster />
      </BrowserRouter>
    </ErrorBoundary>
  );
};

App.displayName = 'App';
