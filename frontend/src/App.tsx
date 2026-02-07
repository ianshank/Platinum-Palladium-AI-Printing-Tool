import { type FC, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import { Layout } from '@/components/Layout';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';
import { Toaster } from '@/components/ui/Toaster';

import { DashboardPage } from '@/pages/DashboardPage';

import { CalibrationPage } from '@/pages/CalibrationPage';
import { CurvesPage } from '@/pages/CurvesPage';

import { ChemistryPage } from '@/pages/ChemistryPage';

import { AIAssistantPage } from '@/pages/AIAssistantPage';

const SessionLogPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Session Log</h1>
    <p className="text-muted-foreground mt-2">Print history and statistics</p>
  </div>
);

const SettingsPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Settings</h1>
    <p className="text-muted-foreground mt-2">Application configuration</p>
  </div>
);

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
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
          <p className="mt-4 text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/calibration" element={<CalibrationPage />} />
          <Route path="/curves" element={<CurvesPage />} />
          <Route path="/chemistry" element={<ChemistryPage />} />
          <Route path="/assistant" element={<AIAssistantPage />} />
          <Route path="/session" element={<SessionLogPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </Layout>
      <Toaster />
    </BrowserRouter>
  );
};

App.displayName = 'App';
