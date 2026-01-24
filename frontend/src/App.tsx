import { type FC, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import { Layout } from '@/components/Layout';
import { useStore } from '@/stores';
import { logger } from '@/lib/logger';
import { Toaster } from '@/components/ui/Toaster';

// Lazy load pages for code splitting
// import { lazy, Suspense } from 'react';
// const Dashboard = lazy(() => import('@/pages/Dashboard'));

// Placeholder pages until components are migrated
const DashboardPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Dashboard</h1>
    <p className="text-muted-foreground mt-2">Overview and metrics</p>
  </div>
);

const CalibrationPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Calibration Wizard</h1>
    <p className="text-muted-foreground mt-2">Step-by-step calibration workflow</p>
  </div>
);

const CurvesPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Curve Editor</h1>
    <p className="text-muted-foreground mt-2">View and edit calibration curves</p>
  </div>
);

const ChemistryPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">Chemistry Calculator</h1>
    <p className="text-muted-foreground mt-2">Coating recipe calculations</p>
  </div>
);

const AIAssistantPage: FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold">AI Assistant</h1>
    <p className="text-muted-foreground mt-2">Chat with the printing assistant</p>
  </div>
);

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
