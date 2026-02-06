/**
 * Main App component.
 * Sets up providers and routing.
 */

import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider } from 'styled-components';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

import { queryClient } from '@/api/queryClient';
import { darkroomTheme, GlobalStyles } from '@/theme';
import { env } from '@/config/env';

// Layout
import { AppShell } from '@/components/layout/AppShell';

// Pages
import { Dashboard } from '@/pages/Dashboard';
import { CalibrationWizard } from '@/pages/CalibrationWizard';
import { ChemistryCalculator } from '@/pages/ChemistryCalculator';
import { AIAssistant } from '@/pages/AIAssistant';
import { SessionLog } from '@/pages/SessionLog';
import { Settings } from '@/pages/Settings';
import { NotFound } from '@/pages/NotFound';

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={darkroomTheme}>
        <GlobalStyles />
        <BrowserRouter>
          <AppShell>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/calibration" element={<CalibrationWizard />} />
              <Route path="/chemistry" element={<ChemistryCalculator />} />
              {env.VITE_ENABLE_AI_ASSISTANT && (
                <Route path="/assistant" element={<AIAssistant />} />
              )}
              <Route path="/sessions" element={<SessionLog />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </ThemeProvider>
      {env.VITE_DEBUG_MODE && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
}
