import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'styled-components';
import { theme } from './styles/theme';

import { App } from './App';
import { config } from './config';
import { logger } from './lib/logger';
import './styles/globals.css';

// Initialize logger
logger.info('Application starting', {
  version: config.app.version,
  environment: config.app.environment,
});

// Configure React Query client with dynamic settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: config.api.staleTime,
      gcTime: config.api.gcTime,
      retry: config.api.retryAttempts,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: config.api.retryAttempts,
    },
  },
});

// Get root element
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error(
    'Root element not found. Ensure index.html has <div id="root"></div>'
  );
}

// Create root and render
const root = createRoot(rootElement);

root.render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <App />
      </ThemeProvider>
    </QueryClientProvider>
  </StrictMode>
);

// Log when app is ready
logger.info('Application rendered successfully');

// Export for potential hot module replacement
export { queryClient };
