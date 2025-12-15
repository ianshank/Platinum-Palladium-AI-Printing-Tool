/**
 * Custom render function with all providers.
 * Wraps components with theme, router, and query providers.
 */

import { ReactElement, ReactNode } from 'react';
import { render, RenderOptions, RenderResult } from '@testing-library/react';
import { ThemeProvider } from 'styled-components';
import { MemoryRouter, MemoryRouterProps } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { darkroomTheme } from '@/theme/darkroomTheme';
import { GlobalStyles } from '@/theme/GlobalStyles';

// Create a fresh QueryClient for each test
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

interface WrapperProps {
  children: ReactNode;
}

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  routerOptions?: MemoryRouterProps;
  queryClient?: QueryClient;
}

/**
 * Creates all providers wrapper for testing.
 */
const createWrapper = (options: CustomRenderOptions = {}) => {
  const { routerOptions = {}, queryClient = createTestQueryClient() } = options;

  return function Wrapper({ children }: WrapperProps) {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={darkroomTheme}>
          <GlobalStyles />
          <MemoryRouter {...routerOptions}>{children}</MemoryRouter>
        </ThemeProvider>
      </QueryClientProvider>
    );
  };
};

/**
 * Custom render function with all providers.
 */
export function renderWithProviders(
  ui: ReactElement,
  options: CustomRenderOptions = {}
): RenderResult {
  const { routerOptions, queryClient, ...renderOptions } = options;

  return render(ui, {
    wrapper: createWrapper({ routerOptions, queryClient }),
    ...renderOptions,
  });
}

/**
 * Re-export everything from testing-library.
 */
export * from '@testing-library/react';
export { renderWithProviders as render };
export { createTestQueryClient };
