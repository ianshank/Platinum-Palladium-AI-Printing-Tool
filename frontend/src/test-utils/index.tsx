/**
 * Test utilities for component testing
 */

import { type ReactElement, type ReactNode } from 'react';
import {
  render,
  type RenderOptions,
  type RenderResult,
} from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from 'styled-components';
import { theme } from '@/styles/theme';
import { createStore, type StoreState } from '@/stores';

// Re-export everything from testing-library
export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';

/**
 * Create a fresh store for testing
 */
export function createTestStore(
  initialState?: Partial<StoreState>
): ReturnType<typeof createStore> {
  const store = createStore();

  // Apply initial state if provided
  if (initialState) {
    store.setState((state) => ({
      ...state,
      ...initialState,
    }));
  }

  return store;
}

/**
 * Create a fresh query client for testing
 */
export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

interface TestProviderProps {
  children: ReactNode;
  store?: ReturnType<typeof createStore>;
  queryClient?: QueryClient;
}

/**
 * Test provider that wraps components with all necessary providers
 */
export function TestProvider({
  children,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  store: _store = createTestStore(),
  queryClient = createTestQueryClient(),
}: TestProviderProps): ReactElement {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <BrowserRouter>{children}</BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  store?: ReturnType<typeof createStore>;
  queryClient?: QueryClient;
  route?: string;
}

/**
 * Custom render function that includes all providers
 */
export function renderWithProviders(
  ui: ReactElement,
  {
    store = createTestStore(),
    queryClient = createTestQueryClient(),
    route = '/',
    ...renderOptions
  }: CustomRenderOptions = {}
): RenderResult & {
  store: ReturnType<typeof createStore>;
  queryClient: QueryClient;
} {
  // Set initial route
  window.history.pushState({}, 'Test page', route);

  function Wrapper({ children }: { children: ReactNode }): ReactElement {
    return (
      <TestProvider store={store} queryClient={queryClient}>
        {children}
      </TestProvider>
    );
  }

  return {
    ...render(ui, { wrapper: Wrapper, ...renderOptions }),
    store,
    queryClient,
  };
}

/**
 * Wait for a condition to be true
 */
export async function waitForCondition(
  condition: () => boolean,
  options: { timeout?: number; interval?: number } = {}
): Promise<void> {
  const { timeout = 5000, interval = 50 } = options;
  const startTime = Date.now();

  while (!condition()) {
    if (Date.now() - startTime > timeout) {
      throw new Error('Timed out waiting for condition');
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }
}

/**
 * Create a mock file for testing file uploads
 */
export function createMockFile(name: string, size: number, type: string): File {
  const content = new Array(size).fill('a').join('');
  const blob = new Blob([content], { type });
  return new File([blob], name, { type });
}

/**
 * Create a mock image file
 */
export function createMockImageFile(name = 'test.png', size = 1024): File {
  return createMockFile(name, size, 'image/png');
}

/**
 * Mock fetch response helper
 */
export function mockFetchResponse<T>(data: T, status = 200): void {
  global.fetch = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  });
}

/**
 * Mock fetch error helper
 */
export function mockFetchError(message: string, status = 500): void {
  global.fetch = vi.fn().mockResolvedValue({
    ok: false,
    status,
    json: async () => ({ message }),
  });
}

// Import vi for use in this file
import { vi } from 'vitest';
