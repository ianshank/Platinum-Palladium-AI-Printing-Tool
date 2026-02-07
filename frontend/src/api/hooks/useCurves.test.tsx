import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useExportCurve, useGenerateCurve, useGetCurve } from './useCurves';

// Mock the API client
vi.mock('../client', () => ({
    api: {
        curves: {
            generate: vi.fn(),
            get: vi.fn(),
            export: vi.fn(),
        },
    },
}));

import { api } from '../client';

function createWrapper() {
    const queryClient = new QueryClient({
        defaultOptions: {
            queries: { retry: false, gcTime: 0 },
            mutations: { retry: false },
        },
    });

    return function Wrapper({ children }: { children: ReactNode }) {
        return (
            <QueryClientProvider client={queryClient}>
                {children}
            </QueryClientProvider>
        );
    };
}

describe('useCurves hooks', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    describe('useGenerateCurve', () => {
        it('calls api.curves.generate on mutate', async () => {
            const mockResponse = { id: 'curve-1', data: 'generated' };
            vi.mocked(api.curves.generate).mockResolvedValue(mockResponse as any);

            const { result } = renderHook(() => useGenerateCurve(), {
                wrapper: createWrapper(),
            });

            result.current.mutate({
                measurements: [0.1, 0.2, 0.3],
                curve_type: 'linear',
                name: 'Test',
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));

            expect(api.curves.generate).toHaveBeenCalledWith({
                measurements: [0.1, 0.2, 0.3],
                type: 'linear',
                name: 'Test',
            });
            expect(result.current.data).toEqual(mockResponse);
        });

        it('handles generation error', async () => {
            vi.mocked(api.curves.generate).mockRejectedValue(new Error('Generation failed'));

            const { result } = renderHook(() => useGenerateCurve(), {
                wrapper: createWrapper(),
            });

            result.current.mutate({ measurements: [0.1] });

            await waitFor(() => expect(result.current.isError).toBe(true));
            expect(result.current.error?.message).toBe('Generation failed');
        });
    });

    describe('useGetCurve', () => {
        it('fetches curve data by id', async () => {
            const mockCurve = { id: 'curve-1', name: 'Test', points: [] };
            vi.mocked(api.curves.get).mockResolvedValue(mockCurve as any);

            const { result } = renderHook(() => useGetCurve('curve-1'), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));

            expect(api.curves.get).toHaveBeenCalledWith('curve-1');
            expect(result.current.data).toEqual(mockCurve);
        });

        it('does not fetch when id is empty', () => {
            renderHook(() => useGetCurve(''), {
                wrapper: createWrapper(),
            });

            expect(api.curves.get).not.toHaveBeenCalled();
        });

        it('handles fetch error', async () => {
            vi.mocked(api.curves.get).mockRejectedValue(new Error('Not found'));

            const { result } = renderHook(() => useGetCurve('nonexistent'), {
                wrapper: createWrapper(),
            });

            await waitFor(() => expect(result.current.isError).toBe(true));
        });
    });

    describe('useExportCurve', () => {
        it('calls api.curves.export on mutate', async () => {
            const mockBlob = new Blob(['data'], { type: 'text/plain' });
            vi.mocked(api.curves.export).mockResolvedValue(mockBlob);

            const { result } = renderHook(() => useExportCurve(), {
                wrapper: createWrapper(),
            });

            result.current.mutate({ curveId: 'curve-1', format: 'csv' });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));

            expect(api.curves.export).toHaveBeenCalledWith({
                curveId: 'curve-1',
                format: 'csv',
            });
        });

        it('handles export error', async () => {
            vi.mocked(api.curves.export).mockRejectedValue(new Error('Export failed'));

            const { result } = renderHook(() => useExportCurve(), {
                wrapper: createWrapper(),
            });

            result.current.mutate({ curveId: 'curve-1', format: 'qtr' });

            await waitFor(() => expect(result.current.isError).toBe(true));
        });
    });
});
