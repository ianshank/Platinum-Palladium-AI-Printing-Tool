import { describe, expect, it, vi } from 'vitest';
import { assessScanQuality, useUploadScan } from './useScan';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

// Mock the API client
vi.mock('../client', () => ({
    api: {
        scan: {
            upload: vi.fn(),
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

describe('useScan hooks', () => {
    describe('useUploadScan', () => {
        it('calls api.scan.upload on mutate', async () => {
            const mockResponse = { id: 'scan-1', measurements: [0.1, 0.2], preview: '/preview.png' };
            vi.mocked(api.scan.upload).mockResolvedValue(mockResponse as any);

            const { result } = renderHook(() => useUploadScan(), {
                wrapper: createWrapper(),
            });

            const file = new File(['scan-data'], 'scan.png', { type: 'image/png' });
            result.current.mutate({ file, tabletType: 'stouffer_21' });

            await waitFor(() => expect(result.current.isSuccess).toBe(true));

            expect(api.scan.upload).toHaveBeenCalledWith(file, 'stouffer_21');
            expect(result.current.data).toEqual(mockResponse);
        });

        it('handles upload error', async () => {
            vi.mocked(api.scan.upload).mockRejectedValue(new Error('Upload failed'));

            const { result } = renderHook(() => useUploadScan(), {
                wrapper: createWrapper(),
            });

            const file = new File(['bad'], 'bad.png', { type: 'image/png' });
            result.current.mutate({ file, tabletType: 'stouffer_21' });

            await waitFor(() => expect(result.current.isError).toBe(true));
            expect(result.current.error?.message).toBe('Upload failed');
        });
    });

    describe('assessScanQuality', () => {
        it('returns quality assessment', () => {
            const result = assessScanQuality({ densities: [0.1, 0.2], dmax: 2.0, dmin: 0.05, range: 1.95, num_patches: 21 });

            expect(result.quality).toBe('excellent');
            expect(result.score).toBe(100);
            expect(result.overall).toBe('excellent');
            expect(result.issues).toEqual([]);
        });
    });
});
