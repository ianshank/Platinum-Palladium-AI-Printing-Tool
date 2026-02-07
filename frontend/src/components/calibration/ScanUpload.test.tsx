import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { userEvent } from '@/test-utils';
import { renderWithProviders, createMockImageFile } from '@/test-utils';
import { ScanUpload } from './ScanUpload';
import { api } from '@/api/client';

// Mock the API client
vi.mock('@/api/client', () => ({
    api: {
        scan: {
            upload: vi.fn(),
        },
    },
}));

describe('ScanUpload', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders dropzone area', () => {
        renderWithProviders(<ScanUpload />);
        expect(screen.getByText(/click to upload or drag and drop/i)).toBeInTheDocument();
    });

    it('handles file selection and upload success', async () => {
        const onUploadComplete = vi.fn();
        const mockFile = createMockImageFile('test-scan.png');

        // Mock API success response
        const mockResponse = {
            success: true,
            extraction_id: '123-456',
            num_patches: 21,
            densities: [],
            quality: 0.95,
            warnings: [],
        };

        // Mock upload implementation to simulate progress
        vi.spyOn(api.scan, 'upload').mockImplementation(async (file, type, onProgress) => {
            onProgress?.(100);
            return mockResponse as any;
        });

        renderWithProviders(<ScanUpload onUploadComplete={onUploadComplete} />);

        // Upload file
        const input = screen.getByTestId('scan-upload-input');

        await userEvent.upload(input, mockFile);

        // Should show file info
        expect(screen.getByText('test-scan.png')).toBeInTheDocument();

        // Click Upload button
        const uploadButton = screen.getByRole('button', { name: /upload scan/i });
        await userEvent.click(uploadButton);

        // Should complete successfully
        await waitFor(() => {
            expect(onUploadComplete).toHaveBeenCalledWith('123-456');
            expect(screen.getByText(/upload complete/i)).toBeInTheDocument();
        });
    });

    it('handles upload failure', async () => {
        const mockFile = createMockImageFile();

        vi.spyOn(api.scan, 'upload').mockRejectedValue(new Error('Network Error'));

        renderWithProviders(<ScanUpload />);

        const input = screen.getByTestId('scan-upload-input');
        await userEvent.upload(input, mockFile);

        const uploadButton = screen.getByRole('button', { name: /upload scan/i });
        await userEvent.click(uploadButton);

        await waitFor(() => {
            expect(screen.getByText(/network error/i)).toBeInTheDocument();
        });
    });
});
