import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { userEvent } from '@/test-utils';
import { createMockImageFile, renderWithProviders } from '@/test-utils';
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
        vi.spyOn(api.scan, 'upload').mockImplementation(async (_file, _type, onProgress) => {
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
            expect(onUploadComplete).toHaveBeenCalledWith(mockResponse);
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

    it('rejects files larger than 20MB', async () => {
        renderWithProviders(<ScanUpload />);

        const largeFile = createMockImageFile('huge.png', 25 * 1024 * 1024); // 25MB
        const input = screen.getByTestId('scan-upload-input');

        await userEvent.upload(input, largeFile);

        await waitFor(() => {
            expect(screen.getByText(/file is too large/i)).toBeInTheDocument();
        });
    });

    it('allows clearing the selected file', async () => {
        renderWithProviders(<ScanUpload />);

        const file = createMockImageFile('test.png');
        const input = screen.getByTestId('scan-upload-input');

        await userEvent.upload(input, file);
        expect(screen.getByText('test.png')).toBeInTheDocument();

        const clearButton = screen.getByRole('button', { name: /remove file/i });
        await userEvent.click(clearButton);

        expect(screen.queryByText('test.png')).not.toBeInTheDocument();
        expect(screen.getByText(/click to upload or drag and drop/i)).toBeInTheDocument();
    });
});
