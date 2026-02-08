import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { createMockFile, renderWithProviders, userEvent } from '@/test-utils';
import { useUploadScan } from '@/api/hooks';
import { Step1Upload } from '../Step1Upload';

// Mock the API hook
const mockMutate = vi.fn();
vi.mock('@/api/hooks', () => ({
    useUploadScan: vi.fn(() => ({
        mutate: mockMutate,
        isPending: false,
        error: null,
    })),
}));

describe('Step1Upload', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useUploadScan).mockReturnValue({
            mutate: mockMutate,
            isPending: false,
            error: null,
        } as any);
    });

    it('renders upload zone and tablet options', () => {
        renderWithProviders(<Step1Upload />);

        expect(screen.getByText(/Upload Scan/i)).toBeInTheDocument();
        expect(screen.getByText(/Drag & drop your scan here/i)).toBeInTheDocument();
        expect(screen.getByText(/Select Tablet Type/i)).toBeInTheDocument();
        expect(screen.getByText(/Stouffer T21/i)).toBeInTheDocument();
    });

    it('handles file selection', async () => {
        renderWithProviders(<Step1Upload />);

        const file = createMockFile('test-scan.png', 1024, 'image/png');
        const fileInput = screen.getByTestId('dropzone-input');

        if (fileInput) {
            await userEvent.upload(fileInput, file);

            await waitFor(() => {
                expect(screen.getByText('test-scan.png')).toBeInTheDocument();
            });
        }
    });

    it('calls uploadScan API when continue is clicked', async () => {
        renderWithProviders(<Step1Upload />);

        const file = createMockFile('test-scan.png', 1024, 'image/png');
        const fileInput = screen.getByTestId('dropzone-input');

        await userEvent.upload(fileInput, file);

        const continueBtn = screen.getByRole('button', { name: /Analyze & Continue/i });

        await waitFor(() => {
            expect(continueBtn).not.toBeDisabled();
        });

        fireEvent.click(continueBtn);

        expect(mockMutate).toHaveBeenCalledTimes(1);
        expect(mockMutate).toHaveBeenCalledWith(
            expect.objectContaining({
                file: expect.any(File),
                tabletType: 'stouffer_21'
            }),
            expect.any(Object)
        );
    });

    it('displays error message when upload fails', async () => {
        const mockError = new Error('Server Error');
        const mockMutateWithError = vi.fn((_data, options) => {
            options.onError(mockError);
        });

        vi.mocked(useUploadScan).mockReturnValue({
            mutate: mockMutateWithError,
            isPending: false,
            error: null,
        } as any);

        renderWithProviders(<Step1Upload />);

        const file = createMockFile('test.png', 1024, 'image/png');
        const fileInput = screen.getByTestId('dropzone-input');
        await userEvent.upload(fileInput, file);

        const continueBtn = screen.getByRole('button', { name: /Analyze & Continue/i });
        fireEvent.click(continueBtn);

        await waitFor(() => {
            expect(screen.getByText(/Server Error/i)).toBeInTheDocument();
        });
    });

    it('handles file replacement', async () => {
        renderWithProviders(<Step1Upload />);

        const file1 = createMockFile('test1.png', 1024, 'image/png');
        const file2 = createMockFile('test2.png', 2048, 'image/png');
        const fileInput = screen.getByTestId('dropzone-input');

        // First drop
        await userEvent.upload(fileInput, file1);
        expect(screen.getByText('test1.png')).toBeInTheDocument();

        // Second drop
        await userEvent.upload(fileInput, file2);

        await waitFor(() => {
            expect(screen.getByText('test2.png')).toBeInTheDocument();
            expect(screen.queryByText('test1.png')).not.toBeInTheDocument();
        });
    });
});
