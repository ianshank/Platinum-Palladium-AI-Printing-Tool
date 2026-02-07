import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { createMockFile, renderWithProviders } from '@/test-utils';
import { Step1Upload } from '../Step1Upload';

// Mock the API hook
const mockMutate = vi.fn();
vi.mock('@/api/hooks', () => ({
    useUploadScan: () => ({
        mutate: mockMutate,
        isPending: false,
        error: null,
    }),
}));

describe('Step1Upload', () => {
    beforeEach(() => {
        vi.clearAllMocks();
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
            await fireEvent.change(fileInput, { target: { files: [file] } });

            await waitFor(() => {
                expect(screen.getByText('test-scan.png')).toBeInTheDocument();
            });
        }
    });

    it('calls uploadScan API when continue is clicked', async () => {
        renderWithProviders(<Step1Upload />);

        const file = createMockFile('test-scan.png', 1024, 'image/png');
        const fileInput = screen.getByTestId('dropzone-input');

        await fireEvent.change(fileInput, { target: { files: [file] } });

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
});
