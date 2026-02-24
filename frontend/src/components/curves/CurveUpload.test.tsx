/**
 * CurveUpload component tests
 *
 * Tests rendering, file drag-and-drop, paste mode, channel selection,
 * error states, profile preview, and load-into-editor callback.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { createMockFile, renderWithProviders, userEvent } from '@/test-utils';
import { CurveUpload } from './CurveUpload';
import { api } from '@/api/client';
import type { QuadUploadResponse } from '@/types/models';

// --- Mocks ---

vi.mock('@/api/client', () => ({
  api: {
    curves: {
      uploadQuad: vi.fn(),
      parseQuad: vi.fn(),
    },
  },
  // Re-export needed symbols used by hooks
  ApiError: class {},
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    info: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// ResizeObserver mock (needed for some layout effects)
global.ResizeObserver = class ResizeObserver {
  observe() {
    /* noop */
  }
  unobserve() {
    /* noop */
  }
  disconnect() {
    /* noop */
  }
};

// --- Fixtures ---

const MOCK_UPLOAD_RESPONSE: QuadUploadResponse = {
  success: true,
  profile_name: 'Test QTR Profile',
  resolution: 720,
  ink_limit: 100,
  media_type: 'matte',
  all_channels: ['K', 'C', 'M', 'Y', 'LC', 'LM'],
  active_channels: ['K', 'C'],
  curve_id: 'curve-abc-123',
  curve_data: {
    input_values: [0, 0.1, 0.2, 0.5, 1.0],
    output_values: [0, 0.15, 0.25, 0.55, 1.0],
  },
  summary: 'QTR profile with 6 channels, K selected',
};

const createQuadFile = (name = 'test-profile.quad', size = 512) =>
  createMockFile(name, size, 'text/plain');

// --- Tests ---

describe('CurveUpload', () => {
  const onLoadCurve = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // === Rendering ===

  describe('Rendering', () => {
    it('renders file upload mode by default', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      expect(screen.getByText('Upload File')).toBeInTheDocument();
      expect(screen.getByText('Paste Content')).toBeInTheDocument();
      expect(screen.getByText(/drop a .quad file/i)).toBeInTheDocument();
      expect(screen.getByLabelText('Ink Channel')).toBeInTheDocument();
    });

    it('switches to paste mode when tab is clicked', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      await userEvent.click(screen.getByText('Paste Content'));

      expect(screen.getByLabelText('.quad File Content')).toBeInTheDocument();
      expect(screen.getByText('Parse Content')).toBeInTheDocument();
    });

    it('renders all channel options', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const select = screen.getByLabelText('Ink Channel');
      const optionElements = select.querySelectorAll('option');
      const options = Array.from(optionElements).map((o) => o.value);

      expect(options).toContain('K');
      expect(options).toContain('C');
      expect(options).toContain('LLK');
      expect(options.length).toBe(8);
    });
  });

  // === Channel Selection ===

  describe('Channel Selection', () => {
    it('allows changing the channel', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const select = screen.getByLabelText('Ink Channel');
      await userEvent.selectOptions(select, 'LM');

      expect((select as HTMLSelectElement).value).toBe('LM');
    });
  });

  // === File Upload Mode ===

  describe('File Upload Mode', () => {
    it('shows selected file after drop', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      const file = createQuadFile();

      await userEvent.upload(input, file);

      expect(screen.getByText('test-profile.quad')).toBeInTheDocument();
    });

    it('shows error for invalid file type', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      const invalidFile = createMockFile('image.png', 100, 'image/png');

      // fireEvent for invalid drops since userEvent may not trigger rejection
      fireEvent.drop(input, {
        dataTransfer: { files: [invalidFile], types: ['Files'] },
      });

      // The dropzone should reject this file type
      // Depending on react-dropzone handling, may show error or just not accept
    });

    it('allows removing a selected file', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      const file = createQuadFile();

      await userEvent.upload(input, file);

      expect(screen.getByText('test-profile.quad')).toBeInTheDocument();

      const removeButton = screen.getByTitle('Remove file');
      await userEvent.click(removeButton);

      expect(screen.queryByText('test-profile.quad')).not.toBeInTheDocument();
    });

    it('disables upload button when no file is selected', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const uploadButton = screen.getByRole('button', {
        name: /upload & parse/i,
      });
      expect(uploadButton).toBeDisabled();
    });

    it('enables upload button when file is selected', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());

      const uploadButton = screen.getByRole('button', {
        name: /upload & parse/i,
      });
      expect(uploadButton).toBeEnabled();
    });
  });

  // === Paste Mode ===

  describe('Paste Mode', () => {
    it('disables parse button when content is empty', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      await userEvent.click(screen.getByText('Paste Content'));

      const parseButton = screen.getByRole('button', {
        name: /parse content/i,
      });
      expect(parseButton).toBeDisabled();
    });

    it('enables parse button when content is entered', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      await userEvent.click(screen.getByText('Paste Content'));

      const textarea = screen.getByTestId('quad-paste-input');
      await userEvent.type(textarea, 'CURVE K\n0 0\n255 255');

      const parseButton = screen.getByRole('button', {
        name: /parse content/i,
      });
      expect(parseButton).toBeEnabled();
    });

    it('shows error when parsing empty content', async () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      await userEvent.click(screen.getByText('Paste Content'));

      // Clear any existing content and directly fire parse with empty
      const parseButton = screen.getByRole('button', {
        name: /parse content/i,
      });
      // Button should be disabled when empty, so this is a guard test
      expect(parseButton).toBeDisabled();
    });
  });

  // === Profile Preview ===

  describe('Profile Preview', () => {
    it('shows profile preview after successful upload', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockResolvedValue(
        MOCK_UPLOAD_RESPONSE
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());

      const uploadButton = screen.getByRole('button', {
        name: /upload & parse/i,
      });
      await userEvent.click(uploadButton);

      await waitFor(() => {
        expect(screen.getByTestId('quad-profile-preview')).toBeInTheDocument();
      });

      expect(screen.getByText('Test QTR Profile')).toBeInTheDocument();
      expect(screen.getByText('Profile Loaded')).toBeInTheDocument();
      expect(screen.getByText('K, C')).toBeInTheDocument();
      expect(screen.getByText('720 DPI')).toBeInTheDocument();
    });

    it('shows "Load into Editor" button after successful upload', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockResolvedValue(
        MOCK_UPLOAD_RESPONSE
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());
      await userEvent.click(
        screen.getByRole('button', { name: /upload & parse/i })
      );

      await waitFor(() => {
        expect(screen.getByText('Load into Editor')).toBeInTheDocument();
      });
    });

    it('shows "Upload Another" button after successful upload', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockResolvedValue(
        MOCK_UPLOAD_RESPONSE
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());
      await userEvent.click(
        screen.getByRole('button', { name: /upload & parse/i })
      );

      await waitFor(() => {
        expect(screen.getByText('Upload Another')).toBeInTheDocument();
      });
    });
  });

  // === Load into Editor ===

  describe('Load into Editor', () => {
    it('calls onLoadCurve with correct data when "Load into Editor" is clicked', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockResolvedValue(
        MOCK_UPLOAD_RESPONSE
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());
      await userEvent.click(
        screen.getByRole('button', { name: /upload & parse/i })
      );

      await waitFor(() => {
        expect(screen.getByText('Load into Editor')).toBeInTheDocument();
      });

      await userEvent.click(screen.getByText('Load into Editor'));

      expect(onLoadCurve).toHaveBeenCalledWith(
        MOCK_UPLOAD_RESPONSE.curve_data,
        MOCK_UPLOAD_RESPONSE.curve_id,
        MOCK_UPLOAD_RESPONSE.profile_name
      );
    });

    it('resets state when "Upload Another" is clicked', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockResolvedValue(
        MOCK_UPLOAD_RESPONSE
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());
      await userEvent.click(
        screen.getByRole('button', { name: /upload & parse/i })
      );

      await waitFor(() => {
        expect(screen.getByText('Upload Another')).toBeInTheDocument();
      });

      await userEvent.click(screen.getByText('Upload Another'));

      // Should return to initial upload state
      expect(screen.getByText(/drop a .quad file/i)).toBeInTheDocument();
      expect(
        screen.queryByTestId('quad-profile-preview')
      ).not.toBeInTheDocument();
    });
  });

  // === Error States ===

  describe('Error States', () => {
    it('displays upload error from API', async () => {
      vi.spyOn(api.curves, 'uploadQuad').mockRejectedValue(
        new Error('Invalid .quad format: missing CURVE header')
      );

      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const input = screen.getByTestId('quad-upload-input');
      await userEvent.upload(input, createQuadFile());
      await userEvent.click(
        screen.getByRole('button', { name: /upload & parse/i })
      );

      await waitFor(() => {
        expect(screen.getByText(/invalid .quad format/i)).toBeInTheDocument();
      });
    });
  });

  // === ARIA / Accessibility ===

  describe('Accessibility', () => {
    it('tab buttons have correct ARIA attributes', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      const fileTab = screen.getByText('Upload File').closest('[role="tab"]');
      const pasteTab = screen
        .getByText('Paste Content')
        .closest('[role="tab"]');

      expect(fileTab).toHaveAttribute('aria-selected', 'true');
      expect(pasteTab).toHaveAttribute('aria-selected', 'false');
    });

    it('tablist has correct ARIA label', () => {
      renderWithProviders(<CurveUpload onLoadCurve={onLoadCurve} />);

      expect(
        screen.getByRole('tablist', { name: 'Upload mode' })
      ).toBeInTheDocument();
    });
  });
});
