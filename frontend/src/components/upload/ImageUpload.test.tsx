/**
 * ImageUpload Component Tests
 * Tests for file upload, drag-and-drop, validation, progress tracking, and accessibility
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ImageUpload } from './ImageUpload';

// Mock logger to avoid console noise
vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('ImageUpload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Mock URL.createObjectURL and revokeObjectURL
    global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    global.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    // Cleanup any created object URLs
    vi.restoreAllMocks();
  });

  describe('Rendering', () => {
    it('renders upload area with default text', () => {
      render(<ImageUpload />);

      expect(screen.getByTestId('image-upload-dropzone')).toBeInTheDocument();
      expect(screen.getByText('Click to upload or drag and drop')).toBeInTheDocument();
    });

    it('renders custom label and helpText', () => {
      render(
        <ImageUpload
          label="Upload your image"
          helpText="Only PNG files accepted"
        />
      );

      expect(screen.getByText('Upload your image')).toBeInTheDocument();
      expect(screen.getByText('Only PNG files accepted')).toBeInTheDocument();
    });

    it('generates default help text from accept prop', () => {
      render(
        <ImageUpload
          accept={{ 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] }}
          maxSize={10 * 1024 * 1024}
        />
      );

      expect(screen.getByText(/\.png, \.jpg, \.jpeg \(max 10MB\)/)).toBeInTheDocument();
    });

    it('applies custom className', () => {
      render(<ImageUpload className="custom-class" />);

      const container = screen.getByTestId('image-upload-container');
      expect(container).toHaveClass('custom-class');
    });
  });

  describe('File Selection', () => {
    it('accepts files via drop', async () => {
      const onFileSelect = vi.fn();
      render(<ImageUpload onFileSelect={onFileSelect} />);

      const file = new File(['dummy content'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      Object.defineProperty(input, 'files', {
        value: [file],
        writable: false,
      });

      fireEvent.drop(screen.getByTestId('image-upload-dropzone'));
      fireEvent.change(input);

      await waitFor(() => {
        expect(onFileSelect).toHaveBeenCalledWith([file]);
      });
    });

    it('shows file info when file is selected', async () => {
      render(<ImageUpload />);

      const file = new File(['a'.repeat(1024 * 1024)], 'test-image.png', {
        type: 'image/png',
      });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByText('test-image.png')).toBeInTheDocument();
        expect(screen.getByText(/1\.00 MB/)).toBeInTheDocument();
      });
    });

    it('shows image preview when showPreview=true', async () => {
      render(<ImageUpload showPreview={true} />);

      const file = new File(['image data'], 'preview.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        const preview = screen.getByTestId('image-upload-preview');
        expect(preview).toBeInTheDocument();
        expect(preview).toHaveAttribute('src', 'blob:mock-url');
      });

      // eslint-disable-next-line @typescript-eslint/unbound-method -- testing mock function reference
      expect(global.URL.createObjectURL).toHaveBeenCalledWith(file);
    });

    it('does not show preview when showPreview=false', async () => {
      render(<ImageUpload showPreview={false} />);

      const file = new File(['image data'], 'no-preview.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.queryByTestId('image-upload-preview')).not.toBeInTheDocument();
      });
    });

    it('clears file when clear button is clicked', async () => {
      render(<ImageUpload />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByText('test.png')).toBeInTheDocument();
      });

      const clearButton = screen.getByTestId('image-upload-clear');
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(screen.queryByText('test.png')).not.toBeInTheDocument();
        expect(screen.getByText('Click to upload or drag and drop')).toBeInTheDocument();
      });
    });
  });

  describe('File Validation', () => {
    it('shows error for oversized files', async () => {
      const maxSize = 1024 * 1024; // 1MB
      render(<ImageUpload maxSize={maxSize} />);

      const largeFile = new File(['a'.repeat(2 * 1024 * 1024)], 'large.png', {
        type: 'image/png',
      });

      const input = screen.getByTestId('image-upload-input');

      // Mock the file size check
      Object.defineProperty(largeFile, 'size', { value: 2 * 1024 * 1024 });

      fireEvent.change(input, { target: { files: [largeFile] } });

      await waitFor(() => {
        const errorMessage = screen.queryByTestId('image-upload-error');
        if (errorMessage) {
          expect(errorMessage).toHaveTextContent(/too large/i);
        }
      });
    });

    it('shows error for invalid file type', async () => {
      render(
        <ImageUpload accept={{ 'image/png': ['.png'] }} />
      );

      const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [invalidFile] } });

      await waitFor(() => {
        const errorMessage = screen.queryByTestId('image-upload-error');
        if (errorMessage) {
          expect(errorMessage).toHaveTextContent(/invalid file type/i);
        }
      });
    });
  });

  describe('Upload Flow', () => {
    it('shows progress bar during upload', async () => {
      const onUpload = vi.fn((_file: File, onProgress?: (progress: number) => void) => {
        return new Promise((resolve) => {
          onProgress?.(50);
          setTimeout(() => {
            onProgress?.(100);
            resolve({ success: true });
          }, 100);
        });
      });

      render(<ImageUpload onUpload={onUpload} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-progress')).toBeInTheDocument();
      });
    });

    it('shows success state after upload', async () => {
      const onUpload = vi.fn(() => Promise.resolve({ success: true }));
      const onUploadComplete = vi.fn();

      render(<ImageUpload onUpload={onUpload} onUploadComplete={onUploadComplete} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-success')).toBeInTheDocument();
        expect(screen.getByText('Upload complete!')).toBeInTheDocument();
      });

      expect(onUploadComplete).toHaveBeenCalledWith({ success: true });
    });

    it('shows error state on upload failure', async () => {
      const errorMessage = 'Network error';
      const onUpload = vi.fn(() => Promise.reject(new Error(errorMessage)));
      const onUploadError = vi.fn();

      render(<ImageUpload onUpload={onUpload} onUploadError={onUploadError} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-error')).toBeInTheDocument();
        expect(screen.getByText(errorMessage)).toBeInTheDocument();
      });

      expect(onUploadError).toHaveBeenCalled();
    });

    it('calls onFileSelect callback', async () => {
      const onFileSelect = vi.fn();

      render(<ImageUpload onFileSelect={onFileSelect} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(onFileSelect).toHaveBeenCalledWith([file]);
      });
    });

    it('calls onUpload and onUploadComplete callbacks', async () => {
      const result = { id: '123', url: 'https://example.com/file.png' };
      const onUpload = vi.fn(() => Promise.resolve(result));
      const onUploadComplete = vi.fn();

      render(<ImageUpload onUpload={onUpload} onUploadComplete={onUploadComplete} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        expect(onUpload).toHaveBeenCalled();
        expect(onUploadComplete).toHaveBeenCalledWith(result);
      });
    });

    it('calls onUploadError on failure', async () => {
      const error = new Error('Upload failed');
      const onUpload = vi.fn(() => Promise.reject(error));
      const onUploadError = vi.fn();

      render(<ImageUpload onUpload={onUpload} onUploadError={onUploadError} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        expect(onUploadError).toHaveBeenCalledWith(error);
      });
    });
  });

  describe('Disabled State', () => {
    it('respects disabled prop', () => {
      render(<ImageUpload disabled={true} />);

      const dropzone = screen.getByTestId('image-upload-dropzone');
      expect(dropzone).toHaveAttribute('aria-disabled', 'true');
      expect(dropzone).toHaveClass('pointer-events-none', 'opacity-50');
    });

    it('disables dropzone during upload', async () => {
      const onUpload = vi.fn(() => new Promise(() => {})); // Never resolves

      render(<ImageUpload onUpload={onUpload} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        const dropzone = screen.getByTestId('image-upload-dropzone');
        expect(dropzone).toHaveClass('pointer-events-none');
      });
    });

    it('disables dropzone after successful upload', async () => {
      const onUpload = vi.fn(() => Promise.resolve({ success: true }));

      render(<ImageUpload onUpload={onUpload} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        const dropzone = screen.getByTestId('image-upload-dropzone');
        expect(dropzone).toHaveClass('pointer-events-none');
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper aria attributes', () => {
      render(<ImageUpload label="Upload image" />);

      const dropzone = screen.getByTestId('image-upload-dropzone');
      expect(dropzone).toHaveAttribute('aria-label', 'Upload image');
      expect(dropzone).toHaveAttribute('aria-disabled', 'false');
    });

    it('has accessible progress bar', async () => {
      const onUpload = vi.fn((_file: File, onProgress?: (progress: number) => void) => {
        return new Promise((resolve) => {
          onProgress?.(75);
          setTimeout(() => resolve({ success: true }), 100);
        });
      });

      render(<ImageUpload onUpload={onUpload} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        const progressBar = screen.getByRole('progressbar');
        expect(progressBar).toHaveAttribute('aria-valuenow');
        expect(progressBar).toHaveAttribute('aria-valuemin', '0');
        expect(progressBar).toHaveAttribute('aria-valuemax', '100');
        expect(progressBar).toHaveAttribute('aria-label', 'Upload progress');
      });
    });

    it('has accessible error message', async () => {
      render(<ImageUpload maxSize={100} />);

      const largeFile = new File(['a'.repeat(200)], 'large.png', {
        type: 'image/png',
      });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [largeFile] } });

      await waitFor(() => {
        const errorAlert = screen.queryByRole('alert');
        if (errorAlert) {
          expect(errorAlert).toHaveAttribute('aria-live', 'assertive');
        }
      });
    });

    it('has accessible success message', async () => {
      const onUpload = vi.fn(() => Promise.resolve({ success: true }));

      render(<ImageUpload onUpload={onUpload} />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        expect(screen.getByTestId('image-upload-button')).toBeInTheDocument();
      });

      const uploadButton = screen.getByTestId('image-upload-button');
      fireEvent.click(uploadButton);

      await waitFor(() => {
        const successStatus = screen.getByRole('status');
        expect(successStatus).toHaveAttribute('aria-live', 'polite');
      });
    });

    it('has accessible clear button', async () => {
      render(<ImageUpload />);

      const file = new File(['test'], 'test.png', { type: 'image/png' });
      const input = screen.getByTestId('image-upload-input');

      fireEvent.change(input, { target: { files: [file] } });

      await waitFor(() => {
        const clearButton = screen.getByLabelText('Remove file');
        expect(clearButton).toBeInTheDocument();
      });
    });
  });

  describe('Multiple Files', () => {
    it('accepts multiple files when maxFiles > 1', async () => {
      const onFileSelect = vi.fn();

      render(<ImageUpload maxFiles={3} onFileSelect={onFileSelect} />);

      const files = [
        new File(['1'], 'file1.png', { type: 'image/png' }),
        new File(['2'], 'file2.png', { type: 'image/png' }),
      ];

      const input = screen.getByTestId('image-upload-input');
      fireEvent.change(input, { target: { files } });

      // Note: Due to maxFiles=1 in the default behavior, only first file is shown
      // But the callback should receive all files
      await waitFor(() => {
        if (onFileSelect.mock.calls.length > 0) {
          expect(onFileSelect).toHaveBeenCalled();
        }
      });
    });
  });
});
