/**
 * ImageUpload Component
 * A reusable file upload component with drag-and-drop, progress tracking,
 * and multiple upload states. Supports image preview and configurable file types.
 */

import { useCallback, useEffect, useState } from 'react';
import { type FileRejection, useDropzone } from 'react-dropzone';
import {
  AlertCircle,
  CheckCircle2,
  FileIcon,
  Image as ImageIcon,
  UploadCloud,
  X,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';

/** Default maximum file size: 20MB */
const DEFAULT_MAX_SIZE = 20 * 1024 * 1024;

/** Default accepted image file types */
const DEFAULT_ACCEPT = {
  'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.webp'],
};

type UploadState = 'idle' | 'selected' | 'uploading' | 'success' | 'error';

export interface ImageUploadProps {
  /** Callback when file(s) are selected */
  onFileSelect?: (files: File[]) => void;
  /** Callback for upload with progress tracking */
  onUpload?: (file: File, onProgress: (percent: number) => void) => Promise<unknown>;
  /** Callback when upload completes */
  onUploadComplete?: (result: unknown) => void;
  /** Callback when upload fails */
  onUploadError?: (error: Error) => void;
  /** Accepted file types for react-dropzone */
  accept?: Record<string, string[]>;
  /** Maximum file size in bytes (default: 20MB) */
  maxSize?: number;
  /** Maximum number of files (default: 1) */
  maxFiles?: number;
  /** Whether to show image thumbnail preview */
  showPreview?: boolean;
  /** Label text for the upload area */
  label?: string;
  /** Help text below the label */
  helpText?: string;
  /** External disabled state */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ImageUpload - Reusable file upload component
 * Supports drag-and-drop, file validation, progress tracking, and image preview
 */
export function ImageUpload({
  onFileSelect,
  onUpload,
  onUploadComplete,
  onUploadError,
  accept = DEFAULT_ACCEPT,
  maxSize = DEFAULT_MAX_SIZE,
  maxFiles = 1,
  showPreview = false,
  label = 'Click to upload or drag and drop',
  helpText,
  disabled = false,
  className = '',
}: ImageUploadProps): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [error, setError] = useState<string | null>(null);

  // Generate help text from accept prop if not provided
  const defaultHelpText = useCallback(() => {
    const extensions = Object.values(accept).flat();
    const maxSizeMB = (maxSize / 1024 / 1024).toFixed(0);
    return `${extensions.slice(0, 3).join(', ')} (max ${maxSizeMB}MB)`;
  }, [accept, maxSize]);

  const finalHelpText = helpText ?? defaultHelpText();

  // Cleanup preview URL on unmount or file change
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      logger.debug('ImageUpload.onDrop', {
        acceptedCount: acceptedFiles.length,
        rejectedCount: fileRejections.length,
      });

      setError(null);
      setUploadState('idle');

      // Handle file rejections
      if (fileRejections.length > 0) {
        const rejection = fileRejections[0];
        if (rejection?.errors[0]?.code === 'file-too-large') {
          setError(`File is too large. Maximum size is ${(maxSize / 1024 / 1024).toFixed(0)}MB.`);
        } else if (rejection?.errors[0]?.code === 'file-invalid-type') {
          setError('Invalid file type. Please select a valid image file.');
        } else {
          setError(rejection?.errors[0]?.message ?? 'File validation failed.');
        }
        setUploadState('error');
        logger.warn('ImageUpload file rejected', {
          error: rejection?.errors[0],
        });
        return;
      }

      // Handle accepted files
      if (acceptedFiles.length > 0) {
        const selectedFile = acceptedFiles[0];
        if (selectedFile) {
          setFile(selectedFile);
          setUploadState('selected');
          logger.info('ImageUpload file selected', {
            name: selectedFile.name,
            size: selectedFile.size,
            type: selectedFile.type,
          });

          // Create preview URL for images
          if (showPreview && selectedFile.type.startsWith('image/')) {
            const url = URL.createObjectURL(selectedFile);
            setPreviewUrl(url);
          }

          // Call onFileSelect callback
          onFileSelect?.(acceptedFiles);
        }
      }
    },
    [maxSize, showPreview, onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles,
    maxSize,
    disabled: disabled || uploadState === 'uploading' || uploadState === 'success',
  });

  const handleUpload = async () => {
    if (!file || !onUpload) return;

    setUploadState('uploading');
    setProgress(0);
    setError(null);

    logger.info('ImageUpload starting upload', { fileName: file.name });

    try {
      const result = await onUpload(file, (percent) => {
        setProgress(Math.min(100, Math.max(0, percent)));
        logger.debug('ImageUpload progress', { percent });
      });

      setUploadState('success');
      logger.info('ImageUpload completed', { fileName: file.name });

      onUploadComplete?.(result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      setError(error.message || 'Upload failed. Please try again.');
      setUploadState('error');
      logger.error('ImageUpload failed', {
        fileName: file.name,
        error: error.message,
      });

      onUploadError?.(error);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    logger.debug('ImageUpload clearing file');

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setFile(null);
    setPreviewUrl(null);
    setProgress(0);
    setError(null);
    setUploadState('idle');
  };

  const isInteractive = !disabled && uploadState !== 'uploading' && uploadState !== 'success';

  return (
    <div className={cn('w-full max-w-md', className)} data-testid="image-upload-container">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={cn(
          'relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition-colors',
          {
            'border-primary bg-primary/5': isDragActive,
            'border-gray-300 hover:border-primary/50': isInteractive && !isDragActive && uploadState === 'idle',
            'border-destructive bg-destructive/5': uploadState === 'error',
            'border-green-500 bg-green-50': uploadState === 'success',
            'pointer-events-none opacity-50': !isInteractive,
          }
        )}
        data-testid="image-upload-dropzone"
        aria-label={label}
        aria-disabled={!isInteractive}
      >
        <input {...getInputProps()} data-testid="image-upload-input" />

        {/* State: Initial / Drag Active */}
        {!file && uploadState !== 'success' && (
          <div className="space-y-4">
            <div className="inline-block rounded-full bg-gray-100 p-4">
              <UploadCloud className="h-8 w-8 text-gray-400" aria-hidden="true" />
            </div>
            <div>
              <p className="text-lg font-medium">{label}</p>
              <p className="mt-1 text-sm text-gray-500">{finalHelpText}</p>
            </div>
          </div>
        )}

        {/* State: File Selected / Uploading / Success */}
        {file && (
          <div className="w-full space-y-4">
            {/* Image Preview */}
            {showPreview && previewUrl && (
              <div className="flex justify-center">
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="h-32 w-32 rounded-md object-cover shadow-sm"
                  data-testid="image-upload-preview"
                />
              </div>
            )}

            {/* File Info Card */}
            <div className="flex items-center justify-between rounded-md border bg-white p-3 shadow-sm">
              <div className="flex items-center gap-3 overflow-hidden">
                <div
                  className={cn('rounded-full p-2', {
                    'bg-green-100': uploadState === 'success',
                    'bg-blue-50': uploadState !== 'success',
                  })}
                >
                  {uploadState === 'success' ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600" aria-hidden="true" />
                  ) : file.type.startsWith('image/') ? (
                    <ImageIcon className="h-5 w-5 text-blue-500" aria-hidden="true" />
                  ) : (
                    <FileIcon className="h-5 w-5 text-blue-500" aria-hidden="true" />
                  )}
                </div>
                <div className="flex flex-col items-start truncate">
                  <span
                    className="max-w-[180px] truncate text-sm font-medium"
                    title={file.name}
                  >
                    {file.name}
                  </span>
                  <span className="text-xs text-gray-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>

              {uploadState !== 'success' && uploadState !== 'uploading' && (
                <button
                  onClick={clearFile}
                  className="rounded-full p-1 transition-colors hover:bg-gray-100"
                  aria-label="Remove file"
                  data-testid="image-upload-clear"
                  type="button"
                >
                  <X className="h-4 w-4 text-gray-500" aria-hidden="true" />
                </button>
              )}
            </div>

            {/* Progress Bar */}
            {uploadState === 'uploading' && (
              <div className="space-y-1" data-testid="image-upload-progress">
                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full bg-primary transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                    role="progressbar"
                    aria-valuenow={progress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label="Upload progress"
                  />
                </div>
                <p className="text-right text-xs text-gray-500">{Math.round(progress)}%</p>
              </div>
            )}

            {/* Success Message */}
            {uploadState === 'success' && (
              <div
                className="flex items-center justify-center gap-2 text-sm text-green-600"
                role="status"
                aria-live="polite"
                data-testid="image-upload-success"
              >
                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                <span>Upload complete!</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && uploadState === 'error' && (
        <div
          className="mt-4 flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive"
          role="alert"
          aria-live="assertive"
          data-testid="image-upload-error"
        >
          <AlertCircle className="h-4 w-4" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {/* Upload Button */}
      {file && uploadState === 'selected' && onUpload && (
        <Button
          onClick={handleUpload}
          className="mt-4 w-full"
          disabled={!!error || disabled}
          data-testid="image-upload-button"
        >
          Upload File
        </Button>
      )}
    </div>
  );
}
