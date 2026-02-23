import type React from 'react';
import { useCallback, useState } from 'react';
import { type FileRejection, useDropzone } from 'react-dropzone';
import { cn } from '@/lib/utils';
import { api } from '@/api/client';
import { Button } from '@/components/ui/Button';
import {
  AlertCircle,
  CheckCircle2,
  File as FileIcon,
  UploadCloud,
  X,
} from 'lucide-react';
import type { ScanUploadResponse } from '@/types/models';

interface ScanUploadProps {
  onUploadComplete?: (response: ScanUploadResponse) => void;
  className?: string;
  targetType?: string;
}

export function ScanUpload({
  onUploadComplete,
  className = '',
  targetType = 'stouffer_21',
}: ScanUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<boolean>(false);

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      setError(null);
      setSuccess(false);

      if (fileRejections.length > 0) {
        setError('Invalid file type. Please upload an image (PNG, JPG, TIFF).');
        return;
      }

      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0] ?? null);
      }
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.tif'],
    },
    maxFiles: 1,
    maxSize: 20 * 1024 * 1024, // 20MB
    disabled: isUploading,
  });

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setProgress(0);
    setError(null);

    try {
      const response = await api.scan.upload(file, targetType, (p) =>
        setProgress(p)
      );

      if (response.success) {
        setSuccess(true);
        if (onUploadComplete) {
          onUploadComplete(response);
        }
      } else {
        setError('Upload failed but no error message returned.');
      }
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : 'Upload failed. Please try again.'
      );
    } finally {
      setIsUploading(false);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setProgress(0);
    setError(null);
    setSuccess(false);
  };

  return (
    <div className={`w-full max-w-md ${className}`}>
      <div
        {...getRootProps()}
        className={cn(
          'relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition-all',
          isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-gray-300 hover:border-primary/50',
          error ? 'border-destructive bg-destructive/5' : '',
          success ? 'border-green-500 bg-green-50' : '',
          isUploading ? 'pointer-events-none opacity-50' : 'hover:bg-accent/5'
        )}
      >
        <input {...getInputProps()} data-testid="scan-upload-input" />

        {/* State: Initial / Drag Assert */}
        {!file && !success && (
          <div className="pointer-events-none space-y-4">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-muted/50">
              <UploadCloud className={cn("h-8 w-8", isDragActive ? "text-primary animate-bounce" : "text-muted-foreground")} />
            </div>
            <div>
              <p className="text-lg font-medium">
                {isDragActive ? 'Drop it here!' : 'Click or drag scan image'}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                Supports PNG, JPG, TIFF (up to 20MB)
              </p>
            </div>
          </div>
        )}

        {/* State: File Selected / Uploading / Success */}
        {file && (
          <div className="w-full space-y-4">
            <div className="flex items-center justify-between rounded-md border bg-white p-3 shadow-sm">
              <div className="flex items-center gap-3 overflow-hidden">
                <div
                  className={`rounded-full p-2 ${success ? 'bg-green-100' : 'bg-blue-50'}`}
                >
                  {success ? (
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                  ) : (
                    <FileIcon className="h-5 w-5 text-blue-500" />
                  )}
                </div>
                <div className="flex flex-col items-start truncate">
                  <span className="max-w-[180px] truncate text-sm font-medium">
                    {file.name}
                  </span>
                  <span className="text-xs text-gray-500">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
              </div>

              {!success && !isUploading && (
                <button
                  onClick={clearFile}
                  className="rounded-full p-1 transition-colors hover:bg-gray-100"
                  aria-label="Remove file"
                >
                  <X className="h-4 w-4 text-gray-500" />
                </button>
              )}
            </div>

            {/* Progress Bar */}
            {isUploading && (
              <div className="space-y-1">
                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                  <div
                    className="h-full bg-primary transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-right text-xs text-gray-500">{progress}%</p>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="flex items-center justify-center gap-2 text-sm text-green-600">
                <CheckCircle2 className="h-4 w-4" />
                <span>Upload complete!</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Upload Button */}
      {file && !success && !isUploading && (
        <Button
          onClick={handleUpload}
          className="mt-4 w-full"
          disabled={!!error}
        >
          Upload Scan
        </Button>
      )}
    </div>
  );
}
