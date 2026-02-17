/**
 * CurveUpload component
 *
 * Allows users to upload .quad (QuadTone RIP) curve files via drag-and-drop
 * or paste raw .quad content directly. Supports channel selection and provides
 * a preview of the parsed curve profile before loading into the editor.
 */

import { useCallback, useMemo, useState } from 'react';
import { type FileRejection, useDropzone } from 'react-dropzone';
import { Button } from '@/components/ui/Button';
import { useParseQuadContent, useUploadQuadFile } from '@/api/hooks';
import { QUAD_CHANNELS } from '@/types/models';
import type {
  QuadChannel,
  QuadCurveValues,
  QuadParseResponse,
  QuadUploadResponse,
} from '@/types/models';
import { logger } from '@/lib/logger';
import {
  AlertCircle,
  CheckCircle2,
  ClipboardPaste,
  File as FileIcon,
  Upload,
  UploadCloud,
  X,
} from 'lucide-react';

// --- Constants ---

/** Maximum upload file size in bytes (5 MB — .quad files are text-based, typically < 100 KB) */
const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;

/** Accepted MIME types and extensions for .quad files */
const ACCEPTED_FILE_TYPES: Record<string, string[]> = {
  'text/plain': ['.quad', '.txt'],
  'application/octet-stream': ['.quad'],
};

// --- Types ---

type UploadMode = 'file' | 'paste';

interface ParsedProfile {
  profileName: string;
  activeChannels: string[];
  curveId: string | null;
  curveData: QuadCurveValues | null;
  summary?: string;
  resolution?: number;
  allChannels?: string[];
}

export interface CurveUploadProps {
  /** Callback when user clicks "Load into Editor" with parsed curve data */
  onLoadCurve?: (
    curveData: QuadCurveValues,
    curveId: string,
    profileName: string
  ) => void;
  className?: string;
}

// --- Helpers ---

/** Normalize both response shapes into a unified ParsedProfile */
function toProfile(
  response: QuadUploadResponse | QuadParseResponse
): ParsedProfile {
  const base: ParsedProfile = {
    profileName: response.profile_name,
    activeChannels: response.active_channels,
    curveId: response.curve_id,
    curveData: response.curve_data,
  };
  // QuadUploadResponse has extra metadata
  if ('summary' in response) {
    base.summary = response.summary;
    base.resolution = response.resolution;
    base.allChannels = response.all_channels;
  }
  return base;
}

// --- Component ---

export function CurveUpload({ onLoadCurve, className = '' }: CurveUploadProps) {
  // --- State ---
  const [mode, setMode] = useState<UploadMode>('file');
  const [file, setFile] = useState<File | null>(null);
  const [channel, setChannel] = useState<QuadChannel>('K');
  const [pasteContent, setPasteContent] = useState('');
  const [pasteName, setPasteName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [parsedProfile, setParsedProfile] = useState<ParsedProfile | null>(
    null
  );

  // --- Mutations ---
  const uploadMutation = useUploadQuadFile();
  const parseMutation = useParseQuadContent();
  const isProcessing = uploadMutation.isPending || parseMutation.isPending;

  // --- File handling ---
  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]) => {
      setError(null);
      setParsedProfile(null);

      if (rejections.length > 0) {
        const isSizeError = rejections.some((r) =>
          r.errors.some((e) => e.code === 'file-too-large')
        );
        setError(
          isSizeError
            ? `File is too large. Max size is ${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB.`
            : 'Invalid file type. Please upload a .quad or .txt file.'
        );
        return;
      }

      if (accepted.length > 0) {
        setFile(accepted[0] ?? null);
        logger.debug('Quad file selected', { name: accepted[0]?.name });
      }
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_FILE_TYPES,
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE_BYTES,
    disabled: isProcessing || !!parsedProfile,
  });

  const clearFile = useCallback((e?: React.MouseEvent) => {
    e?.stopPropagation();
    setFile(null);
    setError(null);
    setParsedProfile(null);
  }, []);

  // --- Upload / Parse ---
  const handleUpload = useCallback(async () => {
    if (!file) return;
    setError(null);

    try {
      const response = await uploadMutation.mutateAsync({ file, channel });
      setParsedProfile(toProfile(response));
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Upload failed. Please try again.';
      setError(message);
    }
  }, [file, channel, uploadMutation]);

  const handleParse = useCallback(async () => {
    const trimmed = pasteContent.trim();
    if (!trimmed) {
      setError('Please paste .quad content first.');
      return;
    }
    setError(null);

    try {
      const trimmedName = pasteName.trim();
      const response = await parseMutation.mutateAsync({
        content: trimmed,
        ...(trimmedName ? { name: trimmedName } : {}),
        channel: channel as string,
      });
      setParsedProfile(toProfile(response));
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Parse failed. Please try again.';
      setError(message);
    }
  }, [pasteContent, pasteName, channel, parseMutation]);

  const handleLoadIntoEditor = useCallback(() => {
    if (!parsedProfile?.curveData || !parsedProfile.curveId) {
      logger.warn('No curve data to load');
      return;
    }
    onLoadCurve?.(
      parsedProfile.curveData,
      parsedProfile.curveId,
      parsedProfile.profileName
    );
  }, [parsedProfile, onLoadCurve]);

  // --- Reset everything ---
  const handleReset = useCallback(() => {
    setFile(null);
    setPasteContent('');
    setPasteName('');
    setError(null);
    setParsedProfile(null);
  }, []);

  // --- Derived ---
  const canUpload =
    mode === 'file' && !!file && !isProcessing && !parsedProfile;
  const canParse =
    mode === 'paste' &&
    pasteContent.trim().length > 0 &&
    !isProcessing &&
    !parsedProfile;
  const canLoadIntoEditor =
    !!parsedProfile?.curveData && !!parsedProfile.curveId;

  // Memoize channel options to prevent re-renders
  const channelOptions = useMemo(
    () => QUAD_CHANNELS.map((ch) => ({ value: ch, label: ch })),
    []
  );

  // --- Render ---
  return (
    <div className={`w-full ${className}`}>
      {/* Mode Tabs */}
      <div className="mb-4 flex gap-2" role="tablist" aria-label="Upload mode">
        <button
          role="tab"
          aria-selected={mode === 'file' ? 'true' : 'false'}
          className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            mode === 'file'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground hover:bg-muted/80'
          }`}
          onClick={() => {
            setMode('file');
            handleReset();
          }}
          disabled={isProcessing}
        >
          <Upload className="h-4 w-4" />
          Upload File
        </button>
        <button
          role="tab"
          aria-selected={mode === 'paste' ? 'true' : 'false'}
          className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            mode === 'paste'
              ? 'bg-primary text-primary-foreground'
              : 'bg-muted text-muted-foreground hover:bg-muted/80'
          }`}
          onClick={() => {
            setMode('paste');
            handleReset();
          }}
          disabled={isProcessing}
        >
          <ClipboardPaste className="h-4 w-4" />
          Paste Content
        </button>
      </div>

      {/* Channel Selector */}
      <div className="mb-4">
        <label
          htmlFor="channel-select"
          className="mb-1 block text-sm font-medium"
        >
          Ink Channel
        </label>
        <select
          id="channel-select"
          value={channel}
          onChange={(e) => setChannel(e.target.value as QuadChannel)}
          className="w-full max-w-[200px] rounded-md border bg-background px-3 py-2 text-sm"
          disabled={isProcessing || !!parsedProfile}
        >
          {channelOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* File Upload Mode */}
      {mode === 'file' && !parsedProfile && (
        <div
          {...getRootProps()}
          className={`relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 text-center transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary/50'} ${error ? 'border-destructive bg-destructive/5' : ''} ${isProcessing ? 'pointer-events-none opacity-50' : ''} `}
        >
          <input {...getInputProps()} data-testid="quad-upload-input" />

          {!file && (
            <div className="space-y-4">
              <div className="inline-block rounded-full bg-gray-100 p-4">
                <UploadCloud className="h-8 w-8 text-gray-400" />
              </div>
              <div>
                <p className="text-lg font-medium">
                  Drop a .quad file here or click to browse
                </p>
                <p className="mt-1 text-sm text-gray-500">
                  QTR profile files (.quad, .txt) up to{' '}
                  {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB
                </p>
              </div>
            </div>
          )}

          {file && (
            <div className="w-full space-y-2">
              <div className="flex items-center justify-between rounded-md border bg-background p-3 shadow-sm">
                <div className="flex items-center gap-3 overflow-hidden">
                  <div className="rounded-full bg-blue-50/10 p-2">
                    <FileIcon className="h-5 w-5 text-blue-500" />
                  </div>
                  <div className="flex flex-col items-start truncate">
                    <span className="max-w-[200px] truncate text-sm font-medium">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-500">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                </div>
                {!isProcessing && (
                  <button
                    onClick={clearFile}
                    className="rounded-full p-1 transition-colors hover:bg-gray-100"
                    title="Remove file"
                    aria-label="Remove file"
                  >
                    <X className="h-4 w-4 text-gray-500" />
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Paste Mode */}
      {mode === 'paste' && !parsedProfile && (
        <div className="space-y-3">
          <div>
            <label
              htmlFor="paste-name"
              className="mb-1 block text-sm font-medium"
            >
              Profile Name (optional)
            </label>
            <input
              id="paste-name"
              type="text"
              value={pasteName}
              onChange={(e) => setPasteName(e.target.value)}
              placeholder="e.g. My Custom Profile"
              className="w-full rounded-md border bg-background px-3 py-2 text-sm"
              disabled={isProcessing}
            />
          </div>
          <div>
            <label
              htmlFor="paste-content"
              className="mb-1 block text-sm font-medium"
            >
              .quad File Content
            </label>
            <textarea
              id="paste-content"
              value={pasteContent}
              onChange={(e) => setPasteContent(e.target.value)}
              placeholder="Paste your .quad file content here..."
              rows={10}
              className="w-full resize-y rounded-md border bg-background px-3 py-2 font-mono text-sm"
              disabled={isProcessing}
              data-testid="quad-paste-input"
            />
          </div>
        </div>
      )}

      {/* Profile Preview */}
      {parsedProfile && (
        <div
          className="space-y-3 rounded-lg border bg-card p-4"
          data-testid="quad-profile-preview"
        >
          <div className="mb-2 flex items-center gap-2 text-green-600">
            <CheckCircle2 className="h-5 w-5" />
            <span className="font-medium">Profile Loaded</span>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Profile Name</dt>
            <dd className="font-medium">{parsedProfile.profileName}</dd>
            <dt className="text-muted-foreground">Selected Channel</dt>
            <dd className="font-medium">{channel}</dd>
            <dt className="text-muted-foreground">Active Channels</dt>
            <dd className="font-medium">
              {parsedProfile.activeChannels.join(', ') || 'None'}
            </dd>
            {parsedProfile.resolution != null && (
              <>
                <dt className="text-muted-foreground">Resolution</dt>
                <dd className="font-medium">{parsedProfile.resolution} DPI</dd>
              </>
            )}
            {parsedProfile.curveData && (
              <>
                <dt className="text-muted-foreground">Data Points</dt>
                <dd className="font-medium">
                  {parsedProfile.curveData.input_values.length}
                </dd>
              </>
            )}
          </dl>
          {parsedProfile.summary && (
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs text-muted-foreground">
              {parsedProfile.summary}
            </pre>
          )}
          <div className="flex gap-2 pt-2">
            <Button
              onClick={handleLoadIntoEditor}
              disabled={!canLoadIntoEditor}
              className="flex-1"
            >
              Load into Editor
            </Button>
            <Button variant="outline" onClick={handleReset}>
              Upload Another
            </Button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 flex items-center gap-2 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Action Button (below dropzone / textarea) */}
      {!parsedProfile && (
        <div className="mt-4">
          {mode === 'file' && (
            <Button
              onClick={handleUpload}
              className="w-full"
              disabled={!canUpload}
            >
              {isProcessing ? 'Uploading...' : 'Upload & Parse'}
            </Button>
          )}
          {mode === 'paste' && (
            <Button
              onClick={handleParse}
              className="w-full"
              disabled={!canParse}
            >
              {isProcessing ? 'Parsing...' : 'Parse Content'}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
