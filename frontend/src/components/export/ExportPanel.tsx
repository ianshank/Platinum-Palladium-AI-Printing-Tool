import { type ComponentPropsWithoutRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileDown,
  RefreshCw,
} from 'lucide-react';
import { cn, downloadFile } from '@/lib/utils';
import { logger } from '@/lib/logger';
import { Button } from '@/components/ui/Button';

export interface ExportFormat {
  id: string;
  label: string;
  extension: string;
  description?: string;
}

export interface ExportPanelProps extends ComponentPropsWithoutRef<'div'> {
  /** Available export formats */
  formats: ExportFormat[];
  /** Default selected format ID */
  defaultFormat?: string;
  /** Called when export is requested. Receives format ID, should return a Blob or throw. */
  onExport: (formatId: string) => Promise<Blob>;
  /** Base filename (without extension) for the downloaded file */
  fileName?: string;
  /** Title text */
  title?: string;
  /** Description text */
  description?: string;
  /** Whether an export is in progress (controlled mode) */
  isExporting?: boolean;
  /** Called when export completes successfully */
  onExportComplete?: (formatId: string) => void;
  /** Called when export fails */
  onExportError?: (error: Error) => void;
  /** Disable the panel */
  disabled?: boolean;
}

type ExportState = 'idle' | 'loading' | 'success' | 'error';

/**
 * ExportPanel component for exporting data in various formats
 * Provides format selection, export triggering, and download functionality
 */
export function ExportPanel({
  formats,
  defaultFormat,
  onExport,
  fileName = 'export',
  title = 'Export',
  description,
  isExporting: controlledIsExporting,
  onExportComplete,
  onExportError,
  disabled = false,
  className,
  ...props
}: ExportPanelProps) {
  // Local state
  const [selectedFormat, setSelectedFormat] = useState<string>(
    defaultFormat ?? formats[0]?.id ?? ''
  );
  const [exportState, setExportState] = useState<ExportState>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  // Use controlled or internal loading state
  const isExporting = controlledIsExporting ?? exportState === 'loading';

  const handleFormatChange = (formatId: string) => {
    if (disabled || isExporting) return;
    logger.debug('ExportPanel: format changed', { formatId });
    setSelectedFormat(formatId);
    setExportState('idle');
    setErrorMessage('');
  };

  const handleExport = async () => {
    if (disabled || isExporting || !selectedFormat) return;

    logger.info('ExportPanel: starting export', { format: selectedFormat });
    setExportState('loading');
    setErrorMessage('');

    try {
      const blob = await onExport(selectedFormat);

      // Find the format to get extension
      const format = formats.find((f) => f.id === selectedFormat);
      const ext = format?.extension ?? '';

      // Sanitize filename (replace spaces with underscores)
      const safeName = fileName.replace(/\s+/g, '_');
      const fullFileName = `${safeName}${ext}`;

      logger.debug('ExportPanel: download starting', { fileName: fullFileName });
      downloadFile(blob, fullFileName);

      setExportState('success');
      logger.info('ExportPanel: export completed', {
        format: selectedFormat,
        fileName: fullFileName,
      });

      onExportComplete?.(selectedFormat);

      // Reset success state after 3 seconds
      setTimeout(() => {
        setExportState('idle');
      }, 3000);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      logger.error('ExportPanel: export failed', {
        format: selectedFormat,
        error: error.message,
      });

      setExportState('error');
      setErrorMessage(error.message || 'Export failed');
      onExportError?.(error);
    }
  };

  const handleRetry = () => {
    logger.debug('ExportPanel: retry requested');
    setExportState('idle');
    setErrorMessage('');
    void handleExport();
  };

  const selectedFormatData = formats.find((f) => f.id === selectedFormat);

  return (
    <div
      className={cn('flex flex-col gap-6', className)}
      data-testid="export-panel"
      {...props}
    >
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <FileDown className="h-5 w-5 text-primary" />
          <h3 className="text-xl font-semibold text-foreground">{title}</h3>
        </div>
        {description && (
          <p className="text-sm text-muted-foreground">{description}</p>
        )}
      </div>

      {/* Format Selection */}
      <div
        role="radiogroup"
        aria-label="Export format selection"
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
      >
        {formats.map((format) => {
          const isSelected = selectedFormat === format.id;
          return (
            <label
              key={format.id}
              className={cn(
                'relative flex cursor-pointer flex-col rounded-md border-2 p-4 transition-all',
                'hover:border-primary/50 focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-2',
                isSelected
                  ? 'border-primary bg-primary/5'
                  : 'border-border bg-background',
                (disabled || isExporting) && 'cursor-not-allowed opacity-50'
              )}
              data-testid={`format-option-${format.id}`}
            >
              <input
                type="radio"
                name="exportFormat"
                value={format.id}
                checked={isSelected}
                onChange={() => handleFormatChange(format.id)}
                disabled={disabled || isExporting}
                className="sr-only"
                aria-label={`${format.label} ${format.extension}`}
              />
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="font-medium text-foreground">
                    {format.label}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {format.extension}
                  </div>
                  {format.description && (
                    <div className="mt-1 text-xs text-muted-foreground">
                      {format.description}
                    </div>
                  )}
                </div>
                {isSelected && (
                  <CheckCircle2 className="h-5 w-5 shrink-0 text-primary" />
                )}
              </div>
            </label>
          );
        })}
      </div>

      {/* Export Button and Status */}
      <div className="flex flex-col gap-4">
        {exportState === 'success' && (
          <div
            className="flex items-center gap-2 rounded-md border border-success bg-success/10 p-3 text-sm text-success"
            data-testid="export-success"
            role="alert"
            aria-live="polite"
          >
            <CheckCircle2 className="h-4 w-4" />
            <span>
              Export successful! File downloaded as{' '}
              <strong>
                {fileName.replace(/\s+/g, '_')}
                {selectedFormatData?.extension}
              </strong>
            </span>
          </div>
        )}

        {exportState === 'error' && (
          <div
            className="flex flex-col gap-2 rounded-md border border-destructive bg-destructive/10 p-3"
            data-testid="export-error"
            role="alert"
            aria-live="assertive"
          >
            <div className="flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4" />
              <span>
                <strong>Export failed:</strong> {errorMessage}
              </span>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRetry}
              disabled={disabled}
              className="w-fit"
              data-testid="export-retry-button"
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </div>
        )}

        <div className="flex items-center gap-3">
          <Button
            variant="default"
            size="default"
            onClick={handleExport}
            disabled={disabled || !selectedFormat}
            isLoading={isExporting}
            loadingText="Preparing download..."
            className="min-w-[160px]"
            data-testid="export-button"
          >
            <Download className="mr-2 h-4 w-4" />
            Download {selectedFormatData?.label ?? 'File'}
          </Button>

          {selectedFormatData && (
            <span className="text-sm text-muted-foreground">
              Format: {selectedFormatData.label} ({selectedFormatData.extension}
              )
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

ExportPanel.displayName = 'ExportPanel';
