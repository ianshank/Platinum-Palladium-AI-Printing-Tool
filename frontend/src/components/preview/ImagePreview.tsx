import { type ComponentPropsWithoutRef, forwardRef, useCallback, useEffect, useState } from 'react';
import { type ReactZoomPanPinchRef, TransformComponent, TransformWrapper } from 'react-zoom-pan-pinch';
import { AlertCircle, Info, Maximize2, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { cn } from '@/lib/utils';
import { logger } from '@/lib/logger';

export interface ImagePreviewProps extends Omit<ComponentPropsWithoutRef<'div'>, 'children'> {
  /** Primary image source (URL or data URL) */
  src: string | null;
  /** Alt text for accessibility */
  alt?: string;
  /** Secondary image for side-by-side comparison */
  compareSrc?: string | null;
  /** Alt text for comparison image */
  compareAlt?: string;
  /** Show zoom controls toolbar */
  showControls?: boolean;
  /** Show image metadata overlay */
  showMetadata?: boolean;
  /** Minimum zoom level */
  minZoom?: number;
  /** Maximum zoom level */
  maxZoom?: number;
  /** Initial zoom level */
  initialZoom?: number;
  /** Called when zoom changes */
  onZoomChange?: (zoom: number) => void;
  /** Loading state */
  isLoading?: boolean;
  /** Error message to display */
  error?: string | null;
}

interface ImageMetadata {
  width: number;
  height: number;
  aspectRatio: number;
}

export const ImagePreview = forwardRef<HTMLDivElement, ImagePreviewProps>(
  (
    {
      src,
      alt = 'Image preview',
      compareSrc,
      compareAlt = 'Comparison image',
      showControls = true,
      showMetadata = true,
      minZoom = 0.5,
      maxZoom = 5,
      initialZoom = 1,
      onZoomChange,
      isLoading = false,
      error = null,
      className,
      ...props
    },
    ref
  ) => {
    const [primaryMetadata, setPrimaryMetadata] = useState<ImageMetadata | null>(null);
    const [compareMetadata, setCompareMetadata] = useState<ImageMetadata | null>(null);
    const [currentZoom, setCurrentZoom] = useState(initialZoom);
    const [showMetadataOverlay, setShowMetadataOverlay] = useState(false);

    const isComparisonMode = Boolean(compareSrc);

    logger.debug('ImagePreview render', {
      hasSrc: Boolean(src),
      hasCompareSrc: Boolean(compareSrc),
      isLoading,
      error,
      currentZoom,
    });

    // Handle primary image load
    const handlePrimaryImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
      const img = e.currentTarget;
      const metadata: ImageMetadata = {
        width: img.naturalWidth,
        height: img.naturalHeight,
        aspectRatio: img.naturalWidth / img.naturalHeight,
      };
      setPrimaryMetadata(metadata);
      logger.debug('Primary image loaded', { ...metadata });
    }, []);

    // Handle comparison image load
    const handleCompareImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
      const img = e.currentTarget;
      const metadata: ImageMetadata = {
        width: img.naturalWidth,
        height: img.naturalHeight,
        aspectRatio: img.naturalWidth / img.naturalHeight,
      };
      setCompareMetadata(metadata);
      logger.debug('Comparison image loaded', { ...metadata });
    }, []);

    // Handle zoom change
    const handleZoomChange = useCallback(
      (ref: ReactZoomPanPinchRef) => {
        const newZoom = ref.state.scale;
        setCurrentZoom(newZoom);
        onZoomChange?.(newZoom);
        logger.debug('Zoom changed', { zoom: newZoom });
      },
      [onZoomChange]
    );

    // Keyboard shortcuts
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        // Only handle shortcuts if not typing in an input
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
          return;
        }

        // We can't easily access the TransformWrapper controls here,
        // so keyboard shortcuts will be handled via the toolbar buttons
      };

      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    // Loading state
    if (isLoading) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center bg-gray-100 rounded-lg border border-gray-300',
            'min-h-[400px]',
            className
          )}
          data-testid="image-preview-loading"
          {...props}
        >
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-gray-300 border-t-blue-600 rounded-full animate-spin" />
            <p className="text-sm text-gray-600">Loading image...</p>
          </div>
        </div>
      );
    }

    // Error state
    if (error) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center bg-red-50 rounded-lg border border-red-300',
            'min-h-[400px]',
            className
          )}
          data-testid="image-preview-error"
          role="alert"
          aria-live="polite"
          {...props}
        >
          <div className="flex flex-col items-center gap-3 max-w-md px-4">
            <AlertCircle className="w-12 h-12 text-red-600" aria-hidden="true" />
            <p className="text-sm font-medium text-red-900">Failed to load image</p>
            <p className="text-xs text-red-700 text-center">{error}</p>
          </div>
        </div>
      );
    }

    // Empty state
    if (!src) {
      return (
        <div
          ref={ref}
          className={cn(
            'flex items-center justify-center bg-gray-50 rounded-lg border border-dashed border-gray-300',
            'min-h-[400px]',
            className
          )}
          data-testid="image-preview-empty"
          {...props}
        >
          <p className="text-sm text-gray-500">No image to display</p>
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn('relative bg-gray-100 rounded-lg border border-gray-300 overflow-hidden', className)}
        data-testid="image-preview"
        {...props}
      >
        <TransformWrapper
          initialScale={initialZoom}
          minScale={minZoom}
          maxScale={maxZoom}
          centerOnInit
          onTransformed={handleZoomChange}
          wheel={{ step: 0.1 }}
          doubleClick={{ mode: 'reset' }}
        >
          {(utils) => (
            <>
              {/* Zoom Controls */}
              {showControls && (
                <div
                  className="absolute top-4 right-4 z-10 flex flex-col gap-1 bg-white rounded-lg shadow-lg border border-gray-200 p-1"
                  data-testid="zoom-controls"
                  role="toolbar"
                  aria-label="Image zoom controls"
                >
                  <button
                    onClick={() => utils.zoomIn()}
                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                    title="Zoom in (+)"
                    aria-label="Zoom in"
                    data-testid="zoom-in-button"
                    type="button"
                  >
                    <ZoomIn className="w-5 h-5 text-gray-700" aria-hidden="true" />
                  </button>
                  <button
                    onClick={() => utils.zoomOut()}
                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                    title="Zoom out (-)"
                    aria-label="Zoom out"
                    data-testid="zoom-out-button"
                    type="button"
                  >
                    <ZoomOut className="w-5 h-5 text-gray-700" aria-hidden="true" />
                  </button>
                  <button
                    onClick={() => utils.resetTransform()}
                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                    title="Reset view (0)"
                    aria-label="Reset view"
                    data-testid="reset-zoom-button"
                    type="button"
                  >
                    <RotateCcw className="w-5 h-5 text-gray-700" aria-hidden="true" />
                  </button>
                  <button
                    onClick={() => {
                      utils.resetTransform();
                      utils.centerView(1);
                    }}
                    className="p-2 hover:bg-gray-100 rounded transition-colors"
                    title="Fit to container"
                    aria-label="Fit to container"
                    data-testid="fit-button"
                    type="button"
                  >
                    <Maximize2 className="w-5 h-5 text-gray-700" aria-hidden="true" />
                  </button>
                  {showMetadata && (
                    <button
                      onClick={() => setShowMetadataOverlay(!showMetadataOverlay)}
                      className={cn(
                        'p-2 hover:bg-gray-100 rounded transition-colors',
                        showMetadataOverlay && 'bg-blue-50'
                      )}
                      title="Toggle image info"
                      aria-label="Toggle image info"
                      aria-pressed={showMetadataOverlay}
                      data-testid="info-button"
                      type="button"
                    >
                      <Info className="w-5 h-5 text-gray-700" aria-hidden="true" />
                    </button>
                  )}
                </div>
              )}

              {/* Metadata Overlay */}
              {showMetadata && showMetadataOverlay && (
                <div
                  className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur-sm rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs"
                  data-testid="metadata-overlay"
                  role="status"
                  aria-live="polite"
                >
                  <h3 className="text-xs font-semibold text-gray-700 mb-2">Image Information</h3>
                  {primaryMetadata && (
                    <div className="space-y-1 text-xs text-gray-600">
                      <div className="flex justify-between gap-4">
                        <span className="font-medium">Dimensions:</span>
                        <span>
                          {primaryMetadata.width} × {primaryMetadata.height}
                        </span>
                      </div>
                      <div className="flex justify-between gap-4">
                        <span className="font-medium">Aspect Ratio:</span>
                        <span>{primaryMetadata.aspectRatio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between gap-4">
                        <span className="font-medium">Zoom:</span>
                        <span>{Math.round(currentZoom * 100)}%</span>
                      </div>
                    </div>
                  )}
                  {isComparisonMode && compareMetadata && (
                    <>
                      <div className="border-t border-gray-200 my-2" />
                      <h4 className="text-xs font-semibold text-gray-700 mb-1">Comparison Image</h4>
                      <div className="space-y-1 text-xs text-gray-600">
                        <div className="flex justify-between gap-4">
                          <span className="font-medium">Dimensions:</span>
                          <span>
                            {compareMetadata.width} × {compareMetadata.height}
                          </span>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Image Content */}
              <TransformComponent
                wrapperClass="!w-full !h-full min-h-[400px]"
                contentClass="!w-full !h-full flex items-center justify-center"
              >
                {isComparisonMode ? (
                  // Side-by-side comparison mode
                  <div
                    className="flex flex-row gap-2 items-center justify-center h-full"
                    data-testid="comparison-container"
                  >
                    <div className="flex-1 flex items-center justify-center">
                      <img
                        src={src}
                        alt={alt}
                        onLoad={handlePrimaryImageLoad}
                        className="max-w-full max-h-full object-contain"
                        data-testid="primary-image"
                      />
                    </div>
                    <div className="w-px bg-gray-400 self-stretch" aria-hidden="true" />
                    <div className="flex-1 flex items-center justify-center">
                      <img
                        src={compareSrc ?? undefined}
                        alt={compareAlt}
                        onLoad={handleCompareImageLoad}
                        className="max-w-full max-h-full object-contain"
                        data-testid="compare-image"
                      />
                    </div>
                  </div>
                ) : (
                  // Single image mode
                  <img
                    src={src}
                    alt={alt}
                    onLoad={handlePrimaryImageLoad}
                    className="max-w-full max-h-full object-contain"
                    data-testid="primary-image"
                  />
                )}
              </TransformComponent>
            </>
          )}
        </TransformWrapper>
      </div>
    );
  }
);

ImagePreview.displayName = 'ImagePreview';
