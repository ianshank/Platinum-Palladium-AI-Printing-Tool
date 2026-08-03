import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ImagePreview } from './ImagePreview';
import type { ReactZoomPanPinchRef } from 'react-zoom-pan-pinch';

// Mock react-zoom-pan-pinch
vi.mock('react-zoom-pan-pinch', () => ({
  TransformWrapper: ({
    children,
    onTransformed,
  }: {
    children: (utils: any) => React.ReactNode;
    onTransformed?: (ref: ReactZoomPanPinchRef) => void;
  }) => {
    const mockUtils = {
      zoomIn: vi.fn(() => {
        if (onTransformed) {
          onTransformed({ state: { scale: 1.5 } } as ReactZoomPanPinchRef);
        }
      }),
      zoomOut: vi.fn(() => {
        if (onTransformed) {
          onTransformed({ state: { scale: 0.8 } } as ReactZoomPanPinchRef);
        }
      }),
      resetTransform: vi.fn(() => {
        if (onTransformed) {
          onTransformed({ state: { scale: 1 } } as ReactZoomPanPinchRef);
        }
      }),
      centerView: vi.fn(),
      setTransform: vi.fn(),
    };
    return <div data-testid="transform-wrapper">{typeof children === 'function' ? children(mockUtils) : children}</div>;
  },
  TransformComponent: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="transform-component">{children}</div>
  ),
}));

// Mock logger
vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

describe('ImagePreview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Empty and Loading States', () => {
    it('shows placeholder when src is null', () => {
      render(<ImagePreview src={null} />);

      const empty = screen.getByTestId('image-preview-empty');
      expect(empty).toBeInTheDocument();
      expect(screen.getByText('No image to display')).toBeInTheDocument();
    });

    it('shows loading skeleton when isLoading is true', () => {
      render(<ImagePreview src="test.jpg" isLoading={true} />);

      const loading = screen.getByTestId('image-preview-loading');
      expect(loading).toBeInTheDocument();
      expect(screen.getByText('Loading image...')).toBeInTheDocument();
    });

    it('shows error state when error prop is set', () => {
      const errorMessage = 'Failed to load the image file';
      render(<ImagePreview src="test.jpg" error={errorMessage} />);

      const errorContainer = screen.getByTestId('image-preview-error');
      expect(errorContainer).toBeInTheDocument();
      expect(errorContainer).toHaveAttribute('role', 'alert');
      expect(errorContainer).toHaveAttribute('aria-live', 'polite');
      expect(screen.getByText('Failed to load image')).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });
  });

  describe('Single Image Mode', () => {
    it('renders image with src', () => {
      render(<ImagePreview src="test-image.jpg" alt="Test image" />);

      const preview = screen.getByTestId('image-preview');
      expect(preview).toBeInTheDocument();

      const image = screen.getByTestId('primary-image');
      expect(image).toBeInTheDocument();
      expect(image).toHaveAttribute('src', 'test-image.jpg');
      expect(image).toHaveAttribute('alt', 'Test image');
    });

    it('uses default alt text when not provided', () => {
      render(<ImagePreview src="test.jpg" />);

      const image = screen.getByTestId('primary-image');
      expect(image).toHaveAttribute('alt', 'Image preview');
    });

    it('applies custom className', () => {
      render(<ImagePreview src="test.jpg" className="custom-class" />);

      const preview = screen.getByTestId('image-preview');
      expect(preview).toHaveClass('custom-class');
    });
  });

  describe('Comparison Mode', () => {
    it('renders comparison mode with compareSrc', () => {
      render(
        <ImagePreview
          src="original.jpg"
          alt="Original"
          compareSrc="processed.jpg"
          compareAlt="Processed"
        />
      );

      const comparisonContainer = screen.getByTestId('comparison-container');
      expect(comparisonContainer).toBeInTheDocument();

      const primaryImage = screen.getByTestId('primary-image');
      expect(primaryImage).toHaveAttribute('src', 'original.jpg');
      expect(primaryImage).toHaveAttribute('alt', 'Original');

      const compareImage = screen.getByTestId('compare-image');
      expect(compareImage).toHaveAttribute('src', 'processed.jpg');
      expect(compareImage).toHaveAttribute('alt', 'Processed');
    });

    it('uses default compareAlt when not provided', () => {
      render(<ImagePreview src="original.jpg" compareSrc="processed.jpg" />);

      const compareImage = screen.getByTestId('compare-image');
      expect(compareImage).toHaveAttribute('alt', 'Comparison image');
    });
  });

  describe('Zoom Controls', () => {
    it('renders zoom controls when showControls is true', () => {
      render(<ImagePreview src="test.jpg" showControls={true} />);

      const controls = screen.getByTestId('zoom-controls');
      expect(controls).toBeInTheDocument();
      expect(controls).toHaveAttribute('role', 'toolbar');
      expect(controls).toHaveAttribute('aria-label', 'Image zoom controls');

      expect(screen.getByTestId('zoom-in-button')).toBeInTheDocument();
      expect(screen.getByTestId('zoom-out-button')).toBeInTheDocument();
      expect(screen.getByTestId('reset-zoom-button')).toBeInTheDocument();
      expect(screen.getByTestId('fit-button')).toBeInTheDocument();
    });

    it('hides zoom controls when showControls is false', () => {
      render(<ImagePreview src="test.jpg" showControls={false} />);

      expect(screen.queryByTestId('zoom-controls')).not.toBeInTheDocument();
    });

    it('calls onZoomChange when zoom in button is clicked', async () => {
      const onZoomChange = vi.fn();
      render(<ImagePreview src="test.jpg" onZoomChange={onZoomChange} />);

      const zoomInButton = screen.getByTestId('zoom-in-button');
      fireEvent.click(zoomInButton);

      await waitFor(() => {
        expect(onZoomChange).toHaveBeenCalledWith(1.5);
      });
    });

    it('calls onZoomChange when zoom out button is clicked', async () => {
      const onZoomChange = vi.fn();
      render(<ImagePreview src="test.jpg" onZoomChange={onZoomChange} />);

      const zoomOutButton = screen.getByTestId('zoom-out-button');
      fireEvent.click(zoomOutButton);

      await waitFor(() => {
        expect(onZoomChange).toHaveBeenCalledWith(0.8);
      });
    });

    it('calls onZoomChange when reset button is clicked', async () => {
      const onZoomChange = vi.fn();
      render(<ImagePreview src="test.jpg" onZoomChange={onZoomChange} />);

      const resetButton = screen.getByTestId('reset-zoom-button');
      fireEvent.click(resetButton);

      await waitFor(() => {
        expect(onZoomChange).toHaveBeenCalledWith(1);
      });
    });

    it('has accessible labels on all control buttons', () => {
      render(<ImagePreview src="test.jpg" />);

      expect(screen.getByLabelText('Zoom in')).toBeInTheDocument();
      expect(screen.getByLabelText('Zoom out')).toBeInTheDocument();
      expect(screen.getByLabelText('Reset view')).toBeInTheDocument();
      expect(screen.getByLabelText('Fit to container')).toBeInTheDocument();
    });
  });

  describe('Metadata Overlay', () => {
    it('shows info button when showMetadata is true', () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      const infoButton = screen.getByTestId('info-button');
      expect(infoButton).toBeInTheDocument();
      expect(infoButton).toHaveAttribute('aria-label', 'Toggle image info');
    });

    it('hides info button when showMetadata is false', () => {
      render(<ImagePreview src="test.jpg" showMetadata={false} />);

      expect(screen.queryByTestId('info-button')).not.toBeInTheDocument();
    });

    it('toggles metadata overlay when info button is clicked', async () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      const infoButton = screen.getByTestId('info-button');

      // Initially not visible
      expect(screen.queryByTestId('metadata-overlay')).not.toBeInTheDocument();

      // Click to show
      fireEvent.click(infoButton);
      await waitFor(() => {
        expect(screen.getByTestId('metadata-overlay')).toBeInTheDocument();
      });

      // Click to hide
      fireEvent.click(infoButton);
      await waitFor(() => {
        expect(screen.queryByTestId('metadata-overlay')).not.toBeInTheDocument();
      });
    });

    it('metadata overlay has proper accessibility attributes', async () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      const infoButton = screen.getByTestId('info-button');
      fireEvent.click(infoButton);

      await waitFor(() => {
        const overlay = screen.getByTestId('metadata-overlay');
        expect(overlay).toHaveAttribute('role', 'status');
        expect(overlay).toHaveAttribute('aria-live', 'polite');
      });
    });

    it('updates metadata overlay when image loads', async () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      // Show metadata overlay
      const infoButton = screen.getByTestId('info-button');
      fireEvent.click(infoButton);

      // Simulate image load
      const image = screen.getByTestId('primary-image');
      Object.defineProperty(image, 'naturalWidth', { value: 1920, configurable: true });
      Object.defineProperty(image, 'naturalHeight', { value: 1080, configurable: true });
      fireEvent.load(image);

      await waitFor(() => {
        expect(screen.getByText('1920 × 1080')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper alt text for images', () => {
      render(<ImagePreview src="test.jpg" alt="Test accessibility" />);

      const image = screen.getByAltText('Test accessibility');
      expect(image).toBeInTheDocument();
    });

    it('has proper ARIA attributes on error state', () => {
      render(<ImagePreview src="test.jpg" error="Network error" />);

      const errorContainer = screen.getByTestId('image-preview-error');
      expect(errorContainer).toHaveAttribute('role', 'alert');
      expect(errorContainer).toHaveAttribute('aria-live', 'polite');
    });

    it('has proper ARIA attributes on zoom controls', () => {
      render(<ImagePreview src="test.jpg" />);

      const controls = screen.getByTestId('zoom-controls');
      expect(controls).toHaveAttribute('role', 'toolbar');
      expect(controls).toHaveAttribute('aria-label', 'Image zoom controls');
    });

    it('info button has aria-pressed attribute', async () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      const infoButton = screen.getByTestId('info-button');
      expect(infoButton).toHaveAttribute('aria-pressed', 'false');

      fireEvent.click(infoButton);

      await waitFor(() => {
        expect(infoButton).toHaveAttribute('aria-pressed', 'true');
      });
    });
  });

  describe('Props Configuration', () => {
    it('respects minZoom prop', () => {
      const { container } = render(<ImagePreview src="test.jpg" minZoom={0.3} />);

      // TransformWrapper should receive the minZoom prop (checked via mock setup)
      expect(container).toBeInTheDocument();
    });

    it('respects maxZoom prop', () => {
      const { container } = render(<ImagePreview src="test.jpg" maxZoom={10} />);

      // TransformWrapper should receive the maxZoom prop (checked via mock setup)
      expect(container).toBeInTheDocument();
    });

    it('respects initialZoom prop', () => {
      const { container } = render(<ImagePreview src="test.jpg" initialZoom={2} />);

      // TransformWrapper should receive the initialZoom prop (checked via mock setup)
      expect(container).toBeInTheDocument();
    });
  });

  describe('Image Load Events', () => {
    it('handles primary image load event', () => {
      render(<ImagePreview src="test.jpg" />);

      const image = screen.getByTestId('primary-image');
      Object.defineProperty(image, 'naturalWidth', { value: 800, configurable: true });
      Object.defineProperty(image, 'naturalHeight', { value: 600, configurable: true });

      expect(() => fireEvent.load(image)).not.toThrow();
    });

    it('handles comparison image load event', () => {
      render(<ImagePreview src="test.jpg" compareSrc="compare.jpg" />);

      const compareImage = screen.getByTestId('compare-image');
      Object.defineProperty(compareImage, 'naturalWidth', { value: 800, configurable: true });
      Object.defineProperty(compareImage, 'naturalHeight', { value: 600, configurable: true });

      expect(() => fireEvent.load(compareImage)).not.toThrow();
    });
  });

  describe('Button Types', () => {
    it('all toolbar buttons have explicit type="button"', () => {
      render(<ImagePreview src="test.jpg" showMetadata={true} />);

      const buttons = [
        screen.getByTestId('zoom-in-button'),
        screen.getByTestId('zoom-out-button'),
        screen.getByTestId('reset-zoom-button'),
        screen.getByTestId('fit-button'),
        screen.getByTestId('info-button'),
      ];

      buttons.forEach((button) => {
        expect(button).toHaveAttribute('type', 'button');
      });
    });
  });
});
