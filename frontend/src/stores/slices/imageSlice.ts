/**
 * Image state slice
 * Manages image upload, preview, and processing state
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface ImageData {
  id: string;
  name: string;
  type: string;
  size: number;
  width: number;
  height: number;
  url: string;
  dataUrl?: string;
  thumbnail?: string;
  uploadedAt: string;
}

export interface ImageHistogram {
  red: number[];
  green: number[];
  blue: number[];
  luminance: number[];
}

export interface ImageSlice {
  // State
  current: ImageData | null;
  preview: string | null;
  processed: string | null;
  histogram: ImageHistogram | null;
  uploadProgress: number;
  isUploading: boolean;
  isProcessing: boolean;
  error: string | null;
  recentImages: ImageData[];
  maxRecentImages: number;

  // Zoom and pan
  zoom: number;
  panX: number;
  panY: number;

  // Actions
  setCurrentImage: (image: ImageData) => void;
  setPreview: (url: string | null) => void;
  setProcessed: (url: string | null) => void;
  setHistogram: (histogram: ImageHistogram | null) => void;

  // Upload
  startUpload: (fileName: string) => void;
  updateUploadProgress: (progress: number) => void;
  completeUpload: (image: ImageData) => void;
  cancelUpload: () => void;

  // Processing
  setProcessing: (processing: boolean) => void;
  setError: (error: string | null) => void;

  // Zoom and pan
  setZoom: (zoom: number) => void;
  setPan: (x: number, y: number) => void;
  resetView: () => void;

  // Recent images
  addToRecent: (image: ImageData) => void;
  clearRecent: () => void;

  // Reset
  clearImage: () => void;
  resetImage: () => void;
}

const initialState = {
  current: null as ImageData | null,
  preview: null as string | null,
  processed: null as string | null,
  histogram: null as ImageHistogram | null,
  uploadProgress: 0,
  isUploading: false,
  isProcessing: false,
  error: null as string | null,
  recentImages: [] as ImageData[],
  maxRecentImages: 10,
  zoom: 1,
  panX: 0,
  panY: 0,
};

export const createImageSlice: StateCreator<
  { image: ImageSlice },
  [['zustand/immer', never]],
  [],
  ImageSlice
> = (set, get) => ({
  ...initialState,

  setCurrentImage: (image) => {
    logger.debug('Image: setCurrentImage', { id: image.id, name: image.name });
    set((state) => {
      state.image.current = image;
      state.image.preview = image.url;
      state.image.error = null;
    });

    get().image.addToRecent(image);
  },

  setPreview: (url) => {
    logger.debug('Image: setPreview', { hasUrl: !!url });
    set((state) => {
      state.image.preview = url;
    });
  },

  setProcessed: (url) => {
    logger.debug('Image: setProcessed', { hasUrl: !!url });
    set((state) => {
      state.image.processed = url;
    });
  },

  setHistogram: (histogram) => {
    logger.debug('Image: setHistogram', { hasHistogram: !!histogram });
    set((state) => {
      state.image.histogram = histogram;
    });
  },

  startUpload: (fileName) => {
    logger.debug('Image: startUpload', { fileName });
    set((state) => {
      state.image.isUploading = true;
      state.image.uploadProgress = 0;
      state.image.error = null;
    });
  },

  updateUploadProgress: (progress) => {
    set((state) => {
      state.image.uploadProgress = Math.max(0, Math.min(100, progress));
    });
  },

  completeUpload: (image) => {
    logger.info('Image: completeUpload', { id: image.id, name: image.name });
    set((state) => {
      state.image.isUploading = false;
      state.image.uploadProgress = 100;
      state.image.current = image;
      state.image.preview = image.url;
    });

    get().image.addToRecent(image);
  },

  cancelUpload: () => {
    logger.debug('Image: cancelUpload');
    set((state) => {
      state.image.isUploading = false;
      state.image.uploadProgress = 0;
    });
  },

  setProcessing: (processing) => {
    logger.debug('Image: setProcessing', { processing });
    set((state) => {
      state.image.isProcessing = processing;
    });
  },

  setError: (error) => {
    if (error) {
      logger.error('Image: error', { error });
    }
    set((state) => {
      state.image.error = error;
      state.image.isUploading = false;
      state.image.isProcessing = false;
    });
  },

  setZoom: (zoom) => {
    const clampedZoom = Math.max(0.1, Math.min(10, zoom));
    logger.debug('Image: setZoom', { zoom: clampedZoom });
    set((state) => {
      state.image.zoom = clampedZoom;
    });
  },

  setPan: (x, y) => {
    set((state) => {
      state.image.panX = x;
      state.image.panY = y;
    });
  },

  resetView: () => {
    logger.debug('Image: resetView');
    set((state) => {
      state.image.zoom = 1;
      state.image.panX = 0;
      state.image.panY = 0;
    });
  },

  addToRecent: (image) => {
    set((state) => {
      // Remove if already exists
      state.image.recentImages = state.image.recentImages.filter(
        (i) => i.id !== image.id
      );
      // Add to beginning
      state.image.recentImages.unshift(image);
      // Trim to max
      if (state.image.recentImages.length > state.image.maxRecentImages) {
        state.image.recentImages = state.image.recentImages.slice(
          0,
          state.image.maxRecentImages
        );
      }
    });
  },

  clearRecent: () => {
    logger.debug('Image: clearRecent');
    set((state) => {
      state.image.recentImages = [];
    });
  },

  clearImage: () => {
    logger.debug('Image: clearImage');
    set((state) => {
      state.image.current = null;
      state.image.preview = null;
      state.image.processed = null;
      state.image.histogram = null;
      state.image.error = null;
      state.image.zoom = 1;
      state.image.panX = 0;
      state.image.panY = 0;
    });
  },

  resetImage: () => {
    logger.debug('Image: resetImage');
    set((state) => {
      Object.assign(state.image, initialState);
    });
  },
});
