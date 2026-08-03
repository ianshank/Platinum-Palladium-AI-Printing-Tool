import { beforeEach, describe, expect, it } from 'vitest';
import { createStore } from '@/stores';
import type { ImageData } from './imageSlice';

describe('imageSlice', () => {
  let store: ReturnType<typeof createStore>;

  const mockImage: ImageData = {
    id: 'img-1',
    name: 'test-scan.png',
    type: 'image/png',
    size: 1024,
    width: 800,
    height: 600,
    url: 'blob:http://localhost/img-1',
    uploadedAt: '2026-02-07T00:00:00Z',
  };

  beforeEach(() => {
    store = createStore();
  });

  describe('Initial State', () => {
    it('has correct initial values', () => {
      const state = store.getState();
      expect(state.image.current).toBeNull();
      expect(state.image.preview).toBeNull();
      expect(state.image.processed).toBeNull();
      expect(state.image.histogram).toBeNull();
      expect(state.image.uploadProgress).toBe(0);
      expect(state.image.isUploading).toBe(false);
      expect(state.image.isProcessing).toBe(false);
      expect(state.image.error).toBeNull();
      expect(state.image.recentImages).toEqual([]);
      expect(state.image.zoom).toBe(1);
      expect(state.image.panX).toBe(0);
      expect(state.image.panY).toBe(0);
    });
  });

  describe('Image State', () => {
    it('setCurrentImage sets image, preview, and adds to recent', () => {
      store.getState().image.setCurrentImage(mockImage);

      const state = store.getState();
      expect(state.image.current).toEqual(mockImage);
      expect(state.image.preview).toBe(mockImage.url);
      expect(state.image.error).toBeNull();
      expect(state.image.recentImages).toHaveLength(1);
    });

    it('setPreview updates preview URL', () => {
      store.getState().image.setPreview('blob:preview');
      expect(store.getState().image.preview).toBe('blob:preview');
    });

    it('setProcessed updates processed URL', () => {
      store.getState().image.setProcessed('blob:processed');
      expect(store.getState().image.processed).toBe('blob:processed');
    });

    it('setHistogram stores histogram data', () => {
      const histogram = {
        red: [0, 1, 2],
        green: [0, 1, 2],
        blue: [0, 1, 2],
        luminance: [0, 1, 2],
      };
      store.getState().image.setHistogram(histogram);
      expect(store.getState().image.histogram).toEqual(histogram);
    });
  });

  describe('Upload Lifecycle', () => {
    it('startUpload sets uploading state', () => {
      store.getState().image.startUpload('scan.png');

      const state = store.getState();
      expect(state.image.isUploading).toBe(true);
      expect(state.image.uploadProgress).toBe(0);
      expect(state.image.error).toBeNull();
    });

    it('updateUploadProgress clamps to 0-100', () => {
      store.getState().image.updateUploadProgress(50);
      expect(store.getState().image.uploadProgress).toBe(50);

      store.getState().image.updateUploadProgress(150);
      expect(store.getState().image.uploadProgress).toBe(100);

      store.getState().image.updateUploadProgress(-10);
      expect(store.getState().image.uploadProgress).toBe(0);
    });

    it('completeUpload finalizes upload state', () => {
      store.getState().image.startUpload('scan.png');
      store.getState().image.completeUpload(mockImage);

      const state = store.getState();
      expect(state.image.isUploading).toBe(false);
      expect(state.image.uploadProgress).toBe(100);
      expect(state.image.current).toEqual(mockImage);
      expect(state.image.preview).toBe(mockImage.url);
    });

    it('cancelUpload resets upload state', () => {
      store.getState().image.startUpload('scan.png');
      store.getState().image.updateUploadProgress(30);
      store.getState().image.cancelUpload();

      expect(store.getState().image.isUploading).toBe(false);
      expect(store.getState().image.uploadProgress).toBe(0);
    });
  });

  describe('Processing', () => {
    it('setProcessing toggles processing flag', () => {
      store.getState().image.setProcessing(true);
      expect(store.getState().image.isProcessing).toBe(true);
    });

    it('setError sets error and stops uploading/processing', () => {
      store.getState().image.startUpload('file.png');
      store.getState().image.setProcessing(true);
      store.getState().image.setError('Upload failed');

      expect(store.getState().image.error).toBe('Upload failed');
      expect(store.getState().image.isUploading).toBe(false);
      expect(store.getState().image.isProcessing).toBe(false);
    });
  });

  describe('Zoom and Pan', () => {
    it('setZoom clamps between 0.1 and 10', () => {
      store.getState().image.setZoom(5);
      expect(store.getState().image.zoom).toBe(5);

      store.getState().image.setZoom(0.01);
      expect(store.getState().image.zoom).toBe(0.1);

      store.getState().image.setZoom(20);
      expect(store.getState().image.zoom).toBe(10);
    });

    it('setPan updates pan coordinates', () => {
      store.getState().image.setPan(100, -50);
      expect(store.getState().image.panX).toBe(100);
      expect(store.getState().image.panY).toBe(-50);
    });

    it('resetView resets zoom and pan', () => {
      store.getState().image.setZoom(3);
      store.getState().image.setPan(50, 50);
      store.getState().image.resetView();

      expect(store.getState().image.zoom).toBe(1);
      expect(store.getState().image.panX).toBe(0);
      expect(store.getState().image.panY).toBe(0);
    });
  });

  describe('Recent Images', () => {
    it('addToRecent adds image to beginning', () => {
      store.getState().image.addToRecent(mockImage);
      const img2 = { ...mockImage, id: 'img-2', name: 'second.png' };
      store.getState().image.addToRecent(img2);

      const recent = store.getState().image.recentImages;
      expect(recent).toHaveLength(2);
      expect(recent[0]?.id).toBe('img-2');
    });

    it('addToRecent deduplicates by id', () => {
      store.getState().image.addToRecent(mockImage);
      store.getState().image.addToRecent(mockImage);

      expect(store.getState().image.recentImages).toHaveLength(1);
    });

    it('addToRecent trims to maxRecentImages', () => {
      for (let i = 0; i < 12; i++) {
        store.getState().image.addToRecent({ ...mockImage, id: `img-${i}` });
      }
      expect(store.getState().image.recentImages).toHaveLength(10);
    });

    it('clearRecent empties recent list', () => {
      store.getState().image.addToRecent(mockImage);
      store.getState().image.clearRecent();

      expect(store.getState().image.recentImages).toEqual([]);
    });
  });

  describe('Reset Operations', () => {
    it('clearImage clears current but keeps recent', () => {
      store.getState().image.setCurrentImage(mockImage);
      store.getState().image.clearImage();

      expect(store.getState().image.current).toBeNull();
      expect(store.getState().image.preview).toBeNull();
      expect(store.getState().image.recentImages).toHaveLength(1);
    });

    it('resetImage restores all initial state', () => {
      store.getState().image.setCurrentImage(mockImage);
      store.getState().image.setZoom(5);
      store.getState().image.setPan(100, 100);

      store.getState().image.resetImage();

      const state = store.getState();
      expect(state.image.current).toBeNull();
      expect(state.image.zoom).toBe(1);
      expect(state.image.recentImages).toEqual([]);
    });
  });
});
