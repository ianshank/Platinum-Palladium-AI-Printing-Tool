/**
 * useWebSocket hook tests
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useWebSocket } from './useWebSocket';
import { getWebSocketManager, resetWebSocketManager } from '@/lib/websocket';

// Mock WebSocket
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;

  url: string;
  readyState = MockWebSocket.OPEN;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(_data: string): void {
    void _data;
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code: code ?? 1000, reason: reason ?? '' }));
    }
  }

  simulateMessage(data: string): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }
}

const OriginalWebSocket = global.WebSocket;

describe('useWebSocket', () => {
  beforeEach(() => {
    resetWebSocketManager();
    global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Always restore real timers to prevent leaking fake timer state
    vi.useRealTimers();
    resetWebSocketManager();
    global.WebSocket = OriginalWebSocket;
  });

  describe('initialization', () => {
    it('returns correct initial state', () => {
      const { result } = renderHook(() => useWebSocket());

      expect(result.current.connectionState).toBe('disconnected');
      expect(result.current.processingStatus).toBeNull();
      expect(typeof result.current.connect).toBe('function');
      expect(typeof result.current.disconnect).toBe('function');
      expect(typeof result.current.send).toBe('function');
    });

    it('does not auto-connect by default', () => {
      const { result } = renderHook(() => useWebSocket());

      expect(result.current.connectionState).toBe('disconnected');
    });

    it('auto-connects when flag is true', async () => {
      const { result } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });
    });
  });

  describe('connection state', () => {
    it('updates connection state on connect', async () => {
      const { result } = renderHook(() => useWebSocket());

      act(() => {
        result.current.connect();
      });

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });
    });

    it('updates connection state on disconnect', async () => {
      const { result } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });

      act(() => {
        result.current.disconnect();
      });

      await waitFor(() => {
        expect(result.current.connectionState).toBe('disconnected');
      });
    });
  });

  describe('processing status updates', () => {
    it('updates processing status from messages', async () => {
      const { result } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;

      const statusMessage = {
        type: 'processing_status',
        payload: {
          operationId: 'test-op-1',
          operation: 'curve_generation',
          progress: 50,
          message: 'Generating curve...',
          status: 'running',
        },
      };

      act(() => {
        ws.simulateMessage(JSON.stringify(statusMessage));
      });

      await waitFor(() => {
        expect(result.current.processingStatus).toEqual(statusMessage.payload);
      });
    });

    it('clears status after completion', () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });

      const { result } = renderHook(() => useWebSocket(true));

      // Advance past the setTimeout(0) in MockWebSocket constructor
      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(result.current.connectionState).toBe('connected');

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;

      const completedMessage = {
        type: 'processing_status',
        payload: {
          operationId: 'test-op-1',
          operation: 'curve_generation',
          progress: 100,
          message: 'Complete',
          status: 'completed',
        },
      };

      act(() => {
        ws.simulateMessage(JSON.stringify(completedMessage));
      });

      expect(result.current.processingStatus).toEqual(completedMessage.payload);

      // Fast-forward 3 seconds for the auto-clear timeout
      act(() => {
        vi.advanceTimersByTime(3000);
      });

      expect(result.current.processingStatus).toBeNull();
    });

    it('clears status after failure', () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });

      const { result } = renderHook(() => useWebSocket(true));

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(result.current.connectionState).toBe('connected');

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;

      const failedMessage = {
        type: 'processing_status',
        payload: {
          operationId: 'test-op-1',
          operation: 'curve_generation',
          progress: 75,
          message: 'Failed',
          status: 'failed',
        },
      };

      act(() => {
        ws.simulateMessage(JSON.stringify(failedMessage));
      });

      expect(result.current.processingStatus).toEqual(failedMessage.payload);

      // Fast-forward 3 seconds for the auto-clear timeout
      act(() => {
        vi.advanceTimersByTime(3000);
      });

      expect(result.current.processingStatus).toBeNull();
    });

    it('ignores non-processing-status messages', async () => {
      const { result } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;

      const otherMessage = {
        type: 'pong',
        payload: {},
      };

      act(() => {
        ws.simulateMessage(JSON.stringify(otherMessage));
      });

      // Processing status should remain null
      expect(result.current.processingStatus).toBeNull();
    });
  });

  describe('send', () => {
    it('sends messages through manager', async () => {
      const { result } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;
      const sendSpy = vi.spyOn(ws, 'send');

      const message = { type: 'custom', payload: { data: 'test' } };
      act(() => {
        result.current.send(message);
      });

      expect(sendSpy).toHaveBeenCalledWith(JSON.stringify(message));
    });
  });

  describe('cleanup', () => {
    it('unsubscribes on unmount', async () => {
      const { result, unmount } = renderHook(() => useWebSocket(true));

      await waitFor(() => {
        expect(result.current.connectionState).toBe('connected');
      });

      const manager = getWebSocketManager();
      const initialHandlerCount = manager['messageHandlers'].size;

      unmount();

      // Handlers should be cleaned up
      expect(manager['messageHandlers'].size).toBeLessThan(initialHandlerCount);
    });
  });

  describe('multiple instances', () => {
    it('shares connection state across instances', async () => {
      const { result: result1 } = renderHook(() => useWebSocket());
      const { result: result2 } = renderHook(() => useWebSocket());

      act(() => {
        result1.current.connect();
      });

      await waitFor(() => {
        expect(result1.current.connectionState).toBe('connected');
      });

      // Second instance should also see the connected state
      await waitFor(() => {
        expect(result2.current.connectionState).toBe('connected');
      });
    });

    it('shares processing status across instances', async () => {
      const { result: result1 } = renderHook(() => useWebSocket(true));
      const { result: result2 } = renderHook(() => useWebSocket());

      await waitFor(() => {
        expect(result1.current.connectionState).toBe('connected');
      });

      const manager = getWebSocketManager();
      const ws = manager['ws'] as unknown as MockWebSocket;

      const statusMessage = {
        type: 'processing_status',
        payload: {
          operationId: 'test-op-1',
          operation: 'test',
          progress: 50,
          message: 'Testing',
          status: 'running',
        },
      };

      act(() => {
        ws.simulateMessage(JSON.stringify(statusMessage));
      });

      await waitFor(() => {
        expect(result1.current.processingStatus).toEqual(statusMessage.payload);
        expect(result2.current.processingStatus).toEqual(statusMessage.payload);
      });
    });
  });
});
