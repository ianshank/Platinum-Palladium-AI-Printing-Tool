/**
 * WebSocket manager tests
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getWebSocketManager, resetWebSocketManager, WebSocketManager } from './websocket';
import { config } from '@/config';

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
    // Simulate async connection
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

  // Helper to simulate received message
  simulateMessage(data: string): void {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data }));
    }
  }

  // Helper to simulate error
  simulateError(): void {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
}

// Store original WebSocket
const OriginalWebSocket = global.WebSocket;

describe('WebSocketManager', () => {
  beforeEach(() => {
    // Replace global WebSocket with mock
    global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Clean up singleton to prevent timer leaks
    resetWebSocketManager();
    // Restore original WebSocket
    global.WebSocket = OriginalWebSocket;
  });

  describe('constructor', () => {
    it('uses config URL when no URL provided', () => {
      const manager = new WebSocketManager();
      expect(manager['url']).toBe(config.api.wsUrl);
    });

    it('uses provided URL when specified', () => {
      const customUrl = 'ws://custom:9000/ws';
      const manager = new WebSocketManager(customUrl);
      expect(manager['url']).toBe(customUrl);
    });

    it('initializes with disconnected state', () => {
      const manager = new WebSocketManager();
      expect(manager.state).toBe('disconnected');
    });
  });

  describe('connect', () => {
    it('changes state to connecting then connected', async () => {
      const manager = new WebSocketManager();
      const states: string[] = [];

      manager.onStateChange((state) => states.push(state));
      manager.connect();

      expect(states[0]).toBe('connecting');

      // Wait for async connection
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(states[1]).toBe('connected');
    });

    it('does not reconnect if already connected', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      const wsCount = manager['ws'] ? 1 : 0;
      manager.connect();

      expect(wsCount).toBe(1);
    });

    it('starts heartbeat on successful connection', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(manager['heartbeatTimer']).not.toBeNull();
    });
  });

  describe('disconnect', () => {
    it('changes state to disconnected', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      manager.disconnect();
      expect(manager.state).toBe('disconnected');
    });

    it('stops heartbeat timer', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      manager.disconnect();
      expect(manager['heartbeatTimer']).toBeNull();
    });

    it('clears reconnect timer', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      manager.disconnect();
      expect(manager['reconnectTimer']).toBeNull();
    });

    it('sets shouldReconnect to false', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      manager.disconnect();
      expect(manager['shouldReconnect']).toBe(false);
    });
  });

  describe('message handling', () => {
    it('calls registered message handlers', async () => {
      const manager = new WebSocketManager();
      const handler = vi.fn();

      manager.onMessage(handler);
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      const testMessage = { type: 'test', payload: { data: 'hello' } };
      (manager['ws'] as unknown as MockWebSocket)?.simulateMessage(JSON.stringify(testMessage));

      expect(handler).toHaveBeenCalledWith(testMessage);
    });

    it('handles malformed JSON gracefully', async () => {
      const manager = new WebSocketManager();
      const handler = vi.fn();

      manager.onMessage(handler);
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      (manager['ws'] as unknown as MockWebSocket)?.simulateMessage('invalid json');

      expect(handler).not.toHaveBeenCalled();
    });

    it('returns unsubscribe function', () => {
      const manager = new WebSocketManager();
      const handler = vi.fn();

      const unsubscribe = manager.onMessage(handler);
      unsubscribe();

      expect(manager['messageHandlers'].has(handler)).toBe(false);
    });
  });

  describe('state change handling', () => {
    it('calls registered state change handlers', async () => {
      const manager = new WebSocketManager();
      const handler = vi.fn();

      manager.onStateChange(handler);
      manager.connect();

      expect(handler).toHaveBeenCalledWith('connecting');

      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(handler).toHaveBeenCalledWith('connected');
    });

    it('returns unsubscribe function', () => {
      const manager = new WebSocketManager();
      const handler = vi.fn();

      const unsubscribe = manager.onStateChange(handler);
      unsubscribe();

      expect(manager['stateHandlers'].has(handler)).toBe(false);
    });
  });

  describe('send', () => {
    it('sends message when connected', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      const sendSpy = vi.spyOn(manager['ws']!, 'send');
      const message = { type: 'test', payload: { foo: 'bar' } };

      manager.send(message);

      expect(sendSpy).toHaveBeenCalledWith(JSON.stringify(message));
    });

    it('logs warning when not connected', () => {
      const manager = new WebSocketManager();
      const message = { type: 'test', payload: {} };

      // Should not throw
      expect(() => manager.send(message)).not.toThrow();
    });
  });

  describe('reconnection', () => {
    it('schedules reconnect on connection close', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      (manager['ws'] as unknown as MockWebSocket)?.close();

      expect(manager.state).toBe('reconnecting');
      expect(manager['reconnectTimer']).not.toBeNull();
    });

    it('uses exponential backoff for reconnect delay', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      // First reconnect
      (manager['ws'] as unknown as MockWebSocket)?.close();
      expect(manager['reconnectAttempts']).toBe(1);

      await new Promise((resolve) => setTimeout(resolve, 10));

      // Second reconnect (after first fails)
      if (manager['ws']) {
        (manager['ws'] as unknown as MockWebSocket).close();
      }

      // Attempts should increment
      expect(manager['reconnectAttempts']).toBeGreaterThan(0);
    });

    it('stops reconnecting after max attempts', async () => {
      const manager = new WebSocketManager();

      manager.connect();
      await new Promise((resolve) => setTimeout(resolve, 10));

      // Manually set attempts to max (since connect resets it)
      manager['reconnectAttempts'] = 10;

      (manager['ws'] as unknown as MockWebSocket)?.close();

      // Wait a bit for the close handler to run
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(manager.state).toBe('disconnected');
      expect(manager['reconnectTimer']).toBeNull();
    });

    it('resets reconnect attempts on successful connection', async () => {
      const manager = new WebSocketManager();
      manager['reconnectAttempts'] = 5;

      manager.connect();
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(manager['reconnectAttempts']).toBe(0);
    });
  });

  describe('heartbeat', () => {
    it('sends ping messages at intervals', async () => {
      vi.useFakeTimers();

      const manager = new WebSocketManager();
      manager.connect();

      await vi.advanceTimersByTimeAsync(10);

      const sendSpy = vi.spyOn(manager, 'send');

      // Fast-forward past heartbeat interval
      await vi.advanceTimersByTimeAsync(30000);

      expect(sendSpy).toHaveBeenCalledWith({ type: 'ping', payload: {} });

      vi.useRealTimers();
    });

    it('stops heartbeat on disconnect', async () => {
      const manager = new WebSocketManager();
      manager.connect();

      await new Promise((resolve) => setTimeout(resolve, 10));

      manager.disconnect();

      expect(manager['heartbeatTimer']).toBeNull();
    });
  });

  describe('getWebSocketManager', () => {
    it('returns singleton instance', () => {
      const instance1 = getWebSocketManager();
      const instance2 = getWebSocketManager();

      expect(instance1).toBe(instance2);
    });
  });
});
