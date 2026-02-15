/**
 * WebSocket connection manager
 *
 * Provides a reconnecting WebSocket client for real-time processing status updates.
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Connection state management
 * - Typed message handling
 * - Heartbeat/ping-pong keepalive
 */

import { config } from '@/config';
import { logger } from '@/lib/logger';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export interface WSMessage {
  type: string;
  payload: Record<string, unknown>;
  timestamp?: string;
}

export interface ProcessingStatus {
  operationId: string;
  operation: string;
  progress: number; // 0-100
  message: string;
  status: 'running' | 'completed' | 'failed';
}

type MessageHandler = (message: WSMessage) => void;
type ConnectionStateHandler = (state: ConnectionState) => void;

const INITIAL_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;
const MAX_RECONNECT_ATTEMPTS = 10;
const HEARTBEAT_INTERVAL_MS = 30000;

/** WebSocket readyState constants (avoid referencing global WebSocket which may not exist in test env) */
const WS_OPEN = 1;

export class WebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private stateHandlers: Set<ConnectionStateHandler> = new Set();
  private _state: ConnectionState = 'disconnected';
  private shouldReconnect = true;

  constructor(url?: string) {
    this.url = url ?? config.api.wsUrl;
  }

  get state(): ConnectionState {
    return this._state;
  }

  private setState(state: ConnectionState): void {
    this._state = state;
    this.stateHandlers.forEach((handler) => handler(state));
  }

  connect(): void {
    if (this.ws?.readyState === WS_OPEN) return;

    this.shouldReconnect = true;
    this.setState('connecting');
    logger.debug('WebSocket: Connecting', { url: this.url });

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setState('connected');
        this.startHeartbeat();
        logger.info('WebSocket: Connected');
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          const message = JSON.parse(event.data as string) as WSMessage;
          this.messageHandlers.forEach((handler) => handler(message));
        } catch {
          logger.warn('WebSocket: Failed to parse message');
        }
      };

      this.ws.onclose = (event) => {
        this.stopHeartbeat();
        logger.debug('WebSocket: Closed', { code: event.code, reason: event.reason });

        if (this.shouldReconnect && this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          this.scheduleReconnect();
        } else {
          this.setState('disconnected');
        }
      };

      this.ws.onerror = () => {
        logger.error('WebSocket: Error');
      };
    } catch (err) {
      logger.error('WebSocket: Failed to create connection', { error: String(err) });
      this.setState('disconnected');
    }
  }

  disconnect(): void {
    this.shouldReconnect = false;
    this.stopHeartbeat();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.setState('disconnected');
    logger.debug('WebSocket: Disconnected');
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  onStateChange(handler: ConnectionStateHandler): () => void {
    this.stateHandlers.add(handler);
    return () => {
      this.stateHandlers.delete(handler);
    };
  }

  send(message: WSMessage): void {
    if (this.ws?.readyState === WS_OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      logger.warn('WebSocket: Cannot send - not connected');
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = Math.min(
      INITIAL_RECONNECT_DELAY_MS * Math.pow(2, this.reconnectAttempts - 1),
      MAX_RECONNECT_DELAY_MS
    );

    this.setState('reconnecting');
    logger.debug('WebSocket: Scheduling reconnect', {
      attempt: this.reconnectAttempts,
      delayMs: delay,
    });

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.send({ type: 'ping', payload: {} });
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}

/** Singleton instance */
let _instance: WebSocketManager | null = null;

export function getWebSocketManager(): WebSocketManager {
  if (!_instance) {
    _instance = new WebSocketManager();
  }
  return _instance;
}

/** Reset the singleton instance (for testing only) */
export function resetWebSocketManager(): void {
  if (_instance) {
    _instance.disconnect();
    _instance = null;
  }
}
