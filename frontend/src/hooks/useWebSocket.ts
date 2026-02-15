/**
 * useWebSocket hook
 *
 * React hook for WebSocket connection management and processing status.
 * Automatically connects on mount and disconnects on unmount.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  type ConnectionState,
  getWebSocketManager,
  type ProcessingStatus,
  type WSMessage,
} from '@/lib/websocket';

export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  processingStatus: ProcessingStatus | null;
  connect: () => void;
  disconnect: () => void;
  send: (message: WSMessage) => void;
}

export function useWebSocket(autoConnect = false): UseWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);

  useEffect(() => {
    const manager = getWebSocketManager();

    const unsubState = manager.onStateChange((state) => {
      setConnectionState(state);
    });

    const unsubMessage = manager.onMessage((message) => {
      if (message.type === 'processing_status') {
        // Cast through unknown to ProcessingStatus
        const status = message.payload as unknown;
        setProcessingStatus(status as ProcessingStatus);

        if ((status as ProcessingStatus).status !== 'running') {
          // Clear status after a delay for completed/failed operations
          setTimeout(() => setProcessingStatus(null), 3000);
        }
      }
    });

    if (autoConnect) {
      manager.connect();
    }

    return () => {
      unsubState();
      unsubMessage();
    };
  }, [autoConnect]);

  const connect = useCallback(() => {
    getWebSocketManager().connect();
  }, []);

  const disconnect = useCallback(() => {
    getWebSocketManager().disconnect();
  }, []);

  const send = useCallback((message: WSMessage) => {
    getWebSocketManager().send(message);
  }, []);

  return {
    connectionState,
    processingStatus,
    connect,
    disconnect,
    send,
  };
}
