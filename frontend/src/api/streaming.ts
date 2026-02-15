/**
 * SSE streaming utilities for chat responses
 * Uses fetch + ReadableStream for POST-based SSE (EventSource only supports GET)
 */

import { config } from '@/config';
import { logger } from '@/lib/logger';

export interface StreamEvent {
  type: 'chunk' | 'done' | 'error';
  content?: string;
  message?: string;
}

export interface StreamOptions {
  onChunk: (content: string) => void;
  onDone: () => void;
  onError: (error: string) => void;
  signal?: AbortSignal;
}

/**
 * Stream chat response from SSE endpoint using fetch API.
 * Uses POST (not EventSource which is GET-only) with ReadableStream parsing.
 */
export async function streamChatResponse(
  message: string,
  includeHistory: boolean,
  options: StreamOptions
): Promise<void> {
  const url = `${config.api.baseUrl}/api/chat/stream`;

  logger.debug('SSE: Starting stream', { url, messageLength: message.length });

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, include_history: includeHistory }),
    ...(options.signal && { signal: options.signal }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`Stream request failed: ${response.status} ${errorText}`);
  }

  if (!response.body) {
    throw new Error('Response body is null - streaming not supported');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? ''; // Keep incomplete last line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const jsonStr = trimmed.slice(6); // Remove 'data: ' prefix
        try {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
          const event: StreamEvent = JSON.parse(jsonStr);

          switch (event.type) {
            case 'chunk':
              if (event.content) {
                options.onChunk(event.content);
              }
              break;
            case 'done':
              options.onDone();
              return;
            case 'error':
              options.onError(event.message ?? 'Unknown streaming error');
              return;
          }
        } catch {
          logger.warn('SSE: Failed to parse event', { line: trimmed });
        }
      }
    }
    // Stream ended without explicit 'done' event
    options.onDone();
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      logger.debug('SSE: Stream aborted by user');
      return;
    }
    throw err;
  }
}
