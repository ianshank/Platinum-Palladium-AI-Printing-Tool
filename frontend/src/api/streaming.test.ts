/**
 * Tests for SSE streaming utilities
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { streamChatResponse } from './streaming';

describe('streamChatResponse', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should stream chunks and call onChunk for each event', async () => {
    const mockChunks = [
      'data: {"type":"chunk","content":"Hello"}\n\n',
      'data: {"type":"chunk","content":" world"}\n\n',
      'data: {"type":"done"}\n\n',
    ];

    let chunkIndex = 0;
    const mockReader = {
      read: vi.fn(() => {
        if (chunkIndex >= mockChunks.length) {
          return Promise.resolve({ done: true as const, value: undefined });
        }
        const chunk = mockChunks[chunkIndex++];
        return Promise.resolve({
          done: false as const,
          value: new TextEncoder().encode(chunk),
        });
      }),
    };

    global.fetch = vi.fn(() => Promise.resolve({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    })) as unknown as typeof fetch;

    const onChunk = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await streamChatResponse('test message', true, {
      onChunk,
      onDone,
      onError,
    });

    expect(onChunk).toHaveBeenCalledTimes(2);
    expect(onChunk).toHaveBeenNthCalledWith(1, 'Hello');
    expect(onChunk).toHaveBeenNthCalledWith(2, ' world');
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });

  it('should call onError when error event is received', async () => {
    const mockChunks = ['data: {"type":"error","message":"Something went wrong"}\n\n'];

    let chunkIndex = 0;
    const mockReader = {
      read: vi.fn(() => {
        if (chunkIndex >= mockChunks.length) {
          return Promise.resolve({ done: true as const, value: undefined });
        }
        const chunk = mockChunks[chunkIndex++];
        return Promise.resolve({
          done: false as const,
          value: new TextEncoder().encode(chunk),
        });
      }),
    };

    global.fetch = vi.fn(() => Promise.resolve({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    })) as unknown as typeof fetch;

    const onChunk = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await streamChatResponse('test message', true, {
      onChunk,
      onDone,
      onError,
    });

    expect(onChunk).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith('Something went wrong');
  });

  it('should throw error on non-ok response', async () => {
    global.fetch = vi.fn(() => Promise.resolve({
      ok: false,
      status: 500,
      text: () => Promise.resolve('Internal server error'),
    })) as unknown as typeof fetch;

    const onChunk = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await expect(
      streamChatResponse('test message', true, {
        onChunk,
        onDone,
        onError,
      })
    ).rejects.toThrow('Stream request failed: 500 Internal server error');
  });

  it('should handle abort signal correctly', async () => {
    const mockReader = {
      read: vi.fn(() => Promise.reject(new DOMException('The user aborted a request', 'AbortError'))),
    };

    global.fetch = vi.fn(() => Promise.resolve({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    })) as unknown as typeof fetch;

    const controller = new AbortController();
    const onChunk = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    // Should not throw when aborted
    await streamChatResponse('test message', true, {
      onChunk,
      onDone,
      onError,
      signal: controller.signal,
    });

    expect(onChunk).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it('should handle invalid JSON gracefully', async () => {
    const mockChunks = [
      'data: {"type":"chunk","content":"Valid"}\n\n',
      'data: invalid json\n\n',
      'data: {"type":"done"}\n\n',
    ];

    let chunkIndex = 0;
    const mockReader = {
      read: vi.fn(() => {
        if (chunkIndex >= mockChunks.length) {
          return Promise.resolve({ done: true as const, value: undefined });
        }
        const chunk = mockChunks[chunkIndex++];
        return Promise.resolve({
          done: false as const,
          value: new TextEncoder().encode(chunk),
        });
      }),
    };

    global.fetch = vi.fn(() => Promise.resolve({
      ok: true,
      body: {
        getReader: () => mockReader,
      },
    })) as unknown as typeof fetch;

    const onChunk = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await streamChatResponse('test message', true, {
      onChunk,
      onDone,
      onError,
    });

    // Should only process valid chunks
    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledWith('Valid');
    expect(onDone).toHaveBeenCalledTimes(1);
  });
});
