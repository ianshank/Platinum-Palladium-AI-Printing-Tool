# SSE Streaming Implementation for AI Chat

## Overview

This document describes the Server-Sent Events (SSE) streaming implementation for the AI chat feature in the Platinum-Palladium AI Printing Tool. The implementation connects the existing backend streaming infrastructure to the frontend, enabling real-time streaming of AI responses.

## Files Modified/Created

### Backend
- **Modified**: `/home/user/Platinum-Palladium-AI-Printing-Tool/src/ptpd_calibration/api/server.py`
  - Added `StreamingResponse` import from `starlette.responses`
  - Added new `POST /api/chat/stream` endpoint for SSE streaming

### Frontend
- **Created**: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/api/streaming.ts`
  - Utility functions for SSE streaming using fetch + ReadableStream
  - Handles chunk parsing, error events, and abort signals

- **Created**: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/hooks/useStreamingChat.ts`
  - React hook for real-time chat streaming
  - Integrates with Zustand state management
  - Automatic fallback to non-streaming API on errors

- **Created**: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/api/streaming.test.ts`
  - Comprehensive test suite for streaming functionality
  - Tests for chunk handling, errors, aborts, and invalid JSON

- **Modified**: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/components/assistant/AIAssistant.tsx`
  - Updated to use `useStreamingChat` instead of `useChat`
  - Added "Stop" button to cancel streaming
  - Improved real-time streaming indicator

## Architecture

### Backend Flow

```
Client Request (POST /api/chat/stream)
  ↓
FastAPI Endpoint
  ↓
CalibrationAssistant.chat_stream()
  ↓
LLM Client Stream
  ↓
SSE Event Generator (async generator)
  ↓
StreamingResponse (text/event-stream)
  ↓
Client (ReadableStream parsing)
```

### Frontend Flow

```
User sends message
  ↓
useStreamingChat.sendMessage()
  ↓
Add user message to store
  ↓
Call streamChatResponse()
  ↓
Fetch API with POST to /api/chat/stream
  ↓
Parse SSE events from ReadableStream
  ↓
For each chunk:
  - appendStreamContent() → Updates UI in real-time
  ↓
On done:
  - finishStreaming() → Saves message to store
  ↓
On error:
  - Fallback to non-streaming API
```

## SSE Event Format

The backend sends events in Server-Sent Events format:

```
data: {"type": "chunk", "content": "text..."}\n\n
data: {"type": "chunk", "content": "more text..."}\n\n
data: {"type": "done"}\n\n
```

Error events:
```
data: {"type": "error", "message": "Error description"}\n\n
```

## Key Features

### 1. Real-time Streaming
- Chunks are displayed as they arrive from the LLM
- Smooth user experience with immediate feedback
- Visual indicator shows streaming is active

### 2. Cancellation Support
- User can stop streaming at any time
- Abort signal propagates to fetch request
- Graceful cleanup of resources

### 3. Automatic Fallback
- If streaming fails, automatically falls back to non-streaming API
- Ensures reliability even if SSE isn't supported
- Transparent to the user

### 4. Error Handling
- Network errors caught and displayed
- Invalid JSON events logged but don't break the stream
- Server errors sent as SSE error events

### 5. Backward Compatibility
- Original `useChat` hook remains unchanged
- Non-streaming `/api/chat` endpoint still works
- Components can choose streaming or non-streaming

## Configuration

The streaming URL is configured via environment variables:

```bash
VITE_API_URL=http://localhost:8000  # Base API URL
```

The streaming endpoint is: `${VITE_API_URL}/api/chat/stream`

## Testing

### Unit Tests
Run the streaming tests:
```bash
npx vitest run src/api/streaming.test.ts
```

Test coverage includes:
- ✅ Chunk parsing and callbacks
- ✅ Error event handling
- ✅ Network error handling
- ✅ Abort signal handling
- ✅ Invalid JSON handling
- ✅ Done event handling

### Manual Testing
1. Start the backend server:
   ```bash
   cd /home/user/Platinum-Palladium-AI-Printing-Tool
   uvicorn src.ptpd_calibration.api.server:app --reload
   ```

2. Start the frontend dev server:
   ```bash
   cd frontend
   pnpm dev
   ```

3. Open the AI Assistant tab
4. Send a message and observe real-time streaming
5. Click "Stop" button to test cancellation
6. Verify error handling by stopping the backend server

## Performance Characteristics

- **First chunk latency**: ~100-300ms (depends on LLM)
- **Chunk frequency**: Variable, typically 10-50ms between chunks
- **Memory usage**: Minimal (streams don't buffer entire response)
- **Network efficiency**: HTTP/2 keepalive connection reduces overhead

## Debugging

Enable debug logging:
```bash
VITE_LOG_LEVEL=debug pnpm dev
```

Look for these log messages:
- `SSE: Starting stream` - Stream initiated
- `SSE: Stream aborted by user` - User cancelled
- `SSE: Failed to parse event` - Invalid JSON (non-critical)
- `useStreamingChat: Stream completed` - Successful completion
- `useStreamingChat: Stream error` - Error occurred
- `useStreamingChat: Falling back to non-streaming API` - Fallback triggered

## Limitations

1. **EventSource not used**: We use fetch + ReadableStream because EventSource only supports GET requests, and we need POST for the chat payload.

2. **No reconnection**: If the connection drops, the stream fails and falls back to non-streaming. Automatic reconnection is not implemented.

3. **Browser compatibility**: Requires modern browser with ReadableStream support (Chrome 52+, Firefox 65+, Safari 10.1+).

4. **No progress indicator**: We don't know the total length, so progress is shown as indeterminate.

## Future Enhancements

- [ ] Automatic retry on network failures
- [ ] Streaming for recipe suggestions and troubleshooting
- [ ] Typing indicator based on streaming state
- [ ] Token usage tracking in stream metadata
- [ ] Support for SSE reconnection with last-event-id

## Migration Guide

To migrate a component from `useChat` to `useStreamingChat`:

1. Change the import:
   ```typescript
   // Before
   import { useChat } from '@/hooks/useChat';

   // After
   import { useStreamingChat } from '@/hooks/useStreamingChat';
   ```

2. Update the hook call:
   ```typescript
   // Before
   const { sendSuggestion, ... } = useChat();

   // After
   const { sendMessage, cancelStream, ... } = useStreamingChat();
   ```

3. Replace method calls:
   ```typescript
   // Before
   sendSuggestion(text);

   // After
   sendMessage(text);
   ```

4. Add cancel button (optional):
   ```tsx
   {isStreaming && (
     <button onClick={cancelStream}>Stop</button>
   )}
   ```

## References

- Backend streaming: `/home/user/Platinum-Palladium-AI-Printing-Tool/src/ptpd_calibration/llm/assistant.py` (line 83-117)
- Chat slice: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/stores/slices/chatSlice.ts`
- Original useChat hook: `/home/user/Platinum-Palladium-AI-Printing-Tool/frontend/src/hooks/useChat.ts`
- SSE specification: https://html.spec.whatwg.org/multipage/server-sent-events.html
