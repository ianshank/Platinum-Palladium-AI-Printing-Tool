/**
 * AIAssistant Component
 *
 * Chat interface for the Pt/Pd printing assistant.
 * Uses the `useChat` hook for API orchestration and Zustand state management.
 *
 * Features:
 * - Message list with user/assistant bubbles
 * - Text input with send on Enter / button click
 * - Streaming indicator while AI responds
 * - Error display with retry
 * - New conversation / clear history actions
 * - Empty state with suggestions
 * - Auto-scroll to latest message
 */

import { type FC, useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { useChat } from '@/hooks/useChat';

export interface AIAssistantProps {
  className?: string;
}

/** Suggestion prompts for the empty state */
const SUGGESTIONS = [
  'What is the ideal Pt/Pd metal ratio for a warm-tone print?',
  'How do I adjust exposure for Arches Platine paper?',
  'My highlights are too flat — how should I change contrast?',
  'What developer temperature gives the best Dmax?',
] as const;

/** Format timestamp for message display */
function formatTime(iso: string): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: 'numeric',
    }).format(new Date(iso));
  } catch {
    return '';
  }
}

export const AIAssistant: FC<AIAssistantProps> = ({ className }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // --- useChat hook (replaces direct api + store wiring) ---
  const {
    messages,
    isStreaming,
    streamContent,
    error,
    isBusy,
    sendSuggestion,
    clear,
    newConversation,
  } = useChat();

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, streamContent]);

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isBusy) return;
    sendSuggestion(trimmed);
    setInput('');
  }, [input, isBusy, sendSuggestion]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      handleSend();
    },
    [handleSend]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleSuggestionClick = useCallback(
    (suggestion: string) => {
      sendSuggestion(suggestion);
    },
    [sendSuggestion]
  );

  const handleNewConversation = useCallback(() => {
    newConversation();
  }, [newConversation]);

  const hasMessages = messages.length > 0;

  return (
    <div
      className={cn(
        'flex h-full min-h-[500px] flex-col rounded-lg border bg-card',
        className
      )}
      data-testid="ai-assistant"
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="text-lg font-semibold">AI Printing Assistant</h2>
          <p className="text-xs text-muted-foreground">
            Ask about Pt/Pd calibration, chemistry, or troubleshooting
          </p>
        </div>
        <div className="flex gap-2">
          {hasMessages && (
            <button
              type="button"
              onClick={clear}
              className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              data-testid="clear-chat-btn"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={handleNewConversation}
            className="rounded-md border px-3 py-1.5 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            data-testid="new-conversation-btn"
          >
            New Chat
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div
        className="flex-1 space-y-4 overflow-y-auto px-4 py-4"
        data-testid="messages-area"
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
      >
        {!hasMessages && !isBusy && (
          <div
            className="flex h-full flex-col items-center justify-center space-y-4 text-center"
            data-testid="empty-chat"
          >
            <div className="rounded-full bg-primary/10 p-4">
              <svg
                className="h-8 w-8 text-primary"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M8.625 12a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 0 1-2.555-.337A5.972 5.972 0 0 1 5.41 20.97a5.969 5.969 0 0 1-.474-.065 4.48 4.48 0 0 0 .978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25Z"
                />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium">How can I help?</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Ask about calibration, chemistry, or troubleshooting
              </p>
            </div>
            <div className="grid w-full max-w-sm gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="w-full rounded-md border px-3 py-2 text-left text-xs transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  data-testid="suggestion-btn"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              'flex',
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            )}
            data-testid={`message-${msg.id}`}
          >
            <div
              className={cn(
                'max-w-[80%] rounded-lg px-4 py-2.5',
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted'
              )}
            >
              <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
              <p
                className={cn(
                  'mt-1 text-[10px]',
                  msg.role === 'user'
                    ? 'text-primary-foreground/70'
                    : 'text-muted-foreground'
                )}
              >
                {formatTime(msg.timestamp)}
              </p>
            </div>
          </div>
        ))}

        {/* Streaming indicator */}
        {isStreaming && streamContent && (
          <div className="flex justify-start" data-testid="streaming-message">
            <div className="max-w-[80%] rounded-lg bg-muted px-4 py-2.5">
              <p className="whitespace-pre-wrap text-sm">{streamContent}</p>
              <span className="mt-1 inline-block h-2 w-2 animate-pulse rounded-full bg-primary" />
            </div>
          </div>
        )}

        {/* Loading dots */}
        {isBusy && !streamContent && (
          <div className="flex justify-start" data-testid="loading-indicator">
            <div className="flex gap-1.5 rounded-lg bg-muted px-4 py-3">
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.3s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.15s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div
          className="mx-4 mb-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive"
          role="alert"
          data-testid="chat-error"
        >
          {error}
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="border-t px-4 py-3"
        data-testid="chat-form"
      >
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about calibration, chemistry…"
            disabled={isBusy}
            rows={1}
            className={cn(
              'flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm',
              'placeholder:text-muted-foreground',
              'focus:outline-none focus:ring-2 focus:ring-primary',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
            data-testid="chat-input"
            aria-label="Chat message"
          />
          <button
            type="submit"
            disabled={isBusy || !input.trim()}
            className={cn(
              'rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground',
              'hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
              'disabled:cursor-not-allowed disabled:opacity-50',
              'transition-colors'
            )}
            data-testid="send-btn"
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

AIAssistant.displayName = 'AIAssistant';
