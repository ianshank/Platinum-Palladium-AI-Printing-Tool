/**
 * AIAssistant Component
 *
 * Chat interface for the Pt/Pd printing assistant.
 * Reads from and writes to the chat Zustand store slice.
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
import { logger } from '@/lib/logger';
import { useStore } from '@/stores';
import { api } from '@/api/client';

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

    // --- Store ---
    const messages = useStore((s) => s.chat.messages);
    const isLoading = useStore((s) => s.chat.isLoading);
    const isStreaming = useStore((s) => s.chat.isStreaming);
    const streamContent = useStore((s) => s.chat.streamContent);
    const error = useStore((s) => s.chat.error);

    const addMessage = useStore((s) => s.chat.addMessage);
    const setLoading = useStore((s) => s.chat.setLoading);
    const setError = useStore((s) => s.chat.setError);
    const startStreaming = useStore((s) => s.chat.startStreaming);
    const appendStreamContent = useStore((s) => s.chat.appendStreamContent);
    const finishStreaming = useStore((s) => s.chat.finishStreaming);
    const clearMessages = useStore((s) => s.chat.clearMessages);
    const startNewConversation = useStore((s) => s.chat.startNewConversation);

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages.length, streamContent]);

    const sendMessage = useCallback(
        async (text: string) => {
            const trimmed = text.trim();
            if (!trimmed || isLoading || isStreaming) return;

            logger.debug('AIAssistant: Sending message', {
                length: trimmed.length,
            });

            // Add user message
            addMessage({ role: 'user', content: trimmed });
            setInput('');
            setError(null);
            setLoading(true);

            try {
                startStreaming();
                const response = await api.chat.send({ message: trimmed });

                if (response?.response) {
                    appendStreamContent(response.response);
                }

                finishStreaming();
                logger.debug('AIAssistant: Response received');
            } catch (err) {
                const msg =
                    err instanceof Error ? err.message : 'Failed to get AI response';
                logger.error('AIAssistant: Error', { error: msg });
                setError(msg);
            } finally {
                setLoading(false);
            }
        },
        [
            isLoading,
            isStreaming,
            addMessage,
            setError,
            setLoading,
            startStreaming,
            appendStreamContent,
            finishStreaming,
        ]
    );

    const handleSubmit = useCallback(
        (e: React.FormEvent) => {
            e.preventDefault();
            sendMessage(input);
        },
        [input, sendMessage]
    );

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(input);
            }
        },
        [input, sendMessage]
    );

    const handleSuggestionClick = useCallback(
        (suggestion: string) => {
            sendMessage(suggestion);
        },
        [sendMessage]
    );

    const handleNewConversation = useCallback(() => {
        logger.debug('AIAssistant: New conversation');
        startNewConversation();
    }, [startNewConversation]);

    const hasMessages = messages.length > 0;
    const isBusy = isLoading || isStreaming;

    return (
        <div
            className={cn(
                'flex flex-col h-full min-h-[500px] rounded-lg border bg-card',
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
                            onClick={clearMessages}
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
                className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
                data-testid="messages-area"
                role="log"
                aria-label="Chat messages"
                aria-live="polite"
            >
                {!hasMessages && !isBusy && (
                    <div
                        className="flex flex-col items-center justify-center h-full text-center space-y-4"
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
                            <p className="text-xs text-muted-foreground mt-1">
                                Ask about calibration, chemistry, or troubleshooting
                            </p>
                        </div>
                        <div className="grid gap-2 w-full max-w-sm">
                            {SUGGESTIONS.map((suggestion) => (
                                <button
                                    key={suggestion}
                                    type="button"
                                    onClick={() => handleSuggestionClick(suggestion)}
                                    className="w-full rounded-md border px-3 py-2 text-left text-xs hover:bg-accent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
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
                            <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
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
                            <p className="text-sm whitespace-pre-wrap">{streamContent}</p>
                            <span className="inline-block mt-1 h-2 w-2 animate-pulse rounded-full bg-primary" />
                        </div>
                    </div>
                )}

                {/* Loading dots */}
                {isBusy && !streamContent && (
                    <div className="flex justify-start" data-testid="loading-indicator">
                        <div className="rounded-lg bg-muted px-4 py-3 flex gap-1.5">
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
