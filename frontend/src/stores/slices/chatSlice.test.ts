import { describe, it, expect, beforeEach } from 'vitest';
import { createStore } from '@/stores';

describe('chatSlice', () => {
    let store: ReturnType<typeof createStore>;

    beforeEach(() => {
        store = createStore();
    });

    describe('Initial State', () => {
        it('has correct initial values', () => {
            const state = store.getState();
            expect(state.chat.messages).toEqual([]);
            expect(state.chat.contexts).toEqual([]);
            expect(state.chat.selectedContextIds).toEqual([]);
            expect(state.chat.isLoading).toBe(false);
            expect(state.chat.isStreaming).toBe(false);
            expect(state.chat.streamContent).toBe('');
            expect(state.chat.error).toBeNull();
            expect(state.chat.conversationId).toBeNull();
        });
    });

    describe('Message Management', () => {
        it('addMessage appends a user message', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'Hello' });

            const messages = store.getState().chat.messages;
            expect(messages).toHaveLength(1);
            expect(messages[0]?.role).toBe('user');
            expect(messages[0]?.content).toBe('Hello');
            expect(messages[0]?.id).toBeDefined();
            expect(messages[0]?.timestamp).toBeDefined();
        });

        it('addMessage appends an assistant message', () => {
            store.getState().chat.addMessage({ role: 'assistant', content: 'Hi there!' });

            expect(store.getState().chat.messages[0]?.role).toBe('assistant');
        });

        it('updateMessage modifies existing message content', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'Original' });
            const id = store.getState().chat.messages[0]?.id;

            if (id) {
                store.getState().chat.updateMessage(id, 'Updated');
                expect(store.getState().chat.messages[0]?.content).toBe('Updated');
            }
        });

        it('deleteMessage removes a message', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'Delete me' });
            const id = store.getState().chat.messages[0]?.id;

            if (id) {
                store.getState().chat.deleteMessage(id);
                expect(store.getState().chat.messages).toHaveLength(0);
            }
        });

        it('clearMessages removes all messages', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'Msg 1' });
            store.getState().chat.addMessage({ role: 'assistant', content: 'Msg 2' });
            store.getState().chat.clearMessages();

            expect(store.getState().chat.messages).toHaveLength(0);
        });
    });

    describe('Context Management', () => {
        it('addContext adds a context', () => {
            store.getState().chat.addContext({ label: 'Curve Data', content: 'Points: ...' });

            const contexts = store.getState().chat.contexts;
            expect(contexts).toHaveLength(1);
            expect(contexts[0]?.label).toBe('Curve Data');
        });

        it('removeContext removes a context and deselects it', () => {
            store.getState().chat.addContext({ label: 'Ctx', content: '...' });
            const id = store.getState().chat.contexts[0]?.id;

            if (id) {
                store.getState().chat.toggleContextSelection(id);
                store.getState().chat.removeContext(id);

                expect(store.getState().chat.contexts).toHaveLength(0);
                expect(store.getState().chat.selectedContextIds).not.toContain(id);
            }
        });

        it('toggleContextSelection selects and deselects', () => {
            store.getState().chat.addContext({ label: 'Ctx', content: '...' });
            const id = store.getState().chat.contexts[0]?.id;

            if (id) {
                store.getState().chat.toggleContextSelection(id);
                expect(store.getState().chat.selectedContextIds).toContain(id);

                store.getState().chat.toggleContextSelection(id);
                expect(store.getState().chat.selectedContextIds).not.toContain(id);
            }
        });

        it('clearContexts removes all contexts and selections', () => {
            store.getState().chat.addContext({ label: 'Ctx1', content: '...' });
            store.getState().chat.addContext({ label: 'Ctx2', content: '...' });
            store.getState().chat.clearContexts();

            expect(store.getState().chat.contexts).toHaveLength(0);
            expect(store.getState().chat.selectedContextIds).toEqual([]);
        });
    });

    describe('Streaming', () => {
        it('startStreaming sets streaming state', () => {
            store.getState().chat.startStreaming();

            expect(store.getState().chat.isStreaming).toBe(true);
            expect(store.getState().chat.streamContent).toBe('');
            expect(store.getState().chat.error).toBeNull();
        });

        it('appendStreamContent accumulates content', () => {
            store.getState().chat.startStreaming();
            store.getState().chat.appendStreamContent('Hello ');
            store.getState().chat.appendStreamContent('World');

            expect(store.getState().chat.streamContent).toBe('Hello World');
        });

        it('finishStreaming creates assistant message from stream', () => {
            store.getState().chat.startStreaming();
            store.getState().chat.appendStreamContent('Streamed response');
            store.getState().chat.finishStreaming();

            expect(store.getState().chat.isStreaming).toBe(false);
            expect(store.getState().chat.streamContent).toBe('');
            expect(store.getState().chat.messages).toHaveLength(1);
            expect(store.getState().chat.messages[0]?.role).toBe('assistant');
            expect(store.getState().chat.messages[0]?.content).toBe('Streamed response');
        });

        it('cancelStreaming clears stream without creating message', () => {
            store.getState().chat.startStreaming();
            store.getState().chat.appendStreamContent('partial...');
            store.getState().chat.cancelStreaming();

            expect(store.getState().chat.isStreaming).toBe(false);
            expect(store.getState().chat.streamContent).toBe('');
            expect(store.getState().chat.messages).toHaveLength(0);
        });
    });

    describe('Loading and Error States', () => {
        it('setLoading toggles loading flag', () => {
            store.getState().chat.setLoading(true);
            expect(store.getState().chat.isLoading).toBe(true);

            store.getState().chat.setLoading(false);
            expect(store.getState().chat.isLoading).toBe(false);
        });

        it('setError sets error and stops loading/streaming', () => {
            store.getState().chat.setLoading(true);
            store.getState().chat.startStreaming();

            store.getState().chat.setError('Network error');

            expect(store.getState().chat.error).toBe('Network error');
            expect(store.getState().chat.isLoading).toBe(false);
            expect(store.getState().chat.isStreaming).toBe(false);
        });

        it('setError with null clears error', () => {
            store.getState().chat.setError('Error');
            store.getState().chat.setError(null);
            expect(store.getState().chat.error).toBeNull();
        });
    });

    describe('Conversation Management', () => {
        it('startNewConversation generates conversationId and clears messages', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'Old msg' });
            store.getState().chat.startNewConversation();

            const state = store.getState();
            expect(state.chat.messages).toHaveLength(0);
            expect(state.chat.conversationId).toBeDefined();
            expect(state.chat.conversationId).toMatch(/^conv-/);
        });
    });

    describe('Reset', () => {
        it('resetChat restores all initial state', () => {
            store.getState().chat.addMessage({ role: 'user', content: 'msg' });
            store.getState().chat.setLoading(true);
            store.getState().chat.setError('err');
            store.getState().chat.addContext({ label: 'Ctx', content: '...' });

            store.getState().chat.resetChat();

            const state = store.getState();
            expect(state.chat.messages).toEqual([]);
            expect(state.chat.isLoading).toBe(false);
            expect(state.chat.error).toBeNull();
            expect(state.chat.contexts).toEqual([]);
        });
    });
});
