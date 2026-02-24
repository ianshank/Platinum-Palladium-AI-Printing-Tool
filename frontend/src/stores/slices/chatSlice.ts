/**
 * Chat state slice
 * Manages AI assistant chat messages and conversation state
 */

import type { StateCreator } from 'zustand';
import { logger } from '@/lib/logger';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: {
    model?: string;
    tokens?: number;
    context?: string[];
  };
}

export interface ChatContext {
  id: string;
  label: string;
  content: string;
}

export interface ChatSlice {
  // State
  messages: ChatMessage[];
  contexts: ChatContext[];
  selectedContextIds: string[];
  isLoading: boolean;
  isStreaming: boolean;
  streamContent: string;
  error: string | null;
  conversationId: string | null;

  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateMessage: (id: string, content: string) => void;
  deleteMessage: (id: string) => void;
  clearMessages: () => void;

  // Context management
  addContext: (context: Omit<ChatContext, 'id'>) => void;
  removeContext: (id: string) => void;
  toggleContextSelection: (id: string) => void;
  clearContexts: () => void;

  // Streaming
  startStreaming: () => void;
  appendStreamContent: (content: string) => void;
  finishStreaming: () => void;
  cancelStreaming: () => void;

  // State management
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  startNewConversation: () => void;

  // Reset
  resetChat: () => void;
}

const initialState = {
  messages: [] as ChatMessage[],
  contexts: [] as ChatContext[],
  selectedContextIds: [] as string[],
  isLoading: false,
  isStreaming: false,
  streamContent: '',
  error: null as string | null,
  conversationId: null as string | null,
};

let messageIdCounter = 0;
let contextIdCounter = 0;

export const createChatSlice: StateCreator<
  { chat: ChatSlice },
  [['zustand/immer', never]],
  [],
  ChatSlice
> = (set, get) => ({
  ...initialState,

  addMessage: (message) => {
    const id = `msg-${++messageIdCounter}`;
    const timestamp = new Date().toISOString();

    logger.debug('Chat: addMessage', { id, role: message.role });

    set((state) => {
      state.chat.messages.push({
        ...message,
        id,
        timestamp,
      });
    });
  },

  updateMessage: (id, content) => {
    logger.debug('Chat: updateMessage', { id });
    set((state) => {
      const message = state.chat.messages.find((m) => m.id === id);
      if (message) {
        message.content = content;
      }
    });
  },

  deleteMessage: (id) => {
    logger.debug('Chat: deleteMessage', { id });
    set((state) => {
      state.chat.messages = state.chat.messages.filter((m) => m.id !== id);
    });
  },

  clearMessages: () => {
    logger.debug('Chat: clearMessages');
    set((state) => {
      state.chat.messages = [];
    });
  },

  addContext: (context) => {
    const id = `ctx-${++contextIdCounter}`;
    logger.debug('Chat: addContext', { id, label: context.label });

    set((state) => {
      state.chat.contexts.push({ ...context, id });
    });
  },

  removeContext: (id) => {
    logger.debug('Chat: removeContext', { id });
    set((state) => {
      state.chat.contexts = state.chat.contexts.filter((c) => c.id !== id);
      state.chat.selectedContextIds = state.chat.selectedContextIds.filter(
        (cid) => cid !== id
      );
    });
  },

  toggleContextSelection: (id) => {
    logger.debug('Chat: toggleContextSelection', { id });
    set((state) => {
      const index = state.chat.selectedContextIds.indexOf(id);
      if (index === -1) {
        state.chat.selectedContextIds.push(id);
      } else {
        state.chat.selectedContextIds.splice(index, 1);
      }
    });
  },

  clearContexts: () => {
    logger.debug('Chat: clearContexts');
    set((state) => {
      state.chat.contexts = [];
      state.chat.selectedContextIds = [];
    });
  },

  startStreaming: () => {
    logger.debug('Chat: startStreaming');
    set((state) => {
      state.chat.isStreaming = true;
      state.chat.streamContent = '';
      state.chat.error = null;
    });
  },

  appendStreamContent: (content) => {
    set((state) => {
      state.chat.streamContent += content;
    });
  },

  finishStreaming: () => {
    logger.debug('Chat: finishStreaming');
    const streamContent = get().chat.streamContent;

    if (streamContent) {
      get().chat.addMessage({
        role: 'assistant',
        content: streamContent,
      });
    }

    set((state) => {
      state.chat.isStreaming = false;
      state.chat.streamContent = '';
    });
  },

  cancelStreaming: () => {
    logger.debug('Chat: cancelStreaming');
    set((state) => {
      state.chat.isStreaming = false;
      state.chat.streamContent = '';
    });
  },

  setLoading: (loading) => {
    logger.debug('Chat: setLoading', { loading });
    set((state) => {
      state.chat.isLoading = loading;
    });
  },

  setError: (error) => {
    if (error) {
      logger.error('Chat: error', { error });
    }
    set((state) => {
      state.chat.error = error;
      state.chat.isLoading = false;
      state.chat.isStreaming = false;
    });
  },

  startNewConversation: () => {
    const conversationId = `conv-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    logger.info('Chat: startNewConversation', { conversationId });

    set((state) => {
      state.chat.messages = [];
      state.chat.conversationId = conversationId;
      state.chat.error = null;
    });
  },

  resetChat: () => {
    logger.debug('Chat: resetChat');
    set((state) => {
      Object.assign(state.chat, initialState);
    });
  },
});
