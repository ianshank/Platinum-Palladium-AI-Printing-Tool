/**
 * AI Assistant page component.
 * Chat interface for getting help with Pt/Pd printing.
 */

import { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useChat, quickPrompts } from '@/api/hooks';

const PageContainer = styled.div`
  max-width: 900px;
  margin: 0 auto;
  height: calc(100vh - 180px);
  display: flex;
  flex-direction: column;
`;

const PageHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const PageTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const ChatContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  overflow: hidden;
`;

const MessagesArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${({ theme }) => theme.spacing[4]};
`;

const MessageBubble = styled.div<{ $isUser: boolean }>`
  max-width: 80%;
  padding: ${({ theme }) => theme.spacing[4]};
  margin-bottom: ${({ theme }) => theme.spacing[3]};
  border-radius: ${({ theme }) => theme.radii.lg};
  background-color: ${({ theme, $isUser }) =>
    $isUser ? theme.colors.accent.primary : theme.colors.background.tertiary};
  color: ${({ theme, $isUser }) =>
    $isUser ? theme.colors.text.inverse : theme.colors.text.primary};
  margin-left: ${({ $isUser }) => ($isUser ? 'auto' : '0')};
  margin-right: ${({ $isUser }) => ($isUser ? '0' : 'auto')};
  white-space: pre-wrap;
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const EmptyStateTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const EmptyStateText = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  margin-bottom: ${({ theme }) => theme.spacing[6]};
`;

const QuickPromptsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[3]};
  max-width: 600px;
`;

const QuickPromptButton = styled.button`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.secondary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  text-align: left;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.background.hover};
    color: ${({ theme }) => theme.colors.text.primary};
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }
`;

const InputArea = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => theme.spacing[4]};
  border-top: 1px solid ${({ theme }) => theme.colors.border.default};
  background-color: ${({ theme }) => theme.colors.background.secondary};
`;

const Input = styled.input`
  flex: 1;
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.subtle};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
    outline: none;
  }

  &::placeholder {
    color: ${({ theme }) => theme.colors.text.disabled};
  }
`;

const SendButton = styled.button`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[6]};
  background-color: ${({ theme }) => theme.colors.accent.primary};
  color: ${({ theme }) => theme.colors.text.inverse};
  border-radius: ${({ theme }) => theme.radii.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover:not(:disabled) {
    background-color: ${({ theme }) => theme.colors.accent.primaryHover};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const LoadingIndicator = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[1]};
  padding: ${({ theme }) => theme.spacing[4]};

  span {
    width: 8px;
    height: 8px;
    background-color: ${({ theme }) => theme.colors.text.secondary};
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out;

    &:nth-child(1) {
      animation-delay: -0.32s;
    }
    &:nth-child(2) {
      animation-delay: -0.16s;
    }
  }

  @keyframes bounce {
    0%,
    80%,
    100% {
      transform: scale(0);
    }
    40% {
      transform: scale(1);
    }
  }
`;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export function AIAssistant() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatMutation = useChat();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (messageText?: string) => {
    const text = messageText || input.trim();
    if (!text || chatMutation.isPending) return;

    const userMessage: Message = {
      id: `user-${Date.now()}-${Math.random().toString(36)}`,
      role: 'user',
      content: text,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      const response = await chatMutation.mutateAsync({ message: text });
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}-${Math.random().toString(36)}`,
        role: 'assistant',
        content: response,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch {
      const errorMessage: Message = {
        id: `error-${Date.now()}-${Math.random().toString(36)}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>AI Assistant</PageTitle>
      </PageHeader>

      <ChatContainer>
        <MessagesArea>
          {messages.length === 0 ? (
            <EmptyState>
              <EmptyStateTitle>Pt/Pd Printing Expert</EmptyStateTitle>
              <EmptyStateText>
                Ask me anything about platinum/palladium printing, calibration,
                chemistry, or troubleshooting.
              </EmptyStateText>
              <QuickPromptsGrid>
                {quickPrompts.map((prompt) => (
                  <QuickPromptButton
                    key={prompt.id}
                    onClick={() => handleSend(prompt.prompt)}
                  >
                    {prompt.label}
                  </QuickPromptButton>
                ))}
              </QuickPromptsGrid>
            </EmptyState>
          ) : (
            <>
              {messages.map((message) => (
                <MessageBubble key={message.id} $isUser={message.role === 'user'}>
                  {message.content}
                </MessageBubble>
              ))}
              {chatMutation.isPending && (
                <LoadingIndicator>
                  <span />
                  <span />
                  <span />
                </LoadingIndicator>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </MessagesArea>

        <InputArea>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about Pt/Pd printing..."
            disabled={chatMutation.isPending}
          />
          <SendButton
            onClick={() => handleSend()}
            disabled={!input.trim() || chatMutation.isPending}
          >
            Send
          </SendButton>
        </InputArea>
      </ChatContainer>
    </PageContainer>
  );
}
