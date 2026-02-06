/**
 * Main application shell component.
 * Provides the layout structure with header, sidebar, and main content area.
 */

import { ReactNode } from 'react';
import styled from 'styled-components';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useUIStore } from '@/store';
import { media } from '@/theme';

interface AppShellProps {
  children: ReactNode;
}

const Container = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: ${({ theme }) => theme.colors.background.primary};
`;

const MainWrapper = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
`;

const MainContent = styled.main<{ $sidebarCollapsed: boolean }>`
  flex: 1;
  overflow-y: auto;
  padding: ${({ theme }) => theme.spacing[6]};
  transition: margin-left ${({ theme }) => theme.transitions.normal};

  ${media.lg} {
    margin-left: ${({ $sidebarCollapsed }) =>
      $sidebarCollapsed ? '64px' : '240px'};
  }

  @media (max-width: 1023px) {
    margin-left: 0;
  }
`;

const SkipLink = styled.a`
  position: absolute;
  top: -100%;
  left: 0;
  padding: ${({ theme }) => theme.spacing[2]} ${({ theme }) => theme.spacing[4]};
  background: ${({ theme }) => theme.colors.accent.primary};
  color: ${({ theme }) => theme.colors.text.inverse};
  z-index: ${({ theme }) => theme.zIndex.tooltip};
  text-decoration: none;

  &:focus {
    top: 0;
  }
`;

export function AppShell({ children }: AppShellProps) {
  const sidebarCollapsed = useUIStore((state) => state.sidebarCollapsed);

  return (
    <Container>
      <SkipLink href="#main-content">Skip to main content</SkipLink>
      <Header />
      <MainWrapper>
        <Sidebar />
        <MainContent id="main-content" $sidebarCollapsed={sidebarCollapsed}>
          {children}
        </MainContent>
      </MainWrapper>
    </Container>
  );
}
