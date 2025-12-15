/**
 * Application sidebar navigation component.
 */

import styled, { css } from 'styled-components';
import { NavLink, useLocation } from 'react-router-dom';
import { useUIStore } from '@/store';
import { env } from '@/config/env';
import { media } from '@/theme';

const SidebarContainer = styled.aside<{ $collapsed: boolean }>`
  position: fixed;
  top: 64px;
  left: 0;
  bottom: 0;
  width: ${({ $collapsed }) => ($collapsed ? '64px' : '240px')};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border-right: 1px solid ${({ theme }) => theme.colors.border.default};
  transition: width ${({ theme }) => theme.transitions.normal};
  z-index: ${({ theme }) => theme.zIndex.sticky};
  overflow-x: hidden;

  @media (max-width: 1023px) {
    transform: translateX(${({ $collapsed }) => ($collapsed ? '-100%' : '0')});
    width: 240px;
  }
`;

const Nav = styled.nav`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing[4]};
  gap: ${({ theme }) => theme.spacing[1]};
`;

const NavSection = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const SectionLabel = styled.span<{ $collapsed: boolean }>`
  display: ${({ $collapsed }) => ($collapsed ? 'none' : 'block')};
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.tertiary};
  text-transform: uppercase;
  letter-spacing: ${({ theme }) => theme.typography.letterSpacing.wider};
  padding: ${({ theme }) => theme.spacing[2]} ${({ theme }) => theme.spacing[3]};
`;

const navItemStyles = css<{ $collapsed: boolean }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => theme.spacing[3]};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.secondary};
  text-decoration: none;
  transition: all ${({ theme }) => theme.transitions.fast};
  white-space: nowrap;

  ${({ $collapsed }) =>
    $collapsed &&
    css`
      justify-content: center;
      padding: ${({ theme }) => theme.spacing[3]};
    `}

  &:hover {
    background-color: ${({ theme }) => theme.colors.background.hover};
    color: ${({ theme }) => theme.colors.text.primary};
  }

  &.active {
    background-color: ${({ theme }) => theme.colors.background.tertiary};
    color: ${({ theme }) => theme.colors.accent.primary};
  }
`;

const StyledNavLink = styled(NavLink)<{ $collapsed: boolean }>`
  ${navItemStyles}
`;

const NavIcon = styled.span`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  flex-shrink: 0;
`;

const NavLabel = styled.span<{ $collapsed: boolean }>`
  display: ${({ $collapsed }) => ($collapsed ? 'none' : 'block')};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
`;

const CollapseButton = styled.button<{ $collapsed: boolean }>`
  display: none;
  align-items: center;
  justify-content: center;
  width: 100%;
  padding: ${({ theme }) => theme.spacing[3]};
  margin-top: auto;
  color: ${({ theme }) => theme.colors.text.tertiary};
  border-top: 1px solid ${({ theme }) => theme.colors.border.default};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.text.secondary};
    background-color: ${({ theme }) => theme.colors.background.hover};
  }

  ${media.lg} {
    display: flex;
  }

  svg {
    transform: rotate(${({ $collapsed }) => ($collapsed ? '180deg' : '0')});
    transition: transform ${({ theme }) => theme.transitions.normal};
  }
`;

// Navigation items configuration
const navItems = [
  {
    section: 'Overview',
    items: [
      { to: '/dashboard', label: 'Dashboard', icon: 'dashboard' },
    ],
  },
  {
    section: 'Calibration',
    items: [
      { to: '/calibration', label: 'Calibration Wizard', icon: 'calibration' },
      { to: '/chemistry', label: 'Chemistry Calculator', icon: 'chemistry' },
    ],
  },
  {
    section: 'Tools',
    items: [
      ...(env.VITE_ENABLE_AI_ASSISTANT
        ? [{ to: '/assistant', label: 'AI Assistant', icon: 'assistant' }]
        : []),
      { to: '/sessions', label: 'Session Log', icon: 'sessions' },
    ],
  },
];

// Icons
const icons: Record<string, React.ReactNode> = {
  dashboard: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="7" height="9" rx="1" />
      <rect x="14" y="3" width="7" height="5" rx="1" />
      <rect x="14" y="12" width="7" height="9" rx="1" />
      <rect x="3" y="16" width="7" height="5" rx="1" />
    </svg>
  ),
  calibration: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <circle cx="12" cy="12" r="8" />
      <path d="M12 12l4 4" />
    </svg>
  ),
  chemistry: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9 3h6v5l4 9H5l4-9V3z" />
      <path d="M9 3h6" />
    </svg>
  ),
  assistant: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  sessions: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  ),
  collapse: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  ),
};

export function Sidebar() {
  const sidebarCollapsed = useUIStore((state) => state.sidebarCollapsed);
  const toggleSidebar = useUIStore((state) => state.toggleSidebar);

  return (
    <SidebarContainer $collapsed={sidebarCollapsed}>
      <Nav role="navigation" aria-label="Main navigation">
        {navItems.map((section) => (
          <NavSection key={section.section}>
            <SectionLabel $collapsed={sidebarCollapsed}>
              {section.section}
            </SectionLabel>
            {section.items.map((item) => (
              <StyledNavLink
                key={item.to}
                to={item.to}
                $collapsed={sidebarCollapsed}
                title={sidebarCollapsed ? item.label : undefined}
              >
                <NavIcon>{icons[item.icon]}</NavIcon>
                <NavLabel $collapsed={sidebarCollapsed}>{item.label}</NavLabel>
              </StyledNavLink>
            ))}
          </NavSection>
        ))}
      </Nav>

      <CollapseButton
        onClick={toggleSidebar}
        $collapsed={sidebarCollapsed}
        aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {icons.collapse}
      </CollapseButton>
    </SidebarContainer>
  );
}
