/**
 * Application header component.
 */

import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { useUIStore } from '@/store';
import { APP_CONSTANTS } from '@/config';

const HeaderContainer = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
  padding: 0 ${({ theme }) => theme.spacing[6]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
  position: sticky;
  top: 0;
  z-index: ${({ theme }) => theme.zIndex.sticky};
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const MenuButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.secondary};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.background.hover};
    color: ${({ theme }) => theme.colors.text.primary};
  }

  @media (min-width: 1024px) {
    display: none;
  }
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  text-decoration: none;
  color: ${({ theme }) => theme.colors.text.primary};
`;

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(
    135deg,
    ${({ theme }) => theme.colors.metals.platinum},
    ${({ theme }) => theme.colors.metals.palladium}
  );
  border-radius: ${({ theme }) => theme.radii.md};
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.inverse};
`;

const LogoText = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};

  @media (max-width: 640px) {
    display: none;
  }
`;

const RightSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const IconButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.secondary};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.background.hover};
    color: ${({ theme }) => theme.colors.text.primary};
  }
`;

const VersionBadge = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.tertiary};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  padding: ${({ theme }) => theme.spacing[1]} ${({ theme }) => theme.spacing[2]};
  border-radius: ${({ theme }) => theme.radii.full};
`;

// Icons as inline SVGs for simplicity
const MenuIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 12h18M3 6h18M3 18h18" />
  </svg>
);

const SettingsIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

export function Header() {
  const toggleSidebar = useUIStore((state) => state.toggleSidebar);

  return (
    <HeaderContainer>
      <LeftSection>
        <MenuButton onClick={toggleSidebar} aria-label="Toggle menu">
          <MenuIcon />
        </MenuButton>
        <Logo to="/dashboard">
          <LogoIcon>Pt</LogoIcon>
          <LogoText>{APP_CONSTANTS.APP_NAME}</LogoText>
        </Logo>
      </LeftSection>

      <RightSection>
        <VersionBadge>v{APP_CONSTANTS.APP_VERSION}</VersionBadge>
        <IconButton as={Link} to="/settings" aria-label="Settings">
          <SettingsIcon />
        </IconButton>
      </RightSection>
    </HeaderContainer>
  );
}
