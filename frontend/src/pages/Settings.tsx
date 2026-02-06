/**
 * Settings page component.
 */

import styled from 'styled-components';
import { useUIStore } from '@/store';

const PageContainer = styled.div`
  max-width: 800px;
  margin: 0 auto;
`;

const PageHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[8]};
`;

const PageTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const SettingsSection = styled.div`
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[6]};
  margin-bottom: ${({ theme }) => theme.spacing[6]};
`;

const SectionTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const SettingRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[4]} 0;
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};

  &:last-child {
    border-bottom: none;
  }
`;

const SettingLabel = styled.div`
  display: flex;
  flex-direction: column;
`;

const SettingName = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const SettingDescription = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-top: ${({ theme }) => theme.spacing[1]};
`;

const Select = styled.select`
  padding: ${({ theme }) => theme.spacing[2]} ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.subtle};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  cursor: pointer;

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
    outline: none;
  }
`;

const Toggle = styled.button<{ $active: boolean }>`
  width: 48px;
  height: 24px;
  border-radius: ${({ theme }) => theme.radii.full};
  background-color: ${({ theme, $active }) =>
    $active ? theme.colors.accent.primary : theme.colors.background.tertiary};
  position: relative;
  transition: all ${({ theme }) => theme.transitions.fast};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.accent.primary : theme.colors.border.default};

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $active }) => ($active ? '24px' : '2px')};
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background-color: ${({ theme }) => theme.colors.text.primary};
    transition: left ${({ theme }) => theme.transitions.fast};
  }
`;

const InfoBox = styled.div`
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

export function Settings() {
  const theme = useUIStore((state) => state.theme);
  const setTheme = useUIStore((state) => state.setTheme);
  const sidebarCollapsed = useUIStore((state) => state.sidebarCollapsed);
  const setSidebarCollapsed = useUIStore((state) => state.setSidebarCollapsed);

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Settings</PageTitle>
      </PageHeader>

      <SettingsSection>
        <SectionTitle>Appearance</SectionTitle>

        <SettingRow>
          <SettingLabel>
            <SettingName>Theme</SettingName>
            <SettingDescription>
              Choose your preferred color scheme
            </SettingDescription>
          </SettingLabel>
          <Select
            value={theme}
            onChange={(e) => setTheme(e.target.value as 'dark' | 'light' | 'system')}
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
            <option value="system">System</option>
          </Select>
        </SettingRow>

        <SettingRow>
          <SettingLabel>
            <SettingName>Compact Sidebar</SettingName>
            <SettingDescription>
              Show collapsed sidebar by default
            </SettingDescription>
          </SettingLabel>
          <Toggle
            $active={sidebarCollapsed}
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            aria-label="Toggle compact sidebar"
          />
        </SettingRow>
      </SettingsSection>

      <SettingsSection>
        <SectionTitle>About</SectionTitle>
        <InfoBox>
          <p>
            <strong>PTPD Calibration Studio</strong>
          </p>
          <p style={{ marginTop: '8px' }}>
            An AI-powered calibration system for platinum/palladium printing.
            Built with React, TypeScript, and FastAPI.
          </p>
          <p style={{ marginTop: '8px' }}>
            For documentation and support, visit the project repository.
          </p>
        </InfoBox>
      </SettingsSection>
    </PageContainer>
  );
}
