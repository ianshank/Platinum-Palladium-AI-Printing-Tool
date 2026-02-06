/**
 * Session Log page component.
 * Displays print session history and statistics.
 */

import styled from 'styled-components';

const PageContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const PageHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[8]};
`;

const PageTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const PageSubtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing[16]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  text-align: center;
`;

const EmptyStateIcon = styled.div`
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.xl};
  color: ${({ theme }) => theme.colors.text.tertiary};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const EmptyStateTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const EmptyStateText = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
  max-width: 400px;
`;

const SessionIcon = () => (
  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>
);

export function SessionLog() {
  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Session Log</PageTitle>
        <PageSubtitle>
          Track your print sessions and results over time.
        </PageSubtitle>
      </PageHeader>

      <EmptyState>
        <EmptyStateIcon>
          <SessionIcon />
        </EmptyStateIcon>
        <EmptyStateTitle>No Sessions Yet</EmptyStateTitle>
        <EmptyStateText>
          Start logging your print sessions to track your progress and
          build a history of your calibrations and results.
        </EmptyStateText>
      </EmptyState>
    </PageContainer>
  );
}
