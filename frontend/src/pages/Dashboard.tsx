/**
 * Dashboard page component.
 * Displays session metrics, quick actions, and recent activity.
 */

import styled from 'styled-components';
import { Link } from 'react-router-dom';

const PageContainer = styled.div`
  max-width: 1400px;
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

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: ${({ theme }) => theme.spacing[6]};
  margin-bottom: ${({ theme }) => theme.spacing[8]};
`;

const StatCard = styled.div`
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[6]};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.border.subtle};
    box-shadow: ${({ theme }) => theme.shadows.md};
  }
`;

const StatLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const StatValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const StatChange = styled.span<{ $positive?: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme, $positive }) =>
    $positive ? theme.colors.semantic.success : theme.colors.text.secondary};
  margin-left: ${({ theme }) => theme.spacing[2]};
`;

const SectionTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const QuickActionsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
  margin-bottom: ${({ theme }) => theme.spacing[8]};
`;

const QuickActionCard = styled(Link)`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[6]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  text-decoration: none;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.accent.primary};
    background-color: ${({ theme }) => theme.colors.background.hover};
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.lg};
  }
`;

const ActionIcon = styled.div`
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.lg};
  color: ${({ theme }) => theme.colors.accent.primary};
  margin-bottom: ${({ theme }) => theme.spacing[3]};
`;

const ActionTitle = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
  text-align: center;
`;

const TwoColumnGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[6]};
`;

const TipCard = styled.div`
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.md};
  margin-bottom: ${({ theme }) => theme.spacing[3]};

  &:last-child {
    margin-bottom: 0;
  }
`;

const TipTitle = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const TipContent = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  line-height: ${({ theme }) => theme.typography.lineHeight.relaxed};
`;

// Icons
const CalibrationIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M2 12h2M20 12h2M12 2v2M12 20v2" />
    <circle cx="12" cy="12" r="8" />
    <path d="M12 12l4 4" />
  </svg>
);

const ChemistryIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M9 3h6v5l4 9H5l4-9V3z" />
    <path d="M9 3h6" />
  </svg>
);

const CurveIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 3v18h18" />
    <path d="M7 16c2-4 4-8 10-10" />
  </svg>
);

const SessionIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
    <line x1="16" y1="2" x2="16" y2="6" />
    <line x1="8" y1="2" x2="8" y2="6" />
    <line x1="3" y1="10" x2="21" y2="10" />
  </svg>
);

export function Dashboard() {
  // In production, these would come from API/state
  const stats = {
    activeCurves: 12,
    printsThisWeek: 8,
    successRate: 87.5,
    calibrations: 24,
  };

  const quickActions = [
    { to: '/calibration', title: 'New Calibration', icon: <CalibrationIcon /> },
    { to: '/chemistry', title: 'Calculate Chemistry', icon: <ChemistryIcon /> },
    { to: '/calibration?mode=curves', title: 'Manage Curves', icon: <CurveIcon /> },
    { to: '/sessions', title: 'Log Session', icon: <SessionIcon /> },
  ];

  const tips = [
    {
      title: 'Optimal Coating Temperature',
      content:
        'For best results, coat your paper at room temperature (68-72°F / 20-22°C). Cold paper can cause uneven absorption.',
    },
    {
      title: 'Na2 Contrast Control',
      content:
        'Start with 1-2 drops of Na2 per 24 drops of metal. Increase for higher contrast with flat negatives.',
    },
  ];

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Dashboard</PageTitle>
        <PageSubtitle>
          Welcome back. Here is your Pt/Pd printing overview.
        </PageSubtitle>
      </PageHeader>

      <StatsGrid>
        <StatCard>
          <StatLabel>Active Curves</StatLabel>
          <StatValue>
            {stats.activeCurves}
            <StatChange $positive>+2 this week</StatChange>
          </StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Prints This Week</StatLabel>
          <StatValue>{stats.printsThisWeek}</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Success Rate</StatLabel>
          <StatValue>{stats.successRate}%</StatValue>
        </StatCard>
        <StatCard>
          <StatLabel>Total Calibrations</StatLabel>
          <StatValue>{stats.calibrations}</StatValue>
        </StatCard>
      </StatsGrid>

      <SectionTitle>Quick Actions</SectionTitle>
      <QuickActionsGrid>
        {quickActions.map((action) => (
          <QuickActionCard key={action.to} to={action.to}>
            <ActionIcon>{action.icon}</ActionIcon>
            <ActionTitle>{action.title}</ActionTitle>
          </QuickActionCard>
        ))}
      </QuickActionsGrid>

      <TwoColumnGrid>
        <Card>
          <SectionTitle>Recent Activity</SectionTitle>
          <TipContent>
            No recent sessions. Start a new calibration or log a print session.
          </TipContent>
        </Card>

        <Card>
          <SectionTitle>Tips & Recommendations</SectionTitle>
          {tips.map((tip, index) => (
            <TipCard key={index}>
              <TipTitle>{tip.title}</TipTitle>
              <TipContent>{tip.content}</TipContent>
            </TipCard>
          ))}
        </Card>
      </TwoColumnGrid>
    </PageContainer>
  );
}
