/**
 * 404 Not Found page component.
 */

import styled from 'styled-components';
import { Link } from 'react-router-dom';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
`;

const ErrorCode = styled.h1`
  font-size: 120px;
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.background.tertiary};
  line-height: 1;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const Title = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize['2xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const Description = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[8]};
  max-width: 400px;
`;

const HomeLink = styled(Link)`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[6]};
  background-color: ${({ theme }) => theme.colors.accent.primary};
  color: ${({ theme }) => theme.colors.text.inverse};
  border-radius: ${({ theme }) => theme.radii.md};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  text-decoration: none;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.accent.primaryHover};
    text-decoration: none;
  }
`;

export function NotFound() {
  return (
    <Container>
      <ErrorCode>404</ErrorCode>
      <Title>Page Not Found</Title>
      <Description>
        The page you are looking for does not exist or has been moved.
      </Description>
      <HomeLink to="/dashboard">Return to Dashboard</HomeLink>
    </Container>
  );
}
