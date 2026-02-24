/**
 * RecommendationCard component tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RecommendationCard } from './RecommendationCard';
import type { MCTSRecommendation } from '@/types/mcts';

describe('RecommendationCard', () => {
  const mockRecommendation: MCTSRecommendation = {
    parameters: {
      exposure_time: 120,
      developer_dilution: 1.5,
      temperature: 20,
    },
    predictedQuality: 0.85,
    rationale: 'This configuration works well for Arches Platine paper',
  };

  it('should render recommendation card', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);

    expect(screen.getByTestId('recommendation-card')).toBeInTheDocument();
  });

  it('should display quality score', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);

    expect(screen.getByText('Quality: 85%')).toBeInTheDocument();
  });

  it('should display rationale', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);

    expect(
      screen.getByText('This configuration works well for Arches Platine paper')
    ).toBeInTheDocument();
  });

  it('should display parameters', () => {
    render(<RecommendationCard recommendation={mockRecommendation} />);

    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('exposure time')).toBeInTheDocument();
    expect(screen.getByText('developer dilution')).toBeInTheDocument();
    expect(screen.getByText('temperature')).toBeInTheDocument();
  });

  it('should show truncated parameters when more than 5', () => {
    const manyParams: MCTSRecommendation = {
      ...mockRecommendation,
      parameters: {
        param1: 1,
        param2: 2,
        param3: 3,
        param4: 4,
        param5: 5,
        param6: 6,
        param7: 7,
      },
    };

    render(<RecommendationCard recommendation={manyParams} />);

    expect(screen.getByText('+2 more...')).toBeInTheDocument();
  });

  it('should show high quality indicator for quality >= 0.8', () => {
    const { container } = render(
      <RecommendationCard recommendation={mockRecommendation} />
    );

    const indicator = container.querySelector('.bg-success');
    expect(indicator).toBeInTheDocument();
  });

  it('should show medium quality indicator for quality >= 0.6', () => {
    const mediumQuality: MCTSRecommendation = {
      ...mockRecommendation,
      predictedQuality: 0.7,
    };

    const { container } = render(<RecommendationCard recommendation={mediumQuality} />);

    const indicator = container.querySelector('.bg-warning');
    expect(indicator).toBeInTheDocument();
  });

  it('should call onUse when button clicked', async () => {
    const onUse = vi.fn();
    const user = userEvent.setup();

    render(<RecommendationCard recommendation={mockRecommendation} onUse={onUse} />);

    const button = screen.getByRole('button', { name: /Use These Parameters/i });
    await user.click(button);

    expect(onUse).toHaveBeenCalledWith(mockRecommendation.parameters);
  });

  it('should not error when onUse is not provided', async () => {
    const user = userEvent.setup();

    render(<RecommendationCard recommendation={mockRecommendation} />);

    const button = screen.getByRole('button', { name: /Use These Parameters/i });
    await expect(user.click(button)).resolves.not.toThrow();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <RecommendationCard recommendation={mockRecommendation} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass('custom-class');
  });
});
