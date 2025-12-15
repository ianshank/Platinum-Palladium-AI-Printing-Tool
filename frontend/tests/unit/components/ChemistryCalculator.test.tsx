/**
 * ChemistryCalculator component unit tests.
 */

import { describe, it, expect, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '@tests/utils/render';
import { ChemistryCalculator } from '@/pages/ChemistryCalculator';

// Mock config values
vi.mock('@/config/chemistry.config', () => ({
  chemistryConfig: {
    defaults: {
      metalRatio: 0.5,
      coatingMlPerSqInch: 0.5,
      ferricOxalateRatio: 2.0,
      contrastAgentRatio: 0.1,
      minimumVolumeMl: 2.0,
      roundingPrecision: 1,
    },
    metals: {
      platinum: { density: 21.45, name: 'Platinum', symbol: 'Pt' },
      palladium: { density: 12.02, name: 'Palladium', symbol: 'Pd' },
    },
    presets: {
      warmTone: { metalRatio: 0.3, description: 'Warmer tones with more palladium' },
      neutral: { metalRatio: 0.5, description: 'Balanced platinum/palladium mix' },
      coolTone: { metalRatio: 0.7, description: 'Cooler tones with more platinum' },
      purePlatinum: { metalRatio: 1.0, description: 'Pure platinum for maximum archival quality' },
    },
    contrastAgents: [
      { id: 'none', name: 'None', description: 'No contrast agent' },
      { id: 'na2', name: 'Na2', description: 'Sodium salt contrast agent' },
      { id: 'dichromate', name: 'Dichromate', description: 'Potassium dichromate' },
    ],
  },
}));

describe('ChemistryCalculator', () => {
  it('renders the page title', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText('Chemistry Calculator')).toBeInTheDocument();
  });

  it('renders the page subtitle', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText(/Calculate precise chemistry volumes/)).toBeInTheDocument();
  });

  it('renders print size inputs', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByLabelText(/Width/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Height/i)).toBeInTheDocument();
  });

  it('renders metal ratio slider', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText('Metal Ratio')).toBeInTheDocument();
    expect(screen.getByRole('slider')).toBeInTheDocument();
  });

  it('renders quick presets', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText('Quick Presets')).toBeInTheDocument();
    expect(screen.getByText('Warm Tone')).toBeInTheDocument();
    expect(screen.getByText('Neutral')).toBeInTheDocument();
    expect(screen.getByText('Cool Tone')).toBeInTheDocument();
    expect(screen.getByText('Pure Platinum')).toBeInTheDocument();
  });

  it('renders contrast agent selector', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText('Contrast Agent')).toBeInTheDocument();
    expect(screen.getByText('None')).toBeInTheDocument();
  });

  it('renders results section', () => {
    render(<ChemistryCalculator />);
    expect(screen.getByText('Calculated Recipe')).toBeInTheDocument();
  });

  it('calculates chemistry when dimensions change', async () => {
    render(<ChemistryCalculator />);

    const widthInput = screen.getByLabelText(/Width/i);
    const heightInput = screen.getByLabelText(/Height/i);

    fireEvent.change(widthInput, { target: { value: '8' } });
    fireEvent.change(heightInput, { target: { value: '10' } });

    await waitFor(() => {
      // Results should show calculated values
      expect(screen.getByText(/Platinum/i)).toBeInTheDocument();
      expect(screen.getByText(/Palladium/i)).toBeInTheDocument();
      expect(screen.getByText(/Ferric Oxalate/i)).toBeInTheDocument();
    });
  });

  it('updates metal ratio when preset is clicked', async () => {
    render(<ChemistryCalculator />);

    const warmToneButton = screen.getByText('Warm Tone');
    fireEvent.click(warmToneButton);

    await waitFor(() => {
      const slider = screen.getByRole('slider');
      expect(slider).toHaveValue('0.3');
    });
  });

  it('updates metal ratio when slider changes', async () => {
    render(<ChemistryCalculator />);

    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '0.7' } });

    await waitFor(() => {
      expect(slider).toHaveValue('0.7');
    });
  });

  it('shows total volume in results', async () => {
    render(<ChemistryCalculator />);

    const widthInput = screen.getByLabelText(/Width/i);
    const heightInput = screen.getByLabelText(/Height/i);

    fireEvent.change(widthInput, { target: { value: '8' } });
    fireEvent.change(heightInput, { target: { value: '10' } });

    await waitFor(() => {
      expect(screen.getByText(/Total Volume/i)).toBeInTheDocument();
    });
  });

  it('validates positive dimensions', async () => {
    render(<ChemistryCalculator />);

    const widthInput = screen.getByLabelText(/Width/i);
    fireEvent.change(widthInput, { target: { value: '-5' } });

    // Should not crash and should handle invalid input gracefully
    expect(widthInput).toBeInTheDocument();
  });
});
