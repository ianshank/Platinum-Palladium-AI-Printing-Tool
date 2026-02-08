/**
 * ChemistryCalculator Component Tests
 *
 * Comprehensive test suite covering:
 * - Form input controls (paper size, metal ratio, coating method, contrast, developer)
 * - Recipe calculation and display
 * - Clear and reset actions
 * - Accessibility (fieldsets, legends, aria-labels)
 * - Store integration
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { ChemistryCalculator } from './ChemistryCalculator';

// Mock the logger
vi.mock('@/lib/logger', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
  },
}));

// Mock the store
const mockChemistryState = {
  paperSize: { name: '8x10', widthInches: 8, heightInches: 10 },
  customSizes: [],
  metalRatio: 0.5,
  coatingMethod: 'brush' as const,
  contrastLevel: 2,
  recipe: null as ReturnType<typeof createMockRecipe> | null,
  developer: {
    type: 'potassium_oxalate' as const,
    concentration: 25,
    temperatureC: 20,
  },
  setPaperSize: vi.fn(),
  addCustomSize: vi.fn(),
  removeCustomSize: vi.fn(),
  setMetalRatio: vi.fn(),
  setCoatingMethod: vi.fn(),
  setContrastLevel: vi.fn(),
  setDeveloper: vi.fn(),
  calculateRecipe: vi.fn(),
  clearRecipe: vi.fn(),
  resetChemistry: vi.fn(),
};

function createMockRecipe() {
  return {
    totalVolume: 2.8,
    platinumMl: 0.56,
    palladiumMl: 0.56,
    ferricOxalateMl: 1.68,
    contrastAgent: { type: 'na2' as const, amount: 1 },
    developer: {
      type: 'potassium_oxalate' as const,
      concentration: 25,
      temperatureC: 20,
    },
  };
}

vi.mock('@/stores', () => ({
  useStore: (selector: (state: any) => any) =>
    selector({ chemistry: mockChemistryState }),
}));

describe('ChemistryCalculator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockChemistryState.recipe = null;
  });

  describe('Rendering', () => {
    it('renders the calculator container', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('chemistry-calculator')).toBeInTheDocument();
    });

    it('renders paper size select with current selection', () => {
      render(<ChemistryCalculator />);
      const select = screen.getByTestId(
        'paper-size-select'
      ) as HTMLSelectElement;
      expect(select.value).toBe('8x10');
    });

    it('renders all standard paper sizes as options', () => {
      render(<ChemistryCalculator />);
      const select = screen.getByTestId('paper-size-select');
      const options = within(select).getAllByRole('option');
      // 7 standard sizes
      expect(options.length).toBeGreaterThanOrEqual(7);
    });

    it('shows paper area calculation', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByText(/80\.0 sq in/)).toBeInTheDocument();
    });

    it('renders metal ratio slider', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('metal-ratio-slider')).toBeInTheDocument();
    });

    it('shows metal ratio percentage', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByText(/50% Pt/)).toBeInTheDocument();
      expect(screen.getByText(/50% Pd/)).toBeInTheDocument();
    });

    it('renders all coating method buttons', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('coating-brush')).toBeInTheDocument();
      expect(screen.getByTestId('coating-rod')).toBeInTheDocument();
      expect(screen.getByTestId('coating-puddle')).toBeInTheDocument();
    });

    it('highlights active coating method', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('coating-brush')).toHaveAttribute(
        'aria-checked',
        'true'
      );
      expect(screen.getByTestId('coating-rod')).toHaveAttribute(
        'aria-checked',
        'false'
      );
    });

    it('renders contrast level slider', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('contrast-slider')).toBeInTheDocument();
    });

    it('shows current contrast grade', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByText(/Grade 2/)).toBeInTheDocument();
    });

    it('renders developer type select', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('developer-type-select')).toBeInTheDocument();
    });

    it('renders developer temperature input', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('developer-temp-input')).toBeInTheDocument();
    });

    it('renders calculate button', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('calculate-btn')).toBeInTheDocument();
    });

    it('renders reset button', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('reset-btn')).toBeInTheDocument();
    });

    it('does not show clear button when no recipe', () => {
      render(<ChemistryCalculator />);
      expect(screen.queryByTestId('clear-btn')).not.toBeInTheDocument();
    });
  });

  describe('Interactions', () => {
    it('calls setPaperSize when paper size changes', () => {
      render(<ChemistryCalculator />);
      fireEvent.change(screen.getByTestId('paper-size-select'), {
        target: { value: '5x7' },
      });
      expect(mockChemistryState.setPaperSize).toHaveBeenCalledWith(
        expect.objectContaining({ name: '5x7' })
      );
    });

    it('calls setMetalRatio when slider changes', () => {
      render(<ChemistryCalculator />);
      fireEvent.change(screen.getByTestId('metal-ratio-slider'), {
        target: { value: '0.75' },
      });
      expect(mockChemistryState.setMetalRatio).toHaveBeenCalledWith(0.75);
    });

    it('calls setCoatingMethod when coating button clicked', () => {
      render(<ChemistryCalculator />);
      fireEvent.click(screen.getByTestId('coating-rod'));
      expect(mockChemistryState.setCoatingMethod).toHaveBeenCalledWith('rod');
    });

    it('calls setContrastLevel when contrast slider changes', () => {
      render(<ChemistryCalculator />);
      fireEvent.change(screen.getByTestId('contrast-slider'), {
        target: { value: '4' },
      });
      expect(mockChemistryState.setContrastLevel).toHaveBeenCalledWith(4);
    });

    it('calls setDeveloper when developer type changes', () => {
      render(<ChemistryCalculator />);
      fireEvent.change(screen.getByTestId('developer-type-select'), {
        target: { value: 'ammonium_citrate' },
      });
      expect(mockChemistryState.setDeveloper).toHaveBeenCalledWith({
        type: 'ammonium_citrate',
      });
    });

    it('calls setDeveloper when temperature changes', () => {
      render(<ChemistryCalculator />);
      fireEvent.change(screen.getByTestId('developer-temp-input'), {
        target: { value: '25' },
      });
      expect(mockChemistryState.setDeveloper).toHaveBeenCalledWith({
        temperatureC: 25,
      });
    });

    it('calls calculateRecipe when calculate button clicked', () => {
      render(<ChemistryCalculator />);
      fireEvent.click(screen.getByTestId('calculate-btn'));
      expect(mockChemistryState.calculateRecipe).toHaveBeenCalledTimes(1);
    });

    it('calls resetChemistry when reset button clicked', () => {
      render(<ChemistryCalculator />);
      fireEvent.click(screen.getByTestId('reset-btn'));
      expect(mockChemistryState.resetChemistry).toHaveBeenCalledTimes(1);
    });
  });

  describe('Recipe Output', () => {
    beforeEach(() => {
      mockChemistryState.recipe = createMockRecipe();
    });

    it('shows recipe output when recipe exists', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('recipe-output')).toBeInTheDocument();
    });

    it('shows clear button when recipe exists', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('clear-btn')).toBeInTheDocument();
    });

    it('calls clearRecipe when clear button clicked', () => {
      render(<ChemistryCalculator />);
      fireEvent.click(screen.getByTestId('clear-btn'));
      expect(mockChemistryState.clearRecipe).toHaveBeenCalledTimes(1);
    });

    it('displays total volume', () => {
      render(<ChemistryCalculator />);
      const totalRow = screen.getByTestId('recipe-total');
      expect(totalRow).toHaveTextContent('2.8 ml');
    });

    it('displays platinum amount', () => {
      render(<ChemistryCalculator />);
      const ptRow = screen.getByTestId('recipe-platinum');
      expect(ptRow).toHaveTextContent('0.56 ml');
    });

    it('displays palladium amount', () => {
      render(<ChemistryCalculator />);
      const pdRow = screen.getByTestId('recipe-palladium');
      expect(pdRow).toHaveTextContent('0.56 ml');
    });

    it('displays ferric oxalate amount', () => {
      render(<ChemistryCalculator />);
      const feRow = screen.getByTestId('recipe-ferric');
      expect(feRow).toHaveTextContent('1.68 ml');
    });

    it('displays contrast agent info', () => {
      render(<ChemistryCalculator />);
      const contrastRow = screen.getByTestId('recipe-contrast');
      expect(contrastRow).toHaveTextContent('NA2');
      expect(contrastRow).toHaveTextContent('1 drops per 10ml');
    });

    it('displays developer type', () => {
      render(<ChemistryCalculator />);
      const devTypeRow = screen.getByTestId('recipe-dev-type');
      expect(devTypeRow).toHaveTextContent('Potassium Oxalate');
    });

    it('displays developer temperature', () => {
      render(<ChemistryCalculator />);
      const devTempRow = screen.getByTestId('recipe-dev-temp');
      expect(devTempRow).toHaveTextContent('20°C');
    });

    it('does not show recipe output when recipe is null', () => {
      mockChemistryState.recipe = null;
      render(<ChemistryCalculator />);
      expect(screen.queryByTestId('recipe-output')).not.toBeInTheDocument();
    });
  });

  describe('Recipe without contrast agent', () => {
    it('hides contrast section when no contrast agent', () => {
      const noContrastRecipe = createMockRecipe();
      (noContrastRecipe as any).contrastAgent = undefined;
      mockChemistryState.recipe = noContrastRecipe;
      render(<ChemistryCalculator />);
      expect(screen.queryByTestId('recipe-contrast')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has fieldset with legend for paper size', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByText('Paper Size')).toBeInTheDocument();
    });

    it('has fieldset with legend for metal ratio', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByText(/Metal Ratio/)).toBeInTheDocument();
    });

    it('has aria-label on paper size select', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('paper-size-select')).toHaveAttribute(
        'aria-label',
        'Paper size'
      );
    });

    it('has aria-label on metal ratio slider', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('metal-ratio-slider')).toHaveAttribute(
        'aria-label',
        'Metal ratio'
      );
    });

    it('coating method buttons use radio role', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByTestId('coating-brush')).toHaveAttribute(
        'role',
        'radio'
      );
    });

    it('has labels for developer inputs', () => {
      render(<ChemistryCalculator />);
      expect(screen.getByLabelText(/Type/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Temperature/)).toBeInTheDocument();
    });
  });

  describe('Customization', () => {
    it('applies custom className', () => {
      render(<ChemistryCalculator className="custom-class" />);
      expect(screen.getByTestId('chemistry-calculator')).toHaveClass(
        'custom-class'
      );
    });
  });
});
