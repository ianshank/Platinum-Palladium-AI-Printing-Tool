/**
 * Chemistry configuration.
 * Values are synchronized with backend ChemistrySettings.
 * These can be overridden by API responses for dynamic configuration.
 */

export const chemistryConfig = {
  // Base coating parameters (from backend ChemistrySettings)
  coating: {
    dropsPerSquareInch: 0.465,
    dropsPerMl: 20.0,
    defaultMarginInches: 0.5,
  },

  // Paper absorbency multipliers
  absorbencyMultipliers: {
    low: 0.8,
    medium: 1.0,
    high: 1.2,
  } as const,

  // Coating method multipliers
  coatingMethodMultipliers: {
    brush: 1.0,
    rod: 0.75,
    puddle_pusher: 0.85,
  } as const,

  // Metal mix presets
  metalPresets: [
    {
      id: 'pure_palladium',
      label: 'Pure Palladium (Warm)',
      platinumRatio: 0,
      description: 'Warm, brown-black tones. Most economical.',
    },
    {
      id: 'warm_mix',
      label: 'Warm Mix (25/75)',
      platinumRatio: 0.25,
      description: 'Warm tones with slightly deeper blacks.',
    },
    {
      id: 'classic_mix',
      label: 'Classic 50/50',
      platinumRatio: 0.5,
      description: 'Balanced tones. Traditional look.',
    },
    {
      id: 'cool_mix',
      label: 'Cool Mix (75/25)',
      platinumRatio: 0.75,
      description: 'Cooler tones with excellent Dmax.',
    },
    {
      id: 'pure_platinum',
      label: 'Pure Platinum (Cool)',
      platinumRatio: 1.0,
      description: 'Neutral/cool blacks. Maximum Dmax.',
    },
  ] as const,

  // Common paper sizes (inches)
  paperSizes: [
    { id: '4x5', label: '4×5"', width: 4, height: 5 },
    { id: '5x7', label: '5×7"', width: 5, height: 7 },
    { id: '8x10', label: '8×10"', width: 8, height: 10 },
    { id: '11x14', label: '11×14"', width: 11, height: 14 },
    { id: '16x20', label: '16×20"', width: 16, height: 20 },
    { id: '20x24', label: '20×24"', width: 20, height: 24 },
  ] as const,

  // Solution costs (USD per ml) for cost estimation
  costs: {
    ferricOxalate: 0.5,
    palladium: 2.0,
    platinum: 8.0,
    na2: 4.0,
  },

  // Contrast agents
  contrastAgents: [
    { id: 'none', label: 'None', dropsRatio: 0 },
    { id: 'na2', label: 'Na2 (Sodium Chloroplatinate)', dropsRatio: 0.25 },
    {
      id: 'potassium_chlorate',
      label: 'Potassium Chlorate (FO#2)',
      dropsRatio: 0.25,
    },
  ] as const,
} as const;

export type PaperAbsorbency = keyof typeof chemistryConfig.absorbencyMultipliers;
export type CoatingMethod = keyof typeof chemistryConfig.coatingMethodMultipliers;
export type MetalPresetId =
  (typeof chemistryConfig.metalPresets)[number]['id'];
export type PaperSizeId = (typeof chemistryConfig.paperSizes)[number]['id'];
export type ContrastAgentId = (typeof chemistryConfig.contrastAgents)[number]['id'];
