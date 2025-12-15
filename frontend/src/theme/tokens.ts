/**
 * Design tokens for the darkroom theme.
 * These values are synchronized with the backend ProLabTheme colors.
 */

export const colors = {
  // Core palette (matching existing Gradio theme)
  background: {
    primary: '#121212', // Main app background
    secondary: '#1E1E1E', // Card background
    tertiary: '#262626', // Input backgrounds
    hover: '#2A2A2A', // Hover states
    elevated: '#2C2C2C', // Modals, dropdowns
  },

  border: {
    default: '#333333',
    subtle: '#404040',
    focus: '#06B6D4', // Cyan accent
    error: '#EF4444',
  },

  text: {
    primary: '#E5E5E5', // Main text
    secondary: '#A3A3A3', // Muted text
    tertiary: '#737373', // Subtle text
    disabled: '#666666',
    inverse: '#121212', // Text on light backgrounds
    accent: '#06B6D4', // Accent text
  },

  // Accent colors (cyan-based for platinum/palladium aesthetic)
  accent: {
    primary: '#06B6D4', // Cyan 500
    primaryHover: '#22D3EE', // Cyan 400
    primaryActive: '#0891B2', // Cyan 600
    secondary: '#3B82F6', // Blue 500
    tertiary: '#8B5CF6', // Violet 500
  },

  // Metal colors (for chemistry visualization)
  metals: {
    platinum: '#C0C0C0', // Silver
    palladium: '#D4A574', // Gold/Bronze
    copper: '#B87333', // For printing related
  },

  // Chemistry colors
  chemistry: {
    ferricOxalate: '#FBBF24', // Amber
    ferricOxalateDark: '#D97706', // Darker amber
    na2: '#F97316', // Orange
    developer: '#10B981', // Emerald
  },

  // Semantic colors
  semantic: {
    success: '#22C55E', // Green 500
    successBg: '#052E16', // Green 950
    warning: '#F59E0B', // Amber 500
    warningBg: '#451A03', // Amber 950
    error: '#EF4444', // Red 500
    errorBg: '#450A0A', // Red 950
    info: '#3B82F6', // Blue 500
    infoBg: '#172554', // Blue 950
  },

  // Result quality colors
  results: {
    excellent: '#22C55E', // Green
    good: '#4ADE80', // Light green
    acceptable: '#FACC15', // Yellow
    poor: '#F87171', // Light red
    failed: '#EF4444', // Red
  },

  // Chart colors
  chart: {
    curve1: '#06B6D4', // Cyan
    curve2: '#8B5CF6', // Violet
    curve3: '#F59E0B', // Amber
    curve4: '#10B981', // Emerald
    curve5: '#EC4899', // Pink
    grid: '#333333',
    axis: '#525252',
    reference: '#525252',
  },
} as const;

export const spacing = {
  0: '0',
  px: '1px',
  0.5: '2px',
  1: '4px',
  1.5: '6px',
  2: '8px',
  2.5: '10px',
  3: '12px',
  3.5: '14px',
  4: '16px',
  5: '20px',
  6: '24px',
  7: '28px',
  8: '32px',
  9: '36px',
  10: '40px',
  11: '44px',
  12: '48px',
  14: '56px',
  16: '64px',
  20: '80px',
  24: '96px',
} as const;

export const typography = {
  fontFamily: {
    sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
  },
  fontSize: {
    xs: '0.75rem', // 12px
    sm: '0.875rem', // 14px
    base: '1rem', // 16px
    lg: '1.125rem', // 18px
    xl: '1.25rem', // 20px
    '2xl': '1.5rem', // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem', // 36px
    '5xl': '3rem', // 48px
  },
  fontWeight: {
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
  },
  lineHeight: {
    none: '1',
    tight: '1.25',
    snug: '1.375',
    normal: '1.5',
    relaxed: '1.625',
    loose: '2',
  },
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em',
  },
} as const;

export const radii = {
  none: '0',
  sm: '4px',
  md: '6px',
  lg: '8px',
  xl: '12px',
  '2xl': '16px',
  full: '9999px',
} as const;

export const shadows = {
  none: 'none',
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.16)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.3)',
  glow: '0 0 15px rgba(6, 182, 212, 0.3)', // Cyan glow
} as const;

export const transitions = {
  none: 'none',
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
  slower: '500ms ease',
} as const;

export const zIndex = {
  hide: -1,
  base: 0,
  dropdown: 1000,
  sticky: 1100,
  modal: 1200,
  popover: 1300,
  tooltip: 1400,
  toast: 1500,
} as const;

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
} as const;
