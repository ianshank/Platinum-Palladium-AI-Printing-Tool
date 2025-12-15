/**
 * Theme exports.
 */

export { darkroomTheme } from './darkroomTheme';
export type { Theme, ThemeColors, ThemeSpacing } from './darkroomTheme';

export {
  colors,
  spacing,
  typography,
  radii,
  shadows,
  transitions,
  zIndex,
  breakpoints,
} from './tokens';

export { GlobalStyles } from './GlobalStyles';

/**
 * Helper function to access theme values in styled-components
 */
export const themeGet = <T>(path: string, fallback?: T) => {
  return (props: { theme: Record<string, unknown> }): T => {
    const keys = path.split('.');
    let value: unknown = props.theme;

    for (const key of keys) {
      if (value && typeof value === 'object' && key in value) {
        value = (value as Record<string, unknown>)[key];
      } else {
        return fallback as T;
      }
    }

    return value as T;
  };
};

/**
 * Media query helpers
 */
export const media = {
  sm: '@media (min-width: 640px)',
  md: '@media (min-width: 768px)',
  lg: '@media (min-width: 1024px)',
  xl: '@media (min-width: 1280px)',
  '2xl': '@media (min-width: 1536px)',
} as const;
