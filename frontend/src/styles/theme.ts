
import { type DefaultTheme } from 'styled-components';

export const theme: DefaultTheme = {
    colors: {
        background: {
            primary: '#ffffff',
            secondary: '#f9fafb',
            tertiary: '#f3f4f6',
            hover: '#f3f4f6',
        },
        text: {
            primary: '#111827',
            secondary: '#6b7280',
            inverse: '#ffffff',
            error: '#ef4444',
        },
        border: {
            default: '#e5e7eb',
            focus: '#3b82f6',
        },
        accent: {
            primary: '#3b82f6',
            primaryHover: '#2563eb',
            secondary: '#8b5cf6',
        },
        semantic: {
            success: '#10b981',
            warning: '#f59e0b',
            error: '#ef4444',
            info: '#3b82f6',
        },
    },
    typography: {
        fontFamily: {
            sans: 'Inter, system-ui, sans-serif',
            mono: 'monospace',
        },
        fontSize: {
            xs: '0.75rem',
            sm: '0.875rem',
            base: '1rem',
            lg: '1.125rem',
            xl: '1.25rem',
            '2xl': '1.5rem',
        },
        fontWeight: {
            regular: 400,
            medium: 500,
            semibold: 600,
            bold: 700,
        },
    },
    spacing: {
        0: '0px',
        1: '0.25rem',
        2: '0.5rem',
        3: '0.75rem',
        4: '1rem',
        5: '1.25rem',
        6: '1.5rem',
        8: '2rem',
        10: '2.5rem',
        12: '3rem',
        16: '4rem',
        20: '5rem',
        24: '6rem',
        32: '8rem',
        40: '10rem',
        48: '12rem',
        56: '14rem',
        64: '16rem',
    },
    radii: {
        sm: '0.125rem',
        md: '0.375rem',
        lg: '0.5rem',
        full: '9999px',
    },
    shadows: {
        sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    },
    transitions: {
        fast: '150ms ease-in-out',
        normal: '300ms ease-in-out',
        slow: '500ms ease-in-out',
    },
    breakpoints: {
        tablet: '640px',
        desktop: '1024px',
        wide: '1280px',
    },
};
