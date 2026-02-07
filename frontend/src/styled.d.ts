import 'styled-components';

declare module 'styled-components' {
    export interface DefaultTheme {
        colors: {
            background: {
                primary: string;
                secondary: string;
                tertiary: string;
                hover: string;
            };
            text: {
                primary: string;
                secondary: string;
                inverse: string;
                error: string;
            };
            border: {
                default: string;
                focus: string;
            };
            accent: {
                primary: string;
                primaryHover: string;
                secondary: string;
            };
            semantic: {
                success: string;
                warning: string;
                error: string;
                info: string;
            };
        };
        typography: {
            fontFamily: {
                sans: string;
                mono: string;
            };
            fontSize: {
                xs: string;
                sm: string;
                base: string;
                lg: string;
                xl: string;
                '2xl': string;
            };
            fontWeight: {
                regular: number;
                medium: number;
                semibold: number;
                bold: number;
            };
        };
        spacing: {
            0: string;
            1: string;
            2: string;
            3: string;
            4: string;
            5: string;
            6: string;
            8: string;
            10: string;
            12: string;
            16: string;
            20: string;
            24: string;
            32: string;
            40: string;
            48: string;
            56: string;
            64: string;
        };
        radii: {
            sm: string;
            md: string;
            lg: string;
            full: string;
        };
        shadows: {
            sm: string;
            md: string;
            lg: string;
        };
        transitions: {
            fast: string;
            normal: string;
            slow: string;
        };
        breakpoints: {
            tablet: string;
            desktop: string;
            wide: string;
        };
    }
}
