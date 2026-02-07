declare module 'jest-axe' {
    import type { AxeResults } from 'axe-core';

    export function axe(html: Element | string): Promise<AxeResults>;

    export const toHaveNoViolations: {
        toHaveNoViolations(received: AxeResults): { pass: boolean; message: () => string };
    };
}
