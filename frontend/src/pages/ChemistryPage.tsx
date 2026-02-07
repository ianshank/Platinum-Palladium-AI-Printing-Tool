/**
 * ChemistryPage — Route page wrapping the ChemistryCalculator component.
 */

import { type FC } from 'react';
import { ChemistryCalculator } from '@/components/chemistry/ChemistryCalculator';

export const ChemistryPage: FC = () => (
    <div className="container mx-auto py-6" data-testid="chemistry-page">
        <div className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight">Chemistry Calculator</h1>
            <p className="text-muted-foreground">
                Calculate coating recipes for your Pt/Pd prints.
            </p>
        </div>
        <div className="max-w-2xl">
            <ChemistryCalculator />
        </div>
    </div>
);

ChemistryPage.displayName = 'ChemistryPage';
