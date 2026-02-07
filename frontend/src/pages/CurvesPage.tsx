import { FC } from 'react';
import { CurveEditor } from '@/components/curves/CurveEditor';

export const CurvesPage: FC = () => {
    return (
        <div className="container mx-auto py-6">
            <div className="mb-6">
                <h1 className="text-3xl font-bold tracking-tight">Curve Editor</h1>
                <p className="text-muted-foreground">View, edit, and generate linearization curves.</p>
            </div>
            <CurveEditor className="bg-white" />
        </div>
    );
};
