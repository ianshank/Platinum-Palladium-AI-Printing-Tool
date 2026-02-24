import { type FC } from 'react';
import { CurveEditor } from '@/components/curves/CurveEditor';

export const CurvesPage: FC = () => {
  return (
    <div className="container mx-auto px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <p className="text-muted-foreground">
          View, edit, and generate linearization curves.
        </p>
      </div>
      <CurveEditor className="bg-white" />
    </div>
  );
};
