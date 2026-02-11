import { type FC } from 'react';
import { CalibrationWizard } from '@/components/calibration/CalibrationWizard';

export const CalibrationPage: FC = () => {
  return (
    <div className="container mx-auto px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">
          Calibration Wizard
        </h1>
        <p className="text-muted-foreground">
          Follow the steps to calibrate your printing process.
        </p>
      </div>
      <CalibrationWizard />
    </div>
  );
};
