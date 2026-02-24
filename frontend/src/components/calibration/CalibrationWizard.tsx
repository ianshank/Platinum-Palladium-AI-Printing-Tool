import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ScanUpload } from './ScanUpload';
import { CurveEditor } from '@/components/curves/CurveEditor';
import { api } from '@/api/client';
import {
  type CalibrationRecord,
  ChemistryType,
  type CurveGenerationResponse,
  CurveType,
  type ScanUploadResponse,
} from '@/types/models';
import { logger } from '@/lib/logger';
import type { CurveData } from '@/types/models';
import { Activity, BarChart, CheckCircle2, Printer, Scan } from 'lucide-react';
import { cn, formatSnakeCaseToTitle } from '@/lib/utils';
import { useStore } from '@/stores';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

// Steps definition
const STEPS = [
  { id: 'setup', title: 'Setup', icon: Activity },
  { id: 'print', title: 'Print Target', icon: Printer },
  { id: 'scan', title: 'Scan Target', icon: Scan },
  { id: 'analyze', title: 'Analyze', icon: Activity },
  { id: 'generate', title: 'Generate Curve', icon: BarChart },
  { id: 'finish', title: 'Finish', icon: CheckCircle2 },
];

export function CalibrationWizard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [data, setData] = useState<Partial<CalibrationRecord>>({
    paper_type: '',
    chemistry_type: ChemistryType.PURE_PLATINUM,
    exposure_time: 0,
  });
  const [scanResult, setScanResult] = useState<ScanUploadResponse | null>(null);
  const [curveResult, setCurveResult] =
    useState<CurveGenerationResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  // ── Zustand store actions for persistence across pages ──
  const startCalibration = useStore((s) => s.calibration.startCalibration);
  const setMeasurements = useStore((s) => s.calibration.setMeasurements);
  const updateMetadata = useStore((s) => s.calibration.updateMetadata);
  const setCurveInStore = useStore((s) => s.curve.setCurve);

  const handleNext = () => {
    setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
  };

  const handleBack = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  const handleScanComplete = (response: ScanUploadResponse) => {
    setScanResult(response);
    setData((prev) => ({ ...prev, extraction_id: response.extraction_id }));

    // ── Persist to Zustand calibration store ──
    // Ensure a calibration session exists before writing measurements
    startCalibration('21-step');

    if (response.densities?.length) {
      const measurements = response.densities.map((d, i) => ({
        step: i + 1,
        targetDensity: i / (response.densities.length - 1),
        measuredDensity: d,
      }));
      setMeasurements(measurements);
      const meta: Record<string, number | string> = {
        num_patches: response.num_patches,
        originalFileName: response.extraction_id,
      };
      if (response.dmin != null) meta['dmin'] = response.dmin;
      if (response.dmax != null) meta['dmax'] = response.dmax;
      if (response.range != null) meta['range'] = response.range;
      updateMetadata(meta);
    }

    handleNext();
  };

  const handleGenerateValues = async () => {
    if (!scanResult?.densities?.length) {
      setGenerateError('No density measurements found in the scan. Please go back and re-scan.');
      logger.warn('Cannot generate curve: no density measurements available');
      return;
    }

    setIsGenerating(true);
    setGenerateError(null);
    try {
      const response = await api.curves.generate({
        measurements: scanResult.densities,
        name: `${data.paper_type || 'Calibration'} ${data.chemistry_type || ''}`.trim(),
        curve_type: 'linear',
      });

      if (response.success) {
        setCurveResult(response);

        // ── Persist to Zustand curve store ──
        const points = response.input_values.map((x, i) => ({
          x,
          y: response.output_values[i] ?? 0,
        }));
        const now = new Date().toISOString();
        setCurveInStore({
          id: response.curve_id,
          name: response.name,
          type: 'linear' as const,
          points,
          createdAt: now,
          updatedAt: now,
          metadata: { curve_type: 'calibration' },
        });

        // Set editing to false to ensure display mode
        useStore.getState().curve.setEditing(false);

        handleNext();
      } else {
        setGenerateError('Curve generation returned an unsuccessful response.');
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setGenerateError(`Curve generation failed: ${msg}`);
      logger.error(
        'Curve generation failed',
        e instanceof Error ? { error: e.message } : undefined
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0: // Setup
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Process Setup</h2>
            <div>
              <label
                htmlFor="paper-type"
                className="mb-1 block text-sm font-medium"
              >
                Paper Type
              </label>
              <Input
                id="paper-type"
                value={data.paper_type}
                onChange={(e) =>
                  setData({ ...data, paper_type: e.target.value })
                }
                placeholder="e.g. Arches Platine"
              />
            </div>
            <div>
              <label
                htmlFor="exposure-time"
                className="mb-1 block text-sm font-medium"
              >
                Exposure Time (s)
              </label>
              <Input
                id="exposure-time"
                type="number"
                value={data.exposure_time}
                onChange={(e) =>
                  setData({ ...data, exposure_time: Number(e.target.value) })
                }
              />
            </div>
            <div className="flex justify-end">
              <Button onClick={handleNext} disabled={!data.paper_type || !data.exposure_time || data.exposure_time <= 0}>
                Next
              </Button>
            </div>
          </div>
        );
      case 1: // Print
        return (
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold">Print Target</h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Follow these steps to create a calibration print for analysis.
              </p>
            </div>

            <div className="rounded-lg border bg-muted/30 p-6">
              <h3 className="mb-4 text-base font-medium">Target Information</h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    1
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Download or prepare target</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Use a standard 21-step Stouffer wedge or similar
                      calibration target. Print this on transparency film to
                      create your digital negative.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    2
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Coat your paper</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Coat {data.paper_type || 'your paper'} with{' '}
                      {data.chemistry_type
                        ? formatSnakeCaseToTitle(data.chemistry_type)
                        : 'your chosen chemistry'}
                      . Allow to dry in darkness.
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    3
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Expose</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Place the calibration target on the coated paper and
                      expose for{' '}
                      <span className="font-semibold">
                        {data.exposure_time} seconds
                      </span>
                      {data.exposure_time === 0 && (
                        <span className="text-amber-600">
                          {' '}
                          (recommended: 180-300 seconds)
                        </span>
                      )}
                      .
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                    4
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Develop and dry</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Develop the print according to your standard process.
                      Allow the print to fully dry before scanning.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-lg border-l-4 border-blue-500 bg-blue-50 p-4 dark:bg-blue-900/20">
              <h4 className="mb-1 flex items-center gap-2 text-sm font-semibold text-blue-900 dark:text-blue-300">
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Important Notes
              </h4>
              <ul className="mt-2 space-y-1 text-sm text-blue-800 dark:text-blue-200">
                <li>
                  • Maintain consistent coating, exposure, and development
                  processes
                </li>
                <li>• Record all parameters for future reference</li>
                <li>• Ensure the print is completely dry before proceeding</li>
              </ul>
            </div>

            <div className="flex justify-between">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
              <Button onClick={handleNext}>I have printed the target</Button>
            </div>
          </div>
        );
      case 2: // Scan
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Scan Target</h2>
            <div className="flex justify-center">
              <ScanUpload onUploadComplete={handleScanComplete} />
            </div>
            <div className="flex justify-start">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
            </div>
          </div>
        );
      case 3: // Analyze
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Analysis</h2>
            <p>Scan received. ID: {data.extraction_id}</p>
            {scanResult && scanResult.densities?.length > 0 && (
              <p className="text-sm text-muted-foreground">
                {scanResult.densities.length} density measurements detected (range: {Math.min(...scanResult.densities).toFixed(2)} - {Math.max(...scanResult.densities).toFixed(2)})
              </p>
            )}
            {generateError && (
              <div className="rounded-lg border-l-4 border-red-500 bg-red-50 p-4 dark:bg-red-900/20">
                <p className="text-sm text-red-800 dark:text-red-200">{generateError}</p>
              </div>
            )}
            {scanResult && scanResult.densities?.length > 0 && (
              <div className="h-64 w-full rounded-lg border bg-card p-4">
                <h3 className="mb-2 text-sm font-medium">Density Response (Step Wedge)</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={scanResult.densities.map((d, i) => ({
                      step: i + 1,
                      density: d,
                    }))}
                    margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="step"
                      label={{ value: 'Step', position: 'insideBottom', offset: -10 }}
                    />
                    <YAxis
                      label={{ value: 'Density', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip
                      formatter={(v: number) => v.toFixed(3)}
                      labelFormatter={(l) => `Step ${l}`}
                    />
                    <Line
                      type="monotone"
                      dataKey="density"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            <div className="flex justify-between">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
              <Button
                onClick={handleGenerateValues}
                disabled={isGenerating || !scanResult?.densities?.length}
              >
                {isGenerating ? 'Generating...' : 'Generate Curve'}
              </Button>
            </div>
          </div>
        );
      case 4: // Generate
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Curve Generation</h2>
            {isGenerating ? <p>Generating...</p> : <p>Curve Generated!</p>}
            <div className="flex justify-between">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
              <Button onClick={handleNext} disabled={!curveResult}>
                Next
              </Button>
            </div>
            {curveResult && (
              <div className="h-64 w-full rounded-lg border bg-card p-4">
                <h3 className="mb-2 text-sm font-medium">Generated Calibration Curve</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={curveResult.input_values.map((x, i) => ({
                      input: x,
                      output: curveResult.output_values[i],
                    }))}
                    margin={{ top: 5, right: 20, bottom: 20, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="input"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                    />
                    <YAxis domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} />
                    <Tooltip formatter={(v: number) => v.toFixed(3)} />
                    <Line
                      type="monotone"
                      dataKey="output"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="linear"
                      dataKey="input"
                      stroke="#ccc"
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        );
      case 5: // Finish
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Complete</h2>
            <p>Your calibration is ready.</p>
            {curveResult && (
              <CurveEditor
                initialCurve={
                  {
                    id: curveResult.curve_id,
                    name: curveResult.name,
                    input_values: curveResult.input_values,
                    output_values: curveResult.output_values,
                    created_at: new Date().toISOString(),
                    curve_type: CurveType.LINEAR,
                  } satisfies CurveData
                }
              />
            )}
            <div className="flex justify-start">
              <Button variant="outline" onClick={handleBack}>
                Back
              </Button>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="mx-auto max-w-4xl p-6">
      <div className="mb-8 overflow-x-auto">
        <div className="flex min-w-max items-center">
          {STEPS.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === currentStep;
            const isCompleted = index < currentStep;

            return (
              <div key={step.id} className="flex items-center">
                <div
                  className={cn(
                    'flex items-center gap-2 rounded-full border px-4 py-2 transition-colors',
                    isActive
                      ? 'border-primary bg-primary/5 text-primary'
                      : isCompleted
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-gray-200 text-gray-400'
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm font-medium">{step.title}</span>
                </div>
                {index < STEPS.length - 1 && (
                  <div className="mx-2 h-[2px] w-8 bg-gray-200" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-xl border bg-white p-6 shadow-sm">
        {renderStepContent()}
      </div>
    </div>
  );
}
