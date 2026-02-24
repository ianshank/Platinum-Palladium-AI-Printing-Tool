import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ScanUpload } from './ScanUpload';
import { CurveEditor } from '@/components/curves/CurveEditor';
import { api } from '@/api/client';
import { type CalibrationRecord, ChemistryType, type CurveGenerationResponse, CurveType, type ScanUploadResponse } from '@/types/models';
import { logger } from '@/lib/logger';
import type { CurveData } from '@/types/models';
import { Activity, BarChart, CheckCircle2, Printer, Scan } from 'lucide-react';
import { cn } from '@/lib/utils';
// import { useToast } from '@/components/ui/use-toast'; // Assuming it exists

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
    const [curveResult, setCurveResult] = useState<CurveGenerationResponse | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);

    const handleNext = () => {
        setCurrentStep((prev) => Math.min(prev + 1, STEPS.length - 1));
    };

    const handleBack = () => {
        setCurrentStep((prev) => Math.max(prev - 1, 0));
    };

    const handleScanComplete = (response: ScanUploadResponse) => {
        setScanResult(response);
        setData(prev => ({ ...prev, extraction_id: response.extraction_id }));
        handleNext();
    };

    const handleGenerateValues = async () => {
        if (!scanResult?.densities?.length) {
            logger.warn('Cannot generate curve: no density measurements available');
            return;
        }

        setIsGenerating(true);
        try {
            const response = await api.curves.generate({
                measurements: scanResult.densities,
                type: 'linearization',
                name: `${data.paper_type} ${data.chemistry_type}`,
            });

            if (response.success) {
                setCurveResult(response);
                handleNext();
            }
        } catch (e) {
            logger.error('Curve generation failed', e instanceof Error ? { error: e.message } : undefined);
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
                            <label className="block text-sm font-medium mb-1">Paper Type</label>
                            <Input
                                value={data.paper_type}
                                onChange={e => setData({ ...data, paper_type: e.target.value })}
                                placeholder="e.g. Arches Platine"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">Exposure Time (s)</label>
                            <Input
                                type="number"
                                value={data.exposure_time}
                                onChange={e => setData({ ...data, exposure_time: Number(e.target.value) })}
                            />
                        </div>
                        <div className="flex justify-end">
                            <Button onClick={handleNext} disabled={!data.paper_type}>Next</Button>
                        </div>
                    </div>
                );
            case 1: // Print
                return (
                    <div className="space-y-4">
                        <h2 className="text-xl font-semibold">Print Target</h2>
                        <p>Please print the standard 21-step Stouffer wedge on your sensitized paper.</p>
                        <p>1. Coat paper with {data.chemistry_type || 'chemistry'}.</p>
                        <p>2. Expose for {data.exposure_time} seconds.</p>
                        <p>3. Develop and dry.</p>
                        <div className="flex justify-between">
                            <Button variant="outline" onClick={handleBack}>Back</Button>
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
                            <Button variant="outline" onClick={handleBack}>Back</Button>
                        </div>
                    </div>
                );
            case 3: // Analyze
                return (
                    <div className="space-y-4">
                        <h2 className="text-xl font-semibold">Analysis</h2>
                        <p>Scan received. ID: {data.extraction_id}</p>
                        {/* Visualization of densities could go here */}
                        <div className="flex justify-between">
                            <Button variant="outline" onClick={handleBack}>Back</Button>
                            <Button onClick={handleGenerateValues}>Generate Curve</Button>
                        </div>
                    </div>
                );
            case 4: // Generate
                return (
                    <div className="space-y-4">
                        <h2 className="text-xl font-semibold">Curve Generation</h2>
                        {isGenerating ? (
                            <p>Generating...</p>
                        ) : (
                            <p>Curve Generated!</p>
                        )}
                        <div className="flex justify-between">
                            <Button variant="outline" onClick={handleBack}>Back</Button>
                            <Button onClick={handleNext} disabled={!curveResult}>Next</Button>
                        </div>
                    </div>
                );
            case 5: // Finish
                return (
                    <div className="space-y-4">
                        <h2 className="text-xl font-semibold">Complete</h2>
                        <p>Your calibration is ready.</p>
                        {curveResult && (
                            <CurveEditor
                                initialCurve={{
                                    id: curveResult.curve_id,
                                    name: curveResult.name,
                                    input_values: curveResult.input_values,
                                    output_values: curveResult.output_values,
                                    created_at: new Date().toISOString(),
                                    curve_type: CurveType.LINEAR,
                                } satisfies CurveData}
                            />
                        )}
                        <div className="flex justify-start">
                            <Button variant="outline" onClick={handleBack}>Back</Button>
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6">
            <div className="mb-8 overflow-x-auto">
                <div className="flex items-center min-w-max">
                    {STEPS.map((step, index) => {
                        const Icon = step.icon;
                        const isActive = index === currentStep;
                        const isCompleted = index < currentStep;

                        return (
                            <div key={step.id} className="flex items-center">
                                <div className={cn(
                                    "flex items-center gap-2 px-4 py-2 rounded-full border transition-colors",
                                    isActive ? "border-primary bg-primary/10 text-primary" :
                                        isCompleted ? "border-green-500 bg-green-500/10 text-green-500" : "border-muted text-muted-foreground"
                                )}>
                                    <Icon className="w-4 h-4" />
                                    <span className="text-sm font-medium">{step.title}</span>
                                </div>
                                {index < STEPS.length - 1 && (
                                    <div className="w-8 h-[2px] bg-gray-200 mx-2" />
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="bg-card p-6 rounded-xl border shadow-sm">
                {renderStepContent()}
            </div>
        </div>
    );
}
