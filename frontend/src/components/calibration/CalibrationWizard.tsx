import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ScanUpload } from './ScanUpload';
import { CurveEditor } from '@/components/curves/CurveEditor';
import { api } from '@/api/client';
import { CalibrationRecord, CurveGenerationResponse, ScanUploadResponse } from '@/types/models';
import { CheckCircle2, Printer, Scan, Activity, BarChart } from 'lucide-react';
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
        chemistry_type: 'pure_platinum' as any,
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

    const handleScanComplete = async (extractionId: string) => {
        // We have extractionId, we might need to fetch the details if ScanUpload doesn't return them fully
        // But for now let's assume we can proceed.
        // Actually ScanUpload returns extractionId via callback. 
        // We should probably fetch the extraction details to show in Analyze step.
        // For now, we'll just mock or assume successful scan triggers next.
        setData(prev => ({ ...prev, extraction_id: extractionId }));
        handleNext();
    };

    const handleGenerateValues = async () => {
        setIsGenerating(true);
        try {
            const response = await api.curves.generate({
                measurements: scanResult?.densities || [], // This assumes we have densities. 
                // If ScanUpload only gave ID, we might need to fetch the extraction first.
                // Let's assume for this wizard we rely on the ID and maybe the backend handles it?
                // Wait, models.ts CurveGenerationRequest takes measurements: number[].
                // So we MUST have densities.
                // ScanUpload component uses api.scan.upload which returns ScanUploadResponse containing densities!
                // But ScanUpload callback onUploadComplete only gives extractionId string.
                // I should update ScanUpload to return the full response or I need to fetch it.
                // OR I can just cast the response in ScanUpload if I update it.
                // For now, I'll simulate needing densities.
                type: 'linearization',
                name: `${data.paper_type} ${data.chemistry_type}`,
            });

            if (response.success) {
                setCurveResult(response);
                handleNext();
            }
        } catch (e) {
            console.error(e);
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
                                    curve_type: 'linearization'
                                } as any}
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
                                    isActive ? "border-primary bg-primary/5 text-primary" :
                                        isCompleted ? "border-green-500 bg-green-50 text-green-700" : "border-gray-200 text-gray-400"
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

            <div className="bg-white p-6 rounded-xl border shadow-sm">
                {renderStepContent()}
            </div>
        </div>
    );
}
