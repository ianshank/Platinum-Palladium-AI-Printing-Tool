import { useStore } from '@/stores';
import { tabletConfig } from '@/config/tablet.config';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';

export function Step3Configure() {
  // Store
  const currentCalibration = useStore((state) => state.calibration.current);
  const saveCalibration = useStore(
    (state) => state.calibration.saveCalibration
  );
  const updateCalibrationMetadata = useStore(
    (state) => state.calibration.updateMetadata
  );

  const linearizationMode =
    currentCalibration?.metadata?.linearizationMode ?? 'linear';
  const targetResponse =
    currentCalibration?.metadata?.targetResponse ??
    tabletConfig.defaults.exportFormat;
  const curveStrategy =
    currentCalibration?.metadata?.curveStrategy ?? 'monotonic';
  const curveName = currentCalibration?.name || '';
  const notes = currentCalibration?.notes || '';

  const nextStep = useStore((state) => state.calibration.nextStep);
  const prevStep = useStore((state) => state.calibration.previousStep);

  const setConfiguration = (config: {
    curveName?: string;
    notes?: string;
    linearizationMode?: string;
    targetResponse?: string;
    curveStrategy?: string;
  }) => {
    if (config.curveName !== undefined || config.notes !== undefined) {
      saveCalibration(config.curveName ?? curveName, config.notes ?? notes);
    }
    if (
      config.linearizationMode ||
      config.targetResponse ||
      config.curveStrategy
    ) {
      updateCalibrationMetadata?.({
        linearizationMode: config.linearizationMode ?? linearizationMode,
        targetResponse: config.targetResponse ?? targetResponse,
        curveStrategy: config.curveStrategy ?? curveStrategy,
      });
    }
  };

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Curve Name & Notes</h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="flex flex-col gap-2">
            <label htmlFor="curve-name" className="text-sm font-medium text-foreground">
              Calibration Name
            </label>
            <Input
              id="curve-name"
              type="text"
              value={curveName}
              onChange={(e) => setConfiguration({ curveName: e.target.value })}
              placeholder="e.g., Arches Platine Pd 12m"
              className="bg-muted/50"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="curve-notes" className="text-sm font-medium text-foreground">
              Notes (Optional)
            </label>
            <textarea
              id="curve-notes"
              value={notes}
              onChange={(e) => setConfiguration({ notes: e.target.value })}
              placeholder="Record exposure time, chemistry details, humidity, etc."
              className="min-h-[46px] w-full rounded-md border border-input bg-muted/50 px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Linearization Method</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {tabletConfig.linearizationMethods.slice(0, 3).map((method) => (
            <label
              key={method.id}
              className={cn(
                "flex flex-col gap-1 rounded-md border-2 p-4 cursor-pointer transition-all hover:border-primary/50",
                curveStrategy === method.id
                  ? "border-primary bg-primary/5"
                  : "border-muted bg-transparent"
              )}
            >
              <input
                type="radio"
                name="curveStrategy"
                className="hidden"
                value={method.id}
                checked={curveStrategy === method.id}
                onChange={() => setConfiguration({ curveStrategy: method.id })}
              />
              <span className="font-medium text-foreground">{method.label}</span>
              <span className="text-xs text-muted-foreground">{method.description}</span>
            </label>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Target Response</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {tabletConfig.targetResponses.slice(0, 3).map((target) => (
            <label
              key={target.id}
              className={cn(
                "flex flex-col gap-1 rounded-md border-2 p-4 cursor-pointer transition-all hover:border-primary/50",
                targetResponse === target.id
                  ? "border-primary bg-primary/5"
                  : "border-muted bg-transparent"
              )}
            >
              <input
                type="radio"
                name="targetResponse"
                className="hidden"
                value={target.id}
                checked={targetResponse === target.id}
                onChange={() => setConfiguration({ targetResponse: target.id })}
              />
              <span className="font-medium text-foreground">{target.label}</span>
              <span className="text-xs text-muted-foreground">{target.description}</span>
            </label>
          ))}
        </div>
      </section>

      {/* Navigation */}
      <div className="mt-8 flex justify-between">
        <Button
          variant="outline"
          onClick={prevStep}
          className="px-6"
        >
          Back
        </Button>
        <Button
          onClick={nextStep}
          className="px-6 font-semibold"
        >
          Generate Curve
        </Button>
      </div>
    </div>
  );
}
