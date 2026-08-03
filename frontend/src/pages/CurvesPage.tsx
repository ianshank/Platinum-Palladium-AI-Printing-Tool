import { useCallback, useState } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { CurveUpload } from '@/components/curves/CurveUpload';
import { CurveEditor } from '@/components/curves/CurveEditor';
import { type ExportFormat, ExportPanel } from '@/components/export/ExportPanel';
import { useExportCurve } from '@/api/hooks';
import type { CurveData, QuadCurveValues } from '@/types/models';
import { CurveType } from '@/types/models';

type CurvesTab = 'upload' | 'edit' | 'export';

const EXPORT_FORMATS: ExportFormat[] = [
  { id: 'qtr', label: 'QuadTone RIP', extension: '.txt', description: 'QTR .quad format' },
  {
    id: 'piezography',
    label: 'Piezography',
    extension: '.ppt',
    description: 'Piezography profile',
  },
  { id: 'csv', label: 'CSV', extension: '.csv', description: 'Comma-separated values' },
  { id: 'json', label: 'JSON', extension: '.json', description: 'JSON data export' },
];

export const CurvesPage = () => {
  const [activeTab, setActiveTab] = useState<CurvesTab>('upload');
  const [selectedCurveId, setSelectedCurveId] = useState<string | null>(null);
  const [selectedCurveName, setSelectedCurveName] = useState<string>('curve');
  const [selectedCurveData, setSelectedCurveData] = useState<QuadCurveValues | null>(null);

  const { mutateAsync: exportCurve } = useExportCurve();

  const handleLoadCurve = useCallback(
    (curveData: QuadCurveValues, curveId: string, profileName: string): void => {
      setSelectedCurveData(curveData);
      setSelectedCurveId(curveId);
      setSelectedCurveName(profileName);
      setActiveTab('edit');
    },
    []
  );

  const handleCurveSaved = useCallback((curve: CurveData): void => {
    setSelectedCurveId(String(curve.id));
    setSelectedCurveName(curve.name);
  }, []);

  const handleExport = useCallback(
    async (formatId: string): Promise<Blob> => {
      if (!selectedCurveId) throw new Error('No curve loaded');
      return exportCurve({ curveId: selectedCurveId, format: formatId });
    },
    [selectedCurveId, exportCurve]
  );

  const initialCurve: CurveData | undefined =
    selectedCurveData && selectedCurveId
      ? {
          id: selectedCurveId,
          name: selectedCurveName,
          created_at: new Date().toISOString(),
          curve_type: CurveType.CUSTOM,
          input_values: selectedCurveData.input_values,
          output_values: selectedCurveData.output_values,
        }
      : undefined;

  return (
    <div className="container mx-auto px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Curve Editor</h1>
        <p className="text-muted-foreground">Upload, edit, and export linearization curves.</p>
      </div>

      <Tabs.Root value={activeTab} onValueChange={(v) => setActiveTab(v as CurvesTab)}>
        <Tabs.List className="mb-6 flex gap-1 rounded-lg border bg-muted p-1">
          <Tabs.Trigger
            value="upload"
            className="flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-background data-[state=active]:shadow-sm"
          >
            Upload .quad
          </Tabs.Trigger>
          <Tabs.Trigger
            value="edit"
            disabled={!selectedCurveId}
            className="flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-background data-[state=active]:shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
          >
            Edit Curve
          </Tabs.Trigger>
          <Tabs.Trigger
            value="export"
            disabled={!selectedCurveId}
            className="flex-1 rounded-md px-4 py-2 text-sm font-medium transition-all data-[state=active]:bg-background data-[state=active]:shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
          >
            Export
          </Tabs.Trigger>
        </Tabs.List>

        <Tabs.Content value="upload">
          <CurveUpload onLoadCurve={handleLoadCurve} />
        </Tabs.Content>

        <Tabs.Content value="edit">
          <CurveEditor
            {...(initialCurve ? { initialCurve } : {})}
            onSave={handleCurveSaved}
            className="bg-white"
          />
        </Tabs.Content>

        <Tabs.Content value="export">
          <ExportPanel
            formats={EXPORT_FORMATS}
            defaultFormat="qtr"
            onExport={handleExport}
            fileName={selectedCurveName}
            title="Export Curve"
            description="Download your calibration curve in the format required by your RIP software."
            disabled={!selectedCurveId}
          />
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};
