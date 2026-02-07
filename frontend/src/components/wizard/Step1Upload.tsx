import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styled, { useTheme } from 'styled-components';
import { useStore } from '@/stores';
import { useUploadScan } from '@/api/hooks/useScan';
import { tabletConfig } from '@/config/tablet.config';
import type { ScanUploadResponse } from '@/types/models';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[8]};
`;

const UploadZone = styled.div<{ $isDragActive: boolean; $hasFile: boolean }>`
  border: 2px dashed
    ${({ theme, $isDragActive, $hasFile }) =>
        $isDragActive
            ? theme.colors.accent.primary
            : $hasFile
                ? theme.colors.semantic.success
                : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[12]};
  text-align: center;
  cursor: pointer;
  background-color: ${({ theme, $isDragActive }) =>
        $isDragActive ? theme.colors.background.tertiary : 'transparent'};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme, $hasFile }) =>
        $hasFile ? theme.colors.semantic.success : theme.colors.accent.primary};
    background-color: ${({ theme }) => theme.colors.background.tertiary};
  }
`;

const UploadText = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const UploadSubtext = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const SelectedFile = styled.div`
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.md};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const SectionTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const TabletGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
`;

const TabletOption = styled.label<{ $selected: boolean }>`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing[4]};
  border: 1px solid
    ${({ theme, $selected }) =>
        $selected ? theme.colors.accent.primary : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  background-color: ${({ theme, $selected }) =>
        $selected ? theme.colors.background.tertiary : 'transparent'};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }

  input {
    display: none;
  }
`;

const TabletLabel = styled.span`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const TabletDetails = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const ErrorMessage = styled.div`
  color: ${({ theme }) => theme.colors.semantic.error};
  background-color: rgba(239, 68, 68, 0.1);
  padding: ${({ theme }) => theme.spacing[4]};
  border-radius: ${({ theme }) => theme.radii.md};
  margin-top: ${({ theme }) => theme.spacing[4]};
`;

export function Step1Upload() {
    const theme = useTheme();

    // Store
    const startCalibration = useStore((state) => state.calibration.startCalibration);
    const setMeasurements = useStore((state) => state.calibration.setMeasurements);
    const updateMetadata = useStore((state) => state.calibration.updateMetadata);
    const nextStep = useStore((state) => state.calibration.nextStep);

    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [tabletType, setTabletType] = useState<string>('stouffer_21');
    const [error, setError] = useState<string | null>(null);

    // API
    const { mutate: uploadScan, isPending, error: uploadError } = useUploadScan();

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles.length > 0) {
                setUploadedFile(acceptedFiles[0]!);
                setError(null);
            }
        },
        []
    );

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg', '.tif', '.tiff'],
        },
        maxFiles: 1,
        multiple: false,
    });

    const handleTabletChange = (typeId: string) => {
        setTabletType(typeId);
    };

    const handleContinue = () => {
        if (!uploadedFile) {
            setError('Please upload a step tablet scan.');
            return;
        }

        uploadScan(
            { file: uploadedFile, tabletType },
            {
                onSuccess: (data: ScanUploadResponse) => {
                    // Initialize calibration
                    const stepCount = tabletConfig.stepTablets.find(t => t.id === tabletType)?.steps ?? 21;
                    const wizardTabletType = `${stepCount}-step` as '21-step' | '31-step' | '41-step' | 'custom';
                    startCalibration(wizardTabletType);

                    // Map densities to measurements
                    const measurements = data.densities.map((d: number, i: number) => ({
                        step: i + 1,
                        targetDensity: 0, // Placeholder, usually comes from reference
                        measuredDensity: d,
                    }));
                    setMeasurements(measurements);

                    // Save analysis metrics
                    updateMetadata({
                        dmin: data.dmin ?? 0,
                        dmax: data.dmax ?? 0,
                        range: data.range ?? 0,
                        num_patches: data.num_patches,
                        // Store the file name if needed for reference, though not persisted scan itself
                        originalFileName: uploadedFile.name
                    });

                    nextStep();
                },
                onError: (err: Error) => {
                    setError(
                        err.message ||
                        'Failed to process scan. Please try again.'
                    );
                },
            }
        );
    };

    return (
        <Container>
            <div>
                <SectionTitle>1. Upload Scan</SectionTitle>
                <UploadZone
                    {...getRootProps()}
                    $isDragActive={isDragActive}
                    $hasFile={!!uploadedFile}
                >
                    <input {...getInputProps()} data-testid="dropzone-input" />
                    {uploadedFile ? (
                        <div>
                            <UploadText>File Selected</UploadText>
                            <SelectedFile>
                                <span>{uploadedFile.name}</span>
                                <span style={{ fontSize: '0.8em', color: theme.colors.text.secondary }}>
                                    {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                                </span>
                            </SelectedFile>
                            <UploadSubtext style={{ marginTop: '1rem' }}>
                                Click or drag to replace
                            </UploadSubtext>
                        </div>
                    ) : (
                        <div>
                            <UploadText>
                                {isDragActive ? 'Drop file here' : 'Drag & drop your scan here'}
                            </UploadText>
                            <UploadSubtext>
                                Supports PNG, JPG, TIFF (16-bit recommended)
                            </UploadSubtext>
                        </div>
                    )}
                </UploadZone>
            </div>

            <div>
                <SectionTitle>2. Select Tablet Type</SectionTitle>
                <TabletGrid>
                    {tabletConfig.stepTablets.map((type) => (
                        <TabletOption
                            key={type.id}
                            $selected={tabletType === type.id}
                        >
                            <input
                                type="radio"
                                name="tabletType"
                                value={type.id}
                                checked={tabletType === type.id}
                                onChange={() => handleTabletChange(type.id)}
                            />
                            <TabletLabel>{type.name}</TabletLabel>
                            <TabletDetails>
                                {type.steps > 0 ? `${type.steps} Steps` : 'Custom'}
                            </TabletDetails>
                        </TabletOption>
                    ))}
                </TabletGrid>
            </div>

            {(error || uploadError) && (
                <ErrorMessage>
                    {error || (uploadError as Error)?.message}
                </ErrorMessage>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '2rem' }}>
                <button
                    onClick={handleContinue}
                    disabled={!uploadedFile || isPending}
                    style={{
                        padding: '12px 24px',
                        backgroundColor: isPending ? theme.colors.background.tertiary : theme.colors.accent.primary,
                        color: theme.colors.text.inverse,
                        border: 'none',
                        borderRadius: theme.radii.md,
                        cursor: isPending ? 'wait' : 'pointer',
                        fontWeight: 600,
                    }}
                >
                    {isPending ? 'Analyzing...' : 'Analyze & Continue'}
                </button>
            </div>
        </Container>
    );
}
