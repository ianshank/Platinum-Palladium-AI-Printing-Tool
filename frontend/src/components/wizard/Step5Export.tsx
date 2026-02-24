import { useState } from 'react';
import styled, { useTheme } from 'styled-components';
import { useStore } from '@/stores';
import { useExportCurve } from '@/api/hooks';
import { tabletConfig } from '@/config/tablet.config';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[8]};
`;

const SuccessCard = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: ${({ theme }) => theme.spacing[8]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.semantic.success};
  text-align: center;
`;

const SuccessIcon = styled.div`
  width: 64px;
  height: 64px;
  border-radius: ${({ theme }) => theme.radii.full};
  background-color: ${({ theme }) => theme.colors.semantic.success + '20'};
  color: ${({ theme }) => theme.colors.semantic.success};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const Title = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const Description = styled.p`
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[6]};
`;

const FormatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
`;

const FormatOption = styled.label<{ $selected: boolean }>`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme, $selected }) =>
    $selected ? theme.colors.background.secondary : 'transparent'};
  border: 1px solid
    ${({ theme, $selected }) =>
      $selected ? theme.colors.accent.primary : theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: pointer;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }

  input {
    display: none;
  }
`;

const FormatName = styled.span`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const FormatExt = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const Button = styled.button<{ $primary?: boolean; disabled?: boolean }>`
  padding: 12px 24px;
  background-color: ${({ theme, $primary }) =>
    $primary ? theme.colors.accent.primary : theme.colors.background.tertiary};
  color: ${({ theme, $primary }) =>
    $primary ? theme.colors.text.inverse : theme.colors.text.secondary};
  border: ${({ theme, $primary }) =>
    $primary ? 'none' : `1px solid ${theme.colors.border.default}`};
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  font-weight: 600;
  opacity: ${({ disabled }) => (disabled ? 0.5 : 1)};
  transition: all 0.2s;

  &:hover:not(:disabled) {
    background-color: ${({ theme, $primary }) =>
      $primary
        ? theme.colors.accent.primaryHover
        : theme.colors.background.hover};
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing[4]};
  justify-content: center;
  margin-top: ${({ theme }) => theme.spacing[8]};
`;

export function Step5Export() {
  const theme = useTheme();

  // Local state
  const [exportFormat, setExportFormat] = useState<string>(
    tabletConfig.defaults.exportFormat
  );

  // Store selectors
  const curveName = useStore(
    (state) => state.calibration.current?.name || 'Untitled Curve'
  );
  const curveId = useStore((state) => state.curve.current?.id);
  const resetCalibration = useStore(
    (state) => state.calibration.resetCalibration
  );
  const prevStep = useStore((state) => state.calibration.previousStep);
  const resetCurve = useStore((state) => state.curve.resetCurve);

  // API
  const { mutate: exportCurve, isPending } = useExportCurve();

  const handleDownload = () => {
    if (!curveId) {
      alert('No curve generated to export.');
      return;
    }

    exportCurve(
      {
        curveId,
        format: exportFormat,
      },
      {
        onSuccess: (data: Blob) => {
          // Create blob link and download
          const url = window.URL.createObjectURL(data);
          const link = document.createElement('a');
          link.href = url;
          // Find extension
          const formatConfig = tabletConfig.exportFormats.find(
            (f) => f.id === exportFormat
          );
          const ext = formatConfig?.extension || '.txt';
          link.setAttribute(
            'download',
            `${curveName.replace(/\s+/g, '_')}${ext}`
          );
          document.body.appendChild(link);
          link.click();
          link.remove();
          window.URL.revokeObjectURL(url);
        },
      }
    );
  };

  const handleFinish = () => {
    if (
      confirm('Are you sure you want to finish? This will reset the wizard.')
    ) {
      resetCalibration();
      resetCurve();
    }
  };

  return (
    <Container>
      <SuccessCard>
        <SuccessIcon>✓</SuccessIcon>
        <Title>Calibration Complete!</Title>
        <Description>
          Your curve has been generated successfully. Choose a format below to
          download your digital negative curve.
        </Description>

        <FormatsGrid>
          {tabletConfig.exportFormats.map((format) => (
            <FormatOption
              key={format.id}
              $selected={exportFormat === format.id}
            >
              <input
                type="radio"
                name="exportFormat"
                value={format.id}
                checked={exportFormat === format.id}
                onChange={() => setExportFormat(format.id)}
              />
              <FormatName>{format.label}</FormatName>
              <FormatExt>{format.extension}</FormatExt>
            </FormatOption>
          ))}
        </FormatsGrid>

        <ButtonGroup>
          <Button onClick={prevStep}>Back</Button>
          <Button $primary onClick={handleDownload} disabled={isPending}>
            {isPending ? 'Preparing Download...' : 'Download Curve'}
          </Button>
          <Button
            onClick={handleFinish}
            style={{ marginLeft: theme.spacing[4] }}
          >
            Start New Calibration
          </Button>
        </ButtonGroup>
      </SuccessCard>
    </Container>
  );
}
