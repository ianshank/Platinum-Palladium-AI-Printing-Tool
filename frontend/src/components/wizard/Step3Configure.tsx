import React from 'react';
import styled, { useTheme } from 'styled-components';
import { useStore } from '@/stores';
import { tabletConfig } from '@/config/tablet.config';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[8]};
`;

const SectionContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const SectionTitle = styled.h3`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const Label = styled.label`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const Select = styled.select`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  cursor: pointer;
  outline: none;

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }
`;

const Input = styled.input`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  outline: none;

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }
`;

const TextArea = styled.textarea`
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  min-height: 100px;
  resize: vertical;
  outline: none;

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
  }
`;

const OptionCard = styled.label<{ $selected: boolean }>`
  display: flex;
  flex-direction: column;
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme, $selected }) =>
    $selected ? theme.colors.background.tertiary : 'transparent'};
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

const OptionTitle = styled.span`
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const OptionDescription = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${({ theme }) => theme.spacing[4]};
`;

export function Step3Configure() {
  const theme = useTheme();

  // Store
  const currentCalibration = useStore((state) => state.calibration.current);
  const saveCalibration = useStore((state) => state.calibration.saveCalibration);
  const updateCalibrationMetadata = useStore((state) => state.calibration.updateMetadata);

  const linearizationMode = currentCalibration?.metadata?.linearizationMode ?? 'linear';
  const targetResponse = currentCalibration?.metadata?.targetResponse ?? tabletConfig.defaults.exportFormat;
  const curveStrategy = currentCalibration?.metadata?.curveStrategy ?? 'monotonic';
  const curveName = currentCalibration?.name || '';
  const notes = currentCalibration?.notes || '';

  const nextStep = useStore((state) => state.calibration.nextStep);
  const prevStep = useStore((state) => state.calibration.previousStep);

  const setConfiguration = (config: { curveName?: string; notes?: string; linearizationMode?: string; targetResponse?: string; curveStrategy?: string }) => {
    if (config.curveName !== undefined || config.notes !== undefined) {
      saveCalibration(config.curveName ?? curveName, config.notes ?? notes);
    }
    if (config.linearizationMode || config.targetResponse || config.curveStrategy) {
      updateCalibrationMetadata?.({
        linearizationMode: config.linearizationMode ?? linearizationMode,
        targetResponse: config.targetResponse ?? targetResponse,
        curveStrategy: config.curveStrategy ?? curveStrategy
      });
    }
  };

  // We could implement preset loading here
  // For now, let's stick to the core curve config

  return (
    <Container>
      <SectionContainer>
        <SectionTitle>Curve Name & Notes</SectionTitle>
        <Grid>
          <FormGroup>
            <Label>Calibration Name</Label>
            <Input
              type="text"
              value={curveName}
              onChange={(e) => setConfiguration({ curveName: e.target.value })}
              placeholder="e.g., Arches Platine Pd 12m"
            />
          </FormGroup>
          <FormGroup>
            <Label>Notes (Optional)</Label>
            <TextArea
              value={notes}
              onChange={(e) => setConfiguration({ notes: e.target.value })}
              placeholder="Record exposure time, chemistry details, humidity, etc."
              style={{ minHeight: '46px' }}
            />
          </FormGroup>
        </Grid>
      </SectionContainer>

      <SectionContainer>
        <SectionTitle>Linearization Method</SectionTitle>
        <Grid>
          {tabletConfig.linearizationMethods.slice(0, 3).map((method) => (
            <OptionCard
              key={method.id}
              $selected={curveStrategy === method.id}
            >
              <input
                type="radio"
                name="curveStrategy"
                value={method.id}
                checked={curveStrategy === method.id}
                onChange={() => setConfiguration({ curveStrategy: method.id })}
              />
              <OptionTitle>{method.label}</OptionTitle>
              <OptionDescription>{method.description}</OptionDescription>
            </OptionCard>
          ))}
        </Grid>
      </SectionContainer>

      <SectionContainer>
        <SectionTitle>Target Response</SectionTitle>
        <Grid>
          {tabletConfig.targetResponses.slice(0, 3).map((target) => (
            <OptionCard
              key={target.id}
              $selected={targetResponse === target.id}
            >
              <input
                type="radio"
                name="targetResponse"
                value={target.id}
                checked={targetResponse === target.id}
                onChange={() => setConfiguration({ targetResponse: target.id })}
              />
              <OptionTitle>{target.label}</OptionTitle>
              <OptionDescription>{target.description}</OptionDescription>
            </OptionCard>
          ))}
        </Grid>
      </SectionContainer>

      {/* Navigation */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2rem' }}>
        <button
          onClick={prevStep}
          style={{
            padding: '12px 24px',
            backgroundColor: theme.colors.background.tertiary,
            color: theme.colors.text.secondary,
            border: `1px solid ${theme.colors.border.default}`,
            borderRadius: theme.radii.md,
            cursor: 'pointer',
          }}
        >
          Back
        </button>
        <button
          onClick={nextStep}
          style={{
            padding: '12px 24px',
            backgroundColor: theme.colors.accent.primary,
            color: theme.colors.text.inverse,
            border: 'none',
            borderRadius: theme.radii.md,
            cursor: 'pointer',
            fontWeight: 600,
          }}
        >
          Generate Curve
        </button>
      </div>
    </Container>
  );
}
