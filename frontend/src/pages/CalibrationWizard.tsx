/**
 * Calibration Wizard page component.
 * Multi-step wizard for creating calibration curves.
 */

import styled from 'styled-components';
import { useWizardStore } from '@/store';

const PageContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
`;

const PageHeader = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[8]};
`;

const PageTitle = styled.h1`
  font-size: ${({ theme }) => theme.typography.fontSize['3xl']};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const PageSubtitle = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  color: ${({ theme }) => theme.colors.text.secondary};
`;

// Stepper
const StepperContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ theme }) => theme.spacing[8]};
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const Step = styled.div<{ $active: boolean; $completed: boolean }>`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const StepNumber = styled.div<{ $active: boolean; $completed: boolean }>`
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: ${({ theme }) => theme.radii.full};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  background-color: ${({ theme, $active, $completed }) =>
    $active
      ? theme.colors.accent.primary
      : $completed
        ? theme.colors.semantic.success
        : theme.colors.background.tertiary};
  color: ${({ theme, $active, $completed }) =>
    $active || $completed ? theme.colors.text.inverse : theme.colors.text.secondary};
  transition: all ${({ theme }) => theme.transitions.fast};
`;

const StepLabel = styled.span<{ $active: boolean }>`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme, $active }) =>
    $active ? theme.typography.fontWeight.semibold : theme.typography.fontWeight.normal};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.text.primary : theme.colors.text.secondary};

  @media (max-width: 768px) {
    display: none;
  }
`;

const StepConnector = styled.div<{ $completed: boolean }>`
  flex: 1;
  height: 2px;
  margin: 0 ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme, $completed }) =>
    $completed ? theme.colors.semantic.success : theme.colors.border.default};
  transition: background-color ${({ theme }) => theme.transitions.fast};

  @media (max-width: 768px) {
    margin: 0 ${({ theme }) => theme.spacing[2]};
  }
`;

// Content area
const ContentCard = styled.div`
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[8]};
  min-height: 400px;
`;

const StepTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const StepDescription = styled.p`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[6]};
`;

const NavigationButtons = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: ${({ theme }) => theme.spacing[8]};
  padding-top: ${({ theme }) => theme.spacing[6]};
  border-top: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const Button = styled.button<{ $variant?: 'primary' | 'secondary' }>`
  padding: ${({ theme }) => theme.spacing[3]} ${({ theme }) => theme.spacing[6]};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  border-radius: ${({ theme }) => theme.radii.md};
  transition: all ${({ theme }) => theme.transitions.fast};

  ${({ theme, $variant }) =>
    $variant === 'primary'
      ? `
        background-color: ${theme.colors.accent.primary};
        color: ${theme.colors.text.inverse};

        &:hover:not(:disabled) {
          background-color: ${theme.colors.accent.primaryHover};
        }
      `
      : `
        background-color: ${theme.colors.background.tertiary};
        color: ${theme.colors.text.secondary};
        border: 1px solid ${theme.colors.border.default};

        &:hover:not(:disabled) {
          background-color: ${theme.colors.background.hover};
          color: ${theme.colors.text.primary};
        }
      `}

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const PlaceholderContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing[12]};
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;
`;

const steps = [
  { number: 1, label: 'Upload', title: 'Upload Step Tablet', description: 'Upload a scan of your step tablet for analysis.' },
  { number: 2, label: 'Analyze', title: 'Review Analysis', description: 'Review the extracted density measurements and quality metrics.' },
  { number: 3, label: 'Configure', title: 'Configure Settings', description: 'Configure linearization settings and paper presets.' },
  { number: 4, label: 'Preview', title: 'Preview Curve', description: 'Preview and fine-tune the generated calibration curve.' },
  { number: 5, label: 'Export', title: 'Export Curve', description: 'Export your curve in QTR or other formats.' },
];

export function CalibrationWizard() {
  const currentStep = useWizardStore((state) => state.currentStep);
  const maxStepReached = useWizardStore((state) => state.maxStepReached);
  const nextStep = useWizardStore((state) => state.nextStep);
  const prevStep = useWizardStore((state) => state.prevStep);
  const setStep = useWizardStore((state) => state.setStep);

  const currentStepInfo = steps[currentStep - 1];

  const canGoNext = currentStep < 5;
  const canGoPrev = currentStep > 1;

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Calibration Wizard</PageTitle>
        <PageSubtitle>
          Create a linearization curve from your step tablet scan.
        </PageSubtitle>
      </PageHeader>

      <StepperContainer>
        {steps.map((step, index) => (
          <React.Fragment key={step.number}>
            <Step
              $active={currentStep === step.number}
              $completed={step.number < currentStep}
            >
              <StepNumber
                $active={currentStep === step.number}
                $completed={step.number < currentStep}
                onClick={() => step.number <= maxStepReached && setStep(step.number)}
                style={{ cursor: step.number <= maxStepReached ? 'pointer' : 'default' }}
              >
                {step.number < currentStep ? '✓' : step.number}
              </StepNumber>
              <StepLabel $active={currentStep === step.number}>
                {step.label}
              </StepLabel>
            </Step>
            {index < steps.length - 1 && (
              <StepConnector $completed={step.number < currentStep} />
            )}
          </React.Fragment>
        ))}
      </StepperContainer>

      <ContentCard>
        <StepTitle>{currentStepInfo.title}</StepTitle>
        <StepDescription>{currentStepInfo.description}</StepDescription>

        <PlaceholderContent>
          <p>Step {currentStep} content will be rendered here.</p>
          <p style={{ marginTop: '8px', fontSize: '14px' }}>
            This is a placeholder. Full implementation includes file upload,
            density analysis, curve generation, and export functionality.
          </p>
        </PlaceholderContent>

        <NavigationButtons>
          <Button
            $variant="secondary"
            onClick={prevStep}
            disabled={!canGoPrev}
          >
            Previous
          </Button>
          <Button
            $variant="primary"
            onClick={nextStep}
            disabled={!canGoNext}
          >
            {currentStep === 5 ? 'Finish' : 'Next'}
          </Button>
        </NavigationButtons>
      </ContentCard>
    </PageContainer>
  );
}
