/**
 * Chemistry Calculator page component.
 * Calculates coating solution amounts for Pt/Pd printing.
 */

import { useState, useMemo } from 'react';
import styled from 'styled-components';
import { chemistryConfig } from '@/config/chemistry.config';

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

const TwoColumnLayout = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${({ theme }) => theme.spacing[6]};

  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background-color: ${({ theme }) => theme.colors.background.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing[6]};
`;

const CardTitle = styled.h2`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
  margin-bottom: ${({ theme }) => theme.spacing[4]};
`;

const FormGroup = styled.div`
  margin-bottom: ${({ theme }) => theme.spacing[5]};
`;

const Label = styled.label`
  display: block;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const Input = styled.input`
  width: 100%;
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.subtle};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
    outline: none;
  }

  &::placeholder {
    color: ${({ theme }) => theme.colors.text.disabled};
  }
`;

const Select = styled.select`
  width: 100%;
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.subtle};
  border-radius: ${({ theme }) => theme.radii.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  cursor: pointer;

  &:focus {
    border-color: ${({ theme }) => theme.colors.accent.primary};
    outline: none;
  }
`;

const SizeButtons = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const SizeButton = styled.button<{ $active: boolean }>`
  padding: ${({ theme }) => theme.spacing[2]} ${({ theme }) => theme.spacing[3]};
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  font-weight: ${({ theme }) => theme.typography.fontWeight.medium};
  background-color: ${({ theme, $active }) =>
    $active ? theme.colors.accent.primary : theme.colors.background.tertiary};
  color: ${({ theme, $active }) =>
    $active ? theme.colors.text.inverse : theme.colors.text.secondary};
  border: 1px solid ${({ theme, $active }) =>
    $active ? theme.colors.accent.primary : theme.colors.border.default};
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme, $active }) =>
      $active ? theme.colors.accent.primaryHover : theme.colors.background.hover};
  }
`;

const SliderContainer = styled.div`
  margin-top: ${({ theme }) => theme.spacing[2]};
`;

const SliderValue = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[2]};
`;

const Slider = styled.input`
  width: 100%;
  height: 6px;
  border-radius: ${({ theme }) => theme.radii.full};
  background: linear-gradient(
    to right,
    ${({ theme }) => theme.colors.metals.palladium},
    ${({ theme }) => theme.colors.metals.platinum}
  );
  appearance: none;
  cursor: pointer;

  &::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.accent.primary};
    border: 2px solid ${({ theme }) => theme.colors.background.secondary};
    cursor: pointer;
  }
`;

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${({ theme }) => theme.spacing[3]};
`;

const ResultItem = styled.div`
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.md};
`;

const ResultLabel = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.xs};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-bottom: ${({ theme }) => theme.spacing[1]};
`;

const ResultValue = styled.div`
  font-size: ${({ theme }) => theme.typography.fontSize.lg};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const ResultUnit = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  margin-left: ${({ theme }) => theme.spacing[1]};
`;

const TotalRow = styled.div`
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[4]};
  background-color: ${({ theme }) => theme.colors.accent.primary}20;
  border: 1px solid ${({ theme }) => theme.colors.accent.primary}40;
  border-radius: ${({ theme }) => theme.radii.md};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const TotalLabel = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.base};
  font-weight: ${({ theme }) => theme.typography.fontWeight.semibold};
  color: ${({ theme }) => theme.colors.text.primary};
`;

const TotalValue = styled.span`
  font-size: ${({ theme }) => theme.typography.fontSize.xl};
  font-weight: ${({ theme }) => theme.typography.fontWeight.bold};
  color: ${({ theme }) => theme.colors.accent.primary};
`;

const CostEstimate = styled.div`
  margin-top: ${({ theme }) => theme.spacing[4]};
  padding: ${({ theme }) => theme.spacing[3]};
  background-color: ${({ theme }) => theme.colors.background.tertiary};
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: ${({ theme }) => theme.typography.fontSize.sm};
  color: ${({ theme }) => theme.colors.text.secondary};
  text-align: center;
`;

export function ChemistryCalculator() {
  const [width, setWidth] = useState(8);
  const [height, setHeight] = useState(10);
  const [platinumRatio, setPlatinumRatio] = useState(0.5);
  const [absorbency, setAbsorbency] = useState<'low' | 'medium' | 'high'>('medium');
  const [coatingMethod, setCoatingMethod] = useState<'brush' | 'rod' | 'puddle_pusher'>('brush');
  const [contrastBoost, setContrastBoost] = useState(0);

  const recipe = useMemo(() => {
    const margin = chemistryConfig.coating.defaultMarginInches;
    const coatingWidth = Math.max(0.5, width - 2 * margin);
    const coatingHeight = Math.max(0.5, height - 2 * margin);
    const coatingArea = coatingWidth * coatingHeight;

    const absorbencyMultiplier = chemistryConfig.absorbencyMultipliers[absorbency];
    const methodMultiplier = chemistryConfig.coatingMethodMultipliers[coatingMethod];

    const baseDrops = coatingArea * chemistryConfig.coating.dropsPerSquareInch;
    const adjustedDrops = baseDrops * absorbencyMultiplier * methodMultiplier;

    const ferricOxalateTotal = adjustedDrops / 2;
    const metalTotal = adjustedDrops / 2;

    const foContrastDrops = ferricOxalateTotal * contrastBoost;
    const foStandardDrops = ferricOxalateTotal - foContrastDrops;

    const platinumDrops = metalTotal * platinumRatio;
    const palladiumDrops = metalTotal * (1 - platinumRatio);

    const na2Ratio = 0.25;
    const na2Drops = metalTotal * na2Ratio;

    const totalDrops = foStandardDrops + foContrastDrops + palladiumDrops + platinumDrops + na2Drops;

    const dropsPerMl = chemistryConfig.coating.dropsPerMl;
    const totalMl = totalDrops / dropsPerMl;

    const cost =
      ((foStandardDrops + foContrastDrops) / dropsPerMl) * chemistryConfig.costs.ferricOxalate +
      (palladiumDrops / dropsPerMl) * chemistryConfig.costs.palladium +
      (platinumDrops / dropsPerMl) * chemistryConfig.costs.platinum +
      (na2Drops / dropsPerMl) * chemistryConfig.costs.na2;

    return {
      ferricOxalate1: { drops: foStandardDrops, ml: foStandardDrops / dropsPerMl },
      ferricOxalate2: { drops: foContrastDrops, ml: foContrastDrops / dropsPerMl },
      palladium: { drops: palladiumDrops, ml: palladiumDrops / dropsPerMl },
      platinum: { drops: platinumDrops, ml: platinumDrops / dropsPerMl },
      na2: { drops: na2Drops, ml: na2Drops / dropsPerMl },
      total: { drops: totalDrops, ml: totalMl },
      cost,
      coatingArea,
    };
  }, [width, height, platinumRatio, absorbency, coatingMethod, contrastBoost]);

  const handleSizeSelect = (w: number, h: number) => {
    setWidth(w);
    setHeight(h);
  };

  return (
    <PageContainer>
      <PageHeader>
        <PageTitle>Chemistry Calculator</PageTitle>
        <PageSubtitle>
          Calculate coating solution amounts for your Pt/Pd prints.
        </PageSubtitle>
      </PageHeader>

      <TwoColumnLayout>
        <Card>
          <CardTitle>Print Settings</CardTitle>

          <FormGroup>
            <Label>Paper Size</Label>
            <SizeButtons>
              {chemistryConfig.paperSizes.map((size) => (
                <SizeButton
                  key={size.id}
                  $active={width === size.width && height === size.height}
                  onClick={() => handleSizeSelect(size.width, size.height)}
                >
                  {size.label}
                </SizeButton>
              ))}
            </SizeButtons>
          </FormGroup>

          <FormGroup style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div>
              <Label htmlFor="width">Width (inches)</Label>
              <Input
                id="width"
                type="number"
                value={width}
                onChange={(e) => setWidth(parseFloat(e.target.value) || 0)}
                min={1}
                max={40}
                step={0.5}
              />
            </div>
            <div>
              <Label htmlFor="height">Height (inches)</Label>
              <Input
                id="height"
                type="number"
                value={height}
                onChange={(e) => setHeight(parseFloat(e.target.value) || 0)}
                min={1}
                max={40}
                step={0.5}
              />
            </div>
          </FormGroup>

          <FormGroup>
            <Label>Metal Ratio (Platinum : Palladium)</Label>
            <SliderContainer>
              <SliderValue>
                <span>Palladium {Math.round((1 - platinumRatio) * 100)}%</span>
                <span>Platinum {Math.round(platinumRatio * 100)}%</span>
              </SliderValue>
              <Slider
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={platinumRatio}
                onChange={(e) => setPlatinumRatio(parseFloat(e.target.value))}
              />
            </SliderContainer>
          </FormGroup>

          <FormGroup>
            <Label htmlFor="absorbency">Paper Absorbency</Label>
            <Select
              id="absorbency"
              value={absorbency}
              onChange={(e) => setAbsorbency(e.target.value as 'low' | 'medium' | 'high')}
            >
              <option value="low">Low (hot press)</option>
              <option value="medium">Medium (standard)</option>
              <option value="high">High (cold press)</option>
            </Select>
          </FormGroup>

          <FormGroup>
            <Label htmlFor="coating-method">Coating Method</Label>
            <Select
              id="coating-method"
              value={coatingMethod}
              onChange={(e) => setCoatingMethod(e.target.value as 'brush' | 'rod' | 'puddle_pusher')}
            >
              <option value="brush">Hake Brush</option>
              <option value="rod">Glass Rod</option>
              <option value="puddle_pusher">Puddle Pusher</option>
            </Select>
          </FormGroup>
        </Card>

        <Card>
          <CardTitle>Recipe</CardTitle>
          <p style={{ fontSize: '14px', color: '#a3a3a3', marginBottom: '16px' }}>
            Coating area: {recipe.coatingArea.toFixed(1)} sq in
          </p>

          <ResultsGrid>
            <ResultItem>
              <ResultLabel>Ferric Oxalate #1</ResultLabel>
              <ResultValue>
                {recipe.ferricOxalate1.drops.toFixed(1)}
                <ResultUnit>drops</ResultUnit>
              </ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>FO #2 (Contrast)</ResultLabel>
              <ResultValue>
                {recipe.ferricOxalate2.drops.toFixed(1)}
                <ResultUnit>drops</ResultUnit>
              </ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Palladium</ResultLabel>
              <ResultValue>
                {recipe.palladium.drops.toFixed(1)}
                <ResultUnit>drops</ResultUnit>
              </ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Platinum</ResultLabel>
              <ResultValue>
                {recipe.platinum.drops.toFixed(1)}
                <ResultUnit>drops</ResultUnit>
              </ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Na2 (Contrast)</ResultLabel>
              <ResultValue>
                {recipe.na2.drops.toFixed(1)}
                <ResultUnit>drops</ResultUnit>
              </ResultValue>
            </ResultItem>
            <ResultItem>
              <ResultLabel>Total Volume</ResultLabel>
              <ResultValue>
                {recipe.total.ml.toFixed(2)}
                <ResultUnit>ml</ResultUnit>
              </ResultValue>
            </ResultItem>
          </ResultsGrid>

          <TotalRow>
            <TotalLabel>Total Drops</TotalLabel>
            <TotalValue>{recipe.total.drops.toFixed(1)}</TotalValue>
          </TotalRow>

          <CostEstimate>
            Estimated cost: ${recipe.cost.toFixed(2)} USD
          </CostEstimate>
        </Card>
      </TwoColumnLayout>
    </PageContainer>
  );
}
