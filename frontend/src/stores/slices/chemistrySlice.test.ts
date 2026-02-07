import { beforeEach, describe, expect, it } from 'vitest';
import { createStore } from '@/stores';
import { STANDARD_PAPER_SIZES } from './chemistrySlice';

describe('chemistrySlice', () => {
    let store: ReturnType<typeof createStore>;

    beforeEach(() => {
        store = createStore();
    });

    describe('Initial State', () => {
        it('has correct defaults', () => {
            const state = store.getState();
            expect(state.chemistry.paperSize.name).toBe('8x10');
            expect(state.chemistry.metalRatio).toBe(0.5);
            expect(state.chemistry.coatingMethod).toBe('brush');
            expect(state.chemistry.contrastLevel).toBe(2);
            expect(state.chemistry.recipe).toBeNull();
            expect(state.chemistry.customSizes).toEqual([]);
        });
    });

    describe('Paper Size Management', () => {
        it('setPaperSize changes paper and clears recipe', () => {
            store.getState().chemistry.calculateRecipe();
            expect(store.getState().chemistry.recipe).not.toBeNull();

            store.getState().chemistry.setPaperSize(STANDARD_PAPER_SIZES[0]!);

            expect(store.getState().chemistry.paperSize.name).toBe('4x5');
            expect(store.getState().chemistry.recipe).toBeNull();
        });

        it('addCustomSize adds a custom paper size', () => {
            store.getState().chemistry.addCustomSize({
                name: '12x16',
                widthInches: 12,
                heightInches: 16,
            });

            const customs = store.getState().chemistry.customSizes;
            expect(customs).toHaveLength(1);
            expect(customs[0]?.name).toBe('12x16');
            expect(customs[0]?.custom).toBe(true);
        });

        it('removeCustomSize removes by name', () => {
            store.getState().chemistry.addCustomSize({
                name: 'TestSize',
                widthInches: 5,
                heightInches: 5,
            });
            store.getState().chemistry.removeCustomSize('TestSize');

            expect(store.getState().chemistry.customSizes).toHaveLength(0);
        });
    });

    describe('Metal Ratio', () => {
        it('setMetalRatio updates ratio and clears recipe', () => {
            store.getState().chemistry.calculateRecipe();
            store.getState().chemistry.setMetalRatio(0.8);

            expect(store.getState().chemistry.metalRatio).toBe(0.8);
            expect(store.getState().chemistry.recipe).toBeNull();
        });

        it('setMetalRatio clamps to 0-1 range', () => {
            store.getState().chemistry.setMetalRatio(1.5);
            expect(store.getState().chemistry.metalRatio).toBe(1);

            store.getState().chemistry.setMetalRatio(-0.5);
            expect(store.getState().chemistry.metalRatio).toBe(0);
        });
    });

    describe('Coating Method', () => {
        it('setCoatingMethod updates method and clears recipe', () => {
            store.getState().chemistry.calculateRecipe();
            store.getState().chemistry.setCoatingMethod('rod');

            expect(store.getState().chemistry.coatingMethod).toBe('rod');
            expect(store.getState().chemistry.recipe).toBeNull();
        });
    });

    describe('Contrast Level', () => {
        it('setContrastLevel updates level and clears recipe', () => {
            store.getState().chemistry.calculateRecipe();
            store.getState().chemistry.setContrastLevel(4);

            expect(store.getState().chemistry.contrastLevel).toBe(4);
            expect(store.getState().chemistry.recipe).toBeNull();
        });

        it('setContrastLevel clamps to 0-5 range', () => {
            store.getState().chemistry.setContrastLevel(10);
            expect(store.getState().chemistry.contrastLevel).toBe(5);

            store.getState().chemistry.setContrastLevel(-1);
            expect(store.getState().chemistry.contrastLevel).toBe(0);
        });
    });

    describe('Developer Settings', () => {
        it('setDeveloper merges partial updates', () => {
            store.getState().chemistry.setDeveloper({ temperatureC: 25 });

            const dev = store.getState().chemistry.developer;
            expect(dev.temperatureC).toBe(25);
            expect(dev.type).toBe('potassium_oxalate'); // Unchanged
        });
    });

    describe('Recipe Calculation', () => {
        it('calculateRecipe produces a valid recipe', () => {
            store.getState().chemistry.calculateRecipe();

            const recipe = store.getState().chemistry.recipe;
            expect(recipe).not.toBeNull();
            expect(recipe!.totalVolume).toBeGreaterThan(0);
            expect(recipe!.platinumMl).toBeGreaterThan(0);
            expect(recipe!.palladiumMl).toBeGreaterThan(0);
            expect(recipe!.ferricOxalateMl).toBeGreaterThan(0);
        });

        it('calculateRecipe respects metalRatio', () => {
            // All platinum
            store.getState().chemistry.setMetalRatio(1.0);
            store.getState().chemistry.calculateRecipe();
            const allPt = store.getState().chemistry.recipe;
            expect(allPt!.palladiumMl).toBe(0);
            expect(allPt!.platinumMl).toBeGreaterThan(0);

            // All palladium
            store.getState().chemistry.setMetalRatio(0);
            store.getState().chemistry.calculateRecipe();
            const allPd = store.getState().chemistry.recipe;
            expect(allPd!.platinumMl).toBe(0);
            expect(allPd!.palladiumMl).toBeGreaterThan(0);
        });

        it('calculateRecipe includes contrast agent when level > 0', () => {
            store.getState().chemistry.setContrastLevel(3);
            store.getState().chemistry.calculateRecipe();

            expect(store.getState().chemistry.recipe!.contrastAgent).toBeDefined();
            expect(store.getState().chemistry.recipe!.contrastAgent!.type).toBe('na2');
        });

        it('calculateRecipe omits contrast agent when level is 0', () => {
            store.getState().chemistry.setContrastLevel(0);
            store.getState().chemistry.calculateRecipe();

            expect(store.getState().chemistry.recipe!.contrastAgent).toBeUndefined();
        });

        it('clearRecipe sets recipe to null', () => {
            store.getState().chemistry.calculateRecipe();
            store.getState().chemistry.clearRecipe();

            expect(store.getState().chemistry.recipe).toBeNull();
        });
    });

    describe('Reset', () => {
        it('resetChemistry restores all defaults', () => {
            store.getState().chemistry.setMetalRatio(0.9);
            store.getState().chemistry.setCoatingMethod('puddle');
            store.getState().chemistry.calculateRecipe();

            store.getState().chemistry.resetChemistry();

            const state = store.getState();
            expect(state.chemistry.metalRatio).toBe(0.5);
            expect(state.chemistry.coatingMethod).toBe('brush');
            expect(state.chemistry.recipe).toBeNull();
        });
    });
});
