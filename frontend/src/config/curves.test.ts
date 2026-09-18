import { describe, expect, it } from 'vitest';
import { CURVE_SAVE_NOOP_ADJUSTMENT } from './curves';

/**
 * Adjustment types dispatched by `POST /api/curves/modify`
 * (src/ptpd_calibration/api/server.py). Anything else is rejected with 400,
 * so a "save" must use one of these with a neutral amount.
 */
const BACKEND_ADJUSTMENT_TYPES = [
  'brightness',
  'contrast',
  'gamma',
  'levels',
  'highlights',
  'shadows',
  'midtones',
] as const;

describe('CURVE_SAVE_NOOP_ADJUSTMENT', () => {
  it('uses an adjustment type the backend accepts (never "none")', () => {
    expect(BACKEND_ADJUSTMENT_TYPES).toContain(
      CURVE_SAVE_NOOP_ADJUSTMENT.adjustment_type
    );
    expect(CURVE_SAVE_NOOP_ADJUSTMENT.adjustment_type).not.toBe('none');
  });

  it('is a no-op: brightness with amount 0 leaves the curve unchanged', () => {
    expect(CURVE_SAVE_NOOP_ADJUSTMENT.adjustment_type).toBe('brightness');
    expect(CURVE_SAVE_NOOP_ADJUSTMENT.amount).toBe(0);
  });
});
