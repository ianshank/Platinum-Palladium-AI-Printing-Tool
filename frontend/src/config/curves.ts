/**
 * Curve editing configuration shared by curve components and API calls.
 *
 * Keep in sync with the backend request model `CurveModifyRequest`
 * (src/ptpd_calibration/api/server.py) and the `AdjustmentType` enum
 * (src/ptpd_calibration/curves/modifier.py).
 */

import { type CurveModificationRequest } from '@/types/models';

/**
 * Payload fragment for persisting a curve through `POST /api/curves/modify`
 * without changing its shape (the editor's "Save" action).
 *
 * Decision (TST-02 item 1c, 2026-09-18): the backend has no neutral
 * adjustment. `AdjustmentType` has no "none" member and the endpoint raises
 * HTTP 400 for any unknown type, which is why `'none'` was rejected on
 * 2026-02-22. `brightness` with `amount: 0` is an exact identity server-side:
 * `CurveModifier.adjust_brightness` adds `amount * 0.3 * shift` (= 0), clips
 * to the modifier's normalised [0, 1] domain (a no-op for a normalised curve)
 * and re-pins both endpoints, so the stored curve equals the input. It is also
 * the request model's own default (`adjustment_type = "brightness"`,
 * `amount = 0.0`). Note: the clip applies to every adjustment type alike, so a
 * curve outside [0, 1] is altered by any modify call, not by this choice.
 */
export const CURVE_SAVE_NOOP_ADJUSTMENT = {
  adjustment_type: 'brightness',
  amount: 0,
} as const satisfies Pick<
  CurveModificationRequest,
  'adjustment_type' | 'amount'
>;
