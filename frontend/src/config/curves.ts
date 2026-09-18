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

/**
 * Curve type sent when the caller expresses no preference.
 *
 * The server validates this against its `CurveType` enum and returns HTTP 400
 * for anything else. The wizard used to send `'linearization'`, which is not a
 * member, under the key `type`, which the server does not read: the request
 * was rejected outright once the field name was corrected, and silently fell
 * back to the default before that. Values live in the generated schema; this
 * constant names the one the wizard intends.
 */
export const DEFAULT_CURVE_TYPE = 'linear';

/** Default output format for a digital negative: the widest the printer can use. */
export const DEFAULT_NEGATIVE_FORMAT = 'tiff_16bit';

/** A negative is a single-channel image; colour is an explicit opt-in. */
export const DEFAULT_NEGATIVE_COLOR_MODE = 'grayscale';

/** Download filename stem when the caller does not supply one. */
export const DEFAULT_NEGATIVE_NAME = 'negative';

/**
 * Backend `CurveType` members that `POST /api/curves/generate` accepts.
 *
 * The wizard's "curve strategy" describes an *interpolation* choice
 * (`monotonic`, `cubic`, `linear`); the backend's `curve_type` names a
 * calibration target. They are different vocabularies, and sending the
 * strategy straight through meant the wizard's own default, `monotonic`,
 * reached `CurveType(request.curve_type)` and returned HTTP 400.
 */
export const BACKEND_CURVE_TYPES = [
  'linear',
  'paper_white',
  'aesthetic',
  'custom',
  'mcts_optimized',
  'spline',
  'polynomial',
] as const;

export type BackendCurveType = (typeof BACKEND_CURVE_TYPES)[number];

/**
 * Translate a wizard curve strategy into a curve type the backend accepts.
 *
 * A strategy that is already a valid backend value passes through, so the two
 * vocabularies can converge later without touching call sites. Anything else,
 * including the interpolation names, falls back to the default: the generate
 * endpoint exposes no interpolation knob, so the strategy is a presentation
 * detail there.
 */
export function toBackendCurveType(
  strategy: string | undefined
): BackendCurveType {
  return (
    BACKEND_CURVE_TYPES.find((value) => value === strategy) ??
    DEFAULT_CURVE_TYPE
  );
}
