/**
 * The wizard's curve strategy and the backend's `curve_type` are different
 * vocabularies. Sending the strategy straight through meant the wizard's own
 * default, `monotonic`, reached `CurveType(request.curve_type)` and returned
 * HTTP 400, so a first-time user could not generate a curve at all.
 */

import { describe, expect, it } from 'vitest';

import {
  BACKEND_CURVE_TYPES,
  DEFAULT_CURVE_TYPE,
  toBackendCurveType,
} from './curves';

describe('toBackendCurveType', () => {
  it.each(['monotonic', 'cubic', 'linearization', '', undefined])(
    'maps the unsupported strategy %p to the default',
    (strategy) => {
      expect(toBackendCurveType(strategy)).toBe(DEFAULT_CURVE_TYPE);
    }
  );

  it.each([...BACKEND_CURVE_TYPES])(
    'passes the backend value %s through',
    (value) => {
      expect(toBackendCurveType(value)).toBe(value);
    }
  );

  it('only ever returns a value the backend accepts', () => {
    const results = ['monotonic', 'cubic', 'spline', 'nonsense'].map(
      toBackendCurveType
    );

    for (const result of results) {
      expect(BACKEND_CURVE_TYPES).toContain(result);
    }
  });
});
