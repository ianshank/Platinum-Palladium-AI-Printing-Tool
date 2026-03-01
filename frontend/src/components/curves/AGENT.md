# Curves Components Directory

## Purpose
Curve visualization and editing UI. The CurveEditor is one of the most critical components — it allows interactive manipulation of calibration curves used for digital negative creation.

## Key Files
- `CurveEditor.tsx` — Interactive curve editor with Canvas API, drag-to-edit control points (`CurveEditor.test.tsx`)
- `CurveUpload.tsx` — Upload QuadTone RIP (.quad) files for parsing (`CurveUpload.test.tsx`)

## Conventions
- **Canvas API**: CurveEditor renders on HTML5 Canvas — not SVG or Plotly
- **Undo/redo**: Integrates with `useUndoRedo` hook from `@/hooks` for edit history
- **Plotly**: CurveDisplay (in pages) uses `react-plotly.js` for read-only curve visualization. CurveEditor uses Canvas for interactive editing
- **60fps target**: Curve preview updates must complete in <16ms — see CLAUDE.md "Performance Targets"

## Data Flow
- Input: `curveSlice.points` (array of `{x, y}` normalized 0-1)
- Edit: Canvas mouse events → `curveSlice.updatePoint(index, newX, newY)`
- Output: Modified curve sent to `/api/curves/modify` for backend processing

## Testing
```bash
pnpm test -- --run src/components/curves/
```
Canvas interactions require `fireEvent.mouseDown/mouseMove/mouseUp` sequences.

## Pitfalls
- Canvas `getContext('2d')` can return null — always guard
- Clean up Canvas event listeners and animation frames in `useEffect` return
- Do NOT convert between pixel coordinates and normalized curve values inline — use dedicated transform functions
- Quad file parsing happens server-side — only upload here, parse via API

## Related
- `../../hooks/useUndoRedo.ts` — Edit history management
- `../../stores/slices/curveSlice.ts` — Curve state
- `../../pages/CurvesPage.tsx` — Parent page composing curve components
- Backend: `../../../../src/ptpd_calibration/curves/` — Curve algorithms
