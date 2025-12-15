/**
 * API hooks exports.
 */

// Curve hooks
export {
  useCurve,
  useGenerateCurve,
  useModifyCurve,
  useSmoothCurve,
  useBlendCurves,
  useEnhanceCurve,
  useUploadQuadFile,
  useParseQuadContent,
  useExportCurve,
} from './useCurves';

// Scan hooks
export {
  useUploadScan,
  useAnalyzeDensities,
  assessScanQuality,
} from './useScan';
export type { ScanQualityAssessment } from './useScan';

// Chat hooks
export {
  useChat,
  useSuggestRecipe,
  useTroubleshoot,
  quickPrompts,
} from './useChat';
export type { ChatMessage } from './useChat';
