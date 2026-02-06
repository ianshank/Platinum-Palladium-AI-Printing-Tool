/**
 * Session and print logging types.
 */

/**
 * Print result quality
 */
export type PrintResult = 'excellent' | 'good' | 'acceptable' | 'poor' | 'failed';

/**
 * Print record for logging
 */
export interface PrintRecord {
  id: string;
  timestamp: string;
  imageName: string;
  negativePath?: string;

  // Chemistry used
  chemistry: {
    type: string;
    metalRatio: number;
    contrastAgent: string;
    contrastAmount: number;
    developer: string;
  };

  // Paper info
  paper: {
    type: string;
    size: string;
    absorbency: string;
  };

  // Exposure
  exposure: {
    timeSeconds: number;
    uvSource: string;
  };

  // Environment
  environment?: {
    humidity: number;
    temperatureC: number;
  };

  // Results
  result: PrintResult;
  densityMeasurements?: number[];
  curveId?: string;

  // Notes
  notes?: string;
  tags: string[];
}

/**
 * Print session (collection of prints)
 */
export interface PrintSession {
  id: string;
  name: string;
  startTime: string;
  endTime?: string;
  prints: PrintRecord[];

  // Summary stats
  totalPrints: number;
  successRate: number;
  averageDmax?: number;

  notes?: string;
}

/**
 * Session statistics
 */
export interface SessionStatistics {
  totalSessions: number;
  totalPrints: number;
  overallSuccessRate: number;

  resultBreakdown: {
    excellent: number;
    good: number;
    acceptable: number;
    poor: number;
    failed: number;
  };

  topPaperTypes: {
    paperType: string;
    count: number;
    successRate: number;
  }[];

  recentActivity: {
    date: string;
    printCount: number;
    successRate: number;
  }[];

  averageExposureTime: number;
  averageDmax: number;
}

/**
 * Dashboard metrics
 */
export interface DashboardMetrics {
  activeCurves: number;
  printsThisWeek: number;
  successRate: number;
  calibrationsCompleted: number;

  recentSessions: PrintSession[];
  pendingTasks: {
    id: string;
    type: 'calibration' | 'export' | 'analysis';
    description: string;
    priority: 'high' | 'medium' | 'low';
  }[];

  tips: {
    id: string;
    title: string;
    content: string;
    category: string;
  }[];
}

/**
 * User preferences
 */
export interface UserPreferences {
  theme: 'dark' | 'light' | 'system';
  animationsEnabled: boolean;
  defaultTabletType: string;
  defaultMetalRatio: number;
  defaultCoatingMethod: string;
  showTips: boolean;
  keyboardShortcutsEnabled: boolean;
  autoSaveInterval: number;
  units: 'imperial' | 'metric';
  language: string;
}

/**
 * Notification
 */
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  action?: {
    label: string;
    href: string;
  };
}
