/**
 * SessionLog Component Tests
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { SessionLog } from './SessionLog';

vi.mock('@/lib/logger', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

const mockSessionState = {
  records: [] as any[],
  stats: {
    totalPrints: 0,
    successRate: 0,
    averageExposure: 0,
    mostUsedPaper: null as string | null,
    recentActivity: [],
  },
  selectedRecordId: null,
  filterDateRange: { start: null, end: null },
  filterResult: 'all' as string,
  isLoading: false,
  setResultFilter: vi.fn(),
  clearFilters: vi.fn(),
  deleteRecord: vi.fn(),
  exportRecords: vi.fn().mockReturnValue([]),
  clearRecords: vi.fn(),
  addRecord: vi.fn(),
  updateRecord: vi.fn(),
  selectRecord: vi.fn(),
  setDateRange: vi.fn(),
  calculateStats: vi.fn(),
  importRecords: vi.fn(),
  resetSession: vi.fn(),
};

vi.mock('@/stores', () => ({
  useStore: (selector: (s: any) => any) =>
    selector({ session: mockSessionState }),
}));

const mockRecord = {
  id: 'rec-1',
  timestamp: '2026-02-07T10:00:00Z',
  paperType: 'Arches Platine',
  paperSize: '8x10',
  chemistry: { platinumMl: 0.5, palladiumMl: 0.5, metalRatio: 0.5 },
  exposure: { timeSeconds: 240, uvSource: 'UV LED' },
  result: 'success' as const,
  notes: 'Great print',
};

describe('SessionLog', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSessionState.records = [];
    mockSessionState.filterResult = 'all';
    mockSessionState.stats = {
      totalPrints: 0,
      successRate: 0,
      averageExposure: 0,
      mostUsedPaper: null,
      recentActivity: [],
    };
  });

  describe('Empty State', () => {
    it('renders the container', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('session-log')).toBeInTheDocument();
    });

    it('shows empty state when no records', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('empty-session')).toBeInTheDocument();
    });

    it('hides stats bar when no records', () => {
      render(<SessionLog />);
      expect(screen.queryByTestId('session-stats')).not.toBeInTheDocument();
    });

    it('hides records table when no records', () => {
      render(<SessionLog />);
      expect(screen.queryByTestId('records-table')).not.toBeInTheDocument();
    });
  });

  describe('With Records', () => {
    beforeEach(() => {
      mockSessionState.records = [mockRecord];
      mockSessionState.stats = {
        totalPrints: 1,
        successRate: 1,
        averageExposure: 240,
        mostUsedPaper: 'Arches Platine' as string | null,
        recentActivity: [],
      };
    });

    it('shows stats bar', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('session-stats')).toBeInTheDocument();
    });

    it('displays total prints stat', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('stat-total')).toHaveTextContent('1');
    });

    it('displays success rate stat', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('stat-success')).toHaveTextContent('100%');
    });

    it('displays average exposure stat', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('stat-exposure')).toHaveTextContent('240s');
    });

    it('displays most used paper stat', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('stat-paper')).toHaveTextContent(
        'Arches Platine'
      );
    });

    it('shows records table', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('records-table')).toBeInTheDocument();
    });

    it('renders record row', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('record-rec-1')).toBeInTheDocument();
    });

    it('shows paper type in row', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('record-rec-1')).toHaveTextContent(
        'Arches Platine'
      );
    });

    it('shows exposure time in row', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('record-rec-1')).toHaveTextContent('240');
    });

    it('hides empty state when records exist', () => {
      render(<SessionLog />);
      expect(screen.queryByTestId('empty-session')).not.toBeInTheDocument();
    });
  });

  describe('Actions', () => {
    beforeEach(() => {
      mockSessionState.records = [mockRecord];
      mockSessionState.stats.totalPrints = 1;
    });

    it('calls deleteRecord when delete clicked', () => {
      render(<SessionLog />);
      fireEvent.click(screen.getByTestId('delete-rec-1'));
      expect(mockSessionState.deleteRecord).toHaveBeenCalledWith('rec-1');
    });

    it('shows export button when records exist', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('export-btn')).toBeInTheDocument();
    });

    it('shows clear all button when records exist', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('clear-all-btn')).toBeInTheDocument();
    });

    it('calls clearRecords when clear all clicked', () => {
      render(<SessionLog />);
      fireEvent.click(screen.getByTestId('clear-all-btn'));
      expect(mockSessionState.clearRecords).toHaveBeenCalledTimes(1);
    });
  });

  describe('Filters', () => {
    it('renders filter buttons', () => {
      render(<SessionLog />);
      expect(screen.getByTestId('filter-all')).toBeInTheDocument();
      expect(screen.getByTestId('filter-success')).toBeInTheDocument();
      expect(screen.getByTestId('filter-partial')).toBeInTheDocument();
      expect(screen.getByTestId('filter-failure')).toBeInTheDocument();
    });

    it('calls setResultFilter on success filter click', () => {
      render(<SessionLog />);
      fireEvent.click(screen.getByTestId('filter-success'));
      expect(mockSessionState.setResultFilter).toHaveBeenCalledWith('success');
    });

    it('calls clearFilters on all filter click', () => {
      render(<SessionLog />);
      fireEvent.click(screen.getByTestId('filter-all'));
      expect(mockSessionState.clearFilters).toHaveBeenCalledTimes(1);
    });
  });

  describe('Customization', () => {
    it('applies custom className', () => {
      render(<SessionLog className="custom" />);
      expect(screen.getByTestId('session-log')).toHaveClass('custom');
    });
  });
});
