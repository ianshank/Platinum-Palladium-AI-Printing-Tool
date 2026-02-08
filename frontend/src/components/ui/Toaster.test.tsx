import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Toaster } from './Toaster';

// Mock Radix UI toast primitives as simple divs/buttons
vi.mock('@radix-ui/react-toast', () => ({
  Provider: ({ children }: any) => (
    <div data-testid="toast-provider">{children}</div>
  ),
  Root: ({ children }: any) => <div data-testid="toast-root">{children}</div>,
  Title: ({ children }: any) => <div data-testid="toast-title">{children}</div>,
  Description: ({ children }: any) => (
    <div data-testid="toast-description">{children}</div>
  ),
  Close: ({ children, ...props }: any) => (
    <button data-testid="toast-close" {...props}>
      {children}
    </button>
  ),
  Viewport: () => <div data-testid="toast-viewport" />,
}));

let mockToasts: any[] = [];
const mockRemoveToast = vi.fn();

vi.mock('@/stores', () => ({
  useStore: (selector: (state: any) => any) => {
    const state = {
      ui: {
        toasts: mockToasts,
        removeToast: mockRemoveToast,
      },
    };
    return selector(state);
  },
}));

vi.mock('lucide-react', () => ({
  X: () => <span data-testid="x-icon">X</span>,
}));

vi.mock('@/lib/utils', () => ({
  cn: (...args: unknown[]) => args.filter(Boolean).join(' '),
}));

describe('Toaster', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockToasts = [];
  });

  it('renders toast provider and viewport', () => {
    render(<Toaster />);

    expect(screen.getByTestId('toast-provider')).toBeInTheDocument();
    expect(screen.getByTestId('toast-viewport')).toBeInTheDocument();
  });

  it('renders nothing when no toasts', () => {
    render(<Toaster />);

    expect(screen.queryByTestId('toast-root')).not.toBeInTheDocument();
  });

  it('renders a toast with title', () => {
    mockToasts = [{ id: 'toast-1', title: 'Success', variant: 'success' }];

    render(<Toaster />);

    expect(screen.getByTestId('toast-root')).toBeInTheDocument();
    expect(screen.getByText('Success')).toBeInTheDocument();
  });

  it('renders toast with description', () => {
    mockToasts = [
      {
        id: 'toast-1',
        title: 'Upload',
        description: 'File uploaded successfully',
        variant: 'success',
      },
    ];

    render(<Toaster />);

    expect(screen.getByText('Upload')).toBeInTheDocument();
    expect(screen.getByText('File uploaded successfully')).toBeInTheDocument();
  });

  it('renders multiple toasts', () => {
    mockToasts = [
      { id: 'toast-1', title: 'Toast 1', variant: 'default' },
      { id: 'toast-2', title: 'Toast 2', variant: 'error' },
    ];

    render(<Toaster />);

    const roots = screen.getAllByTestId('toast-root');
    expect(roots).toHaveLength(2);
  });

  it('renders close button for each toast', () => {
    mockToasts = [{ id: 'toast-1', title: 'Closeable', variant: 'default' }];

    render(<Toaster />);

    expect(screen.getByTestId('toast-close')).toBeInTheDocument();
    expect(screen.getByLabelText('Close')).toBeInTheDocument();
  });
});
