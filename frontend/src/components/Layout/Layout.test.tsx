import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Layout } from './Layout';

// Mock lucide-react icons - render empty spans instead of text to avoid duplicating nav labels
vi.mock('lucide-react', () => ({
    LayoutDashboard: () => <span data-testid="icon-dashboard" />,
    SlidersHorizontal: () => <span data-testid="icon-calibration" />,
    LineChart: () => <span data-testid="icon-curves" />,
    FlaskConical: () => <span data-testid="icon-chemistry" />,
    MessageSquare: () => <span data-testid="icon-assistant" />,
    History: () => <span data-testid="icon-session" />,
    Settings: () => <span data-testid="icon-settings" />,
    Menu: () => <span data-testid="icon-menu" />,
    X: () => <span data-testid="icon-close" />,
}));

vi.mock('@/lib/utils', () => ({
    cn: (...args: unknown[]) => args.filter(Boolean).join(' '),
}));

vi.mock('@/lib/logger', () => ({
    logger: { debug: vi.fn(), info: vi.fn(), error: vi.fn(), warn: vi.fn() },
}));

vi.mock('@/components/ui/Button', () => ({
    Button: ({ children, onClick, ...props }: any) => (
        <button onClick={onClick} {...props}>{children}</button>
    ),
}));

let mockSidebarOpen = true;
let mockIsProcessing = false;
const mockToggleSidebar = vi.fn();

vi.mock('@/stores', () => ({
    useStore: (selector: (state: any) => any) => {
        const state = {
            ui: {
                sidebarOpen: mockSidebarOpen,
                toggleSidebar: mockToggleSidebar,
                isProcessing: mockIsProcessing,
            },
        };
        return selector(state);
    },
}));

function renderLayout(route = '/') {
    return render(
        <MemoryRouter initialEntries={[route]}>
            <Layout><div data-testid="child">Page Content</div></Layout>
        </MemoryRouter>
    );
}

describe('Layout', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockSidebarOpen = true;
        mockIsProcessing = false;
    });

    it('renders logo and branding', () => {
        renderLayout();

        expect(screen.getByText('Pt')).toBeInTheDocument();
        expect(screen.getByText('Pt/Pd Tool')).toBeInTheDocument();
    });

    it('renders all navigation links', () => {
        renderLayout();

        // 'Dashboard' appears both in nav and as the h1 heading
        expect(screen.getAllByText('Dashboard').length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText('Calibration')).toBeInTheDocument();
        expect(screen.getByText('Curves')).toBeInTheDocument();
        expect(screen.getByText('Chemistry')).toBeInTheDocument();
        expect(screen.getByText('AI Assistant')).toBeInTheDocument();
        expect(screen.getByText('Session Log')).toBeInTheDocument();
        expect(screen.getByText('Settings')).toBeInTheDocument();
    });

    it('renders keyboard shortcuts in nav', () => {
        renderLayout();

        expect(screen.getByText('Ctrl+1')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+2')).toBeInTheDocument();
    });

    it('renders children content', () => {
        renderLayout();

        expect(screen.getByTestId('child')).toBeInTheDocument();
        expect(screen.getByText('Page Content')).toBeInTheDocument();
    });

    it('renders page title from current route', () => {
        renderLayout('/');

        const heading = screen.getByRole('heading', { level: 1 });
        expect(heading).toHaveTextContent('Dashboard');
    });

    it('shows processing indicator when isProcessing', () => {
        mockIsProcessing = true;
        renderLayout();

        expect(screen.getByText('Processing...')).toBeInTheDocument();
    });

    it('hides processing indicator when not processing', () => {
        mockIsProcessing = false;
        renderLayout();

        expect(screen.queryByText('Processing...')).not.toBeInTheDocument();
    });

    it('renders footer text', () => {
        renderLayout();

        expect(screen.getByText('Pt/Pd AI Printing Tool')).toBeInTheDocument();
        expect(screen.getByText('v0.1.0')).toBeInTheDocument();
    });

    it('calls toggleSidebar on menu button click', () => {
        renderLayout();

        const menuBtn = screen.getByLabelText('Open sidebar');
        fireEvent.click(menuBtn);

        expect(mockToggleSidebar).toHaveBeenCalledTimes(1);
    });
});
