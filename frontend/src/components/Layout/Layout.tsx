import { type FC, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  SlidersHorizontal,
  LineChart,
  FlaskConical,
  MessageSquare,
  History,
  Settings,
  Menu,
  X,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useStore } from '@/stores';
import { Button } from '@/components/ui/Button';
import { logger } from '@/lib/logger';

interface NavItem {
  path: string;
  label: string;
  icon: typeof LayoutDashboard;
  shortcut?: string;
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard, shortcut: 'Ctrl+1' },
  { path: '/calibration', label: 'Calibration', icon: SlidersHorizontal, shortcut: 'Ctrl+2' },
  { path: '/curves', label: 'Curves', icon: LineChart, shortcut: 'Ctrl+3' },
  { path: '/chemistry', label: 'Chemistry', icon: FlaskConical, shortcut: 'Ctrl+4' },
  { path: '/assistant', label: 'AI Assistant', icon: MessageSquare, shortcut: 'Ctrl+5' },
  { path: '/session', label: 'Session Log', icon: History },
  { path: '/settings', label: 'Settings', icon: Settings },
];

interface LayoutProps {
  children: ReactNode;
}

/**
 * Main application layout with sidebar navigation
 */
export const Layout: FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  const sidebarOpen = useStore((state) => state.ui.sidebarOpen);
  const toggleSidebar = useStore((state) => state.ui.toggleSidebar);
  const isProcessing = useStore((state) => state.ui.isProcessing);

  const handleNavClick = (path: string): void => {
    logger.debug('Layout: navigation', { path });
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r bg-card transition-transform duration-200 lg:static lg:translate-x-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b px-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
              <span className="text-lg font-bold text-primary-foreground">Pt</span>
            </div>
            <span className="text-lg font-semibold">Pt/Pd Tool</span>
          </Link>
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={toggleSidebar}
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 overflow-y-auto p-4">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => handleNavClick(item.path)}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                aria-current={isActive ? 'page' : undefined}
              >
                <Icon className="h-5 w-5" />
                <span className="flex-1">{item.label}</span>
                {item.shortcut && (
                  <kbd className="hidden rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground lg:inline-block">
                    {item.shortcut}
                  </kbd>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Processing indicator */}
        {isProcessing && (
          <div className="border-t p-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
              Processing...
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="border-t p-4">
          <p className="text-xs text-muted-foreground">
            Pt/Pd AI Printing Tool
          </p>
          <p className="text-xs text-muted-foreground">v0.1.0</p>
        </div>
      </aside>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm lg:hidden"
          onClick={toggleSidebar}
          aria-hidden="true"
        />
      )}

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top bar */}
        <header className="flex h-16 items-center gap-4 border-b bg-card px-4 lg:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={toggleSidebar}
            aria-label="Open sidebar"
          >
            <Menu className="h-5 w-5" />
          </Button>

          {/* Page title */}
          <h1 className="text-lg font-semibold">
            {NAV_ITEMS.find((item) => item.path === location.pathname)?.label ?? 'Dashboard'}
          </h1>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Actions */}
          {/* Add global actions here */}
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
};

Layout.displayName = 'Layout';
