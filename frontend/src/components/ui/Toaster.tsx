import { type FC } from 'react';
import * as ToastPrimitive from '@radix-ui/react-toast';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useStore } from '@/stores';

/**
 * Toast notification component using Radix UI primitives
 */
export const Toaster: FC = () => {
  const toasts = useStore((state) => state.ui.toasts);
  const removeToast = useStore((state) => state.ui.removeToast);

  return (
    <ToastPrimitive.Provider swipeDirection="right">
      {toasts.map((toast) => (
        <ToastPrimitive.Root
          key={toast.id}
          className={cn(
            'group pointer-events-auto relative flex w-full items-center justify-between space-x-4 overflow-hidden rounded-md border p-6 pr-8 shadow-lg transition-all',
            toast.variant, // Enable group styling based on variant
            'data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none',
            'data-[state=open]:animate-in data-[state=open]:animate-slide-in-from-top-full',
            'data-[state=closed]:animate-fade-out-80 data-[state=closed]:animate-out data-[state=closed]:animate-slide-out-to-right-full',
            'data-[swipe=end]:animate-out',
            {
              'border-border bg-background text-foreground':
                toast.variant === 'default' || !toast.variant,
              'border-success/50 bg-success/10 text-success':
                toast.variant === 'success',
              'border-warning/50 bg-warning/10 text-warning':
                toast.variant === 'warning',
              'border-destructive/50 bg-destructive/10 text-destructive':
                toast.variant === 'error',
            }
          )}
          onOpenChange={(open) => {
            if (!open) {
              removeToast(toast.id);
            }
          }}
        >
          <div className="grid gap-1">
            <ToastPrimitive.Title className="text-sm font-semibold">
              {toast.title}
            </ToastPrimitive.Title>
            {toast.description && (
              <ToastPrimitive.Description className="text-sm opacity-90">
                {toast.description}
              </ToastPrimitive.Description>
            )}
          </div>
          <ToastPrimitive.Close
            className={cn(
              'absolute right-2 top-2 rounded-md p-1 opacity-0 transition-opacity',
              'hover:opacity-100 focus:opacity-100 focus:outline-none focus:ring-2 group-hover:opacity-100',
              'group-[.error]:text-destructive group-[.success]:text-success'
            )}
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </ToastPrimitive.Close>
        </ToastPrimitive.Root>
      ))}
      <ToastPrimitive.Viewport className="fixed bottom-0 right-0 z-[100] flex max-h-screen w-full flex-col-reverse p-4 sm:max-w-[420px]" />
    </ToastPrimitive.Provider>
  );
};

Toaster.displayName = 'Toaster';
