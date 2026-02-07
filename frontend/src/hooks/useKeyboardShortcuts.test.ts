import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useKeyboardShortcuts, type ShortcutConfig } from './useKeyboardShortcuts';

describe('useKeyboardShortcuts', () => {
    let addEventSpy: ReturnType<typeof vi.spyOn>;
    let removeEventSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(() => {
        addEventSpy = vi.spyOn(document, 'addEventListener');
        removeEventSpy = vi.spyOn(document, 'removeEventListener');
    });

    afterEach(() => {
        addEventSpy.mockRestore();
        removeEventSpy.mockRestore();
    });

    it('registers keydown event listener on mount', () => {
        const shortcuts: ShortcutConfig[] = [
            { key: 'a', action: vi.fn(), description: 'Test' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        expect(addEventSpy).toHaveBeenCalledWith('keydown', expect.any(Function));
    });

    it('removes keydown event listener on unmount', () => {
        const shortcuts: ShortcutConfig[] = [
            { key: 'a', action: vi.fn(), description: 'Test' },
        ];

        const { unmount } = renderHook(() => useKeyboardShortcuts(shortcuts));
        unmount();

        expect(removeEventSpy).toHaveBeenCalledWith('keydown', expect.any(Function));
    });

    it('fires action on matching key press', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 's', ctrl: true, action, description: 'Save' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const event = new KeyboardEvent('keydown', {
            key: 's',
            ctrlKey: true,
            bubbles: true,
        });
        Object.defineProperty(event, 'target', { value: document.body });
        document.dispatchEvent(event);

        expect(action).toHaveBeenCalledTimes(1);
    });

    it('does not fire action on wrong key', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 's', ctrl: true, action, description: 'Save' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const event = new KeyboardEvent('keydown', {
            key: 'a',
            ctrlKey: true,
            bubbles: true,
        });
        Object.defineProperty(event, 'target', { value: document.body });
        document.dispatchEvent(event);

        expect(action).not.toHaveBeenCalled();
    });

    it('does not fire action when typing in an input', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 's', ctrl: true, action, description: 'Save' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const input = document.createElement('input');
        document.body.appendChild(input);

        const event = new KeyboardEvent('keydown', {
            key: 's',
            ctrlKey: true,
            bubbles: true,
        });
        Object.defineProperty(event, 'target', { value: input });
        document.dispatchEvent(event);

        expect(action).not.toHaveBeenCalled();
        document.body.removeChild(input);
    });

    it('does not fire action when typing in a textarea', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 's', ctrl: true, action, description: 'Save' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const textarea = document.createElement('textarea');
        document.body.appendChild(textarea);

        const event = new KeyboardEvent('keydown', {
            key: 's',
            ctrlKey: true,
            bubbles: true,
        });
        Object.defineProperty(event, 'target', { value: textarea });
        document.dispatchEvent(event);

        expect(action).not.toHaveBeenCalled();
        document.body.removeChild(textarea);
    });

    it('respects shift modifier', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 'z', ctrl: true, shift: true, action, description: 'Redo' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        // Without shift - should not fire
        const event1 = new KeyboardEvent('keydown', {
            key: 'z',
            ctrlKey: true,
            shiftKey: false,
            bubbles: true,
        });
        Object.defineProperty(event1, 'target', { value: document.body });
        document.dispatchEvent(event1);
        expect(action).not.toHaveBeenCalled();

        // With shift - should fire
        const event2 = new KeyboardEvent('keydown', {
            key: 'z',
            ctrlKey: true,
            shiftKey: true,
            bubbles: true,
        });
        Object.defineProperty(event2, 'target', { value: document.body });
        document.dispatchEvent(event2);
        expect(action).toHaveBeenCalledTimes(1);
    });

    it('respects alt modifier', () => {
        const action = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: 'p', alt: true, action, description: 'Preview' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const event = new KeyboardEvent('keydown', {
            key: 'p',
            altKey: true,
            bubbles: true,
        });
        Object.defineProperty(event, 'target', { value: document.body });
        document.dispatchEvent(event);

        expect(action).toHaveBeenCalledTimes(1);
    });

    it('prevents default on matching shortcut', () => {
        const shortcuts: ShortcutConfig[] = [
            { key: 's', ctrl: true, action: vi.fn(), description: 'Save' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const event = new KeyboardEvent('keydown', {
            key: 's',
            ctrlKey: true,
            bubbles: true,
            cancelable: true,
        });
        Object.defineProperty(event, 'target', { value: document.body });
        const preventSpy = vi.spyOn(event, 'preventDefault');
        document.dispatchEvent(event);

        expect(preventSpy).toHaveBeenCalled();
    });

    it('handles multiple shortcuts', () => {
        const action1 = vi.fn();
        const action2 = vi.fn();
        const shortcuts: ShortcutConfig[] = [
            { key: '1', ctrl: true, action: action1, description: 'Action 1' },
            { key: '2', ctrl: true, action: action2, description: 'Action 2' },
        ];

        renderHook(() => useKeyboardShortcuts(shortcuts));

        const event1 = new KeyboardEvent('keydown', { key: '1', ctrlKey: true, bubbles: true });
        Object.defineProperty(event1, 'target', { value: document.body });
        document.dispatchEvent(event1);

        const event2 = new KeyboardEvent('keydown', { key: '2', ctrlKey: true, bubbles: true });
        Object.defineProperty(event2, 'target', { value: document.body });
        document.dispatchEvent(event2);

        expect(action1).toHaveBeenCalledTimes(1);
        expect(action2).toHaveBeenCalledTimes(1);
    });
});
