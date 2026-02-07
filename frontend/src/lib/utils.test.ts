import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
    cn,
    debounce,
    throttle,
    formatNumber,
    formatPercent,
    formatDate,
    clamp,
    lerp,
    mapRange,
    generateId,
    deepClone,
    isEmpty,
    sleep,
    retry,
    downloadFile,
    readFileAsText,
    readFileAsDataURL,
    copyToClipboard,
    createEventEmitter,
} from './utils';

describe('utils', () => {
    describe('cn', () => {
        it('merges class names', () => {
            expect(cn('foo', 'bar')).toBe('foo bar');
        });

        it('handles conditional classes', () => {
            expect(cn('foo', false && 'bar', 'baz')).toBe('foo baz');
        });
    });

    describe('debounce', () => {
        beforeEach(() => { vi.useFakeTimers(); });
        afterEach(() => { vi.useRealTimers(); });

        it('delays function execution', () => {
            const fn = vi.fn();
            const debounced = debounce(fn, 100);
            debounced();
            expect(fn).not.toHaveBeenCalled();
            vi.advanceTimersByTime(100);
            expect(fn).toHaveBeenCalledTimes(1);
        });

        it('resets timer on repeated calls', () => {
            const fn = vi.fn();
            const debounced = debounce(fn, 100);
            debounced();
            vi.advanceTimersByTime(50);
            debounced();
            vi.advanceTimersByTime(50);
            expect(fn).not.toHaveBeenCalled();
            vi.advanceTimersByTime(50);
            expect(fn).toHaveBeenCalledTimes(1);
        });
    });

    describe('throttle', () => {
        beforeEach(() => { vi.useFakeTimers(); });
        afterEach(() => { vi.useRealTimers(); });

        it('calls immediately on first invocation', () => {
            const fn = vi.fn();
            const throttled = throttle(fn, 100);
            throttled();
            expect(fn).toHaveBeenCalledTimes(1);
        });

        it('blocks subsequent calls within limit', () => {
            const fn = vi.fn();
            const throttled = throttle(fn, 100);
            throttled();
            throttled();
            throttled();
            expect(fn).toHaveBeenCalledTimes(1);
        });

        it('allows call after limit expires', () => {
            const fn = vi.fn();
            const throttled = throttle(fn, 100);
            throttled();
            vi.advanceTimersByTime(100);
            throttled();
            expect(fn).toHaveBeenCalledTimes(2);
        });
    });

    describe('formatNumber', () => {
        it('formats numbers', () => {
            expect(formatNumber(1234.567)).toBeDefined();
        });

        it('respects options', () => {
            const result = formatNumber(0.5, { style: 'percent' });
            expect(result).toContain('50');
        });
    });

    describe('formatPercent', () => {
        it('formats a decimal as percentage', () => {
            expect(formatPercent(0.856)).toBe('85.6%');
        });

        it('respects custom decimals', () => {
            expect(formatPercent(0.856, 0)).toBe('86%');
        });
    });

    describe('formatDate', () => {
        it('formats a Date object', () => {
            const result = formatDate(new Date('2026-01-15T12:00:00Z'));
            expect(result).toBeDefined();
        });

        it('formats from string', () => {
            expect(formatDate('2026-01-15')).toBeDefined();
        });

        it('formats from number', () => {
            expect(formatDate(1737000000000)).toBeDefined();
        });
    });

    describe('clamp', () => {
        it('clamps below min', () => {
            expect(clamp(-5, 0, 10)).toBe(0);
        });

        it('clamps above max', () => {
            expect(clamp(15, 0, 10)).toBe(10);
        });

        it('passes through in range', () => {
            expect(clamp(5, 0, 10)).toBe(5);
        });
    });

    describe('lerp', () => {
        it('returns a at t=0', () => {
            expect(lerp(0, 10, 0)).toBe(0);
        });

        it('returns b at t=1', () => {
            expect(lerp(0, 10, 1)).toBe(10);
        });

        it('returns midpoint at t=0.5', () => {
            expect(lerp(0, 10, 0.5)).toBe(5);
        });

        it('clamps t to [0, 1]', () => {
            expect(lerp(0, 10, 2)).toBe(10);
            expect(lerp(0, 10, -1)).toBe(0);
        });
    });

    describe('mapRange', () => {
        it('maps value between ranges', () => {
            expect(mapRange(5, 0, 10, 0, 100)).toBe(50);
        });

        it('maps across different ranges', () => {
            expect(mapRange(0, 0, 10, 50, 100)).toBe(50);
            expect(mapRange(10, 0, 10, 50, 100)).toBe(100);
        });
    });

    describe('generateId', () => {
        it('generates unique IDs', () => {
            const id1 = generateId();
            const id2 = generateId();
            expect(id1).not.toBe(id2);
        });

        it('uses custom prefix', () => {
            expect(generateId('test')).toMatch(/^test_/);
        });
    });

    describe('deepClone', () => {
        it('clones objects without reference', () => {
            const original = { a: 1, b: { c: 2 } };
            const cloned = deepClone(original);
            cloned.b.c = 99;
            expect(original.b.c).toBe(2);
        });

        it('returns primitives directly', () => {
            expect(deepClone(42)).toBe(42);
            expect(deepClone(null)).toBeNull();
            expect(deepClone('str')).toBe('str');
        });
    });

    describe('isEmpty', () => {
        it('returns true for null/undefined', () => {
            expect(isEmpty(null)).toBe(true);
            expect(isEmpty(undefined)).toBe(true);
        });

        it('returns true for empty string', () => {
            expect(isEmpty('')).toBe(true);
            expect(isEmpty('  ')).toBe(true);
        });

        it('returns true for empty array', () => {
            expect(isEmpty([])).toBe(true);
        });

        it('returns true for empty object', () => {
            expect(isEmpty({})).toBe(true);
        });

        it('returns false for non-empty values', () => {
            expect(isEmpty('hello')).toBe(false);
            expect(isEmpty([1])).toBe(false);
            expect(isEmpty({ a: 1 })).toBe(false);
            expect(isEmpty(0)).toBe(false);
        });
    });

    describe('sleep', () => {
        beforeEach(() => { vi.useFakeTimers(); });
        afterEach(() => { vi.useRealTimers(); });

        it('resolves after specified time', async () => {
            const promise = sleep(100);
            vi.advanceTimersByTime(100);
            await expect(promise).resolves.toBeUndefined();
        });
    });

    describe('retry', () => {
        it('returns on first success', async () => {
            const fn = vi.fn().mockResolvedValue('ok');
            const result = await retry(fn, { attempts: 3, delay: 0 });
            expect(result).toBe('ok');
            expect(fn).toHaveBeenCalledTimes(1);
        });

        it('retries on failure and succeeds', async () => {
            const fn = vi.fn()
                .mockRejectedValueOnce(new Error('fail'))
                .mockResolvedValue('ok');
            const result = await retry(fn, { attempts: 3, delay: 0 });
            expect(result).toBe('ok');
            expect(fn).toHaveBeenCalledTimes(2);
        });

        it('throws after all attempts fail', async () => {
            const fn = vi.fn().mockRejectedValue(new Error('fail'));
            await expect(retry(fn, { attempts: 2, delay: 0 })).rejects.toThrow('fail');
            expect(fn).toHaveBeenCalledTimes(2);
        });

        it('calls onError callback', async () => {
            const onError = vi.fn();
            const fn = vi.fn().mockRejectedValue(new Error('fail'));
            await expect(retry(fn, { attempts: 2, delay: 0, onError })).rejects.toThrow();
            expect(onError).toHaveBeenCalledTimes(2);
        });
    });

    describe('downloadFile', () => {
        it('creates and clicks a link element', () => {
            const createObjectURL = vi.fn().mockReturnValue('blob:test');
            const revokeObjectURL = vi.fn();
            vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });

            const appendChild = vi.spyOn(document.body, 'appendChild').mockImplementation((node) => node);
            const removeChild = vi.spyOn(document.body, 'removeChild').mockImplementation((node) => node);

            downloadFile('test data', 'test.txt', 'text/plain');

            expect(createObjectURL).toHaveBeenCalled();
            expect(appendChild).toHaveBeenCalled();
            expect(removeChild).toHaveBeenCalled();
            expect(revokeObjectURL).toHaveBeenCalled();

            appendChild.mockRestore();
            removeChild.mockRestore();
            vi.unstubAllGlobals();
        });

        it('handles Blob input', () => {
            const createObjectURL = vi.fn().mockReturnValue('blob:test');
            const revokeObjectURL = vi.fn();
            vi.stubGlobal('URL', { createObjectURL, revokeObjectURL });

            const appendChild = vi.spyOn(document.body, 'appendChild').mockImplementation((node) => node);
            const removeChild = vi.spyOn(document.body, 'removeChild').mockImplementation((node) => node);

            const blob = new Blob(['data'], { type: 'text/plain' });
            downloadFile(blob, 'test.txt');

            expect(createObjectURL).toHaveBeenCalledWith(blob);

            appendChild.mockRestore();
            removeChild.mockRestore();
            vi.unstubAllGlobals();
        });
    });

    describe('createEventEmitter', () => {
        it('emits events to handlers', () => {
            const emitter = createEventEmitter<{ test: string }>();
            const handler = vi.fn();
            emitter.on('test', handler);
            emitter.emit('test', 'hello');
            expect(handler).toHaveBeenCalledWith('hello');
        });

        it('removes handlers with off', () => {
            const emitter = createEventEmitter<{ test: string }>();
            const handler = vi.fn();
            emitter.on('test', handler);
            emitter.off('test', handler);
            emitter.emit('test', 'hello');
            expect(handler).not.toHaveBeenCalled();
        });

        it('returns unsubscribe function from on', () => {
            const emitter = createEventEmitter<{ test: string }>();
            const handler = vi.fn();
            const unsub = emitter.on('test', handler);
            unsub();
            emitter.emit('test', 'hello');
            expect(handler).not.toHaveBeenCalled();
        });

        it('handles emit on non-existent event gracefully', () => {
            const emitter = createEventEmitter<{ test: string }>();
            expect(() => emitter.emit('test', 'hello')).not.toThrow();
        });
    });
});
