import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock config before importing Logger
vi.mock('@/config', () => ({
  config: {
    logging: {
      level: 'debug',
      enableConsole: true,
      enableRemote: false,
      maxLogSize: 100,
    },
    api: {
      baseUrl: 'http://localhost:8000',
    },
  },
}));

import { Logger } from './logger';

describe('Logger', () => {
  let consoleSpy: {
    debug: ReturnType<typeof vi.spyOn>;
    info: ReturnType<typeof vi.spyOn>;
    warn: ReturnType<typeof vi.spyOn>;
    error: ReturnType<typeof vi.spyOn>;
  };

  beforeEach(() => {
    consoleSpy = {
      debug: vi.spyOn(console, 'debug').mockImplementation(() => {}),
      info: vi.spyOn(console, 'info').mockImplementation(() => {}),
      warn: vi.spyOn(console, 'warn').mockImplementation(() => {}),
      error: vi.spyOn(console, 'error').mockImplementation(() => {}),
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Log level methods', () => {
    it('logs debug messages', () => {
      const logger = new Logger({ level: 'debug', enableConsole: true });
      logger.debug('debug test');
      expect(consoleSpy.debug).toHaveBeenCalledOnce();
    });

    it('logs info messages', () => {
      const logger = new Logger({ level: 'debug', enableConsole: true });
      logger.info('info test');
      expect(consoleSpy.info).toHaveBeenCalledOnce();
    });

    it('logs warn messages', () => {
      const logger = new Logger({ level: 'debug', enableConsole: true });
      logger.warn('warn test');
      expect(consoleSpy.warn).toHaveBeenCalledOnce();
    });

    it('logs error messages', () => {
      const logger = new Logger({ level: 'debug', enableConsole: true });
      logger.error('error test');
      expect(consoleSpy.error).toHaveBeenCalledOnce();
    });

    it('passes context to console', () => {
      const logger = new Logger({ level: 'debug', enableConsole: true });
      const ctx = { userId: 42 };
      logger.info('with context', ctx);
      expect(consoleSpy.info).toHaveBeenCalledOnce();
    });
  });

  describe('Log level filtering', () => {
    it('suppresses debug when level is info', () => {
      const logger = new Logger({ level: 'info', enableConsole: true });
      logger.debug('should not appear');
      expect(consoleSpy.debug).not.toHaveBeenCalled();
    });

    it('suppresses debug and info when level is warn', () => {
      const logger = new Logger({ level: 'warn', enableConsole: true });
      logger.debug('nope');
      logger.info('nope');
      expect(consoleSpy.debug).not.toHaveBeenCalled();
      expect(consoleSpy.info).not.toHaveBeenCalled();
    });

    it('only allows error when level is error', () => {
      const logger = new Logger({ level: 'error', enableConsole: true });
      logger.debug('no');
      logger.info('no');
      logger.warn('no');
      logger.error('yes');
      expect(consoleSpy.debug).not.toHaveBeenCalled();
      expect(consoleSpy.info).not.toHaveBeenCalled();
      expect(consoleSpy.warn).not.toHaveBeenCalled();
      expect(consoleSpy.error).toHaveBeenCalledOnce();
    });
  });

  describe('setLevel', () => {
    it('changes log level dynamically', () => {
      const logger = new Logger({ level: 'error', enableConsole: true });
      logger.debug('suppressed');
      expect(consoleSpy.debug).not.toHaveBeenCalled();

      logger.setLevel('debug');
      logger.debug('now visible');
      expect(consoleSpy.debug).toHaveBeenCalledOnce();
    });
  });

  describe('Log buffer', () => {
    it('stores entries in getRecentLogs', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      logger.info('first');
      logger.warn('second');
      logger.error('third');

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(3);
      expect(logs[0]!.level).toBe('info');
      expect(logs[1]!.level).toBe('warn');
      expect(logs[2]!.level).toBe('error');
    });

    it('getRecentLogs respects count parameter', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      logger.info('a');
      logger.info('b');
      logger.info('c');

      const logs = logger.getRecentLogs(2);
      expect(logs).toHaveLength(2);
      expect(logs[0]!.message).toBe('b');
      expect(logs[1]!.message).toBe('c');
    });

    it('clearLogs empties the buffer', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      logger.info('entry');
      expect(logger.getRecentLogs()).toHaveLength(1);

      logger.clearLogs();
      expect(logger.getRecentLogs()).toHaveLength(0);
    });

    it('trims buffer when exceeding maxLogSize', () => {
      const logger = new Logger({
        level: 'debug',
        enableConsole: false,
        maxLogSize: 3,
      });
      logger.info('1');
      logger.info('2');
      logger.info('3');
      logger.info('4');

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(3);
      expect(logs[0]!.message).toBe('2');
    });
  });

  describe('Child logger', () => {
    it('creates a child logger with source prefix', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      const child = logger.child('TestSource');

      child.info('hello');

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(1);
      expect(logs[0]!.message).toBe('[TestSource] hello');
    });

    it('child logger delegates all levels to parent', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      const child = logger.child('Src');

      child.debug('d');
      child.info('i');
      child.warn('w');
      child.error('e');

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(4);
      expect(logs.map((l) => l.level)).toEqual([
        'debug',
        'info',
        'warn',
        'error',
      ]);
    });
  });

  describe('Performance timing', () => {
    it('time() returns a stop function that logs duration', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      const stop = logger.time('operation');

      // Simulate some work
      stop();

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(1);
      expect(logs[0]!.message).toContain('operation completed');
      expect(logs[0]!.context).toHaveProperty('durationMs');
    });

    it('timeAsync() measures and returns result', async () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });

      const result = await logger.timeAsync('async-op', async () => {
        return 'done';
      });

      expect(result).toBe('done');
      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(1);
      expect(logs[0]!.message).toContain('async-op completed');
    });

    it('timeAsync() logs error on failure', async () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });

      await expect(
        logger.timeAsync('failing-op', async () => {
          throw new Error('boom');
        })
      ).rejects.toThrow('boom');

      const logs = logger.getRecentLogs();
      expect(logs).toHaveLength(1);
      expect(logs[0]!.level).toBe('error');
      expect(logs[0]!.message).toContain('failing-op failed');
    });
  });

  describe('Console disabled', () => {
    it('does not output to console when enableConsole is false', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      logger.debug('silent');
      logger.info('silent');
      logger.warn('silent');
      logger.error('silent');

      expect(consoleSpy.debug).not.toHaveBeenCalled();
      expect(consoleSpy.info).not.toHaveBeenCalled();
      expect(consoleSpy.warn).not.toHaveBeenCalled();
      expect(consoleSpy.error).not.toHaveBeenCalled();
    });
  });

  describe('Log entry format', () => {
    it('creates entries with correct structure', () => {
      const logger = new Logger({ level: 'debug', enableConsole: false });
      logger.info('test message', { key: 'value' });

      const entry = logger.getRecentLogs()[0]!;
      expect(entry).toMatchObject({
        level: 'info',
        message: 'test message',
        context: { key: 'value' },
      });
      expect(entry.timestamp).toBeDefined();
      expect(new Date(entry.timestamp).getTime()).not.toBeNaN();
    });
  });
});
