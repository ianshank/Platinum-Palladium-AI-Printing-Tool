/**
 * Structured logging utility
 * Provides consistent logging with levels, timestamps, and context
 */

import { config, type LogLevel } from '@/config';

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: Record<string, unknown> | undefined;
  source?: string | undefined;
}

interface LoggerOptions {
  level?: LogLevel;
  enableConsole?: boolean;
  enableRemote?: boolean;
  maxLogSize?: number;
}

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const LOG_COLORS: Record<LogLevel, string> = {
  debug: '#6b7280', // gray
  info: '#3b82f6', // blue
  warn: '#f59e0b', // amber
  error: '#ef4444', // red
};

class Logger {
  private level: LogLevel;
  private enableConsole: boolean;
  private enableRemote: boolean;
  private maxLogSize: number;
  private logBuffer: LogEntry[] = [];

  constructor(options: LoggerOptions = {}) {
    this.level = options.level ?? config.logging.level;
    this.enableConsole = options.enableConsole ?? config.logging.enableConsole;
    this.enableRemote = options.enableRemote ?? config.logging.enableRemote;
    this.maxLogSize = options.maxLogSize ?? config.logging.maxLogSize;
  }

  /**
   * Check if a log level should be output
   */
  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[this.level];
  }

  /**
   * Format log entry for console output
   */
  private formatForConsole(entry: LogEntry): string[] {
    const args: string[] = [
      `%c[${entry.level.toUpperCase()}]%c ${entry.timestamp} - ${entry.message}`,
      `color: ${LOG_COLORS[entry.level]}; font-weight: bold`,
      'color: inherit',
    ];

    return args;
  }

  /**
   * Create a log entry
   */
  private createEntry(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    source?: string
  ): LogEntry {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
      source,
    };
  }

  /**
   * Add entry to buffer and trim if necessary
   */
  private addToBuffer(entry: LogEntry): void {
    this.logBuffer.push(entry);
    if (this.logBuffer.length > this.maxLogSize) {
      this.logBuffer.shift();
    }
  }

  /**
   * Output log to console
   */
  private outputToConsole(entry: LogEntry): void {
    if (!this.enableConsole) return;

    const [format, ...styles] = this.formatForConsole(entry);

    switch (entry.level) {
      case 'debug':
        // eslint-disable-next-line no-console
        console.debug(format, ...styles, entry.context ?? '');
        break;
      case 'info':
        // eslint-disable-next-line no-console
        console.info(format, ...styles, entry.context ?? '');
        break;
      case 'warn':
        console.warn(format, ...styles, entry.context ?? '');
        break;
      case 'error':
        console.error(format, ...styles, entry.context ?? '');
        break;
    }
  }

  /**
   * Send log to remote endpoint (if enabled)
   */
  private async sendToRemote(entry: LogEntry): Promise<void> {
    if (!this.enableRemote) return;

    try {
      await fetch(`${config.api.baseUrl}/api/logs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry),
      });
    } catch {
      // Silently fail - don't cause issues if logging fails
    }
  }

  /**
   * Core logging method
   */
  private log(
    level: LogLevel,
    message: string,
    context?: Record<string, unknown>,
    source?: string
  ): void {
    if (!this.shouldLog(level)) return;

    const entry = this.createEntry(level, message, context, source);
    this.addToBuffer(entry);
    this.outputToConsole(entry);

    // Only send errors and warns to remote by default
    if (level === 'error' || level === 'warn') {
      void this.sendToRemote(entry);
    }
  }

  /**
   * Log debug message
   */
  debug(message: string, context?: Record<string, unknown>): void {
    this.log('debug', message, context);
  }

  /**
   * Log info message
   */
  info(message: string, context?: Record<string, unknown>): void {
    this.log('info', message, context);
  }

  /**
   * Log warning message
   */
  warn(message: string, context?: Record<string, unknown>): void {
    this.log('warn', message, context);
  }

  /**
   * Log error message
   */
  error(message: string, context?: Record<string, unknown>): void {
    this.log('error', message, context);
  }

  /**
   * Create a child logger with a source prefix
   */
  child(source: string): ChildLogger {
    return new ChildLogger(this, source);
  }

  /**
   * Get recent log entries
   */
  getRecentLogs(count?: number): LogEntry[] {
    const n = count ?? this.logBuffer.length;
    return this.logBuffer.slice(-n);
  }

  /**
   * Clear log buffer
   */
  clearLogs(): void {
    this.logBuffer = [];
  }

  /**
   * Set log level dynamically
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Start a performance timer
   */
  time(label: string): () => void {
    const start = performance.now();
    return () => {
      const duration = performance.now() - start;
      this.debug(`${label} completed`, { durationMs: duration.toFixed(2) });
    };
  }

  /**
   * Log with performance timing
   */
  async timeAsync<T>(
    label: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const endTimer = this.time(label);
    try {
      const result = await fn();
      endTimer();
      return result;
    } catch (error) {
      this.error(`${label} failed`, { error: String(error) });
      throw error;
    }
  }
}

/**
 * Child logger with source prefix
 */
class ChildLogger {
  constructor(
    private parent: Logger,
    private source: string
  ) { }

  debug(message: string, context?: Record<string, unknown>): void {
    this.parent.debug(`[${this.source}] ${message}`, context);
  }

  info(message: string, context?: Record<string, unknown>): void {
    this.parent.info(`[${this.source}] ${message}`, context);
  }

  warn(message: string, context?: Record<string, unknown>): void {
    this.parent.warn(`[${this.source}] ${message}`, context);
  }

  error(message: string, context?: Record<string, unknown>): void {
    this.parent.error(`[${this.source}] ${message}`, context);
  }

  time(label: string): () => void {
    return this.parent.time(`[${this.source}] ${label}`);
  }
}

// Export singleton instance
export const logger = new Logger();

// Export class for custom instances
export { Logger, type LogEntry, type LoggerOptions };
