/**
 * Environment configuration with Zod validation.
 * All environment variables are validated at runtime to ensure type safety.
 */

import { z } from 'zod';

const envSchema = z.object({
  // API Configuration
  VITE_API_BASE_URL: z.string().url().default('http://localhost:8000'),
  VITE_API_TIMEOUT: z.coerce.number().positive().default(30000),

  // Feature Flags
  VITE_ENABLE_AI_ASSISTANT: z
    .string()
    .transform((v) => v === 'true')
    .default('true'),
  VITE_ENABLE_DEEP_LEARNING: z
    .string()
    .transform((v) => v === 'true')
    .default('false'),
  VITE_ENABLE_CLOUD_SYNC: z
    .string()
    .transform((v) => v === 'true')
    .default('false'),

  // UI Configuration
  VITE_DEFAULT_THEME: z.enum(['dark', 'light']).default('dark'),
  VITE_ANIMATION_ENABLED: z
    .string()
    .transform((v) => v === 'true')
    .default('true'),

  // Development
  VITE_DEBUG_MODE: z
    .string()
    .transform((v) => v === 'true')
    .default('false'),
});

export type EnvConfig = z.infer<typeof envSchema>;

const parseEnv = (): EnvConfig => {
  const result = envSchema.safeParse(import.meta.env);

  if (!result.success) {
    console.error('Environment validation failed:', result.error.format());
    // In production, we might want to throw here
    // For now, return defaults
    return envSchema.parse({});
  }

  return result.data;
};

export const env = parseEnv();

/**
 * Type-safe accessor for environment variables
 */
export const getEnv = <K extends keyof EnvConfig>(key: K): EnvConfig[K] =>
  env[key];
