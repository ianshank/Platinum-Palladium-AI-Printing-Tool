import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],

    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@/components': path.resolve(__dirname, './src/components'),
        '@/stores': path.resolve(__dirname, './src/stores'),
        '@/hooks': path.resolve(__dirname, './src/hooks'),
        '@/lib': path.resolve(__dirname, './src/lib'),
        '@/api': path.resolve(__dirname, './src/api'),
        '@/utils': path.resolve(__dirname, './src/utils'),
        '@/types': path.resolve(__dirname, './src/types'),
        '@/config': path.resolve(__dirname, './src/config'),
        '@/test-utils': path.resolve(__dirname, './src/test-utils'),
      },
    },

    server: {
      port: 3000,
      host: true,
      proxy: {
        '/api': {
          target: env['VITE_API_URL'] || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
        },
      },
    },

    preview: {
      port: 3000,
    },

    build: {
      target: 'ES2022',
      outDir: 'dist',
      sourcemap: mode === 'development',
      minify: mode === 'production' ? 'esbuild' : false,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'react-router-dom'],
            ui: [
              '@radix-ui/react-tabs',
              '@radix-ui/react-dialog',
              '@radix-ui/react-dropdown-menu',
              '@radix-ui/react-accordion',
            ],
            charts: ['plotly.js', 'react-plotly.js', 'recharts'],
            query: ['@tanstack/react-query', 'axios'],
            state: ['zustand', 'immer'],
          },
        },
      },
      chunkSizeWarningLimit: 1000,
    },

    define: {
      __APP_VERSION__: JSON.stringify(process.env['npm_package_version']),
      __DEV__: mode === 'development',
    },

    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'zustand',
        'immer',
        '@tanstack/react-query',
      ],
    },
  };
});
