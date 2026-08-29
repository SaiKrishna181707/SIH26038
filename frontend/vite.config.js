import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The React plugin was previously a dependency but never registered, so Fast
// Refresh did not work and every edit forced a full page reload.
//
// The dev server proxies /api to the FastAPI backend so the browser sees a
// single origin during development and CORS never enters the picture. Set
// VITE_API_BASE_URL at build time to point a production bundle at a deployed
// backend instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_ORIGIN ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
