import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-only: where Vite proxies `/api` (not exposed to the browser bundle).
const apiProxyTarget = process.env.API_PROXY_TARGET ?? 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
