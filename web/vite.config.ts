import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/predictions': 'http://localhost:8010',
      '/models': 'http://localhost:8010',
      '/lines': 'http://localhost:8010',
      '/health': 'http://localhost:8010',
      '/ready': 'http://localhost:8010',
    },
  },
})
