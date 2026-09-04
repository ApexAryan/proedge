import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', '')
  const apiKey = env.API_KEY
  const target = `http://localhost:${env.API_PORT || '8010'}`
  const proxy = Object.fromEntries(
    ['/predictions', '/models', '/lines', '/health', '/ready'].map((path) => [
      path,
      {
        target,
        headers: apiKey ? { 'X-API-Key': apiKey } : undefined,
      },
    ]),
  )

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true,
      proxy,
    },
  }
})
