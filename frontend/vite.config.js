import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  if (mode === 'production' && !env.VITE_API_BASE_URL) {
    console.warn(
      '[vite] VITE_API_BASE_URL is not set — production build will not reach the backend until configured on Vercel.',
    )
  }

  return {
    plugins: [react()],
  }
})
