import { defineConfig } from 'vite'
import { realpathSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const dependencies = realpathSync(fileURLToPath(new URL('./node_modules', import.meta.url)))

export default defineConfig({
  build: { cssMinify: false },
  server: { fs: { allow: ['.', dependencies] } },
})
