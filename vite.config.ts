import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Relative base so the build works on any static host / sub-path (e.g. GitHub Pages).
  base: './',
  plugins: [react(), tailwindcss()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // onnxruntime-web ships its own wasm/worker files; pre-bundling breaks their resolution.
    exclude: ['@huggingface/transformers'],
  },
})
