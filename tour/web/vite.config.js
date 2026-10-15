import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// Built into dist/, which tour/app.py serves at / (index.html) and /assets/.
export default defineConfig({ plugins: [react()], base: "/", build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 6000 } })
