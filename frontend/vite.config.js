import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app and the API share one origin: Vite forwards /api and /health
// to FastAPI, so there is no CORS setup and no IP address to edit. A phone on the
// same Wi-Fi can open http://<your-computer-ip>:5173 and it just works.
const target = process.env.VITE_PROXY_TARGET || "http://localhost:8000";
const proxy = { "/api": { target, changeOrigin: true }, "/health": { target, changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173, proxy },
  preview: { host: true, port: 5173, proxy },
});
