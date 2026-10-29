// "" means same origin: the Vite dev proxy in development, or FastAPI itself when it serves the built app.
// Set VITE_API_URL (see .env.example) only when the API lives at a different address.
export const API_BASE = (import.meta.env.VITE_API_URL ?? "").replace(/\/+$/, "");
