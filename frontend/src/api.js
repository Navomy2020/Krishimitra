import { API_BASE } from "./config.js";
import { createApi } from "./lib/api.js";

export const api = createApi({ baseUrl: API_BASE });
