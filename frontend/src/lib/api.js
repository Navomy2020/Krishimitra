// Talks to the FastAPI backend. Plain JavaScript with injectable fetch so it can be tested in Node.

export class ApiError extends Error {
  constructor(kind, message, status) {
    super(message);
    this.name = "ApiError";
    this.kind = kind; // "network" | "timeout" | "cancelled" | "server"
    this.status = status;
  }
}

export const MESSAGES = {
  network: "Can't reach the Krishi Mithra server. Make sure the backend is running, then try again.",
  timeout: "The check took too long. Check your connection and try again.",
  unavailable: "The disease model is not ready yet. Wait a moment and try again.",
  badRequest: "The photo could not be sent. Choose it again.",
  serverError: "The server had a problem with this photo. Try again, or use a different photo.",
  badResponse: "The server sent an answer the app could not read. Try again.",
};

async function toApiError(res) {
  let detail;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail; // FastAPI's own validation errors are arrays: ignored
  } catch {
    /* the error body was not JSON */
  }
  const fallback =
    res.status === 422 ? MESSAGES.badRequest
    : res.status === 503 ? MESSAGES.unavailable
    : res.status >= 500 ? MESSAGES.serverError
    : `The request failed (${res.status}).`;
  return new ApiError("server", detail ?? fallback, res.status);
}

export function createApi({ baseUrl = "", fetchImpl = (...args) => fetch(...args), timeoutMs = 30000 } = {}) {
  async function request(path, { method = "GET", body, signal, timeout = timeoutMs } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeout);
    const onAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) controller.abort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    try {
      // No Content-Type header: the browser adds the multipart boundary itself.
      const res = await fetchImpl(`${baseUrl}${path}`, {
        method,
        body,
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      if (!res.ok) throw await toApiError(res);
      try {
        return await res.json();
      } catch {
        throw new ApiError("server", MESSAGES.badResponse, res.status);
      }
    } catch (err) {
      if (err instanceof ApiError) throw err;
      if (err?.name === "AbortError") {
        throw timedOut ? new ApiError("timeout", MESSAGES.timeout) : new ApiError("cancelled", "Cancelled.");
      }
      throw new ApiError("network", MESSAGES.network);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
  }

  return {
    health: (opts) => request("/health", { timeout: 8000, ...opts }),
    predict(blob, opts = {}) {
      const form = new FormData();
      form.append("file", blob, "leaf.jpg");
      return request("/api/leaf/predict", { method: "POST", body: form, ...opts });
    },
  };
}
