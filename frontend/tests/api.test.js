import test from "node:test";
import assert from "node:assert/strict";
import { ApiError, MESSAGES, createApi } from "../src/lib/api.js";

const json = (status, body) => ({ ok: status < 400, status, json: async () => body });
const blob = () => new Blob([new Uint8Array([1, 2, 3])], { type: "image/jpeg" });

test("predict posts the photo as multipart field 'file' and returns the JSON", async () => {
  let seen;
  const api = createApi({
    baseUrl: "http://x",
    fetchImpl: async (url, init) => {
      seen = { url, init };
      return json(200, { label: "blast" });
    },
  });
  assert.deepEqual(await api.predict(blob()), { label: "blast" });
  assert.equal(seen.url, "http://x/api/leaf/predict");
  assert.equal(seen.init.method, "POST");
  assert.ok(seen.init.body instanceof FormData);
  assert.ok(seen.init.body.get("file") instanceof Blob);
  assert.equal(seen.init.headers["Content-Type"], undefined); // the browser must set the boundary
});

test("server detail messages are passed through to the user", async () => {
  const api = createApi({ fetchImpl: async () => json(400, { detail: "The photo is too small." }) });
  await assert.rejects(api.predict(blob()), (e) => e instanceof ApiError && e.kind === "server" && e.status === 400 && e.message === "The photo is too small.");
});

test("503 and 422 and 500 get helpful fallback text", async () => {
  const cases = [[503, MESSAGES.unavailable], [422, MESSAGES.badRequest], [500, MESSAGES.serverError]];
  for (const [status, text] of cases) {
    const api = createApi({ fetchImpl: async () => json(status, status === 422 ? { detail: [{ msg: "field required" }] } : {}) });
    await assert.rejects(api.predict(blob()), (e) => e.message === text);
  }
});

test("a network failure is reported as a connection problem", async () => {
  const api = createApi({ fetchImpl: async () => { throw new TypeError("Failed to fetch"); } });
  await assert.rejects(api.predict(blob()), (e) => e.kind === "network" && e.message === MESSAGES.network);
});

const hangUntilAborted = (_url, { signal }) =>
  new Promise((_, reject) => signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }))));

test("a slow server times out", async () => {
  const api = createApi({ fetchImpl: hangUntilAborted, timeoutMs: 20 });
  await assert.rejects(api.predict(blob()), (e) => e.kind === "timeout");
});

test("the user can cancel a request", async () => {
  const api = createApi({ fetchImpl: hangUntilAborted, timeoutMs: 5000 });
  const controller = new AbortController();
  const p = api.predict(blob(), { signal: controller.signal });
  controller.abort();
  await assert.rejects(p, (e) => e.kind === "cancelled");
});

test("an unreadable 200 response is a server error, not a network error", async () => {
  const api = createApi({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("bad"); } }) });
  await assert.rejects(api.predict(blob()), (e) => e.kind === "server" && e.message === MESSAGES.badResponse);
});

test("health works", async () => {
  const api = createApi({ fetchImpl: async () => json(200, { model_loaded: true }) });
  assert.equal((await api.health()).model_loaded, true);
});
