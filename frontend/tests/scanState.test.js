import test from "node:test";
import assert from "node:assert/strict";
import { initialState, scanReducer as r } from "../src/lib/scanState.js";

test("happy path: scan -> preview -> analyzing -> result -> reset", () => {
  let s = r(initialState, { type: "PICK", file: "f", previewUrl: "blob:1" });
  assert.equal(s.step, "preview");
  s = r(s, { type: "ANALYZE_START" });
  assert.equal(s.step, "analyzing");
  s = r(s, { type: "ANALYZE_OK", result: { label: "blast" } });
  assert.equal(s.step, "result");
  assert.equal(s.result.label, "blast");
  assert.deepEqual(r(s, { type: "RESET" }), initialState);
});

test("a failed check returns to the preview with the error, keeping the photo", () => {
  let s = r(initialState, { type: "PICK", file: "f", previewUrl: "blob:1" });
  s = r(s, { type: "ANALYZE_START" });
  s = r(s, { type: "ANALYZE_FAIL", message: "Can't reach the server." });
  assert.equal(s.step, "preview");
  assert.equal(s.error, "Can't reach the server.");
  assert.equal(s.file, "f");
  assert.equal(r(s, { type: "ANALYZE_START" }).error, null); // retrying clears the message
});

test("choosing a new photo clears the old result and error", () => {
  let s = r(initialState, { type: "PICK", file: "a", previewUrl: "blob:a" });
  s = r(s, { type: "ANALYZE_OK", result: { label: "blast" } });
  s = r(s, { type: "PICK", file: "b", previewUrl: "blob:b" });
  assert.equal(s.result, null);
  assert.equal(s.file, "b");
  assert.equal(s.step, "preview");
});

test("going back from the result keeps the photo", () => {
  let s = r(initialState, { type: "PICK", file: "a", previewUrl: "blob:a" });
  s = r(s, { type: "ANALYZE_OK", result: {} });
  s = r(s, { type: "BACK_TO_PREVIEW" });
  assert.equal(s.step, "preview");
  assert.equal(s.file, "a");
});

test("a rejected file shows an error on the scan screen", () => {
  const s = r(initialState, { type: "PICK_ERROR", message: "Choose a photo." });
  assert.equal(s.step, "scan");
  assert.equal(s.error, "Choose a photo.");
});
