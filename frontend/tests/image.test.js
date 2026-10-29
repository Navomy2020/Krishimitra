import test from "node:test";
import assert from "node:assert/strict";
import { fitWithin, isImageFile } from "../src/lib/image.js";

test("large photos are scaled down, keeping the aspect ratio", () => {
  assert.deepEqual(fitWithin(4000, 3000, 1024), { width: 1024, height: 768 });
  assert.deepEqual(fitWithin(3000, 4000, 1024), { width: 768, height: 1024 });
});

test("small photos are never upscaled", () => {
  assert.deepEqual(fitWithin(640, 480, 1024), { width: 640, height: 480 });
  assert.deepEqual(fitWithin(1024, 1024, 1024), { width: 1024, height: 1024 });
});

test("only images are accepted; an unknown type is let through", () => {
  assert.equal(isImageFile({ type: "image/jpeg" }), true);
  assert.equal(isImageFile({ type: "" }), true);
  assert.equal(isImageFile({ type: "application/pdf" }), false);
  assert.equal(isImageFile(null), false);
});
