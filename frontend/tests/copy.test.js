import test from "node:test";
import assert from "node:assert/strict";
import { adviceIntro, headline, possibilities, rankWord } from "../src/lib/copy.js";

const preds = [
  { rank: 1, label: "tungro", name: "Tungro" },
  { rank: 2, label: "bacterial_leaf_blight", name: "Bacterial Leaf Blight" },
  { rank: 3, label: "brown_spot", name: "Brown Spot" },
];
const base = { disease: "Tungro", is_healthy: false, needs_expert: false, confidence_level: "high", predictions: preds };

test("headline says so when the scan is not certain", () => {
  assert.equal(headline({ ...base, needs_expert: true, confidence_level: "low" }), "Not sure yet");
  assert.equal(headline({ ...base, confidence_level: "medium" }), "Likely Tungro");
  assert.equal(headline(base), "Tungro");
  assert.equal(headline({ ...base, disease: "Healthy Leaf", is_healthy: true, confidence_level: "medium" }), "Healthy Leaf");
});

test("a low-confidence scan still lists the top guess among the possibilities", () => {
  const p = possibilities({ ...base, needs_expert: true, confidence_level: "low" });
  assert.equal(p.title, "Possible causes");
  assert.deepEqual(p.items.map((x) => x.name), ["Tungro", "Bacterial Leaf Blight", "Brown Spot"]);
});

test("a confident diagnosis lists only the runners-up", () => {
  const p = possibilities(base);
  assert.equal(p.title, "Other possibilities");
  assert.deepEqual(p.items.map((x) => x.rank), [2, 3]);
});

test("a confidently healthy leaf shows no list", () => {
  assert.deepEqual(possibilities({ ...base, is_healthy: true }), { title: null, items: [] });
});

test("rank words and advice titles", () => {
  assert.equal(rankWord(1), "Most likely");
  assert.equal(rankWord(2), "Also possible");
  assert.equal(rankWord(3), "Less likely");
  assert.equal(adviceIntro({ ...base, confidence_level: "medium" }), "General steps while you confirm the diagnosis");
  assert.equal(adviceIntro(base), "What you can do");
});
