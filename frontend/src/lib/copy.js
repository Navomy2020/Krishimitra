// All wording that depends on the result, in one place. Uncertainty is always stated
// in words, never as a raw probability.

export const LEVELS = {
  high: { label: "High confidence" },
  medium: { label: "Moderate confidence" },
  low: { label: "Low confidence" },
};

const RANK_WORDS = { 1: "Most likely", 2: "Also possible", 3: "Less likely" };

export function rankWord(rank) {
  return RANK_WORDS[rank] ?? "Possible";
}

export function headline(result) {
  if (result.needs_expert) return "Not sure yet";
  if (result.confidence_level === "medium" && !result.is_healthy) return `Likely ${result.disease}`;
  return result.disease;
}

// Which candidates to list under the diagnosis, and what to call the list.
export function possibilities(result) {
  const all = result.predictions ?? [];
  if (result.confidence_level === "high") {
    if (result.is_healthy) return { title: null, items: [] };
    return { title: "Other possibilities", items: all.slice(1) };
  }
  return { title: "Possible causes", items: all };
}

export function adviceIntro(result) {
  return result.confidence_level === "medium"
    ? "General steps while you confirm the diagnosis"
    : result.is_healthy
      ? "Keep it that way"
      : "What you can do";
}
