import { describe, expect, it } from "vitest";

import { calculateCost, getModel } from "../src/models.js";

describe("service tier pricing", () => {
  it("applies OpenAI flex/priority multipliers", () => {
    const model = getModel("openai", "gpt-5.2");

    const usage = {
      inputTokens: 1_000_000,
      outputTokens: 1_000_000,
      cacheReadTokens: 1_000_000,
      cacheWriteTokens: 0,
    };

    const standard = calculateCost(model, usage, "standard");
    expect(standard.input).toBeCloseTo(1.75);
    expect(standard.output).toBeCloseTo(14);
    expect(standard.cacheRead).toBeCloseTo(0.175);
    expect(standard.total).toBeCloseTo(15.925);

    const flex = calculateCost(model, usage, "flex");
    expect(flex.total).toBeCloseTo(15.925 * 0.5);

    const priority = calculateCost(model, usage, "priority");
    expect(priority.total).toBeCloseTo(15.925 * 2);
  });

  it("does not apply OpenAI multipliers to other providers", () => {
    const model = getModel("anthropic", "opus-4.5");

    const usage = {
      inputTokens: 1_000_000,
      outputTokens: 1_000_000,
      cacheReadTokens: 1_000_000,
      cacheWriteTokens: 1_000_000,
    };

    const base = calculateCost(model, usage);
    const priority = calculateCost(model, usage, "priority");

    expect(priority.total).toBeCloseTo(base.total);
  });
});
