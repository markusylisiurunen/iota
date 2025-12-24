import { beforeEach, describe, expect, it, vi } from "vitest";

let capturedParams: any;

vi.mock("@anthropic-ai/sdk", () => {
  class AnthropicMock {
    messages: {
      stream: (params: any) => AsyncIterable<unknown>;
    };

    constructor(_options: any) {
      this.messages = {
        stream: (params: any) => {
          capturedParams = params;
          return (async function* () {})();
        },
      };
    }
  }

  return { default: AnthropicMock };
});

import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type { Context } from "../src/types.js";

describe("anthropic thinking budget", () => {
  beforeEach(() => {
    capturedParams = undefined;
  });

  it("uses 1024 budget tokens for minimal", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "minimal" });
    await s.result();

    expect(capturedParams.thinking).toMatchObject({ type: "enabled", budget_tokens: 1024 });
  });

  it("uses 32768 budget tokens for high", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "high" });
    await s.result();

    expect(capturedParams.thinking).toMatchObject({ type: "enabled", budget_tokens: 32768 });
  });

  it("caps budget tokens to 80% of maxTokens", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "high", maxTokens: 1000 });
    await s.result();

    expect(capturedParams.thinking).toMatchObject({ type: "enabled", budget_tokens: 800 });
  });

  it("uses 32768 budget tokens for xhigh", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "xhigh" });
    await s.result();

    expect(capturedParams.thinking).toMatchObject({ type: "enabled", budget_tokens: 32768 });
  });
});
