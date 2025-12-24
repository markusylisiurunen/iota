import { beforeEach, describe, expect, it, vi } from "vitest";

let capturedParams: any;

vi.mock("@google/genai", () => {
  const FinishReason = {
    STOP: "STOP",
    MAX_TOKENS: "MAX_TOKENS",
  };

  const ThinkingLevel = {
    THINKING_LEVEL_UNSPECIFIED: "THINKING_LEVEL_UNSPECIFIED",
    LOW: "LOW",
    MEDIUM: "MEDIUM",
    HIGH: "HIGH",
    MINIMAL: "MINIMAL",
  };

  class GoogleGenAIMock {
    models: {
      generateContentStream: (params: any) => Promise<AsyncIterable<unknown>>;
    };

    constructor(_options: any) {
      this.models = {
        generateContentStream: async (params: any) => {
          capturedParams = params;
          return (async function* () {})();
        },
      };
    }
  }

  return {
    GoogleGenAI: GoogleGenAIMock,
    FinishReason,
    ThinkingLevel,
  };
});

import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type { Context } from "../src/types.js";

describe("google thinking config for reasoning:none", () => {
  beforeEach(() => {
    capturedParams = undefined;
  });

  it("sets thinkingBudget=0 for flash", async () => {
    const model = getModel("google", "gemini-3-flash-preview");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "none" });
    await s.result();

    expect(capturedParams.config.thinkingConfig).toMatchObject({
      includeThoughts: false,
      thinkingBudget: 0,
    });
  });

  it("sets thinkingLevel=LOW for pro", async () => {
    const model = getModel("google", "gemini-3-pro-preview");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "none" });
    await s.result();

    expect(capturedParams.config.thinkingConfig).toMatchObject({
      includeThoughts: false,
      thinkingLevel: "LOW",
    });
  });
});
