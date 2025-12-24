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

describe("google tool result grouping", () => {
  beforeEach(() => {
    capturedParams = undefined;
  });

  it("merges consecutive tool results into a single user content turn", async () => {
    const model = getModel("google", "gemini-3-flash-preview");

    const context: Context = {
      messages: [
        { role: "tool", toolCallId: "c1", toolName: "add", content: "1", isError: false },
        { role: "tool", toolCallId: "c2", toolName: "add", content: "2", isError: false },
      ],
    };

    const s = stream(model, context, { apiKey: "test" });
    await s.result();

    expect(capturedParams.contents).toHaveLength(1);
    expect(capturedParams.contents[0].role).toBe("user");
    expect(capturedParams.contents[0].parts).toHaveLength(2);
    expect(capturedParams.contents[0].parts[0].functionResponse).toMatchObject({
      id: "c1",
      name: "add",
    });
    expect(capturedParams.contents[0].parts[1].functionResponse).toMatchObject({
      id: "c2",
      name: "add",
    });
  });

  it("does not merge tool results across non-functionResponse user turns", async () => {
    const model = getModel("google", "gemini-3-flash-preview");

    const context: Context = {
      messages: [
        { role: "tool", toolCallId: "c1", toolName: "add", content: "1", isError: false },
        { role: "user", content: "hello" },
        { role: "tool", toolCallId: "c2", toolName: "add", content: "2", isError: false },
      ],
    };

    const s = stream(model, context, { apiKey: "test" });
    await s.result();

    expect(capturedParams.contents).toHaveLength(3);
    expect(capturedParams.contents[0].parts).toHaveLength(1);
    expect(capturedParams.contents[0].parts[0].functionResponse).toMatchObject({ id: "c1" });

    expect(capturedParams.contents[1].parts).toHaveLength(1);
    expect(capturedParams.contents[1].parts[0]).toMatchObject({ text: "hello" });

    expect(capturedParams.contents[2].parts).toHaveLength(1);
    expect(capturedParams.contents[2].parts[0].functionResponse).toMatchObject({ id: "c2" });
  });
});
