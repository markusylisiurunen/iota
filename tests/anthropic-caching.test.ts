import { beforeEach, describe, expect, it, vi } from "vitest";

let capturedParams: any;
let capturedClientOptions: any;

vi.mock("@anthropic-ai/sdk", () => {
  class AnthropicMock {
    messages: {
      stream: (params: any) => AsyncIterable<unknown>;
    };

    constructor(options: any) {
      capturedClientOptions = options;
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

describe("anthropic prompt caching", () => {
  beforeEach(() => {
    capturedParams = undefined;
    capturedClientOptions = undefined;
  });

  it("adds cache_control to the system prompt", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      system: "system",
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test" });
    await s.result();

    expect(Array.isArray(capturedParams.system)).toBe(true);
    expect(capturedParams.system[0]).toMatchObject({
      type: "text",
      text: "system",
      cache_control: { type: "ephemeral" },
    });

    expect(capturedClientOptions.defaultHeaders).toMatchObject({
      accept: "application/json",
    });
  });

  it("adds cache_control to the last user content block", async () => {
    const model = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [
        {
          role: "assistant",
          content: [{ type: "tool_call", id: "c1", name: "add", args: { a: 1 } }],
        },
        {
          role: "tool",
          toolCallId: "c1",
          toolName: "add",
          content: '{"sum":3}',
        },
      ],
    };

    const s = stream(model, context, { apiKey: "test" });
    await s.result();

    const last = capturedParams.messages.at(-1);
    expect(last.role).toBe("user");
    expect(Array.isArray(last.content)).toBe(true);

    const lastBlock = last.content.at(-1);
    expect(lastBlock).toMatchObject({
      type: "tool_result",
      tool_use_id: "c1",
      cache_control: { type: "ephemeral" },
    });
  });
});
