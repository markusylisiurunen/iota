import { describe, expect, it } from "vitest";

import { getModel } from "../src/models.js";
import { normalizeContextForTarget } from "../src/stream.js";
import type { Context } from "../src/types.js";

describe("normalizeContextForTarget", () => {
  it("merges system messages into context.system and removes system role messages", () => {
    const target = getModel("openai", "gpt-5.2");

    const context: Context = {
      system: "system A",
      messages: [
        { role: "system", content: "system B" },
        { role: "user", content: "hello" },
      ],
    };

    const normalized = normalizeContextForTarget(target, context);

    expect(normalized.system).toBe("system A\n\nsystem B");
    expect(normalized.messages.some((m) => (m as any).role === "system")).toBe(false);
    expect(normalized.messages).toEqual([{ role: "user", content: "hello" }]);
  });

  it("keeps tool calls and tool results across provider/model boundaries", () => {
    const target = getModel("anthropic", "opus-4.5");

    const context: Context = {
      messages: [
        {
          role: "assistant",
          provider: "openai",
          model: "gpt-5.2",
          content: [
            {
              type: "thinking",
              text: "secret",
              meta: { provider: "openai", type: "reasoning_item", item: {} },
            },
            { type: "tool_call", id: "c1", name: "add", args: { a: 1 } },
          ],
        },
        {
          role: "tool",
          toolCallId: "c1",
          toolName: "add",
          content: '{"sum":3}',
        },
      ],
    };

    const normalized = normalizeContextForTarget(target, context);

    expect(normalized.messages).toHaveLength(2);

    const assistant = normalized.messages[0] as any;
    expect(assistant.role).toBe("assistant");
    expect(assistant.provider).toBeUndefined();
    expect(assistant.model).toBeUndefined();
    expect(assistant.content).toEqual([
      { type: "tool_call", id: "c1", name: "add", args: { a: 1 } },
    ]);

    const tool = normalized.messages[1] as any;
    expect(tool.role).toBe("tool");
    expect(tool.toolCallId).toBe("c1");
    expect(tool.toolName).toBe("add");
    expect(tool.content).toBe('{"sum":3}');
    expect(tool.isError).toBe(false);
  });

  it("normalizes tool results to include isError=false by default", () => {
    const target = getModel("openai", "gpt-5.2");

    const context: Context = {
      messages: [
        {
          role: "assistant",
          provider: "openai",
          model: "gpt-5.2",
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

    const normalized = normalizeContextForTarget(target, context);

    const tool = normalized.messages[1] as any;
    expect(tool.role).toBe("tool");
    expect(tool.isError).toBe(false);
  });
});
