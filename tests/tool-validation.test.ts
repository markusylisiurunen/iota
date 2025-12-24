import { describe, expect, it, vi } from "vitest";

vi.mock("../src/providers/openai.js", async () => {
  const { AssistantStream } = await import("../src/assistant-stream.js");

  return {
    streamOpenAI: (model: { provider: "openai"; id: string }) => {
      const s = new AssistantStream();
      queueMicrotask(() => {
        s.push({
          type: "done",
          message: {
            role: "assistant",
            provider: model.provider,
            model: model.id,
            content: [],
            stopReason: "stop",
            usage: {
              inputTokens: 0,
              outputTokens: 0,
              cacheReadTokens: 0,
              cacheWriteTokens: 0,
              totalTokens: 0,
              cost: {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
                total: 0,
              },
            },
          },
        });
      });
      return s;
    },
  };
});

import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";

describe("tool validation", () => {
  it("rejects duplicate tool names", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/duplicate tool name/);
  });

  it("rejects tool names that are not provider-safe", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "not ok",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/tool\.name/);
  });

  it("rejects required keys not present in properties", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {
                  a: { type: "number" },
                },
                required: ["a", "b"],
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/required/);
  });

  it("accepts minimum/maximum for number schemas", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "set_volume",
              parameters: {
                type: "object",
                properties: {
                  volume: { type: "number", minimum: 0, maximum: 10 },
                },
                required: ["volume"],
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).not.toThrow();
  });

  it("rejects minimum/maximum for non-numeric schemas", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "bad",
              parameters: {
                type: "object",
                properties: {
                  s: { type: "string", minimum: 0 },
                },
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/minimum: only valid when type is 'number' or 'integer'/);
  });

  it("rejects non-finite minimum/maximum values", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "bad",
              parameters: {
                type: "object",
                properties: {
                  n: { type: "number", minimum: Number.NaN },
                },
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/minimum: expected a finite number/);
  });

  it("rejects minimum greater than maximum", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "bad",
              parameters: {
                type: "object",
                properties: {
                  n: { type: "number", minimum: 10, maximum: 0 },
                },
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/minimum \(10\) cannot be greater than maximum \(0\)/);
  });
});
