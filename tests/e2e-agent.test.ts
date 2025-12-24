import { randomInt } from "node:crypto";
import { describe, expect } from "vitest";

import { getModel } from "../src/models.js";
import { agent } from "../src/stream.js";
import type { AgentStreamEvent, AssistantPart, Context, Tool } from "../src/types.js";

import {
  assistantText,
  assistantToolCalls,
  expectAgentLifecycle,
  expectNoToolCalls,
  expectNoVisibleText,
  expectToolCalls,
  itIfSmokeAndEnv,
  parseJsonObjectFromText,
} from "./e2e-utils.js";

const tools: Tool[] = [
  {
    name: "random_number",
    description: "Generate a random integer in [min, max] inclusive.",
    parameters: {
      type: "object",
      properties: {
        min: { type: "integer" },
        max: { type: "integer" },
      },
      required: ["min", "max"],
      additionalProperties: false,
    },
  },
  {
    name: "multiply",
    description: "Multiply two numbers.",
    parameters: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
      additionalProperties: false,
    },
  },
];

type ToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

describe("agent() tool calling loop", () => {
  const cases = [
    {
      name: "openai/gpt-5.2",
      model: getModel("openai", "gpt-5.2"),
      env: "OPENAI_API_KEY",
    },
    {
      name: "anthropic/haiku-4.5",
      model: getModel("anthropic", "haiku-4.5"),
      env: "ANTHROPIC_API_KEY",
    },
    {
      name: "gemini/gemini-3-flash-preview",
      model: getModel("gemini", "gemini-3-flash-preview"),
      env: "GEMINI_API_KEY",
    },
  ] as const;

  for (const c of cases) {
    itIfSmokeAndEnv(c.env, c.name, async () => {
      const context: Context = {
        system:
          "You are a deterministic tool-using agent. You must follow the instructions exactly.",
        tools,
        messages: [
          {
            role: "user",
            content:
              "Do the following steps strictly:\n" +
              "1) In your first assistant turn, call the tool random_number twice (two separate tool calls in parallel) with {min: 2, max: 9}. Do not output any text in that turn.\n" +
              "2) After receiving both tool results, call the tool multiply exactly once with {a: <first random>, b: <second random>}. Do not output any text in that turn.\n" +
              '3) After receiving the multiply tool result, output ONLY a JSON object: {"a": number, "b": number, "product": number}.',
          },
        ],
      };

      const randoms: number[] = [];

      const s = agent(
        c.model,
        context,
        {
          random_number: async (args) => {
            if (!args || typeof args !== "object") throw new Error("Unexpected tool args");

            const record = args as Record<string, unknown>;
            const min = Math.trunc(Number(record.min));
            const max = Math.trunc(Number(record.max));

            expect(min).toBe(2);
            expect(max).toBe(9);

            const value = randomInt(min, max + 1);
            randoms.push(value);

            return JSON.stringify({ value });
          },
          multiply: async (args) => {
            if (!args || typeof args !== "object") throw new Error("Unexpected tool args");

            const record = args as Record<string, unknown>;
            const a = Number(record.a);
            const b = Number(record.b);

            expect(a).toBe(randoms[0]);
            expect(b).toBe(randoms[1]);

            return JSON.stringify({ product: a * b });
          },
        },
        {
          maxTokens: 8192,
          reasoning: "low",
        },
      );

      const events: AgentStreamEvent[] = [];
      for await (const e of s) events.push(e);

      const result = await s.resultOrThrow();
      expectAgentLifecycle(events);

      expect(randoms).toHaveLength(2);

      const assistantMessages = result.messages.filter(
        (m): m is Extract<typeof m, { role: "assistant" }> => m.role === "assistant",
      );

      expect(assistantMessages.length).toBeGreaterThanOrEqual(3);

      const t1 = assistantMessages[0];
      expectNoVisibleText(t1);
      expectToolCalls(t1, { random_number: 2 });

      const t2 = assistantMessages[1];
      expectNoVisibleText(t2);
      expectToolCalls(t2, { multiply: 1 });

      const final = assistantMessages[assistantMessages.length - 1];
      expectNoToolCalls(final);

      const obj = parseJsonObjectFromText(assistantText(final));
      expect(obj.a).toBe(randoms[0]);
      expect(obj.b).toBe(randoms[1]);
      expect(obj.product).toBe(randoms[0] * randoms[1]);

      // Sanity: intermediate messages are tool-only.
      const intermediateCalls = assistantMessages
        .slice(0, -1)
        .flatMap((m) => assistantToolCalls(m));
      expect(
        intermediateCalls.every(
          (p: ToolCallPart) => p.name === "random_number" || p.name === "multiply",
        ),
      ).toBe(true);
    });
  }

  itIfSmokeAndEnv("OPENAI_API_KEY", "unknown tool handler is a hard error", async () => {
    const s = agent(
      getModel("openai", "gpt-5.2"),
      {
        system: "You must call the provided tool.",
        tools: [
          {
            name: "random_number",
            parameters: {
              type: "object",
              properties: { min: { type: "integer" }, max: { type: "integer" } },
              required: ["min", "max"],
              additionalProperties: false,
            },
          },
        ],
        messages: [
          {
            role: "user",
            content: "Call the tool random_number with {min: 1, max: 1}. Do not output any text.",
          },
        ],
      },
      {},
      {
        maxTurns: 1,
      },
    );

    const events: AgentStreamEvent[] = [];
    for await (const e of s) events.push(e);

    const result = await s.result();
    if (result.stopReason !== "error") {
      throw new Error(`Expected stopReason error, got ${result.stopReason}`);
    }

    expect(result.errorMessage?.includes("Unknown tool")).toBe(true);
    expect(result.messages.some((m) => m.role === "assistant")).toBe(true);
    expect(events.filter((e) => e.type === "error")).toHaveLength(1);
  });
});
