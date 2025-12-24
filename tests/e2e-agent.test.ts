import { randomInt } from "node:crypto";
import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";
import { agent } from "../src/stream.js";
import type {
  AgentStreamEvent,
  AssistantMessage,
  AssistantPart,
  Context,
  Tool,
} from "../src/types.js";

const smokeEnabled = process.env.IOTA_SMOKE === "1";

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

function itIf(condition: boolean, name: string, fn: () => Promise<void>) {
  return (condition ? it.concurrent : it.skip)(name, fn, 30_000);
}

type ToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

type AgentAssistantEvent = Extract<AgentStreamEvent, { type: "assistant_event" }>;

function toolCalls(message: Pick<AssistantMessage, "content">): ToolCallPart[] {
  return message.content.filter((p): p is ToolCallPart => p.type === "tool_call");
}

function textContent(message: Pick<AssistantMessage, "content">) {
  return message.content
    .filter((p): p is Extract<AssistantPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");
}

function parseFirstJsonObject(text: string): unknown {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;

  const candidate = text.slice(start, end + 1);
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

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
    itIf(smokeEnabled && Boolean(process.env[c.env]), c.name, async () => {
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

      expect(events.some((e) => e.type === "turn_start")).toBe(true);
      expect(events.some((e) => e.type === "tool_result")).toBe(true);
      expect(events.filter((e) => e.type === "done").length).toBe(1);
      expect(events.filter((e) => e.type === "error").length).toBe(0);

      const assistantEvents = events.filter(
        (e): e is AgentAssistantEvent => e.type === "assistant_event",
      );
      expect(assistantEvents.some((e) => e.event.type === "start")).toBe(true);
      expect(assistantEvents.some((e) => e.event.type === "done" || e.event.type === "error")).toBe(
        true,
      );

      expect(randoms.length).toBe(2);

      const assistantMessages = result.messages.filter(
        (m): m is Extract<typeof m, { role: "assistant" }> => m.role === "assistant",
      );

      expect(assistantMessages.length).toBeGreaterThanOrEqual(3);

      expect(textContent(assistantMessages[0]).trim()).toBe("");
      expect(toolCalls(assistantMessages[0]).filter((c) => c.name === "random_number").length).toBe(
        2,
      );

      expect(textContent(assistantMessages[1]).trim()).toBe("");
      expect(toolCalls(assistantMessages[1]).filter((c) => c.name === "multiply").length).toBe(1);

      const final = assistantMessages[assistantMessages.length - 1];
      expect(toolCalls(final).length).toBe(0);

      const parsed = parseFirstJsonObject(textContent(final));
      expect(parsed).not.toBeNull();

      const obj = parsed as Record<string, unknown>;
      expect(obj.a).toBe(randoms[0]);
      expect(obj.b).toBe(randoms[1]);
      expect(obj.product).toBe(randoms[0] * randoms[1]);
    });
  }

  itIf(
    smokeEnabled && Boolean(process.env.OPENAI_API_KEY),
    "unknown tool handler is a hard error",
    async () => {
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
      expect(events.filter((e) => e.type === "error").length).toBe(1);
    },
  );
});
