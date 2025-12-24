import { randomInt } from "node:crypto";
import { describe, expect, it } from "vitest";
import type { AnyModel } from "../src/models.js";
import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type {
  AssistantMessage,
  AssistantPart,
  AssistantStreamEvent,
  Context,
  Tool,
  ToolMessage,
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

function toolCalls(message: Pick<AssistantMessage, "content">): ToolCallPart[] {
  return message.content.filter((p): p is ToolCallPart => p.type === "tool_call");
}

function textContent(message: Pick<AssistantMessage, "content">) {
  return message.content
    .filter((p): p is Extract<AssistantPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");
}

async function runTurn(model: AnyModel, context: Context) {
  const s = await stream(model, context, {
    maxTokens: 8192,
    reasoning: "low",
  });

  const events: AssistantStreamEvent[] = [];
  for await (const e of s) events.push(e);
  const message = await s.result();

  expect(events.some((e) => e.type === "start")).toBe(true);
  expect(events.some((e) => e.type === "done" || e.type === "error")).toBe(true);

  return { events, message };
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

function numberFromToolResult(toolMsg: ToolMessage, key: string) {
  const parsed: unknown = JSON.parse(toolMsg.content);
  if (!parsed || typeof parsed !== "object") throw new Error("Unexpected tool result JSON");

  const v = (parsed as Record<string, unknown>)[key];
  expect(typeof v).toBe("number");
  return v as number;
}

async function runScenario(model: AnyModel) {
  const context: Context = {
    system: "You are a deterministic tool-using agent. You must follow the instructions exactly.",
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

  const turn1 = await runTurn(model, context);
  context.messages.push(turn1.message);

  expect(textContent(turn1.message).trim()).toBe("");

  const calls1 = toolCalls(turn1.message);
  expect(calls1.filter((c) => c.name === "random_number").length).toBe(2);
  expect(calls1.filter((c) => c.name === "multiply").length).toBe(0);

  const randomResults: ToolMessage[] = await Promise.all(
    calls1.map(async (call) => {
      expect(call.name).toBe("random_number");

      if (!call.args || typeof call.args !== "object") throw new Error("Unexpected tool args");

      const args = call.args as Record<string, unknown>;
      const minRaw = Number(args.min);
      const maxRaw = Number(args.max);
      expect(Number.isFinite(minRaw)).toBe(true);
      expect(Number.isFinite(maxRaw)).toBe(true);

      const min = Math.trunc(minRaw);
      const max = Math.trunc(maxRaw);
      expect(min).toBe(2);
      expect(max).toBe(9);

      const value = randomInt(min, max + 1);

      return {
        role: "tool",
        toolCallId: call.id,
        toolName: call.name,
        content: JSON.stringify({ value }),
      };
    }),
  );

  context.messages.push(...randomResults);

  const turn2 = await runTurn(model, context);
  context.messages.push(turn2.message);

  expect(textContent(turn2.message).trim()).toBe("");

  const calls2 = toolCalls(turn2.message);
  expect(calls2.length).toBe(1);
  expect(calls2[0].name).toBe("multiply");

  const a = numberFromToolResult(randomResults[0], "value");
  const b = numberFromToolResult(randomResults[1], "value");

  if (!calls2[0].args || typeof calls2[0].args !== "object")
    throw new Error("Unexpected tool args");
  const args2 = calls2[0].args as Record<string, unknown>;

  expect(Number(args2.a)).toBe(a);
  expect(Number(args2.b)).toBe(b);

  const product = a * b;
  const multiplyResult: ToolMessage = {
    role: "tool",
    toolCallId: calls2[0].id,
    toolName: calls2[0].name,
    content: JSON.stringify({ product }),
  };

  context.messages.push(multiplyResult);

  const turn3 = await runTurn(model, context);

  const parsed = parseFirstJsonObject(textContent(turn3.message));
  expect(parsed).not.toBeNull();

  const obj = parsed as Record<string, unknown>;
  expect(obj.a).toBe(a);
  expect(obj.b).toBe(b);
  expect(obj.product).toBe(product);
}

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

describe("tool calling agent loop", () => {
  for (const c of cases) {
    itIf(smokeEnabled && Boolean(process.env[c.env]), c.name, async () => {
      await runScenario(c.model);
    });
  }
});
