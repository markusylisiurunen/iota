import { randomInt } from "node:crypto";
import { describe, expect, it } from "vitest";
import { getModel } from "../src/models.js";

const smokeEnabled = process.env.IOTA_SMOKE === "1";

let streamFn: undefined | ((model: any, context: any, options?: any) => any);

async function stream(model: any, context: any, options?: any) {
  if (!streamFn) {
    const mod = await import("../dist/index.js");
    streamFn = mod.stream;
  }
  return streamFn(model, context, options);
}

const tools = [
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
] as const;

function itIf(condition: boolean, name: string, fn: () => Promise<void>) {
  return (condition ? it.concurrent : it.skip)(name, fn, 30_000);
}

function toolCalls(message: { content: Array<{ type: string }> }) {
  return (message.content as any[]).filter((p) => p.type === "tool_call");
}

function textContent(message: { content: Array<{ type: string; text?: string }> }) {
  return message.content
    .filter((p) => p.type === "text")
    .map((p) => p.text ?? "")
    .join("");
}

async function runTurn(model: any, context: any) {
  const s = await stream(model, context, {
    maxTokens: 8192,
    reasoning: "low",
  });

  const events: any[] = [];
  for await (const e of s) events.push(e);
  const message = await s.result();

  expect(events.some((e) => e.type === "start")).toBe(true);
  expect(events.some((e) => e.type === "done" || e.type === "error")).toBe(true);

  return { events, message };
}

function parseFirstJsonObject(text: string) {
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

function numberFromToolResult(toolMsg: { content: string }, key: string) {
  const parsed = JSON.parse(toolMsg.content);
  const v = parsed?.[key];
  expect(typeof v).toBe("number");
  return v as number;
}

async function runScenario(model: any) {
  const context: any = {
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

  const randomResults = await Promise.all(
    calls1.map(async (call) => {
      expect(call.name).toBe("random_number");

      const minRaw = Number(call.args?.min);
      const maxRaw = Number(call.args?.max);
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

  expect(Number(calls2[0].args?.a)).toBe(a);
  expect(Number(calls2[0].args?.b)).toBe(b);

  const product = a * b;
  const multiplyResult = {
    role: "tool",
    toolCallId: calls2[0].id,
    toolName: calls2[0].name,
    content: JSON.stringify({ product }),
  };

  context.messages.push(multiplyResult);

  const turn3 = await runTurn(model, context);

  const parsed = parseFirstJsonObject(textContent(turn3.message));
  expect(parsed).not.toBeNull();
  expect(parsed.a).toBe(a);
  expect(parsed.b).toBe(b);
  expect(parsed.product).toBe(product);
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
