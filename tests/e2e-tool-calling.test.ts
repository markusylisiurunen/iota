import { randomInt } from "node:crypto";
import { describe, expect } from "vitest";

import type { AnyModel } from "../src/models.js";
import { getModel } from "../src/models.js";
import type { Context, Tool, ToolMessage } from "../src/types.js";

import {
  expectIntField,
  expectNoToolCalls,
  expectNoVisibleText,
  expectToolCallArgs,
  expectToolCalls,
  itIf,
  numberFromToolResult,
  parseJsonObjectFromMessage,
  runTurn,
  smokeEnabled,
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

function itIfSmokeAndEnv(envVar: string, name: string, fn: () => Promise<void>) {
  return itIf(smokeEnabled && Boolean(process.env[envVar]), name, fn);
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

  const turn1 = await runTurn(model, context, { reasoning: "low" });
  context.messages.push(turn1.message);

  expectNoVisibleText(turn1.message);
  const calls1 = expectToolCalls(turn1.message, { random_number: 2 });

  const randomResults: ToolMessage[] = calls1.map((call) => {
    expect(call.name).toBe("random_number");

    const args = expectToolCallArgs(call);
    const min = expectIntField(args, "min");
    const max = expectIntField(args, "max");

    expect(min).toBe(2);
    expect(max).toBe(9);

    const value = randomInt(min, max + 1);

    return {
      role: "tool",
      toolCallId: call.id,
      toolName: call.name,
      content: JSON.stringify({ value }),
    };
  });

  context.messages.push(...randomResults);

  const turn2 = await runTurn(model, context, { reasoning: "low" });
  context.messages.push(turn2.message);

  expectNoVisibleText(turn2.message);

  const calls2 = expectToolCalls(turn2.message, { multiply: 1 });
  const multiplyCall = calls2[0];

  const a = numberFromToolResult(randomResults[0], "value");
  const b = numberFromToolResult(randomResults[1], "value");

  const args2 = expectToolCallArgs(multiplyCall);
  expect(Number(args2.a)).toBe(a);
  expect(Number(args2.b)).toBe(b);

  const product = a * b;
  const multiplyResult: ToolMessage = {
    role: "tool",
    toolCallId: multiplyCall.id,
    toolName: multiplyCall.name,
    content: JSON.stringify({ product }),
  };

  context.messages.push(multiplyResult);

  const turn3 = await runTurn(model, context, { reasoning: "low" });

  expectNoToolCalls(turn3.message);

  const obj = parseJsonObjectFromMessage(turn3.message);
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
    itIfSmokeAndEnv(c.env, c.name, async () => {
      await runScenario(c.model);
    });
  }
});
