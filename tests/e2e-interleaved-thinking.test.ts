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

function itIf(condition: boolean, name: string, fn: () => Promise<void>) {
  return (condition ? it.concurrent : it.skip)(name, fn, 30_000);
}

type ToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

type ThinkingPart = Extract<AssistantPart, { type: "thinking" }>;

function toolCalls(message: Pick<AssistantMessage, "content">): ToolCallPart[] {
  return message.content.filter((p): p is ToolCallPart => p.type === "tool_call");
}

function thinkingParts(message: Pick<AssistantMessage, "content">): ThinkingPart[] {
  return message.content.filter((p): p is ThinkingPart => p.type === "thinking");
}

function textContent(message: Pick<AssistantMessage, "content">) {
  return message.content
    .filter((p): p is Extract<AssistantPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");
}

function significantParts(message: Pick<AssistantMessage, "content">) {
  return message.content.filter((p) => {
    if (p.type !== "text") return true;
    return p.text.trim().length > 0;
  });
}

async function runTurn(model: AnyModel, context: Context) {
  const s = stream(model, context, {
    maxTokens: 8192,
    reasoning: "high",
  });

  const events: AssistantStreamEvent[] = [];
  for await (const e of s) events.push(e);
  const message = await s.result();

  expect(events.some((e) => e.type === "start")).toBe(true);
  expect(events.some((e) => e.type === "done" || e.type === "error")).toBe(true);

  return { events, message };
}

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
];

describe("interleaved thinking", () => {
  const model = getModel("anthropic", "haiku-4.5");

  itIf(smokeEnabled && Boolean(process.env.ANTHROPIC_API_KEY), "anthropic/haiku-4.5", async () => {
    const a = 3;
    const b = 7;
    const sum = a + b;

    const context: Context = {
      system: "You are a deterministic tool-using agent. Follow instructions exactly.",
      tools,
      messages: [
        {
          role: "user",
          content:
            "Follow these steps strictly:\n" +
            "1) In your first assistant turn, think about the task, then call the tool random_number twice (in parallel) with {min: 2, max: 9}. Do not output any user-visible text in that turn.\n" +
            "2) After receiving both tool results, think about the sum, then output ONLY the sum as digits.",
        },
      ],
    };

    const turn1 = await runTurn(model, context);
    context.messages.push(turn1.message);

    expect(textContent(turn1.message).trim()).toBe("");

    const sig1 = significantParts(turn1.message);
    const firstToolCallIndex = sig1.findIndex((p) => p.type === "tool_call");
    expect(firstToolCallIndex).toBeGreaterThan(0);

    const preTools = sig1.slice(0, firstToolCallIndex);
    expect(preTools.every((p) => p.type === "thinking")).toBe(true);

    expect(sig1[firstToolCallIndex]?.type).toBe("tool_call");
    expect(sig1[firstToolCallIndex + 1]?.type).toBe("tool_call");
    expect(sig1.slice(firstToolCallIndex + 2).every((p) => p.type !== "tool_call")).toBe(true);

    const calls1 = toolCalls(turn1.message);
    expect(calls1.length).toBe(2);
    expect(calls1.every((c) => c.name === "random_number")).toBe(true);

    const t1Thinking = thinkingParts(turn1.message);
    expect(t1Thinking.length).toBeGreaterThan(0);
    expect(
      t1Thinking.every(
        (p) =>
          p.meta?.provider === "anthropic" &&
          p.meta.type === "thinking_signature" &&
          p.meta.signature.trim().length > 0,
      ),
    ).toBe(true);

    const toolResults: ToolMessage[] = calls1.map((call, i) => ({
      role: "tool",
      toolCallId: call.id,
      toolName: call.name,
      content: JSON.stringify({ value: i === 0 ? a : b }),
    }));

    context.messages.push(...toolResults);

    const turn2 = await runTurn(model, context);

    expect(toolCalls(turn2.message).length).toBe(0);

    const sig2 = significantParts(turn2.message);
    const firstTextIndex = sig2.findIndex((p) => p.type === "text");
    expect(firstTextIndex).toBeGreaterThanOrEqual(0);

    const preText = sig2.slice(0, firstTextIndex);
    expect(preText.length).toBeGreaterThan(0);
    expect(preText.every((p) => p.type === "thinking")).toBe(true);

    const postText = sig2.slice(firstTextIndex);
    expect(postText.length).toBeGreaterThan(0);
    expect(postText.every((p) => p.type === "text")).toBe(true);

    expect(textContent(turn2.message).trim()).toBe(String(sum));

    const t2Thinking = thinkingParts(turn2.message);
    expect(t2Thinking.length).toBeGreaterThan(0);
    expect(
      t2Thinking.every(
        (p) =>
          p.meta?.provider === "anthropic" &&
          p.meta.type === "thinking_signature" &&
          p.meta.signature.trim().length > 0,
      ),
    ).toBe(true);
  });
});
