import { expect, it } from "vitest";
import type { AnyModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type {
  AgentStreamEvent,
  AssistantMessage,
  AssistantPart,
  AssistantStreamEvent,
  Context,
  StreamOptions,
  ToolMessage,
} from "../src/types.js";

export const smokeEnabled = process.env.IOTA_SMOKE === "1";

export function itIf(condition: boolean, name: string, fn: () => Promise<void>) {
  return (condition ? it.concurrent : it.skip)(name, fn, 30_000);
}

export function itIfSmokeAndEnv(envVar: string, name: string, fn: () => Promise<void>) {
  return itIf(smokeEnabled && Boolean(process.env[envVar]), name, fn);
}

type TextPart = Extract<AssistantPart, { type: "text" }>;
export type ToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;
type ThinkingPart = Extract<AssistantPart, { type: "thinking" }>;

export function assistantText(message: Pick<AssistantMessage, "content">) {
  return message.content
    .filter((p): p is TextPart => p.type === "text")
    .map((p) => p.text)
    .join("");
}

export function assistantToolCalls(message: Pick<AssistantMessage, "content">): ToolCallPart[] {
  return message.content.filter((p): p is ToolCallPart => p.type === "tool_call");
}

export function assistantThinkingParts(message: Pick<AssistantMessage, "content">): ThinkingPart[] {
  return message.content.filter((p): p is ThinkingPart => p.type === "thinking");
}

export function significantParts(message: Pick<AssistantMessage, "content">) {
  return message.content.filter((p) => {
    if (p.type !== "text") return true;
    return p.text.trim().length > 0;
  });
}

export function expectNoVisibleText(message: Pick<AssistantMessage, "content">) {
  expect(assistantText(message).trim()).toBe("");
}

export function expectToolCalls(
  message: Pick<AssistantMessage, "content">,
  expected: Record<string, number>,
) {
  const calls = assistantToolCalls(message);

  const counts = new Map<string, number>();
  for (const call of calls) counts.set(call.name, (counts.get(call.name) ?? 0) + 1);

  for (const [name, expectedCount] of Object.entries(expected)) {
    expect(counts.get(name) ?? 0).toBe(expectedCount);
  }

  const expectedTotal = Object.values(expected).reduce((sum, n) => sum + n, 0);
  expect(calls.length).toBe(expectedTotal);

  return calls;
}

export function expectNoToolCalls(message: Pick<AssistantMessage, "content">) {
  expect(assistantToolCalls(message)).toHaveLength(0);
}

export function expectThinkingSignaturesAnthropic(message: Pick<AssistantMessage, "content">) {
  const thinking = assistantThinkingParts(message);
  expect(thinking.length).toBeGreaterThan(0);
  expect(
    thinking.every(
      (p) =>
        p.meta?.provider === "anthropic" &&
        p.meta.type === "thinking_signature" &&
        p.meta.signature.trim().length > 0,
    ),
  ).toBe(true);
}

export function expectThinkingThenToolCalls(
  message: Pick<AssistantMessage, "content">,
  expected: Record<string, number>,
) {
  const sig = significantParts(message);
  const firstToolCallIndex = sig.findIndex((p) => p.type === "tool_call");
  expect(firstToolCallIndex).toBeGreaterThan(0);

  const pre = sig.slice(0, firstToolCallIndex);
  expect(pre.length).toBeGreaterThan(0);
  expect(pre.every((p) => p.type === "thinking")).toBe(true);

  const calls = expectToolCalls(message, expected);

  const lastToolCallIndex = sig.findLastIndex((p) => p.type === "tool_call");
  const post = sig.slice(lastToolCallIndex + 1);
  expect(post.every((p) => p.type !== "tool_call")).toBe(true);

  return calls;
}

export function expectThinkingThenText(message: Pick<AssistantMessage, "content">) {
  const sig = significantParts(message);
  const firstTextIndex = sig.findIndex((p) => p.type === "text");
  expect(firstTextIndex).toBeGreaterThan(0);

  const pre = sig.slice(0, firstTextIndex);
  expect(pre.length).toBeGreaterThan(0);
  expect(pre.every((p) => p.type === "thinking")).toBe(true);

  const post = sig.slice(firstTextIndex);
  expect(post.length).toBeGreaterThan(0);
  expect(post.every((p) => p.type === "text")).toBe(true);
}

export function parseJsonObjectFromText(text: string): Record<string, unknown> {
  const trimmed = text.trim();

  const tryParse = (candidate: string) => {
    try {
      return JSON.parse(candidate) as unknown;
    } catch {
      return null;
    }
  };

  const extracted = (() => {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start === -1 || end === -1 || end <= start) return null;
    return trimmed.slice(start, end + 1);
  })();

  const parsed = tryParse(trimmed) ?? (extracted ? tryParse(extracted) : null);

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Expected a JSON object, got: ${trimmed}`);
  }

  return parsed as Record<string, unknown>;
}

export function parseJsonObjectFromMessage(message: Pick<AssistantMessage, "content">) {
  return parseJsonObjectFromText(assistantText(message));
}

export function expectPlainObject(value: unknown, _label: string): Record<string, unknown> {
  expect(value).toBeTruthy();
  expect(typeof value).toBe("object");
  expect(Array.isArray(value)).toBe(false);
  return value as Record<string, unknown>;
}

export function expectToolCallArgs(call: ToolCallPart) {
  return expectPlainObject(call.args, `tool call args for ${call.name}`);
}

export function expectNumberField(record: Record<string, unknown>, key: string) {
  const n = Number(record[key]);
  expect(Number.isFinite(n)).toBe(true);
  return n;
}

export function expectIntField(record: Record<string, unknown>, key: string) {
  const n = expectNumberField(record, key);
  return Math.trunc(n);
}

export function numberFromToolResult(toolMsg: ToolMessage, key: string) {
  const parsed = parseJsonObjectFromText(toolMsg.content);
  return expectNumberField(parsed, key);
}

export async function runTurn(
  model: AnyModel,
  context: Context,
  options?: { reasoning?: StreamOptions["reasoning"] },
) {
  const s = stream(model, context, {
    maxTokens: 8192,
    reasoning: options?.reasoning ?? "low",
  });

  const events: AssistantStreamEvent[] = [];
  for await (const e of s) events.push(e);

  const message = await s.result();

  expect(events.some((e) => e.type === "start")).toBe(true);
  expect(events.some((e) => e.type === "done" || e.type === "error")).toBe(true);

  return { events, message };
}

type AgentAssistantEvent = Extract<AgentStreamEvent, { type: "assistant_event" }>;

export function expectAgentLifecycle(events: AgentStreamEvent[]) {
  expect(events.some((e) => e.type === "turn_start")).toBe(true);
  expect(events.some((e) => e.type === "tool_result")).toBe(true);
  expect(events.filter((e) => e.type === "done")).toHaveLength(1);
  expect(events.filter((e) => e.type === "error")).toHaveLength(0);

  const assistantEvents = events.filter(
    (e): e is AgentAssistantEvent => e.type === "assistant_event",
  );
  expect(assistantEvents.some((e) => e.event.type === "start")).toBe(true);
  expect(assistantEvents.some((e) => e.event.type === "done" || e.event.type === "error")).toBe(
    true,
  );
}
