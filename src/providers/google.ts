import {
  type Content,
  FinishReason,
  GoogleGenAI,
  type Tool as GoogleTool,
  type HttpOptions,
  type Part,
  type Schema,
  type ThinkingConfig,
  ThinkingLevel,
  type UsageMetadata,
} from "@google/genai";
import type { AssistantStream } from "../assistant-stream.js";
import type { AnyModel } from "../models.js";
import { StreamController } from "../stream-controller.js";
import type {
  AssistantPart,
  NormalizedContext,
  ReasoningEffort,
  ResolvedStreamOptions,
  StopReason,
  Tool,
  Usage,
} from "../types.js";
import { createDebugLogger } from "../utils/debug-log.js";
import { exhaustive } from "../utils/exhaustive.js";
import { sanitizeSurrogates } from "../utils/sanitize.js";

type GoogleModel = Extract<AnyModel, { provider: "google" }>;

const googleHttpOptions: HttpOptions = {
  baseUrl: "https://generativelanguage.googleapis.com",
  apiVersion: "v1beta",
};

export function streamGoogle(
  model: GoogleModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
): AssistantStream {
  const ctrl = new StreamController(model, options);
  ctrl.start();

  (async () => {
    const output = ctrl.output;

    const debug = createDebugLogger("google");

    try {
      const client = new GoogleGenAI({
        apiKey: options.apiKey,
        httpOptions: googleHttpOptions,
      });

      const params = buildParams(model, context, options);
      debug.logRequest(params);
      const googleStream = await client.models.generateContentStream(params);

      let currentIndex: number | null = null;
      let currentType: "text" | "thinking" | null = null;

      const startPart = (type: "text" | "thinking") => {
        const part: AssistantPart =
          type === "text" ? { type: "text", text: "" } : { type: "thinking", text: "" };
        const index = ctrl.addPart(part);
        currentIndex = index;
        currentType = type;
        return index;
      };

      const endCurrent = () => {
        if (currentIndex === null) return;
        ctrl.endPart(currentIndex);
        currentIndex = null;
        currentType = null;
      };

      for await (const chunk of googleStream) {
        debug.logResponseEvent(chunk);
        const candidate = chunk.candidates?.[0];

        if (candidate?.content?.parts) {
          for (const p of candidate.content.parts) {
            if (p.text !== undefined) {
              const isThinking = p.thought === true;
              if (isThinking && (options.reasoning ?? "none") === "none") continue;

              const desired = isThinking ? "thinking" : "text";
              if (currentIndex === null || currentType !== desired) endCurrent();

              const index = currentIndex === null ? startPart(desired) : currentIndex;
              const part = output.content[index];
              const delta = p.text ?? "";
              if (delta.length === 0) continue;

              if (part?.type === "text") {
                part.text += delta;
              } else if (part?.type === "thinking") {
                part.text += delta;
                if (typeof p.thoughtSignature === "string") {
                  part.meta = {
                    provider: "google",
                    type: "thought_signature",
                    signature: p.thoughtSignature,
                  };
                }
              }

              ctrl.delta(index, delta);
            }

            if (p.functionCall) {
              endCurrent();

              const id = resolveToolCallId(p.functionCall.id, p.functionCall.name, output.content);
              const toolCall: Extract<AssistantPart, { type: "tool_call" }> = {
                type: "tool_call",
                id,
                name: p.functionCall.name ?? "",
                args: asRecord(p.functionCall.args),
              };

              if (typeof p.thoughtSignature === "string") {
                toolCall.meta = {
                  provider: "google",
                  type: "thought_signature",
                  signature: p.thoughtSignature,
                };
              }

              const index = ctrl.addPart(toolCall);
              ctrl.delta(index, JSON.stringify(toolCall.args));
              ctrl.endPart(index);
            }
          }
        }

        if (candidate?.finishReason) {
          ctrl.setStopReason(mapStopReason(candidate.finishReason));
        }

        if (chunk.usageMetadata) {
          ctrl.setUsage(usageFromGoogle(chunk.usageMetadata));
        }
      }

      endCurrent();
      ctrl.finish();
    } catch (error) {
      ctrl.fail(error);
    } finally {
      await debug.flushResponse();
    }
  })();

  return ctrl.stream;
}

function buildParams(
  model: GoogleModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
) {
  const contents = convertMessages(context);

  const config: Record<string, unknown> = {};

  if (options.temperature !== undefined) config.temperature = options.temperature;
  if (options.maxTokens !== undefined) config.maxOutputTokens = options.maxTokens;

  if (context.system && context.system.trim().length > 0) {
    config.systemInstruction = sanitizeSurrogates(context.system);
  }

  if (context.tools && context.tools.length > 0) {
    config.tools = convertTools(context.tools);
  }

  const reasoning = options.reasoning ?? "none";
  if (model.supports.reasoning) {
    const thinkingConfig: ThinkingConfig = {};

    if (reasoning === "none") {
      thinkingConfig.includeThoughts = false;

      if (model.id.includes("3-flash")) {
        thinkingConfig.thinkingBudget = 0;
      } else if (model.id.includes("3-pro")) {
        thinkingConfig.thinkingLevel = ThinkingLevel.LOW;
      } else {
        thinkingConfig.thinkingBudget = 0;
      }
    } else {
      thinkingConfig.includeThoughts = true;

      const effort: Exclude<ReasoningEffort, "none" | "xhigh"> =
        reasoning === "xhigh" ? "high" : reasoning;
      const thinkingLevel = googleThinkingLevel(model.id, effort);
      if (thinkingLevel) thinkingConfig.thinkingLevel = thinkingLevel;
    }

    config.thinkingConfig = thinkingConfig;
  }

  if (options.signal) {
    if (options.signal.aborted) throw new Error("Request aborted");
    config.abortSignal = options.signal;
  }

  return {
    model: model.id,
    contents,
    config,
  };
}

function convertMessages(context: NormalizedContext): Content[] {
  const contents: Content[] = [];

  for (const msg of context.messages) {
    switch (msg.role) {
      case "user": {
        if (msg.content.trim().length === 0) continue;
        contents.push({
          role: "user",
          parts: [{ text: sanitizeSurrogates(msg.content) }],
        });
        continue;
      }

      case "assistant": {
        const outParts: Part[] = [];
        for (const part of msg.content) {
          switch (part.type) {
            case "text": {
              if (part.text.trim().length === 0) continue;
              outParts.push({ text: sanitizeSurrogates(part.text) });
              continue;
            }

            case "thinking": {
              if (part.text.trim().length === 0) continue;
              if (part.meta?.provider !== "google" || part.meta.type !== "thought_signature") {
                continue;
              }

              outParts.push({
                thought: true,
                text: sanitizeSurrogates(part.text),
                thoughtSignature: part.meta.signature,
              });
              continue;
            }

            case "tool_call": {
              const p: Part = {
                functionCall: {
                  id: part.id,
                  name: part.name,
                  args: asRecord(part.args),
                },
              };

              if (part.meta?.provider === "google" && part.meta.type === "thought_signature") {
                p.thoughtSignature = part.meta.signature;
              }

              outParts.push(p);
              continue;
            }

            default:
              return exhaustive(part);
          }
        }

        if (outParts.length > 0) contents.push({ role: "model", parts: outParts });
        continue;
      }

      case "tool": {
        const part: Part = {
          functionResponse: {
            id: msg.toolCallId,
            name: msg.toolName,
            response: msg.isError
              ? { error: sanitizeSurrogates(msg.content) }
              : { output: sanitizeSurrogates(msg.content) },
          },
        };

        const last = contents.at(-1);
        if (
          last?.role === "user" &&
          Array.isArray(last.parts) &&
          last.parts.some((p) => p.functionResponse)
        ) {
          last.parts.push(part);
        } else {
          contents.push({ role: "user", parts: [part] });
        }

        continue;
      }

      default:
        return exhaustive(msg);
    }
  }

  return contents;
}

function convertTools(tools: Tool[]): GoogleTool[] | undefined {
  if (tools.length === 0) return undefined;

  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.parameters as Schema,
      })),
    },
  ];
}

function resolveToolCallId(
  providedId: string | undefined,
  name: string | undefined,
  parts: AssistantPart[],
): string {
  const existing = new Set(parts.filter((p) => p.type === "tool_call").map((p) => p.id));
  if (providedId && !existing.has(providedId)) return providedId;

  const base = name && name.trim().length > 0 ? name : "tool";
  let id = `${base}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  while (existing.has(id)) id = `${base}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  return id;
}

function googleThinkingLevel(modelId: string, effort: Exclude<ReasoningEffort, "none" | "xhigh">) {
  if (modelId.includes("3-pro")) {
    switch (effort) {
      case "minimal":
      case "low":
        return ThinkingLevel.LOW;
      case "medium":
      case "high":
        return ThinkingLevel.HIGH;
      default:
        return exhaustive(effort);
    }
  }

  switch (effort) {
    case "minimal":
      return ThinkingLevel.MINIMAL;
    case "low":
      return ThinkingLevel.LOW;
    case "medium":
      return ThinkingLevel.MEDIUM;
    case "high":
      return ThinkingLevel.HIGH;
    default:
      return exhaustive(effort);
  }
}

function mapStopReason(reason: FinishReason | string): StopReason {
  const r = typeof reason === "string" ? reason : reason;

  switch (r) {
    case FinishReason.STOP:
    case "STOP":
      return "stop";
    case FinishReason.MAX_TOKENS:
    case "MAX_TOKENS":
      return "length";
    default:
      return "error";
  }
}

function usageFromGoogle(u: UsageMetadata): Usage {
  const cached = u.cachedContentTokenCount || 0;
  const toolUsePrompt = u.toolUsePromptTokenCount || 0;
  const input = Math.max(0, (u.promptTokenCount || 0) - cached) + toolUsePrompt;
  const output = (u.responseTokenCount || 0) + (u.thoughtsTokenCount || 0);

  return {
    inputTokens: input,
    outputTokens: output,
    cacheReadTokens: cached,
    cacheWriteTokens: 0,
    totalTokens: u.totalTokenCount || input + output + cached,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as Record<string, unknown>;
}
