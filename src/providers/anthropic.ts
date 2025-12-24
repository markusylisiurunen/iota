import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  MessageCreateParamsStreaming,
  MessageParam,
} from "@anthropic-ai/sdk/resources/messages.js";
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
import { exhaustive } from "../utils/exhaustive.js";
import { parseStreamingJson } from "../utils/json.js";
import { sanitizeSurrogates } from "../utils/sanitize.js";

type AnthropicModel = Extract<AnyModel, { provider: "anthropic" }>;

type AnthropicToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

type AnthropicPart = AssistantPart;

export function streamAnthropic(
  model: AnthropicModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
): AssistantStream {
  const ctrl = new StreamController(model, options);
  ctrl.start();

  (async () => {
    const output = ctrl.output;

    const contentIndexByBlockIndex = new Map<number, number>();
    const toolCallPartialJsonByIndex = new Map<number, string>();
    const thinkingSignatureByIndex = new Map<number, string>();

    try {
      const client = createClient(options.apiKey, options.reasoning !== "none");
      const params = buildParams(model, context, options);
      const anthropicStream = client.messages.stream(params, {
        signal: options.signal,
      });

      for await (const event of anthropicStream) {
        if (event.type === "message_start") {
          ctrl.setUsage(
            usageFromAnthropic({
              inputTokens: event.message.usage.input_tokens || 0,
              outputTokens: event.message.usage.output_tokens || 0,
              cacheReadTokens: event.message.usage.cache_read_input_tokens || 0,
              cacheWriteTokens: event.message.usage.cache_creation_input_tokens || 0,
            }),
          );
        } else if (event.type === "content_block_start") {
          if (event.content_block.type === "text") {
            const part: AnthropicPart = { type: "text", text: "" };
            const idx = ctrl.addPart(part);
            contentIndexByBlockIndex.set(event.index, idx);
          } else if (event.content_block.type === "thinking") {
            const part: AnthropicPart = { type: "thinking", text: "" };
            const idx = ctrl.addPart(part);
            contentIndexByBlockIndex.set(event.index, idx);
          } else if (event.content_block.type === "tool_use") {
            const part: AnthropicToolCallPart = {
              type: "tool_call",
              id: event.content_block.id,
              name: event.content_block.name,
              args: {},
            };
            const idx = ctrl.addPart(part);
            contentIndexByBlockIndex.set(event.index, idx);
            toolCallPartialJsonByIndex.set(idx, "");
          }
        } else if (event.type === "content_block_delta") {
          const idx = contentIndexByBlockIndex.get(event.index);
          if (idx === undefined) continue;

          const part = output.content[idx] as AnthropicPart | undefined;
          if (!part) continue;

          if (event.delta.type === "text_delta" && part.type === "text") {
            part.text += event.delta.text;
            ctrl.delta(idx, event.delta.text);
          } else if (event.delta.type === "thinking_delta" && part.type === "thinking") {
            part.text += event.delta.thinking;
            ctrl.delta(idx, event.delta.thinking);
          } else if (event.delta.type === "input_json_delta" && part.type === "tool_call") {
            const current = toolCallPartialJsonByIndex.get(idx) ?? "";
            const next = current + event.delta.partial_json;
            toolCallPartialJsonByIndex.set(idx, next);

            ctrl.delta(idx, event.delta.partial_json);
          } else if (event.delta.type === "signature_delta" && part.type === "thinking") {
            const current = thinkingSignatureByIndex.get(idx) ?? "";
            const next = current + event.delta.signature;
            thinkingSignatureByIndex.set(idx, next);

            part.meta = { provider: "anthropic", type: "thinking_signature", signature: next };
          }
        } else if (event.type === "content_block_stop") {
          const idx = contentIndexByBlockIndex.get(event.index);
          if (idx === undefined) continue;

          const part = output.content[idx] as AnthropicPart | undefined;
          if (!part) continue;

          if (part.type === "tool_call") {
            const json = toolCallPartialJsonByIndex.get(idx);
            if (json !== undefined) {
              part.args = parseStreamingJson(json);
              toolCallPartialJsonByIndex.delete(idx);
            }
          }

          ctrl.endPart(idx);
        } else if (event.type === "message_delta") {
          if (event.delta.stop_reason) ctrl.setStopReason(mapStopReason(event.delta.stop_reason));
          ctrl.setUsage(
            usageFromAnthropic({
              inputTokens: event.usage.input_tokens || 0,
              outputTokens: event.usage.output_tokens || 0,
              cacheReadTokens: event.usage.cache_read_input_tokens || 0,
              cacheWriteTokens: event.usage.cache_creation_input_tokens || 0,
            }),
          );
        }
      }

      output.content = output.content.filter((p) => {
        if (p.type !== "thinking") return true;
        if (p.text.trim().length === 0) return false;
        if (p.meta?.provider !== "anthropic" || p.meta.type !== "thinking_signature") return false;
        return p.meta.signature.trim().length > 0;
      });

      ctrl.finish();
    } catch (error) {
      ctrl.fail(error);
    }
  })();

  return ctrl.stream;
}

function buildParams(
  model: AnthropicModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
): MessageCreateParamsStreaming {
  const params: MessageCreateParamsStreaming = {
    model: model.id,
    max_tokens: options.maxTokens,
    messages: convertMessages(context),
    stream: true,
  };

  if (context.system && context.system.trim().length > 0) {
    params.system = sanitizeSurrogates(context.system);
  }

  if (options.temperature !== undefined) {
    params.temperature = options.temperature;
  }

  if (context.tools && context.tools.length > 0) {
    params.tools = convertTools(context.tools);
  }

  const reasoning = options.reasoning ?? "none";
  if (model.supports.reasoning && reasoning !== "none") {
    params.thinking = {
      type: "enabled",
      budget_tokens: anthropicBudget(reasoning, options.maxTokens),
    };
  }

  return params;
}

function convertMessages(context: NormalizedContext): MessageParam[] {
  const out: MessageParam[] = [];

  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (msg.content.trim().length > 0)
        out.push({ role: "user", content: sanitizeSurrogates(msg.content) });
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: ContentBlockParam[] = [];
      for (const part of msg.content) {
        if (part.type === "text") {
          if (part.text.trim().length === 0) continue;
          blocks.push({ type: "text", text: sanitizeSurrogates(part.text) });
        } else if (part.type === "thinking") {
          if (part.text.trim().length === 0) continue;
          if (part.meta?.provider !== "anthropic" || part.meta.type !== "thinking_signature")
            continue;
          if (part.meta.signature.trim().length === 0) continue;

          blocks.push({
            type: "thinking",
            thinking: sanitizeSurrogates(part.text),
            signature: part.meta.signature,
          });
        } else if (part.type === "tool_call") {
          blocks.push({
            type: "tool_use",
            id: sanitizeToolCallId(part.id),
            name: part.name,
            input: part.args,
          });
        }
      }
      if (blocks.length > 0) out.push({ role: "assistant", content: blocks });
      continue;
    }

    if (msg.role === "tool") {
      out.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: sanitizeToolCallId(msg.toolCallId),
            content: sanitizeSurrogates(msg.content),
            is_error: msg.isError,
          },
        ],
      });
    }
  }

  return out;
}

function convertTools(tools: Tool[]) {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.parameters as any,
  }));
}

function createClient(apiKey: string, interleavedThinking: boolean): Anthropic {
  const betaFeatures = ["fine-grained-tool-streaming-2025-05-14"];
  if (interleavedThinking) betaFeatures.push("interleaved-thinking-2025-05-14");

  const defaultHeaders: Record<string, string> = {
    accept: "application/json",
    "anthropic-beta": betaFeatures.join(","),
  };

  return new Anthropic({ apiKey, defaultHeaders });
}

function sanitizeToolCallId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function anthropicBudget(effort: Exclude<ReasoningEffort, "none">, maxTokens?: number): number {
  const clamp = (n: number) => Math.max(128, Math.min(8192, n));

  if (typeof maxTokens === "number") {
    const ratio = (() => {
      switch (effort) {
        case "minimal":
          return 0.1;
        case "low":
          return 0.2;
        case "medium":
          return 0.3;
        case "high":
          return 0.5;
        case "xhigh":
          return 0.7;
        default:
          return exhaustive(effort);
      }
    })();

    return clamp(Math.round(maxTokens * ratio));
  }

  switch (effort) {
    case "minimal":
      return 256;
    case "low":
      return 512;
    case "medium":
      return 1024;
    case "high":
      return 2048;
    case "xhigh":
      return 4096;
    default:
      return exhaustive(effort);
  }
}

function mapStopReason(reason: string): StopReason {
  switch (reason) {
    case "end_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_use";
    case "stop_sequence":
      return "stop";
    default:
      return "error";
  }
}

function usageFromAnthropic(u: {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
}): Usage {
  const totalTokens = u.inputTokens + u.outputTokens + u.cacheReadTokens + u.cacheWriteTokens;
  return {
    inputTokens: u.inputTokens,
    outputTokens: u.outputTokens,
    cacheReadTokens: u.cacheReadTokens,
    cacheWriteTokens: u.cacheWriteTokens,
    totalTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}
