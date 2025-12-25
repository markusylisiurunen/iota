import Anthropic from "@anthropic-ai/sdk";
import type {
  StopReason as AnthropicStopReason,
  Tool as AnthropicTool,
  ContentBlockParam,
  MessageCreateParamsStreaming,
  MessageParam,
  TextBlockParam,
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
import { createDebugLogger } from "../utils/debug-log.js";
import { exhaustive } from "../utils/exhaustive.js";
import { parseStreamingJson } from "../utils/json.js";
import { sanitizeSurrogates } from "../utils/sanitize.js";

type AnthropicModel = Extract<AnyModel, { provider: "anthropic" }>;

type AnthropicToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

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

    const debug = createDebugLogger("anthropic");

    try {
      const client = createClient(options.apiKey, options.reasoning !== "none");
      const params = buildParams(model, context, options);
      debug.logRequest(params);
      const anthropicStream = client.messages.stream(params, {
        signal: options.signal,
      });

      for await (const event of anthropicStream) {
        debug.logResponseEvent(event);
        switch (event.type) {
          case "message_start": {
            ctrl.setUsage(
              usageFromAnthropic({
                inputTokens: event.message.usage.input_tokens || 0,
                outputTokens: event.message.usage.output_tokens || 0,
                cacheReadTokens: event.message.usage.cache_read_input_tokens || 0,
                cacheWriteTokens: event.message.usage.cache_creation_input_tokens || 0,
              }),
            );
            continue;
          }

          case "content_block_start": {
            const block = event.content_block;

            switch (block.type) {
              case "text": {
                const idx = ctrl.addPart({ type: "text", text: "" });
                contentIndexByBlockIndex.set(event.index, idx);
                continue;
              }

              case "thinking": {
                const idx = ctrl.addPart({ type: "thinking", text: "" });
                contentIndexByBlockIndex.set(event.index, idx);
                continue;
              }

              case "tool_use": {
                const part: AnthropicToolCallPart = {
                  type: "tool_call",
                  id: block.id,
                  name: block.name,
                  args: {},
                };
                const idx = ctrl.addPart(part);
                contentIndexByBlockIndex.set(event.index, idx);
                toolCallPartialJsonByIndex.set(idx, "");
                continue;
              }

              case "redacted_thinking":
              case "server_tool_use":
              case "web_search_tool_result":
                continue;

              default:
                return exhaustive(block);
            }
          }

          case "content_block_delta": {
            const idx = contentIndexByBlockIndex.get(event.index);
            if (idx === undefined) continue;

            const part = output.content[idx];
            if (!part) continue;

            switch (event.delta.type) {
              case "text_delta": {
                if (part.type !== "text") continue;
                part.text += event.delta.text;
                ctrl.delta(idx, event.delta.text);
                continue;
              }

              case "thinking_delta": {
                if (part.type !== "thinking") continue;
                part.text += event.delta.thinking;
                ctrl.delta(idx, event.delta.thinking);
                continue;
              }

              case "input_json_delta": {
                if (part.type !== "tool_call") continue;

                const current = toolCallPartialJsonByIndex.get(idx) ?? "";
                const next = current + event.delta.partial_json;
                toolCallPartialJsonByIndex.set(idx, next);

                ctrl.delta(idx, event.delta.partial_json);
                continue;
              }

              case "signature_delta": {
                if (part.type !== "thinking") continue;

                const current = thinkingSignatureByIndex.get(idx) ?? "";
                const next = current + event.delta.signature;
                thinkingSignatureByIndex.set(idx, next);

                part.meta = { provider: "anthropic", type: "thinking_signature", signature: next };
                continue;
              }

              case "citations_delta":
                continue;

              default:
                return exhaustive(event.delta);
            }
          }

          case "content_block_stop": {
            const idx = contentIndexByBlockIndex.get(event.index);
            if (idx === undefined) continue;

            const part = output.content[idx];
            if (!part) continue;

            if (part.type === "tool_call") {
              const json = toolCallPartialJsonByIndex.get(idx);
              if (json !== undefined) {
                part.args = parseStreamingJson(json);
                toolCallPartialJsonByIndex.delete(idx);
              }
            }

            ctrl.endPart(idx);
            continue;
          }

          case "message_delta": {
            if (event.delta.stop_reason) {
              ctrl.setStopReason(mapStopReason(event.delta.stop_reason));
            }

            ctrl.setUsage(
              usageFromAnthropic({
                inputTokens: event.usage.input_tokens || 0,
                outputTokens: event.usage.output_tokens || 0,
                cacheReadTokens: event.usage.cache_read_input_tokens || 0,
                cacheWriteTokens: event.usage.cache_creation_input_tokens || 0,
              }),
            );
            continue;
          }

          case "message_stop":
            continue;

          default:
            return exhaustive(event);
        }
      }

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
    params.system = [
      {
        type: "text",
        text: sanitizeSurrogates(context.system),
        cache_control: { type: "ephemeral" },
      },
    ] satisfies TextBlockParam[];
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

  for (let i = 0; i < context.messages.length; i++) {
    const msg = context.messages[i];
    if (!msg) continue;

    switch (msg.role) {
      case "user": {
        if (msg.content.trim().length > 0) {
          out.push({ role: "user", content: sanitizeSurrogates(msg.content) });
        }
        continue;
      }

      case "assistant": {
        const blocks: ContentBlockParam[] = [];
        for (const part of msg.content) {
          switch (part.type) {
            case "text": {
              if (part.text.trim().length === 0) continue;
              blocks.push({ type: "text", text: sanitizeSurrogates(part.text) });
              continue;
            }

            case "thinking": {
              if (part.text.trim().length === 0) continue;
              if (part.meta?.provider !== "anthropic" || part.meta.type !== "thinking_signature") {
                continue;
              }
              if (part.meta.signature.trim().length === 0) continue;

              blocks.push({
                type: "thinking",
                thinking: sanitizeSurrogates(part.text),
                signature: part.meta.signature,
              });
              continue;
            }

            case "tool_call": {
              blocks.push({
                type: "tool_use",
                id: sanitizeToolCallId(part.id),
                name: part.name,
                input: part.args,
              });
              continue;
            }

            default:
              return exhaustive(part);
          }
        }

        if (blocks.length > 0) out.push({ role: "assistant", content: blocks });
        continue;
      }

      case "tool": {
        const toolResults: ContentBlockParam[] = [];

        for (; i < context.messages.length; i++) {
          const next = context.messages[i];
          if (!next || next.role !== "tool") break;

          toolResults.push({
            type: "tool_result",
            tool_use_id: sanitizeToolCallId(next.toolCallId),
            content: sanitizeSurrogates(next.content),
            is_error: next.isError,
          });
        }

        i--;
        out.push({ role: "user", content: toolResults });
        continue;
      }

      default:
        return exhaustive(msg);
    }
  }

  let lastUserIndex = -1;
  for (let i = out.length - 1; i >= 0; i--) {
    if (out[i]?.role === "user") {
      lastUserIndex = i;
      break;
    }
  }

  const lastUser = out[lastUserIndex];
  if (!lastUser || lastUser.role !== "user") return out;

  if (typeof lastUser.content === "string") {
    out[lastUserIndex] = {
      role: "user",
      content: [
        {
          type: "text",
          text: lastUser.content,
          cache_control: { type: "ephemeral" },
        },
      ],
    };

    return out;
  }

  if (Array.isArray(lastUser.content)) {
    const lastBlock = lastUser.content.at(-1);
    if (!lastBlock) return out;

    switch (lastBlock.type) {
      case "text":
      case "image":
      case "tool_result":
      case "tool_use":
        lastBlock.cache_control = { type: "ephemeral" };
        break;
      default:
        break;
    }
  }

  return out;
}

function convertTools(tools: Tool[]): AnthropicTool[] {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.parameters as AnthropicTool.InputSchema,
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
  const desired = (() => {
    switch (effort) {
      case "minimal":
        return 1024;
      case "low":
        return 8192;
      case "medium":
        return 16384;
      case "high":
      case "xhigh":
        return 32768;
      default:
        return exhaustive(effort);
    }
  })();

  if (typeof maxTokens !== "number") return desired;

  const upper = Math.max(0, Math.floor(maxTokens * 0.8));
  return Math.min(desired, upper);
}

function mapStopReason(reason: AnthropicStopReason): StopReason {
  switch (reason) {
    case "end_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_use";
    case "stop_sequence":
    case "pause_turn":
    case "refusal":
      return "stop";
    default:
      return exhaustive(reason);
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
