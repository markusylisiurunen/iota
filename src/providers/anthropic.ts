import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  MessageCreateParamsStreaming,
  MessageParam,
} from "@anthropic-ai/sdk/resources/messages.js";
import { AssistantStream } from "../assistant-stream.js";
import type { AnyModel } from "../models.js";
import { calculateCost } from "../models.js";
import { emptyUsage } from "../stream.js";
import type {
  AssistantMessage,
  AssistantMessageDraft,
  AssistantPart,
  Context,
  ReasoningEffort,
  StopReason,
  StreamOptions,
  Tool,
  Usage,
} from "../types.js";
import { parseStreamingJson } from "../utils/json.js";
import { sanitizeSurrogates } from "../utils/sanitize.js";

type AnthropicModel = Extract<AnyModel, { provider: "anthropic" }>;

export function streamAnthropic(
  model: AnthropicModel,
  context: Context,
  options: StreamOptions,
): AssistantStream {
  const stream = new AssistantStream();

  (async () => {
    const output: AssistantMessageDraft = {
      role: "assistant",
      provider: model.provider,
      model: model.id,
      content: [],
      stopReason: "stop",
      usage: emptyUsage(),
    };

    try {
      const client = new Anthropic({ apiKey: options.apiKey! });
      const params = buildParams(model, context, options);
      const anthropicStream = client.messages.stream(params, {
        signal: options.signal,
      });

      stream.push({ type: "start", partial: output });

      const blocks = output.content as Array<
        AssistantPart & { index?: number; partialJson?: string }
      >;

      for await (const event of anthropicStream) {
        if (event.type === "message_start") {
          output.usage = usageFromAnthropic({
            inputTokens: event.message.usage.input_tokens || 0,
            outputTokens: event.message.usage.output_tokens || 0,
            cacheReadTokens: event.message.usage.cache_read_input_tokens || 0,
            cacheWriteTokens: event.message.usage.cache_creation_input_tokens || 0,
          });
          calculateCost(model, output.usage);
        } else if (event.type === "content_block_start") {
          if (event.content_block.type === "text") {
            const part: AssistantPart & { index: number } = {
              type: "text",
              text: "",
              index: event.index,
            };
            blocks.push(part);
            stream.push({ type: "part_start", index: blocks.length - 1, partial: output });
          } else if (event.content_block.type === "thinking") {
            const part: AssistantPart & { index: number } = {
              type: "thinking",
              text: "",
              signature: "",
              index: event.index,
            };
            blocks.push(part);
            stream.push({ type: "part_start", index: blocks.length - 1, partial: output });
          } else if (event.content_block.type === "tool_use") {
            const part: AssistantPart & { index: number; partialJson: string } = {
              type: "tool_call",
              id: event.content_block.id,
              name: event.content_block.name,
              args: event.content_block.input as Record<string, unknown>,
              partialJson: "",
              index: event.index,
            };
            blocks.push(part);
            stream.push({ type: "part_start", index: blocks.length - 1, partial: output });
          }
        } else if (event.type === "content_block_delta") {
          const idx = blocks.findIndex((b) => (b as any)?.index === event.index);
          const part = blocks[idx] as any;
          if (!part) continue;

          if (event.delta.type === "text_delta" && part.type === "text") {
            part.text += event.delta.text;
            stream.push({
              type: "part_delta",
              index: idx,
              delta: event.delta.text,
              partial: output,
            });
          } else if (event.delta.type === "thinking_delta" && part.type === "thinking") {
            part.text += event.delta.thinking;
            stream.push({
              type: "part_delta",
              index: idx,
              delta: event.delta.thinking,
              partial: output,
            });
          } else if (event.delta.type === "input_json_delta" && part.type === "tool_call") {
            part.partialJson = (part.partialJson || "") + event.delta.partial_json;
            part.args = parseStreamingJson(part.partialJson);
            stream.push({
              type: "part_delta",
              index: idx,
              delta: event.delta.partial_json,
              partial: output,
            });
          } else if (event.delta.type === "signature_delta" && part.type === "thinking") {
            part.signature = (part.signature || "") + event.delta.signature;
          }
        } else if (event.type === "content_block_stop") {
          const idx = blocks.findIndex((b) => (b as any)?.index === event.index);
          const part = blocks[idx] as any;
          if (!part) continue;

          delete part.index;
          if (part.type === "tool_call") {
            part.args = parseStreamingJson(part.partialJson);
            delete part.partialJson;
          }
          stream.push({ type: "part_end", index: idx, partial: output });
        } else if (event.type === "message_delta") {
          if (event.delta.stop_reason) output.stopReason = mapStopReason(event.delta.stop_reason);
          output.usage = usageFromAnthropic({
            inputTokens: event.usage.input_tokens || 0,
            outputTokens: event.usage.output_tokens || 0,
            cacheReadTokens: event.usage.cache_read_input_tokens || 0,
            cacheWriteTokens: event.usage.cache_creation_input_tokens || 0,
          });
          calculateCost(model, output.usage);
        }
      }

      output.content = output.content.filter(
        (p) => p.type !== "thinking" || (p.signature && p.signature.trim().length > 0),
      );

      if (output.content.some((p) => p.type === "tool_call") && output.stopReason === "stop") {
        output.stopReason = "tool_use";
      }

      if (options.signal?.aborted) {
        output.stopReason = "aborted";
        output.errorMessage = output.errorMessage ?? "Request was aborted";
      }

      if (!output.usage) output.usage = emptyUsage();
      if (!output.stopReason) output.stopReason = "stop";

      const final = output as AssistantMessage;

      if (final.stopReason === "error" || final.stopReason === "aborted") {
        if (!final.errorMessage) final.errorMessage = "Request failed";
        stream.push({ type: "error", error: final });
      } else {
        stream.push({ type: "done", message: final });
      }

      stream.end();
    } catch (error) {
      for (const p of output.content as any[]) delete p.index;
      output.stopReason = options.signal?.aborted ? "aborted" : "error";
      output.usage = output.usage ?? emptyUsage();
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", error: output as AssistantMessage });
      stream.end();
    }
  })();

  return stream;
}

function buildParams(
  model: AnthropicModel,
  context: Context,
  options: StreamOptions,
): MessageCreateParamsStreaming {
  const params: MessageCreateParamsStreaming = {
    model: model.id,
    max_tokens: options.maxTokens ?? model.maxOutputTokens,
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

function convertMessages(context: Context): MessageParam[] {
  const out: MessageParam[] = [];

  for (const msg of context.messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      if (msg.content.trim().length > 0)
        out.push({ role: "user", content: sanitizeSurrogates(msg.content) });
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: ContentBlockParam[] = [];
      const parts: AssistantPart[] =
        typeof msg.content === "string" ? [{ type: "text", text: msg.content }] : msg.content;
      for (const part of parts) {
        if (part.type === "text") {
          if (part.text.trim().length === 0) continue;
          blocks.push({ type: "text", text: sanitizeSurrogates(part.text) });
        } else if (part.type === "thinking") {
          if (!part.signature || part.signature.trim().length === 0) continue;
          if (part.text.trim().length === 0) continue;
          blocks.push({
            type: "thinking",
            thinking: sanitizeSurrogates(part.text),
            signature: part.signature,
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
            is_error: msg.isError ?? false,
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
