import OpenAI from "openai";
import type {
  ResponseCreateParamsStreaming,
  ResponseFunctionToolCall,
  ResponseInput,
  ResponseOutputMessage,
  ResponseReasoningItem,
  ResponseStreamEvent,
} from "openai/resources/responses/responses.js";
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

type OpenAIModel = Extract<AnyModel, { provider: "openai" }>;

type OpenAIToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

type OpenAITextPart = Extract<AssistantPart, { type: "text" }>;

type OpenAIThinkingPart = Extract<AssistantPart, { type: "thinking" }>;

const openaiBaseUrl = "https://api.openai.com/v1";

export function streamOpenAI(
  model: OpenAIModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
): AssistantStream {
  const ctrl = new StreamController(model, options);
  ctrl.start();

  (async () => {
    const output = ctrl.output;

    const partIndexByItemId = new Map<string, number>();
    const ignoredItemIds = new Set<string>();

    const toolCallArgsJsonByIndex = new Map<number, string>();

    try {
      const client = new OpenAI({
        apiKey: options.apiKey,
        baseURL: openaiBaseUrl,
      });

      const params = buildParams(model, context, options);
      const openaiStream = await client.responses.create(params, { signal: options.signal });

      for await (const event of openaiStream) {
        handleEvent(event);
      }

      ctrl.finish();
    } catch (error) {
      ctrl.fail(error);
    }

    function handleEvent(event: ResponseStreamEvent): void {
      switch (event.type) {
        case "response.output_item.added": {
          const item = event.item;
          const itemId = item.id ?? `output_${event.output_index}`;

          if (item.type === "reasoning") {
            if ((options.reasoning ?? "none") === "none") {
              ignoredItemIds.add(itemId);
              return;
            }

            const part: OpenAIThinkingPart = { type: "thinking", text: "" };
            const idx = ctrl.addPart(part);
            partIndexByItemId.set(itemId, idx);
            return;
          }

          if (item.type === "message") {
            const part: OpenAITextPart = { type: "text", text: "" };
            const idx = ctrl.addPart(part);
            partIndexByItemId.set(itemId, idx);
            return;
          }

          if (item.type === "function_call") {
            const part: OpenAIToolCallPart = {
              type: "tool_call",
              id: item.call_id,
              name: item.name,
              args: {},
            };

            if (item.id) {
              part.meta = { provider: "openai", type: "function_call_item_id", id: item.id };
            }

            const idx = ctrl.addPart(part);
            partIndexByItemId.set(itemId, idx);

            const initial = item.arguments || "";
            toolCallArgsJsonByIndex.set(idx, initial);
            if (initial.trim().length > 0) {
              part.args = parseStreamingJson(initial);
            }

            return;
          }

          return;
        }

        case "response.reasoning_summary_text.delta": {
          if (ignoredItemIds.has(event.item_id)) return;

          const idx = partIndexByItemId.get(event.item_id);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part || part.type !== "thinking") return;

          part.text += event.delta;
          ctrl.delta(idx, event.delta);
          return;
        }

        case "response.reasoning_summary_part.done": {
          if (ignoredItemIds.has(event.item_id)) return;

          const idx = partIndexByItemId.get(event.item_id);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part || part.type !== "thinking") return;

          part.text += "\n\n";
          ctrl.delta(idx, "\n\n");
          return;
        }

        case "response.output_text.delta": {
          const idx = partIndexByItemId.get(event.item_id);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part || part.type !== "text") return;

          part.text += event.delta;
          ctrl.delta(idx, event.delta);
          return;
        }

        case "response.refusal.delta": {
          const idx = partIndexByItemId.get(event.item_id);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part || part.type !== "text") return;

          part.text += event.delta;
          ctrl.delta(idx, event.delta);
          return;
        }

        case "response.function_call_arguments.delta": {
          const idx = partIndexByItemId.get(event.item_id);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part || part.type !== "tool_call") return;

          const current = toolCallArgsJsonByIndex.get(idx) ?? "";
          const next = current + event.delta;
          toolCallArgsJsonByIndex.set(idx, next);

          part.args = parseStreamingJson(next);
          ctrl.delta(idx, event.delta);
          return;
        }

        case "response.output_item.done": {
          const item = event.item;
          const itemId = item.id ?? `output_${event.output_index}`;

          if (ignoredItemIds.has(itemId)) return;

          const idx = partIndexByItemId.get(itemId);
          if (idx === undefined) return;

          const part = output.content[idx];
          if (!part) return;

          if (item.type === "reasoning" && part.type === "thinking") {
            part.meta = { provider: "openai", type: "reasoning_item", item };
            ctrl.endPart(idx);
            return;
          }

          if (item.type === "message" && part.type === "text") {
            part.meta = { provider: "openai", type: "message_id", id: item.id };
            ctrl.endPart(idx);
            return;
          }

          if (item.type === "function_call" && part.type === "tool_call") {
            if (item.id) {
              part.meta = { provider: "openai", type: "function_call_item_id", id: item.id };
            }

            part.args = parseFinalToolArgs(item);
            toolCallArgsJsonByIndex.delete(idx);

            ctrl.endPart(idx);
            return;
          }

          ctrl.endPart(idx);
          return;
        }

        case "response.completed": {
          const response = event.response;
          if (response?.usage) {
            const cached = response.usage.input_tokens_details?.cached_tokens || 0;
            ctrl.setUsage(
              usageFromOpenAI({
                inputTokens: (response.usage.input_tokens || 0) - cached,
                outputTokens: response.usage.output_tokens || 0,
                cacheReadTokens: cached,
                totalTokens: response.usage.total_tokens || 0,
              }),
            );
          }

          ctrl.setStopReason(mapStopReason(response?.status));
          return;
        }

        case "error": {
          throw new Error(event.message || "OpenAI error");
        }

        case "response.failed": {
          throw new Error("OpenAI request failed");
        }

        default:
          return;
      }
    }
  })();

  return ctrl.stream;
}

function parseFinalToolArgs(item: ResponseFunctionToolCall): unknown {
  const raw = item.arguments || "";
  if (raw.trim().length === 0) return {};

  try {
    return JSON.parse(raw);
  } catch {
    return parseStreamingJson(raw);
  }
}

function buildParams(
  model: OpenAIModel,
  context: NormalizedContext,
  options: ResolvedStreamOptions,
): ResponseCreateParamsStreaming {
  const messages = convertMessages(context);

  const params: ResponseCreateParamsStreaming = {
    model: model.id,
    input: messages,
    stream: true,
  };

  if (options.maxTokens !== undefined) {
    params.max_output_tokens = options.maxTokens;
  }

  if (options.temperature !== undefined) {
    params.temperature = options.temperature;
  }

  if (context.tools && context.tools.length > 0) {
    params.tools = convertTools(context.tools);
  }

  if (model.supports.reasoning && (options.reasoning ?? "none") !== "none") {
    params.reasoning = {
      effort: mapReasoningEffort(options.reasoning),
      summary: "auto",
    };
    params.include = ["reasoning.encrypted_content"];
  }

  return params;
}

function convertMessages(context: NormalizedContext): ResponseInput {
  const input: ResponseInput = [];

  if (context.system && context.system.trim().length > 0) {
    input.push({ role: "system", content: sanitizeSurrogates(context.system) });
  }

  let msgIndex = 0;
  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (msg.content.trim().length > 0) {
        input.push({
          role: "user",
          content: [{ type: "input_text", text: sanitizeSurrogates(msg.content) }],
        });
      }
      continue;
    }

    if (msg.role === "assistant") {
      for (const part of msg.content) {
        if (part.type === "thinking") {
          if (part.meta?.provider === "openai" && part.meta.type === "reasoning_item") {
            input.push(part.meta.item as ResponseReasoningItem);
          }
          continue;
        }

        if (part.type === "text") {
          if (part.text.trim().length === 0) continue;

          const id =
            part.meta?.provider === "openai" && part.meta.type === "message_id"
              ? part.meta.id
              : `msg_${msgIndex++}`;

          input.push({
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: sanitizeSurrogates(part.text), annotations: [] },
            ],
            status: "completed",
            id,
          } satisfies ResponseOutputMessage);
          continue;
        }

        if (part.type === "tool_call") {
          const id =
            part.meta?.provider === "openai" && part.meta.type === "function_call_item_id"
              ? part.meta.id
              : `fc_${msgIndex++}`;

          input.push({
            type: "function_call",
            id,
            call_id: part.id,
            name: part.name,
            arguments: JSON.stringify(part.args),
          });
        }
      }
      continue;
    }

    if (msg.role === "tool") {
      input.push({
        type: "function_call_output",
        call_id: msg.toolCallId,
        output: sanitizeSurrogates(msg.content),
      });
    }
  }

  return input;
}

function convertTools(tools: Tool[]) {
  return tools.map((tool) => ({
    type: "function" as const,
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters as Record<string, unknown>,
    strict: null,
  }));
}

function mapReasoningEffort(effort: ReasoningEffort): Exclude<ReasoningEffort, "none" | "xhigh"> {
  const value = effort === "none" ? "minimal" : effort;
  if (value === "xhigh") return "high";
  return value;
}

function usageFromOpenAI(u: {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  totalTokens: number;
}): Usage {
  return {
    inputTokens: u.inputTokens,
    outputTokens: u.outputTokens,
    cacheReadTokens: u.cacheReadTokens,
    cacheWriteTokens: 0,
    totalTokens: u.totalTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function mapStopReason(status: OpenAI.Responses.ResponseStatus | undefined): StopReason {
  if (!status) return "stop";
  switch (status) {
    case "completed":
      return "stop";
    case "incomplete":
      return "length";
    case "cancelled":
      return "aborted";
    case "failed":
      return "error";
    case "in_progress":
    case "queued":
      return "stop";
    default:
      return exhaustive(status);
  }
}
