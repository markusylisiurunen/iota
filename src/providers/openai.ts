import OpenAI from "openai";
import type {
  ResponseCreateParamsStreaming,
  ResponseFunctionToolCall,
  ResponseInput,
  ResponseOutputMessage,
  ResponseReasoningItem,
} from "openai/resources/responses/responses.js";
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

type OpenAIModel = Extract<AnyModel, { provider: "openai" }>;

export function streamOpenAI(
  model: OpenAIModel,
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
      const client = new OpenAI({
        apiKey: options.apiKey!,
        baseURL: model.baseUrl,
        dangerouslyAllowBrowser: true,
      });

      const params = buildParams(model, context, options);
      const openaiStream = await client.responses.create(params, { signal: options.signal });

      stream.push({ type: "start", partial: output });

      let currentItem:
        | ResponseReasoningItem
        | ResponseOutputMessage
        | ResponseFunctionToolCall
        | null = null;
      let currentPart: (AssistantPart & { partialJson?: string }) | null = null;
      const parts = output.content;
      const partIndex = () => parts.length - 1;

      for await (const event of openaiStream) {
        if (event.type === "response.output_item.added") {
          const item = event.item;
          if (item.type === "reasoning") {
            if ((options.reasoning ?? "none") === "none") continue;
            currentItem = item;
            currentPart = { type: "thinking", text: "" };
            output.content.push(currentPart);
            stream.push({ type: "part_start", index: partIndex(), partial: output });
          } else if (item.type === "message") {
            currentItem = item;
            currentPart = { type: "text", text: "" };
            output.content.push(currentPart);
            stream.push({ type: "part_start", index: partIndex(), partial: output });
          } else if (item.type === "function_call") {
            currentItem = item;
            currentPart = {
              type: "tool_call",
              id: item.call_id,
              name: item.name,
              args: {},
              signature: item.id,
              partialJson: item.arguments || "",
            };
            output.content.push(currentPart);
            stream.push({ type: "part_start", index: partIndex(), partial: output });
          }
        } else if (event.type === "response.reasoning_summary_part.added") {
          if (currentItem && currentItem.type === "reasoning") {
            currentItem.summary = currentItem.summary || [];
            currentItem.summary.push(event.part);
          }
        } else if (event.type === "response.reasoning_summary_text.delta") {
          if (currentItem?.type !== "reasoning") continue;
          if (!currentPart || currentPart.type !== "thinking") continue;

          currentItem.summary = currentItem.summary || [];
          const last = currentItem.summary[currentItem.summary.length - 1];
          if (!last) continue;

          currentPart.text += event.delta;
          last.text += event.delta;
          stream.push({
            type: "part_delta",
            index: partIndex(),
            delta: event.delta,
            partial: output,
          });
        } else if (event.type === "response.reasoning_summary_part.done") {
          if (currentItem?.type !== "reasoning") continue;
          if (!currentPart || currentPart.type !== "thinking") continue;

          currentItem.summary = currentItem.summary || [];
          const last = currentItem.summary[currentItem.summary.length - 1];
          if (!last) continue;

          currentPart.text += "\n\n";
          last.text += "\n\n";
          stream.push({ type: "part_delta", index: partIndex(), delta: "\n\n", partial: output });
        } else if (event.type === "response.content_part.added") {
          if (currentItem?.type !== "message") continue;
          currentItem.content = currentItem.content || [];
          if (event.part.type === "output_text" || event.part.type === "refusal") {
            currentItem.content.push(event.part);
          }
        } else if (event.type === "response.output_text.delta") {
          if (currentItem?.type !== "message") continue;
          if (!currentPart || currentPart.type !== "text") continue;

          const last = currentItem.content[currentItem.content.length - 1];
          if (!last || last.type !== "output_text") continue;
          currentPart.text += event.delta;
          last.text += event.delta;
          stream.push({
            type: "part_delta",
            index: partIndex(),
            delta: event.delta,
            partial: output,
          });
        } else if (event.type === "response.refusal.delta") {
          if (currentItem?.type !== "message") continue;
          if (!currentPart || currentPart.type !== "text") continue;

          const last = currentItem.content[currentItem.content.length - 1];
          if (!last || last.type !== "refusal") continue;
          currentPart.text += event.delta;
          last.refusal += event.delta;
          stream.push({
            type: "part_delta",
            index: partIndex(),
            delta: event.delta,
            partial: output,
          });
        } else if (event.type === "response.function_call_arguments.delta") {
          if (currentItem?.type !== "function_call") continue;
          if (!currentPart || currentPart.type !== "tool_call") continue;

          (currentPart as any).partialJson = ((currentPart as any).partialJson ?? "") + event.delta;
          (currentPart as any).args = parseStreamingJson((currentPart as any).partialJson);
          stream.push({
            type: "part_delta",
            index: partIndex(),
            delta: event.delta,
            partial: output,
          });
        } else if (event.type === "response.output_item.done") {
          const item = event.item;

          if (item.type === "reasoning" && currentPart?.type === "thinking") {
            currentPart.text = item.summary?.map((p) => p.text).join("\n\n") || "";
            currentPart.signature = JSON.stringify(item);
            stream.push({ type: "part_end", index: partIndex(), partial: output });
            currentPart = null;
            currentItem = null;
          } else if (item.type === "message" && currentPart?.type === "text") {
            currentPart.text = item.content
              .map((c) => (c.type === "output_text" ? c.text : c.refusal))
              .join("");
            currentPart.signature = item.id;
            stream.push({ type: "part_end", index: partIndex(), partial: output });
            currentPart = null;
            currentItem = null;
          } else if (item.type === "function_call" && currentPart?.type === "tool_call") {
            (currentPart as any).args = JSON.parse(item.arguments);
            delete (currentPart as any).partialJson;
            stream.push({ type: "part_end", index: partIndex(), partial: output });
            currentPart = null;
            currentItem = null;
          }
        } else if (event.type === "response.completed") {
          const response = event.response;
          if (response?.usage) {
            const cached = response.usage.input_tokens_details?.cached_tokens || 0;
            output.usage = usageFromOpenAI({
              inputTokens: (response.usage.input_tokens || 0) - cached,
              outputTokens: response.usage.output_tokens || 0,
              cacheReadTokens: cached,
              totalTokens: response.usage.total_tokens || 0,
            });
            calculateCost(model, output.usage);
          }

          output.stopReason = mapStopReason(response?.status);
          if (output.content.some((p) => p.type === "tool_call") && output.stopReason === "stop") {
            output.stopReason = "tool_use";
          }
        } else if (event.type === "error") {
          throw new Error(event.message || "OpenAI error");
        } else if (event.type === "response.failed") {
          throw new Error("OpenAI request failed");
        }
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
  model: OpenAIModel,
  context: Context,
  options: StreamOptions,
): ResponseCreateParamsStreaming {
  const messages = convertMessages(model, context);

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
      effort: mapReasoningEffort(options.reasoning ?? "none"),
      summary: "auto",
    };
    params.include = ["reasoning.encrypted_content"];
  }

  return params;
}

function convertMessages(_model: OpenAIModel, context: Context): ResponseInput {
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
      const parts: AssistantPart[] =
        typeof msg.content === "string" ? [{ type: "text", text: msg.content }] : msg.content;
      for (const part of parts) {
        if (part.type === "thinking") {
          if (!part.signature) continue;
          try {
            input.push(JSON.parse(part.signature));
          } catch {
            continue;
          }
          continue;
        }

        if (part.type === "text") {
          if (part.text.trim().length === 0) continue;
          input.push({
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: sanitizeSurrogates(part.text), annotations: [] },
            ],
            status: "completed",
            id: part.signature || `msg_${msgIndex++}`,
          } satisfies ResponseOutputMessage);
          continue;
        }

        if (part.type === "tool_call") {
          input.push({
            type: "function_call",
            id: part.signature || `fc_${msgIndex++}`,
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
    parameters: tool.parameters as any,
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
    default: {
      const _exhaustive: never = status;
      throw new Error(`Unhandled OpenAI status: ${_exhaustive}`);
    }
  }
}
