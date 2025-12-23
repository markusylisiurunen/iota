import { GoogleGenAI } from "@google/genai";
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
import { sanitizeSurrogates } from "../utils/sanitize.js";

type GeminiModel = Extract<AnyModel, { provider: "gemini" }>;

type GeminiThinkingLevel = "minimal" | "low" | "medium" | "high";

export function streamGemini(
  model: GeminiModel,
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
      const client = new GoogleGenAI({
        apiKey: options.apiKey!,
        httpOptions: model.baseUrl
          ? {
              baseUrl: model.baseUrl,
              apiVersion: "", // baseUrl already includes version path
            }
          : undefined,
      });

      const params = buildParams(model, context, options);
      const geminiStream = await client.interactions.create(params, { signal: options.signal });

      stream.push({ type: "start", partial: output });

      const started = new Set<number>();
      const indexToPartIndex = new Map<number, number>();

      const partIndex = () => output.content.length - 1;

      for await (const event of geminiStream as any) {
        if (event.event_type === "content.start") {
          const idx = typeof event.index === "number" ? event.index : null;
          const content = event.content;
          if (idx === null || !content?.type) continue;
          started.add(idx);

          if (content.type === "text") {
            const part: Extract<AssistantPart, { type: "text" }> = { type: "text", text: "" };
            output.content.push(part);
            const pidx = partIndex();
            indexToPartIndex.set(idx, pidx);
            stream.push({ type: "part_start", index: pidx, partial: output });

            const initial = content.text ?? "";
            if (initial.length > 0) {
              part.text += initial;
              stream.push({ type: "part_delta", index: pidx, delta: initial, partial: output });
            }
          } else if (content.type === "thought") {
            if ((options.reasoning ?? "none") === "none") {
              indexToPartIndex.set(idx, -1);
              continue;
            }

            const part: Extract<AssistantPart, { type: "thinking" }> = {
              type: "thinking",
              text: "",
              signature: content.signature,
            };
            output.content.push(part);
            const pidx = partIndex();
            indexToPartIndex.set(idx, pidx);
            stream.push({ type: "part_start", index: pidx, partial: output });

            const initialSummary = Array.isArray(content.summary)
              ? content.summary.map((p: any) => (p?.type === "text" ? (p.text ?? "") : "")).join("")
              : "";

            if (initialSummary.length > 0) {
              part.text += initialSummary;
              stream.push({
                type: "part_delta",
                index: pidx,
                delta: initialSummary,
                partial: output,
              });
            }
          } else if (content.type === "function_call") {
            const part: Extract<AssistantPart, { type: "tool_call" }> = {
              type: "tool_call",
              id: content.id,
              name: content.name,
              args: content.arguments ?? {},
            };
            output.content.push(part);
            indexToPartIndex.set(idx, partIndex());
            stream.push({ type: "part_start", index: partIndex(), partial: output });
            stream.push({
              type: "part_delta",
              index: partIndex(),
              delta: JSON.stringify(part.args),
              partial: output,
            });
          } else {
            indexToPartIndex.set(idx, -1);
          }
        } else if (event.event_type === "content.delta") {
          const idx = typeof event.index === "number" ? event.index : null;
          const delta = event.delta;
          if (idx === null || !delta?.type) continue;

          const mapped = indexToPartIndex.get(idx);
          if (mapped === undefined || mapped === -1) continue;

          const currentPart = output.content[mapped];
          if (!currentPart) continue;

          if (delta.type === "text") {
            if (currentPart.type !== "text") continue;
            const text = delta.text ?? "";
            if (text.length === 0) continue;
            currentPart.text += text;
            stream.push({ type: "part_delta", index: mapped, delta: text, partial: output });
          } else if (delta.type === "thought" || delta.type === "thought_summary") {
            if (currentPart.type !== "thinking") continue;

            const text =
              typeof delta.thought === "string"
                ? delta.thought
                : delta.content?.type === "text"
                  ? (delta.content.text ?? "")
                  : "";

            if (text.length === 0) continue;
            currentPart.text += text;
            stream.push({ type: "part_delta", index: mapped, delta: text, partial: output });
          } else if (delta.type === "thought_signature") {
            if (currentPart.type !== "thinking") continue;
            if (typeof delta.signature === "string") currentPart.signature = delta.signature;
          } else if (delta.type === "function_call") {
            if (currentPart.type !== "tool_call") continue;
            if (typeof delta.name === "string") currentPart.name = delta.name;
            if (typeof delta.id === "string") currentPart.id = delta.id;
            if (delta.arguments && typeof delta.arguments === "object") {
              currentPart.args = { ...(currentPart.args as any), ...(delta.arguments as any) };
              stream.push({
                type: "part_delta",
                index: mapped,
                delta: JSON.stringify(delta.arguments),
                partial: output,
              });
            }
          }
        } else if (event.event_type === "content.stop") {
          const idx = typeof event.index === "number" ? event.index : null;
          if (idx === null) continue;
          started.delete(idx);

          const mapped = indexToPartIndex.get(idx);
          if (mapped === undefined || mapped === -1) continue;

          stream.push({ type: "part_end", index: mapped, partial: output });
        } else if (event.event_type === "interaction.complete") {
          const interaction = event.interaction;

          if (interaction?.usage) {
            output.usage = usageFromGeminiInteractions(interaction.usage);
            calculateCost(model, output.usage);
          }

          output.stopReason = mapStopReason(interaction?.status);
          if (output.content.some((p) => p.type === "tool_call") && output.stopReason === "stop") {
            output.stopReason = "tool_use";
          }
        } else if (event.event_type === "error") {
          throw new Error(event.error?.message || "Gemini error");
        }
      }

      for (const idx of started) {
        const mapped = indexToPartIndex.get(idx);
        if (mapped === undefined || mapped === -1) continue;
        stream.push({ type: "part_end", index: mapped, partial: output });
      }

      output.content = output.content.filter(
        (p) => p.type !== "thinking" || (p.signature && p.signature.trim().length > 0),
      );

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

function buildParams(model: GeminiModel, context: Context, options: StreamOptions) {
  const input = convertMessages(context);

  const generation_config: any = {};

  if (options.temperature !== undefined) {
    generation_config.temperature = options.temperature;
  }

  if (options.maxTokens !== undefined) {
    generation_config.max_output_tokens = options.maxTokens;
  }

  const reasoning = options.reasoning ?? "none";
  if (model.supports.reasoning) {
    if (reasoning === "none") {
      generation_config.thinking_summaries = "none";
    } else {
      const effort = (reasoning === "xhigh" ? "high" : reasoning) as Exclude<
        ReasoningEffort,
        "none" | "xhigh"
      >;
      const thinkingLevel = geminiThinkingLevel(model.id, effort);
      if (thinkingLevel) generation_config.thinking_level = thinkingLevel;
      generation_config.thinking_summaries = "auto";
    }
  }

  const params: any = {
    model: model.id,
    input,
    stream: true,
    store: false,
  };

  if (Object.keys(generation_config).length > 0) {
    params.generation_config = generation_config;
  }

  if (context.system && context.system.trim().length > 0) {
    params.system_instruction = sanitizeSurrogates(context.system);
  }

  if (context.tools && context.tools.length > 0) {
    params.tools = convertTools(context.tools);
  }

  return params;
}

function convertMessages(context: Context) {
  const turns: any[] = [];

  for (const msg of context.messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      if (msg.content.trim().length === 0) continue;
      turns.push({
        role: "user",
        content: [{ type: "text", text: sanitizeSurrogates(msg.content) }],
      });
      continue;
    }

    if (msg.role === "assistant") {
      const parts: AssistantPart[] =
        typeof msg.content === "string" ? [{ type: "text", text: msg.content }] : msg.content;

      const content: any[] = [];
      for (const part of parts) {
        if (part.type === "text") {
          if (part.text.trim().length === 0) continue;
          content.push({ type: "text", text: sanitizeSurrogates(part.text) });
        } else if (part.type === "thinking") {
          if (!part.signature || part.signature.trim().length === 0) continue;
          if (part.text.trim().length === 0) continue;
          content.push({
            type: "thought",
            signature: part.signature,
            summary: [{ type: "text", text: sanitizeSurrogates(part.text) }],
          });
        } else if (part.type === "tool_call") {
          content.push({
            type: "function_call",
            id: part.id,
            name: part.name,
            arguments: part.args && typeof part.args === "object" ? part.args : {},
          });
        }
      }

      if (content.length > 0) turns.push({ role: "model", content });
      continue;
    }

    if (msg.role === "tool") {
      turns.push({
        role: "user",
        content: [
          {
            type: "function_result",
            call_id: msg.toolCallId,
            name: msg.toolName,
            is_error: msg.isError ?? false,
            result: sanitizeSurrogates(msg.content),
          },
        ],
      });
    }
  }

  return turns;
}

function convertTools(tools: Tool[]) {
  return tools.map((t) => ({
    type: "function" as const,
    name: t.name,
    description: t.description,
    parameters: t.parameters,
  }));
}

function geminiThinkingLevel(
  modelId: string,
  effort: Exclude<ReasoningEffort, "none" | "xhigh">,
): GeminiThinkingLevel | undefined {
  if (modelId.includes("3-pro")) {
    switch (effort) {
      case "minimal":
      case "low":
      case "medium":
        return "low";
      case "high":
        return "high";
    }
  }

  switch (effort) {
    case "minimal":
      return "minimal";
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
  }
}

function mapStopReason(status: string | undefined): StopReason {
  switch (status) {
    case "completed":
      return "stop";
    case "requires_action":
      return "tool_use";
    case "cancelled":
      return "aborted";
    case "failed":
      return "error";
    default:
      return "stop";
  }
}

function usageFromGeminiInteractions(u: {
  total_input_tokens?: number;
  total_output_tokens?: number;
  total_cached_tokens?: number;
  total_reasoning_tokens?: number;
  total_tokens?: number;
}): Usage {
  const cached = u.total_cached_tokens || 0;
  const input = Math.max(0, (u.total_input_tokens || 0) - cached);
  const output = (u.total_output_tokens || 0) + (u.total_reasoning_tokens || 0);

  return {
    inputTokens: input,
    outputTokens: output,
    cacheReadTokens: cached,
    cacheWriteTokens: 0,
    totalTokens: u.total_tokens || input + output + cached,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}
