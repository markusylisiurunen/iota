import { AgentStream } from "./agent-stream.js";
import type { AssistantStream } from "./assistant-stream.js";
import type { AnyModel, Model } from "./models.js";
import { clampReasoningForModel } from "./models.js";
import { streamAnthropic } from "./providers/anthropic.js";
import { streamGemini } from "./providers/gemini.js";
import { streamOpenAI } from "./providers/openai.js";
import type {
  AgentOptions,
  AssistantMessage,
  AssistantMessageInput,
  AssistantPart,
  Context,
  JsonSchema,
  Message,
  NormalizedContext,
  NormalizedMessage,
  Provider,
  ResolvedStreamOptions,
  StreamOptions,
  Tool,
  ToolHandlers,
  ToolMessage,
} from "./types.js";
import { exhaustive } from "./utils/exhaustive.js";

export function stream(
  model: AnyModel,
  context: Context,
  options: StreamOptions = {},
): AssistantStream {
  const apiKey = options.apiKey ?? getApiKey(model.provider);
  if (!apiKey) throw new Error(`Missing API key for provider: ${model.provider}`);

  const reasoning = clampReasoningForModel(model, options.reasoning ?? "none");
  const normalized = normalizeContextForTarget(model, context);

  if (!model.supports.tools && contextUsesTools(normalized)) {
    throw new Error(`Model does not support tools: ${model.provider}/${model.id}`);
  }

  validateTools(normalized.tools);

  const resolvedOptions: ResolvedStreamOptions = {
    ...options,
    apiKey,
    reasoning,
    maxTokens: options.maxTokens ?? model.maxOutputTokens,
  };

  switch (model.provider) {
    case "openai":
      return streamOpenAI(model, normalized, resolvedOptions);
    case "anthropic":
      return streamAnthropic(model, normalized, resolvedOptions);
    case "gemini":
      return streamGemini(model, normalized, resolvedOptions);
    default:
      return exhaustive(model);
  }
}

export function agent(
  model: AnyModel,
  context: Context,
  handlers: ToolHandlers,
  options: AgentOptions = {},
): AgentStream {
  if (options.maxTurns !== undefined) {
    if (!Number.isInteger(options.maxTurns) || options.maxTurns < 1) {
      throw new Error("Invalid agent options: maxTurns must be a positive integer");
    }
  }

  if (!handlers || typeof handlers !== "object" || Array.isArray(handlers)) {
    throw new Error("Invalid tool handlers: expected a map of toolName -> handler");
  }

  const output = new AgentStream();

  const { maxTurns, ...streamOptions } = options;

  type ToolCallPart = Extract<AssistantPart, { type: "tool_call" }>;

  const toolCalls = (message: Pick<AssistantMessage, "content">): ToolCallPart[] =>
    message.content.filter((p): p is ToolCallPart => p.type === "tool_call");

  const errorToString = (error: unknown): string =>
    error instanceof Error ? error.message : String(error);

  const toolResultToString = (value: unknown): string => {
    if (typeof value === "string") return value;

    try {
      const json = JSON.stringify(value);
      return json ?? String(value);
    } catch {
      return String(value);
    }
  };

  const messages: Message[] = [];

  (async () => {
    for (let turn = 0; ; turn++) {
      if (maxTurns !== undefined && turn >= maxTurns) {
        output.push({
          type: "error",
          error: {
            messages,
            stopReason: "error",
            errorMessage: "maxTurns exceeded",
          },
        });
        output.end();
        return;
      }

      output.push({ type: "turn_start", turn });

      const turnContext: Context = {
        ...context,
        messages: [...context.messages, ...messages],
      };

      const s = stream(model, turnContext, streamOptions);
      for await (const e of s) {
        output.push({ type: "assistant_event", turn, event: e });
      }

      const assistant = await s.result();
      messages.push(assistant);

      if (assistant.stopReason === "error" || assistant.stopReason === "aborted") {
        output.push({
          type: "error",
          error: {
            messages,
            stopReason: assistant.stopReason,
            errorMessage: assistant.errorMessage,
          },
        });
        output.end();
        return;
      }

      const calls = toolCalls(assistant);
      if (calls.length === 0) {
        output.push({
          type: "done",
          result: {
            messages,
            stopReason: assistant.stopReason,
            errorMessage: assistant.errorMessage,
          },
        });
        output.end();
        return;
      }

      for (const call of calls) {
        const handler = handlers[call.name];
        if (!handler) {
          output.push({
            type: "error",
            error: {
              messages,
              stopReason: "error",
              errorMessage: `Unknown tool: ${call.name}`,
            },
          });
          output.end();
          return;
        }

        let content: string;
        let isError: boolean;

        try {
          const result = await handler(call.args, call);
          content = toolResultToString(result);
          isError = false;
        } catch (error) {
          content = errorToString(error);
          isError = true;
        }

        const toolMessage: ToolMessage = {
          role: "tool",
          toolCallId: call.id,
          toolName: call.name,
          content,
          isError,
        };

        messages.push(toolMessage);
        output.push({ type: "tool_result", turn, message: toolMessage });
      }
    }
  })().catch((error) => {
    output.push({
      type: "error",
      error: {
        messages,
        stopReason: options.signal?.aborted ? "aborted" : "error",
        errorMessage: errorToString(error),
      },
    });
    output.end();
  });

  return output;
}

export async function complete(
  model: AnyModel,
  context: Context,
  options?: StreamOptions,
): Promise<AssistantMessage> {
  const s = stream(model, context, options);
  return s.result();
}

export async function completeOrThrow(
  model: AnyModel,
  context: Context,
  options?: StreamOptions,
): Promise<AssistantMessage> {
  const s = stream(model, context, options);
  return s.resultOrThrow();
}

export function getApiKey(provider: Provider): string | undefined {
  switch (provider) {
    case "openai":
      return process.env.OPENAI_API_KEY;
    case "anthropic":
      return process.env.ANTHROPIC_API_KEY;
    case "gemini":
      return process.env.GEMINI_API_KEY;
    default:
      return exhaustive(provider);
  }
}

export function normalizeContextForTarget(target: Model, context: Context): NormalizedContext {
  const targetProvider = target.provider;
  const targetModel = target.id;

  const systemParts: string[] = [];
  if (context.system && context.system.trim().length > 0) systemParts.push(context.system.trim());
  for (const msg of context.messages) {
    if (msg.role === "system" && msg.content.trim().length > 0)
      systemParts.push(msg.content.trim());
  }
  const system = systemParts.length > 0 ? systemParts.join("\n\n") : undefined;

  const messages: NormalizedMessage[] = [];
  for (const msg of context.messages) {
    if (msg.role === "system") continue;

    if (msg.role === "user") {
      messages.push(msg);
      continue;
    }

    if (msg.role === "assistant") {
      messages.push(normalizeAssistantMessage(targetProvider, targetModel, msg));
      continue;
    }

    if (msg.role === "tool") {
      messages.push({ ...msg, isError: msg.isError ?? false });
    }
  }

  return {
    system,
    messages: messages.filter((m) => {
      if (m.role === "assistant") return m.content.length > 0;
      if (m.role === "user") return m.content.trim().length > 0;
      return true;
    }),
    tools: context.tools,
  };
}

function normalizeAssistantMessage(
  targetProvider: Provider,
  targetModel: string,
  msg: AssistantMessageInput,
): Extract<NormalizedMessage, { role: "assistant" }> {
  const parts = coerceAssistantContent(msg.content);

  if (msg.provider === targetProvider && msg.model === targetModel) {
    return {
      ...msg,
      content: parts.map((p) => ({ ...p })),
    };
  }

  const content: AssistantPart[] = [];
  for (const part of parts) {
    if (part.type === "text") {
      content.push({ type: "text", text: part.text, ...(part.meta ? { meta: part.meta } : {}) });
      continue;
    }
    if (part.type === "thinking") {
      continue;
    }
    if (part.type === "tool_call") {
      content.push({ ...part, args: part.args });
    }
  }

  return {
    role: "assistant",
    content,
  };
}

function coerceAssistantContent(content: AssistantMessageInput["content"]): AssistantPart[] {
  return typeof content === "string" ? [{ type: "text", text: content }] : content;
}

function contextUsesTools(context: NormalizedContext): boolean {
  if (context.tools && context.tools.length > 0) return true;
  for (const m of context.messages) {
    if (m.role === "tool") return true;
    if (m.role === "assistant" && m.content.some((p) => p.type === "tool_call")) return true;
  }
  return false;
}

const toolNameRe = /^[a-zA-Z0-9_-]{1,64}$/;

function validateTools(tools?: Tool[]): void {
  if (!tools || tools.length === 0) return;

  const names = new Set<string>();
  for (const tool of tools) {
    if (typeof tool.name !== "string" || tool.name.trim().length === 0) {
      throw new Error("Invalid tool definition: tool.name must be a non-empty string");
    }

    if (!toolNameRe.test(tool.name)) {
      throw new Error(
        `Invalid tool definition: tool.name '${tool.name}' must match ${toolNameRe.source} and be <= 64 chars`,
      );
    }

    if (names.has(tool.name)) {
      throw new Error(`Invalid tool definition: duplicate tool name '${tool.name}'`);
    }

    names.add(tool.name);

    if (tool.description !== undefined && typeof tool.description !== "string") {
      throw new Error(
        `Invalid tool definition: tool.description for '${tool.name}' must be a string`,
      );
    }

    validateJsonSchema(tool.parameters, `tool:${tool.name}`, true);
  }
}

const supportedSchemaKeys = new Set([
  "type",
  "description",
  "properties",
  "required",
  "enum",
  "items",
  "additionalProperties",
]);

const unsupportedSchemaKeys = new Set([
  "$ref",
  "definitions",
  "$defs",
  "oneOf",
  "anyOf",
  "allOf",
  "patternProperties",
  "propertyNames",
  "format",
  "pattern",
  "if",
  "then",
  "else",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "multipleOf",
  "minLength",
  "maxLength",
  "minItems",
  "maxItems",
  "uniqueItems",
  "minProperties",
  "maxProperties",
]);

function validateJsonSchema(schema: JsonSchema, path: string, isRoot: boolean): void {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error(`Invalid tool JSON Schema at ${path}: expected an object schema`);
  }

  for (const key of Object.keys(schema)) {
    if (unsupportedSchemaKeys.has(key)) {
      throw new Error(`Unsupported tool JSON Schema keyword at ${path}.${key}`);
    }
    if (!supportedSchemaKeys.has(key)) {
      throw new Error(`Unsupported tool JSON Schema keyword at ${path}.${key}`);
    }
  }

  const type = schema.type;
  if (type !== undefined) {
    if (typeof type !== "string") {
      throw new Error(`Invalid tool JSON Schema at ${path}.type: expected a string`);
    }
    const allowed = new Set(["object", "string", "number", "integer", "boolean", "array"]);
    if (!allowed.has(type)) {
      throw new Error(`Invalid tool JSON Schema at ${path}.type: unsupported type ${type}`);
    }
  }

  if (schema.description !== undefined && typeof schema.description !== "string") {
    throw new Error(`Invalid tool JSON Schema at ${path}.description: expected a string`);
  }

  if (isRoot) {
    if (schema.type !== "object") {
      throw new Error(`Invalid tool JSON Schema at ${path}: root schema must have type 'object'`);
    }
    if (
      !schema.properties ||
      typeof schema.properties !== "object" ||
      Array.isArray(schema.properties)
    ) {
      throw new Error(`Invalid tool JSON Schema at ${path}: root schema must have properties`);
    }
  }

  if (schema.properties !== undefined) {
    if (
      typeof schema.properties !== "object" ||
      !schema.properties ||
      Array.isArray(schema.properties)
    ) {
      throw new Error(`Invalid tool JSON Schema at ${path}.properties: expected an object`);
    }

    for (const [key, value] of Object.entries(schema.properties as Record<string, unknown>)) {
      validateJsonSchema(value as JsonSchema, `${path}.properties.${key}`, false);
    }
  }

  if (schema.required !== undefined) {
    if (!Array.isArray(schema.required) || schema.required.some((v) => typeof v !== "string")) {
      throw new Error(`Invalid tool JSON Schema at ${path}.required: expected string[]`);
    }

    if (
      schema.properties &&
      typeof schema.properties === "object" &&
      !Array.isArray(schema.properties)
    ) {
      for (const key of schema.required as string[]) {
        if (!(key in (schema.properties as Record<string, unknown>))) {
          throw new Error(
            `Invalid tool JSON Schema at ${path}.required: '${key}' is not present in properties`,
          );
        }
      }
    }
  }

  if (schema.enum !== undefined) {
    if (!Array.isArray(schema.enum)) {
      throw new Error(`Invalid tool JSON Schema at ${path}.enum: expected an array`);
    }

    for (const v of schema.enum) {
      const t = typeof v;
      if (v === null || (t !== "string" && t !== "number" && t !== "boolean")) {
        throw new Error(
          `Invalid tool JSON Schema at ${path}.enum: only string/number/boolean values are supported`,
        );
      }
    }
  }

  if (schema.items !== undefined) {
    if (schema.type !== "array") {
      throw new Error(`Invalid tool JSON Schema at ${path}.items: only valid when type is 'array'`);
    }
    validateJsonSchema(schema.items as JsonSchema, `${path}.items`, false);
  }

  if (schema.additionalProperties !== undefined) {
    if (typeof schema.additionalProperties !== "boolean") {
      throw new Error(`Invalid tool JSON Schema at ${path}.additionalProperties: expected boolean`);
    }
  }
}
