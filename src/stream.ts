import type { AssistantStream } from "./assistant-stream.js";
import type { AnyModel, Model } from "./models.js";
import { streamAnthropic } from "./providers/anthropic.js";
import { streamGemini } from "./providers/gemini.js";
import { streamOpenAI } from "./providers/openai.js";
import type {
  AssistantMessage,
  AssistantMessageInput,
  AssistantPart,
  Context,
  JsonSchema,
  Message,
  Provider,
  ReasoningEffort,
  StreamOptions,
  Tool,
  ToolMessage,
  Usage,
} from "./types.js";

export function stream(
  model: AnyModel,
  context: Context,
  options: StreamOptions = {},
): AssistantStream {
  const apiKey = options.apiKey ?? getApiKey(model.provider);
  if (!apiKey) throw new Error(`Missing API key for provider: ${model.provider}`);

  const reasoning = options.reasoning ?? "none";
  const normalized = normalizeContextForTarget(model, context);
  validateToolSchemas(normalized.tools);

  const resolvedOptions: StreamOptions = {
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
  }
}

export async function complete(
  model: AnyModel,
  context: Context,
  options?: StreamOptions,
): Promise<AssistantMessage> {
  const s = stream(model, context, options);
  return s.result();
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
      return undefined;
  }
}

function normalizeContextForTarget(target: Model, context: Context): Context {
  const targetProvider = target.provider;
  const targetModel = target.id;

  const systemParts: string[] = [];
  if (context.system && context.system.trim().length > 0) systemParts.push(context.system.trim());
  for (const msg of context.messages) {
    if (msg.role === "system" && msg.content.trim().length > 0)
      systemParts.push(msg.content.trim());
  }
  const system = systemParts.length > 0 ? systemParts.join("\n\n") : undefined;

  const toolCallIds = new Set<string>();
  for (const msg of context.messages) {
    if (msg.role !== "assistant") continue;
    if (msg.provider !== targetProvider || msg.model !== targetModel) continue;
    for (const part of coerceAssistantContent(msg.content)) {
      if (part.type === "tool_call") toolCallIds.add(part.id);
    }
  }

  const messages: Message[] = [];
  for (const msg of context.messages) {
    if (msg.role === "system") continue;

    if (msg.role === "assistant") {
      messages.push(normalizeAssistantMessage(targetProvider, targetModel, msg));
      continue;
    }

    if (msg.role === "tool") {
      if (toolCallIds.has(msg.toolCallId)) {
        messages.push({ ...msg, isError: msg.isError ?? false });
      } else {
        messages.push({ role: "user", content: toolResultTranscript(msg) });
      }
      continue;
    }

    messages.push(msg);
  }

  return {
    system,
    messages: messages.filter((m) => {
      if (m.role === "assistant") return coerceAssistantContent(m.content).length > 0;
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
): AssistantMessageInput {
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
      content.push({ type: "text", text: part.text });
      continue;
    }
    if (part.type === "thinking") {
      continue;
    }
    if (part.type === "tool_call") {
      content.push({
        type: "text",
        text: toolCallTranscript(part),
      });
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

function toolCallTranscript(part: Extract<AssistantPart, { type: "tool_call" }>): string {
  const args = safeJson(part.args);
  return `[iota tool_call] name=${part.name} id=${part.id} args=${args}`;
}

function toolResultTranscript(msg: ToolMessage): string {
  const isError = msg.isError ?? false;
  return `[iota tool_result] name=${msg.toolName} id=${msg.toolCallId} isError=${String(isError)}\n${msg.content}`;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "[unserializable]";
  }
}

export function emptyUsage(): Usage {
  return {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    totalTokens: 0,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: 0,
    },
  };
}

export function clampReasoning(effort: ReasoningEffort): Exclude<ReasoningEffort, "xhigh"> {
  return effort === "xhigh" ? "high" : effort;
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

function validateToolSchemas(tools?: Tool[]): void {
  if (!tools || tools.length === 0) return;

  for (const tool of tools) {
    validateJsonSchema(tool.parameters, `tool:${tool.name}`, true);
  }
}

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
