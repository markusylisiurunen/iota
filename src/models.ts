import type { Provider, Usage } from "./types.js";

export type Pricing = {
  inputPer1M: number;
  outputPer1M: number;
  cacheReadPer1M: number;
  cacheWritePer1M: number;
};

export type Model<P extends Provider = Provider> = {
  provider: P;
  id: string;
  name: string;
  baseUrl: string;
  contextWindow: number;
  maxOutputTokens: number;
  supports: {
    reasoning: boolean;
    tools: boolean;
  };
  pricing: Pricing;
};

export function calculateCost(model: Model, usage: Usage): Usage["cost"] {
  usage.cost.input = (model.pricing.inputPer1M / 1_000_000) * usage.inputTokens;
  usage.cost.output = (model.pricing.outputPer1M / 1_000_000) * usage.outputTokens;
  usage.cost.cacheRead = (model.pricing.cacheReadPer1M / 1_000_000) * usage.cacheReadTokens;
  usage.cost.cacheWrite = (model.pricing.cacheWritePer1M / 1_000_000) * usage.cacheWriteTokens;
  usage.cost.total =
    usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite;
  return usage.cost;
}

const gpt52Chat = {
  provider: "openai",
  id: "gpt-5.2-chat-latest",
  name: "GPT-5.2 Chat",
  baseUrl: "https://api.openai.com/v1",
  contextWindow: 128000,
  maxOutputTokens: 16384,
  supports: { reasoning: true, tools: true },
  pricing: {
    inputPer1M: 1.75,
    outputPer1M: 14,
    cacheReadPer1M: 0.175,
    cacheWritePer1M: 0,
  },
} as const satisfies Model<"openai">;

export const openaiModels = {
  "gpt-5.2": gpt52Chat,
  "gpt-5.2-chat-latest": gpt52Chat,
} as const;

export type OpenAIModelId = keyof typeof openaiModels;
export type OpenAIModel = (typeof openaiModels)[OpenAIModelId];

export const anthropicModels = {
  "opus-4.5": {
    provider: "anthropic",
    id: "claude-opus-4-5",
    name: "Claude Opus 4.5 (latest)",
    baseUrl: "https://api.anthropic.com",
    contextWindow: 200000,
    maxOutputTokens: 64000,
    supports: { reasoning: true, tools: true },
    pricing: {
      inputPer1M: 5,
      outputPer1M: 25,
      cacheReadPer1M: 0.5,
      cacheWritePer1M: 6.25,
    },
  } as const satisfies Model<"anthropic">,
  "haiku-4.5": {
    provider: "anthropic",
    id: "claude-haiku-4-5",
    name: "Claude Haiku 4.5 (latest)",
    baseUrl: "https://api.anthropic.com",
    contextWindow: 200000,
    maxOutputTokens: 64000,
    supports: { reasoning: true, tools: true },
    pricing: {
      inputPer1M: 1,
      outputPer1M: 5,
      cacheReadPer1M: 0.1,
      cacheWritePer1M: 1.25,
    },
  } as const satisfies Model<"anthropic">,
} as const;

export type AnthropicModelId = keyof typeof anthropicModels;
export type AnthropicModel = (typeof anthropicModels)[AnthropicModelId];

export const geminiModels = {
  "gemini-3-pro-preview": {
    provider: "gemini",
    id: "gemini-3-pro-preview",
    name: "Gemini 3 Pro Preview",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    contextWindow: 1000000,
    maxOutputTokens: 64000,
    supports: { reasoning: true, tools: true },
    pricing: {
      inputPer1M: 2,
      outputPer1M: 12,
      cacheReadPer1M: 0.2,
      cacheWritePer1M: 0,
    },
  } as const satisfies Model<"gemini">,
  "gemini-3-flash-preview": {
    provider: "gemini",
    id: "gemini-3-flash-preview",
    name: "Gemini 3 Flash Preview",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    contextWindow: 1048576,
    maxOutputTokens: 65536,
    supports: { reasoning: true, tools: true },
    pricing: {
      inputPer1M: 0.5,
      outputPer1M: 3,
      cacheReadPer1M: 0.05,
      cacheWritePer1M: 0,
    },
  } as const satisfies Model<"gemini">,
} as const;

export type GeminiModelId = keyof typeof geminiModels;
export type GeminiModel = (typeof geminiModels)[GeminiModelId];

export type AnyModel = OpenAIModel | AnthropicModel | GeminiModel;

export function getModel(provider: "openai", id: OpenAIModelId): OpenAIModel;
export function getModel(provider: "anthropic", id: AnthropicModelId): AnthropicModel;
export function getModel(provider: "gemini", id: GeminiModelId): GeminiModel;
export function getModel(provider: Provider, id: string): AnyModel {
  switch (provider) {
    case "openai": {
      const m = (openaiModels as Record<string, AnyModel>)[id];
      if (!m) throw new Error(`Unknown OpenAI model: ${id}`);
      return m;
    }
    case "anthropic": {
      const m = (anthropicModels as Record<string, AnyModel>)[id];
      if (!m) throw new Error(`Unknown Anthropic model: ${id}`);
      return m;
    }
    case "gemini": {
      const m = (geminiModels as Record<string, AnyModel>)[id];
      if (!m) throw new Error(`Unknown Gemini model: ${id}`);
      return m;
    }
  }
}

export function supportsXhigh(model: Model<"openai">): boolean {
  return model.id.startsWith("gpt-5.2");
}
