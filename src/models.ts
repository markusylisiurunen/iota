import type { Provider, ReasoningEffort, ServiceTier, Usage } from "./types.js";
import { exhaustive } from "./utils/exhaustive.js";

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
  contextWindow: number;
  maxOutputTokens: number;
  supports: {
    reasoning: boolean;
    tools: boolean;
    reasoningXhigh: boolean;
  };
  pricing: Pricing;
};

export function calculateCost(
  model: Model,
  usage: Pick<Usage, "inputTokens" | "outputTokens" | "cacheReadTokens" | "cacheWriteTokens">,
  serviceTier?: ServiceTier,
): Usage["cost"] {
  const inputBase = (model.pricing.inputPer1M / 1_000_000) * usage.inputTokens;
  const outputBase = (model.pricing.outputPer1M / 1_000_000) * usage.outputTokens;
  const cacheReadBase = (model.pricing.cacheReadPer1M / 1_000_000) * usage.cacheReadTokens;
  const cacheWriteBase = (model.pricing.cacheWritePer1M / 1_000_000) * usage.cacheWriteTokens;

  const multiplier = openaiServiceTierCostMultiplier(model.provider, serviceTier);

  const input = inputBase * multiplier;
  const output = outputBase * multiplier;
  const cacheRead = cacheReadBase * multiplier;
  const cacheWrite = cacheWriteBase * multiplier;
  const total = input + output + cacheRead + cacheWrite;

  return { input, output, cacheRead, cacheWrite, total };
}

function openaiServiceTierCostMultiplier(provider: Provider, tier?: ServiceTier): number {
  if (provider !== "openai") return 1;

  switch (tier) {
    case undefined:
    case "standard":
      return 1;
    case "flex":
      return 0.5;
    case "priority":
      return 2;
    default:
      return exhaustive(tier);
  }
}

const gpt52 = {
  provider: "openai",
  id: "gpt-5.2",
  name: "GPT-5.2",
  contextWindow: 400000,
  maxOutputTokens: 128000,
  supports: { reasoning: true, tools: true, reasoningXhigh: true },
  pricing: {
    inputPer1M: 1.75,
    outputPer1M: 14,
    cacheReadPer1M: 0.175,
    cacheWritePer1M: 0,
  },
} as const satisfies Model<"openai">;

export const openaiModels = {
  "gpt-5.2": gpt52,
} as const;

export type OpenAIModelId = keyof typeof openaiModels;
export type OpenAIModel = (typeof openaiModels)[OpenAIModelId];

export const anthropicModels = {
  "opus-4.5": {
    provider: "anthropic",
    id: "claude-opus-4-5",
    name: "Claude Opus 4.5",
    contextWindow: 200000,
    maxOutputTokens: 64000,
    supports: { reasoning: true, tools: true, reasoningXhigh: true },
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
    name: "Claude Haiku 4.5",
    contextWindow: 200000,
    maxOutputTokens: 64000,
    supports: { reasoning: true, tools: true, reasoningXhigh: true },
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

export const googleModels = {
  "gemini-3-pro-preview": {
    provider: "google",
    id: "gemini-3-pro-preview",
    name: "Gemini 3 Pro Preview",
    contextWindow: 1048576,
    maxOutputTokens: 65536,
    supports: { reasoning: true, tools: true, reasoningXhigh: false },
    pricing: {
      inputPer1M: 2,
      outputPer1M: 12,
      cacheReadPer1M: 0.2,
      cacheWritePer1M: 0,
    },
  } as const satisfies Model<"google">,
  "gemini-3-flash-preview": {
    provider: "google",
    id: "gemini-3-flash-preview",
    name: "Gemini 3 Flash Preview",
    contextWindow: 1048576,
    maxOutputTokens: 65536,
    supports: { reasoning: true, tools: true, reasoningXhigh: false },
    pricing: {
      inputPer1M: 0.5,
      outputPer1M: 3,
      cacheReadPer1M: 0.05,
      cacheWritePer1M: 0,
    },
  } as const satisfies Model<"google">,
} as const;

export type GoogleModelId = keyof typeof googleModels;
export type GoogleModel = (typeof googleModels)[GoogleModelId];

export type AnyModel = OpenAIModel | AnthropicModel | GoogleModel;

export function getModel(provider: "openai", id: OpenAIModelId): OpenAIModel;
export function getModel(provider: "anthropic", id: AnthropicModelId): AnthropicModel;
export function getModel(provider: "google", id: GoogleModelId): GoogleModel;
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
    case "google": {
      const m = (googleModels as Record<string, AnyModel>)[id];
      if (!m) throw new Error(`Unknown Google model: ${id}`);
      return m;
    }
    default:
      return exhaustive(provider);
  }
}

export function supportsXhigh(model: { supports: { reasoningXhigh: boolean } }): boolean {
  return model.supports.reasoningXhigh;
}

export function clampReasoning(effort: ReasoningEffort): Exclude<ReasoningEffort, "xhigh"> {
  return effort === "xhigh" ? "high" : effort;
}

export function clampReasoningForModel(model: Model, effort: ReasoningEffort): ReasoningEffort {
  if (effort === "none") return "none";
  if (!model.supports.reasoning) return "none";
  if (!supportsXhigh(model)) return clampReasoning(effort);
  return effort;
}
