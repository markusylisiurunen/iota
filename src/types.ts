export type Provider = "openai" | "anthropic" | "gemini";

export type ReasoningEffort = "none" | "minimal" | "low" | "medium" | "high" | "xhigh";

export type StopReason = "stop" | "length" | "tool_use" | "error" | "aborted";

export type Usage = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
};

export type AssistantPartMeta =
  | { provider: "openai"; type: "reasoning_item"; item: unknown }
  | { provider: "openai"; type: "message_id"; id: string }
  | { provider: "openai"; type: "function_call_item_id"; id: string }
  | { provider: "anthropic"; type: "thinking_signature"; signature: string }
  | { provider: "gemini"; type: "thought_signature"; signature: string };

export type AssistantPart =
  | { type: "text"; text: string; meta?: AssistantPartMeta }
  | { type: "thinking"; text: string; meta?: AssistantPartMeta }
  | { type: "tool_call"; id: string; name: string; args: unknown; meta?: AssistantPartMeta };

export type SystemMessage = { role: "system"; content: string };
export type UserMessage = { role: "user"; content: string };

export type ToolMessage = {
  role: "tool";
  toolCallId: string;
  toolName: string;
  content: string;
  isError?: boolean;
};

export type AssistantMessageInput = {
  role: "assistant";
  content: string | AssistantPart[];

  // Optional to support “bring your own history”.
  provider?: Provider;
  model?: string;

  stopReason?: StopReason;
  usage?: Usage;
  errorMessage?: string;
};

export type AssistantMessageDraft = {
  role: "assistant";
  provider: Provider;
  model: string;
  content: AssistantPart[];

  stopReason?: StopReason;
  usage?: Usage;
  errorMessage?: string;
};

export type AssistantMessage = {
  role: "assistant";
  provider: Provider;
  model: string;
  content: AssistantPart[];
  stopReason: StopReason;
  usage: Usage;
  errorMessage?: string;
};

export type Message = SystemMessage | UserMessage | AssistantMessageInput | ToolMessage;

export type NormalizedToolMessage = ToolMessage & { isError: boolean };
export type NormalizedAssistantMessageInput = Omit<AssistantMessageInput, "content"> & {
  content: AssistantPart[];
};
export type NormalizedMessage =
  | UserMessage
  | NormalizedAssistantMessageInput
  | NormalizedToolMessage;

export type JsonSchema = Record<string, unknown>;

export type Tool = {
  name: string;
  description?: string;
  parameters: JsonSchema;
};

export type Context = {
  system?: string;
  messages: Message[];
  tools?: Tool[];
};

export type NormalizedContext = {
  system?: string;
  messages: NormalizedMessage[];
  tools?: Tool[];
};

export type StreamOptions = {
  apiKey?: string;
  temperature?: number;
  maxTokens?: number;
  reasoning?: ReasoningEffort;
  signal?: AbortSignal;
};

export type ResolvedStreamOptions = Omit<StreamOptions, "apiKey" | "maxTokens" | "reasoning"> & {
  apiKey: string;
  maxTokens: number;
  reasoning: ReasoningEffort;
};

export type AssistantStreamEvent =
  | { type: "start"; partial: AssistantMessageDraft }
  | { type: "part_start"; index: number; partial: AssistantMessageDraft }
  | { type: "part_delta"; index: number; delta: string; partial: AssistantMessageDraft }
  | { type: "part_end"; index: number; partial: AssistantMessageDraft }
  | { type: "done"; message: AssistantMessage }
  | { type: "error"; error: AssistantMessage };
