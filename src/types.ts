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

export type AssistantPart =
  | { type: "text"; text: string; signature?: string }
  | { type: "thinking"; text: string; signature?: string }
  | { type: "tool_call"; id: string; name: string; args: unknown; signature?: string };

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
  // If missing (or mismatched to the target), the message is treated as provider-agnostic plain text.
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

export type StreamOptions = {
  apiKey?: string;
  temperature?: number;
  maxTokens?: number;
  reasoning?: ReasoningEffort;
  signal?: AbortSignal;
};

export type AssistantStreamEvent =
  | { type: "start"; partial: AssistantMessageDraft }
  | { type: "part_start"; index: number; partial: AssistantMessageDraft }
  | { type: "part_delta"; index: number; delta: string; partial: AssistantMessageDraft }
  | { type: "part_end"; index: number; partial: AssistantMessageDraft }
  | { type: "done"; message: AssistantMessage }
  | { type: "error"; error: AssistantMessage };
