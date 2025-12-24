# iota

a typescript library for streaming LLM completions across OpenAI, Anthropic, and Google. provides a unified event interface for streaming responses with tool calling, reasoning summaries, and token usage with cost tracking.

## installation

```sh
npm install @markusylisiurunen/iota@latest
```

## api keys

iota looks for API keys in environment variables:

```sh
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
```

you can also pass a key explicitly via `options.apiKey`.

## quickstart

```ts
import { getModel, stream } from "@markusylisiurunen/iota";

const model = getModel("anthropic", "opus-4.5");

const s = stream(model, {
  messages: [{ role: "user", content: "write a haiku about typescript" }],
});

for await (const event of s) {
  if (event.type === "part_delta") {
    process.stdout.write(event.delta);
  }
}

const message = await s.result();
console.log("\nstop reason:", message.stopReason);
console.log("tokens:", message.usage.totalTokens);
console.log("cost:", message.usage.cost.total);
```

## context

the `stream()` function accepts a model and a context object:

```ts
type Context = {
  system?: string;
  messages: Message[];
  tools?: Tool[];
};
```

messages can be user messages, assistant messages, or tool results:

```ts
// user message
{ role: "user", content: "hello" }

// assistant message (from a previous response)
{ role: "assistant", content: [{ type: "text", text: "hi there" }], provider: "anthropic", model: "claude-opus-4-5" }

// tool result
{ role: "tool", toolCallId: "call_123", toolName: "add", content: "5" }
```

assistant messages from a different provider/model are automatically normalized. thinking parts are stripped (they're not portable), while text and tool calls are preserved.

## streaming events

iterate over the stream to receive events as they arrive:

```ts
for await (const event of stream(model, context)) {
  switch (event.type) {
    case "start":
      // stream began
      break;
    case "part_start":
      // new content part at event.index
      break;
    case "part_delta":
      // incremental text for part at event.index
      console.log(event.delta);
      break;
    case "part_end":
      // part at event.index is complete
      break;
    case "done":
      // stream finished, event.message is the final AssistantMessage
      break;
    case "error":
      // stream failed, event.error contains partial message
      break;
  }
}
```

every event includes `partial`, a draft of the message so far.

## assistant messages

the final message contains structured content parts:

```ts
const message = await stream(model, context).result();

for (const part of message.content) {
  switch (part.type) {
    case "text":
      console.log("text:", part.text);
      break;
    case "thinking":
      console.log("reasoning:", part.text);
      break;
    case "tool_call":
      console.log("tool call:", part.name, part.args);
      break;
  }
}
```

the message also includes `stopReason` (`"stop"`, `"length"`, `"tool_use"`, `"error"`, or `"aborted"`), `usage` with token counts and costs, and optionally `errorMessage`.

## tools

define tools with a name, description, and JSON Schema parameters:

```ts
const tools: Tool[] = [
  {
    name: "add",
    description: "add two numbers",
    parameters: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
    },
  },
];

const s = stream(model, { messages, tools });
```

tool names must match `/^[a-zA-Z0-9_-]{1,64}$/` and be unique.

### schema restrictions

iota validates tool schemas against a strict subset of JSON Schema that all providers support:

- root must be `type: "object"` with `properties`
- allowed keywords: `type`, `description`, `properties`, `required`, `enum`, `minimum`, `maximum`, `items`, `additionalProperties`
- rejected keywords: `$ref`, `definitions`, `oneOf`, `anyOf`, `allOf`, `pattern`, `format`, and others

this keeps tool definitions portable across providers.

### manual tool execution

```ts
const messages: Message[] = [{ role: "user", content: "what is 2 + 3?" }];

while (true) {
  const assistant = await stream(model, { messages, tools }).result();
  messages.push(assistant);

  const toolCalls = assistant.content.filter((p) => p.type === "tool_call");
  if (toolCalls.length === 0) break;

  for (const call of toolCalls) {
    const result = executeMyTool(call.name, call.args);
    messages.push({
      role: "tool",
      toolCallId: call.id,
      toolName: call.name,
      content: String(result),
    });
  }
}
```

### automatic tool execution with agent()

`agent()` runs a multi-turn loop, executing tools automatically:

```ts
import { getModel, agent } from "@markusylisiurunen/iota";

const model = getModel("openai", "gpt-5.2");

const tools: Tool[] = [
  {
    name: "add",
    description: "add two numbers",
    parameters: {
      type: "object",
      properties: {
        a: { type: "number" },
        b: { type: "number" },
      },
      required: ["a", "b"],
    },
  },
];

const handlers = {
  add: (args: unknown) => {
    const { a, b } = args as { a: number; b: number };
    return a + b;
  },
};

const s = agent(
  model,
  { messages: [{ role: "user", content: "what is 2 + 3?" }], tools },
  handlers,
  { maxTurns: 10 },
);

for await (const event of s) {
  switch (event.type) {
    case "turn_start":
      console.log(`turn ${event.turn}`);
      break;
    case "assistant_event":
      if (event.event.type === "part_delta") {
        process.stdout.write(event.event.delta);
      }
      break;
    case "tool_result":
      console.log(`tool result: ${event.message.content}`);
      break;
    case "done":
      console.log("complete");
      break;
    case "error":
      console.log("error:", event.error.errorMessage);
      break;
  }
}

const result = await s.result();
```

the loop terminates when the model responds without tool calls, when `maxTurns` is exceeded, when an error occurs, or when an unknown tool is called.

## reasoning

request extended thinking with `options.reasoning`:

```ts
const s = stream(
  model,
  { messages: [{ role: "user", content: "prove that sqrt(2) is irrational" }] },
  { reasoning: "high" },
);
```

levels: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`.

reasoning is automatically clamped to what the model supports. thinking content appears as `{ type: "thinking" }` parts in the response.

each provider handles reasoning differently:

- **openai**: uses the Responses API with `reasoning.effort` and reasoning summaries
- **anthropic**: uses `thinking` blocks with a budget calculated from effort level and max tokens
- **google**: uses `thinkingConfig` with provider-specific thinking levels

thinking parts include provider-specific metadata (signatures, encrypted content) that enables round-tripping within the same provider.

## options

```ts
type StreamOptions = {
  apiKey?: string; // provider API key
  temperature?: number; // sampling temperature
  maxTokens?: number; // max output tokens (default: model.maxOutputTokens)
  reasoning?: ReasoningEffort;
  serviceTier?: ServiceTier; // openai only: "flex", "standard", "priority"
  signal?: AbortSignal; // cancel the request
};
```

## cancellation

pass an `AbortSignal` to cancel a request:

```ts
const controller = new AbortController();

const s = stream(model, context, { signal: controller.signal });

// later...
controller.abort();

const message = await s.result();
console.log(message.stopReason); // "aborted"
```

## error handling

streaming errors are captured and emitted as events:

```ts
const message = await s.result();

if (message.stopReason === "error") {
  console.log("failed:", message.errorMessage);
}
```

use `resultOrThrow()` to throw on error or abort:

```ts
try {
  const message = await s.resultOrThrow();
} catch (error) {
  // error.assistantMessage contains the partial response
  console.log(error.message);
}
```

pre-stream validation errors (missing API key, invalid schema) are thrown synchronously.

## convenience functions

```ts
// non-streaming, returns AssistantMessage
const message = await complete(model, context, options);

// throws on error/abort
const message = await completeOrThrow(model, context, options);
```

## models

iota includes a registry of supported models with pricing and capabilities:

```ts
import { getModel, openaiModels, anthropicModels, googleModels } from "@markusylisiurunen/iota";

const opus = getModel("anthropic", "opus-4.5");
const haiku = getModel("anthropic", "haiku-4.5");
const gpt = getModel("openai", "gpt-5.2");
const geminiPro = getModel("google", "gemini-3-pro-preview");
const geminiFlash = getModel("google", "gemini-3-flash-preview");
```

each model includes:

- `provider`, `id`, `name`
- `contextWindow`, `maxOutputTokens`
- `supports.reasoning`, `supports.tools`, `supports.reasoningXhigh`
- `pricing` (per 1M tokens for input, output, cache read, cache write)

## cost tracking

token usage and costs are calculated automatically:

```ts
const message = await stream(model, context).result();

console.log("input tokens:", message.usage.inputTokens);
console.log("output tokens:", message.usage.outputTokens);
console.log("cache read:", message.usage.cacheReadTokens);
console.log("cache write:", message.usage.cacheWriteTokens);
console.log("total tokens:", message.usage.totalTokens);

console.log("input cost:", message.usage.cost.input);
console.log("output cost:", message.usage.cost.output);
console.log("total cost:", message.usage.cost.total);
```

costs are in USD based on the model's pricing. for OpenAI, costs are adjusted by service tier (flex = 0.5x, priority = 2x).

## api reference

### functions

| function                                    | description                                       |
| ------------------------------------------- | ------------------------------------------------- |
| `stream(model, context, options?)`          | stream a completion, returns `AssistantStream`    |
| `agent(model, context, handlers, options?)` | run a multi-turn tool loop, returns `AgentStream` |
| `complete(model, context, options?)`        | non-streaming completion                          |
| `completeOrThrow(model, context, options?)` | non-streaming, throws on error                    |
| `getModel(provider, id)`                    | get a model from the registry                     |
| `getApiKey(provider)`                       | get API key from environment                      |
| `calculateCost(model, usage, serviceTier?)` | calculate cost breakdown                          |
| `normalizeContextForTarget(model, context)` | normalize context for a target model              |
| `clampReasoning(effort)`                    | clamp xhigh to high                               |
| `clampReasoningForModel(model, effort)`     | clamp reasoning to model capabilities             |
| `supportsXhigh(model)`                      | check if model supports xhigh reasoning           |

### classes

| class             | description                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| `AssistantStream` | async iterable of `AssistantStreamEvent`, with `result()` and `resultOrThrow()` |
| `AgentStream`     | async iterable of `AgentStreamEvent`, with `result()` and `resultOrThrow()`     |

### types

exported from the package: `Provider`, `ReasoningEffort`, `ServiceTier`, `StopReason`, `Usage`, `AssistantPart`, `AssistantPartMeta`, `SystemMessage`, `UserMessage`, `ToolMessage`, `AssistantMessage`, `AssistantMessageInput`, `Message`, `Tool`, `JsonSchema`, `Context`, `StreamOptions`, `AgentOptions`, `AgentResult`, `AssistantStreamEvent`, `AgentStreamEvent`, `ToolHandler`, `ToolHandlers`, `Model`, `Pricing`.

## development

requires Node.js 20+.

```sh
npm install
npm run check    # biome format + typecheck
npm run build    # esm + cjs to dist/
npm test         # unit tests (no API calls)
npm run smoke    # e2e tests (requires API keys, IOTA_SMOKE=1)
```

## releasing

publishing happens automatically via GitHub Actions when a release is created.

```sh
npm run check
npm run build
npm version patch   # or minor, major
git push --follow-tags
gh release create v$(node -p "require('./package.json').version") --generate-notes
```
