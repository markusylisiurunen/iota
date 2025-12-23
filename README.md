# iota

a small typescript library for streaming chat completions across **openai**, **anthropic**, and **gemini**, with a unified event stream, tool calling parts, reasoning summaries (when available), and token usage + cost tracking.

## installation

```sh
npm install @markusylisiurunen/iota
```

## api keys

iota looks up API keys from environment variables by default:

```sh
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...
```

you can also pass a key explicitly per call via `stream(..., { apiKey })`.

## quickstart

```ts
import { getModel, stream } from "@markusylisiurunen/iota";

const model = getModel("anthropic", "opus-4.5");

const s = stream(model, {
  messages: [{ role: "user", content: "write a haiku about typescript" }],
});

for await (const e of s) {
  if (e.type === "part_delta") process.stdout.write(e.delta);
}

const final = await s.result();
console.log("\nstopReason:", final.stopReason);
console.log("tokens:", final.usage.totalTokens);
console.log("cost (usd):", final.usage.cost.total);
```

## tools (function calling)

provide tool definitions via `context.tools`. tools use a restricted subset of json schema (root must be an `object` with `properties`).

iota does not execute tools for you, it streams `tool_call` parts that you can handle.

```ts
import { getModel, stream, type Message, type Tool } from "@markusylisiurunen/iota";

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
      additionalProperties: false,
    },
  },
];

const messages: Message[] = [
  { role: "user", content: "what is 2+3? use the add tool." },
];

while (true) {
  const s = stream(model, { messages, tools });
  const assistant = await s.result();
  messages.push(assistant);

  const toolCalls = assistant.content.filter((p) => p.type === "tool_call");
  if (toolCalls.length === 0) break;

  for (const call of toolCalls) {
    if (call.name !== "add") continue;
    const { a, b } = call.args as any;

    messages.push({
      role: "tool",
      toolCallId: call.id,
      toolName: call.name,
      content: String(Number(a) + Number(b)),
    });
  }
}
```

## reasoning

set `options.reasoning` to request reasoning summaries when supported:

- openai: uses responses api reasoning summaries
- anthropic: uses `thinking` blocks (budget depends on `reasoning` + `maxTokens`)
- gemini: uses thinking summaries + a provider-specific thinking level

```ts
import { getModel, complete } from "@markusylisiurunen/iota";

const model = getModel("gemini", "gemini-3-pro-preview");

const msg = await complete(
  model,
  { messages: [{ role: "user", content: "prove that 1+1=2" }] },
  { reasoning: "high" },
);
```

## development

iota requires Node.js 20+.

```sh
npm install
npm run check
npm run build
```

## creating a release

publishing to npm happens automatically via github actions when a github release is published.

release steps:

- run checks and build:

```sh
npm run check
npm run build
```

- bump the version (creates a git tag):

```sh
npm version patch
```

- push the commit and tag:

```sh
git push --follow-tags
```

- create a github release (this triggers the publish workflow):

```sh
gh release create v$(node -p "require('./package.json').version") --generate-notes
```
