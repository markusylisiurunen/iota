# Iota

Multi-provider LLM streaming library supporting OpenAI, Anthropic, and Gemini. Iota provides a unified interface for streaming completions with tool calling, reasoning summaries (extended thinking), and token usage + cost tracking.

## Architecture

- **stream.ts** (`src/stream.ts`): Public entry point. Normalizes context across providers, validates tool schemas, resolves API keys, and dispatches to provider-specific implementations.
- **providers/** (`src/providers/*.ts`): Provider-specific adapters that map `Context` ↔ SDK formats and emit a unified `AssistantStreamEvent` stream.
- **assistant-stream.ts** (`src/assistant-stream.ts`): Specialized `EventStream` that completes on `done`/`error` and exposes `result()`.
- **event-stream.ts** (`src/event-stream.ts`): Generic async-iterable queue with a final result promise.
- **models.ts** (`src/models.ts`): Model registry, capabilities, pricing, and `calculateCost()`.
- **utils/json.ts**: Streaming JSON parsing for incremental tool call args.
- **utils/sanitize.ts**: Replaces invalid unicode surrogate pairs to avoid SDK / JSON failures.

**Data flow**: `stream(model, context, options)` → normalize + validate → provider stream → `AssistantStreamEvent` events → final `AssistantMessage`.

## Key types

- `Context`: `{ system?, messages, tools? }`
- `Message`: `system` | `user` | `assistant` | `tool`
- `AssistantPart`: `text` | `thinking` | `tool_call`
- `AssistantStreamEvent`: `start`/`part_start`/`part_delta`/`part_end`/`done`/`error`

Iota is intentionally low-level: it surfaces tool calls and tool results in the message history, but it does not run tools or implement an agent loop.

## Context normalization

Context normalization lives in `src/stream.ts` and is designed to make histories portable across providers.

- System messages: `context.system` plus any `messages[]` with `role: "system"` are merged into a single system string. System messages are removed from the regular message list.
- Cross-provider assistant history:
  - If an assistant message does not match the target `{ provider, model }`, it is treated as provider-agnostic.
  - `thinking` parts are dropped.
  - `tool_call` parts are converted into a plain text transcript (`[iota tool_call] ...`) so the history remains understandable even when the target provider cannot round-trip the tool call.
- Tool results:
  - If a `tool` message refers to a tool call id that exists in the target provider/model history, it is kept as a tool result.
  - Otherwise, it is converted into a user transcript (`[iota tool_result] ...`) so the conversation stays consistent.

## Tool schema support

Tool parameters are validated in `src/stream.ts`.

- Root schema must have `type: "object"` and `properties`.
- Supported keywords: `type`, `description`, `properties`, `required`, `enum`, `items`, `additionalProperties`.
- Explicitly rejected keywords include: `$ref`, `definitions`, `$defs`, `oneOf`, `anyOf`, `allOf`, regex/pattern-related keywords, and numeric/string constraints.

This is a deliberate compatibility subset across providers.

## Provider notes

- **OpenAI** (`src/providers/openai.ts`)
  - Uses the OpenAI **Responses API** streaming events.
  - Reasoning is represented as `thinking` parts. For round-tripping, OpenAI reasoning items are stored in `AssistantPart.signature` as JSON.
  - Tool call args stream incrementally and are parsed via `partial-json`.
- **Anthropic** (`src/providers/anthropic.ts`)
  - Uses `@anthropic-ai/sdk` streaming events.
  - Tool call ids are sanitized to match Anthropic requirements (`sanitizeToolCallId()`), so do not assume ids are preserved byte-for-byte across providers.
  - Thinking blocks are included only when `options.reasoning !== "none"` and are filtered from final output if they do not have a signature.
- **Gemini** (`src/providers/gemini.ts`)
  - Uses `@google/genai` interactions streaming.
  - Reasoning summaries are surfaced as `thinking` parts when enabled.

All providers fill `message.usage` and compute `message.usage.cost` via `calculateCost(model, usage)`.

## Development

- `npm run check` - Format (Biome) + typecheck
- `npm run build` - Build ESM + CJS to `dist/`

**Style**: Biome (2-space indent, 100 line width). Types `PascalCase`, values/functions `camelCase`, files `lowercase.ts`.

## Releasing

Publishing to npm happens automatically via GitHub Actions when a GitHub Release is published (see `.github/workflows/npm-publish.yml`).

1. Ensure you are on `main` with a clean working tree.
2. Verify and build:
   - `npm run check`
   - `npm run build`
3. Bump the version and create a tag:
   - `npm version patch|minor|major` (creates a `vX.Y.Z` tag)
4. Push the commit and tag:
   - `git push --follow-tags`
5. Create a GitHub Release (triggers publish):
   - `gh release create v$(node -p "require('./package.json').version") --generate-notes`

## Maintaining this file

Keep AGENTS.md and README.md in sync with the codebase. When changes affect architecture, exports, provider behavior, or supported schema surface area, update the relevant sections here.
