import { describe, expect } from "vitest";

import { getModel } from "../src/models.js";
import type { Context, Tool, ToolMessage } from "../src/types.js";

import {
  assistantText,
  expectNoToolCalls,
  expectNoVisibleText,
  expectThinkingSignaturesAnthropic,
  expectThinkingThenText,
  expectThinkingThenToolCalls,
  itIfSmokeAndEnv,
  parseJsonObjectFromText,
  runTurn,
} from "./e2e-utils.js";

const tools: Tool[] = [
  {
    name: "random_number",
    description: "Generate a random integer in [min, max] inclusive.",
    parameters: {
      type: "object",
      properties: {
        min: { type: "integer" },
        max: { type: "integer" },
      },
      required: ["min", "max"],
      additionalProperties: false,
    },
  },
];

describe("interleaved thinking", () => {
  const model = getModel("anthropic", "haiku-4.5");

  itIfSmokeAndEnv("ANTHROPIC_API_KEY", "anthropic/haiku-4.5", async () => {
    const a = 3;
    const b = 7;
    const sum = a + b;

    const context: Context = {
      system: "You are a deterministic tool-using agent. Follow instructions exactly.",
      tools,
      messages: [
        {
          role: "user",
          content:
            "Follow these steps strictly:\n" +
            "1) In your first assistant turn, think about the task, then call the tool random_number twice (in parallel) with {min: 2, max: 9}. Do not output any user-visible text in that turn.\n" +
            "2) After receiving both tool results, think about the sum, then output ONLY the sum as digits.",
        },
      ],
    };

    const turn1 = await runTurn(model, context, { reasoning: "high" });
    context.messages.push(turn1.message);

    expectNoVisibleText(turn1.message);
    const calls1 = expectThinkingThenToolCalls(turn1.message, { random_number: 2 });
    expectThinkingSignaturesAnthropic(turn1.message);

    const toolResults: ToolMessage[] = calls1.map((call, i) => ({
      role: "tool",
      toolCallId: call.id,
      toolName: call.name,
      content: JSON.stringify({ value: i === 0 ? a : b }),
    }));

    context.messages.push(...toolResults);

    const turn2 = await runTurn(model, context, { reasoning: "high" });

    expectNoToolCalls(turn2.message);
    expectThinkingThenText(turn2.message);

    expect(turn2.message.stopReason).toBe("stop");
    expect(turn2.message.errorMessage).toBeUndefined();

    expect(assistantText(turn2.message).trim()).toBe(String(sum));

    expectThinkingSignaturesAnthropic(turn2.message);

    // Sanity: tool results are JSON objects.
    for (const msg of toolResults) parseJsonObjectFromText(msg.content);
  });
});
