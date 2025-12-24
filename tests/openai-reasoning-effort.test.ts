import { beforeEach, describe, expect, it, vi } from "vitest";

let capturedParams: any;

vi.mock("openai", () => {
  class OpenAIMock {
    responses: {
      create: (params: any) => Promise<AsyncIterable<unknown>>;
    };

    constructor(_options: any) {
      this.responses = {
        create: async (params: any) => {
          capturedParams = params;
          return (async function* () {})();
        },
      };
    }
  }

  return { default: OpenAIMock };
});

import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";
import type { Context } from "../src/types.js";

describe("openai reasoning effort", () => {
  beforeEach(() => {
    capturedParams = undefined;
  });

  it("passes xhigh through for gpt-5.2", async () => {
    const model = getModel("openai", "gpt-5.2");

    const context: Context = {
      messages: [{ role: "user", content: "hello" }],
    };

    const s = stream(model, context, { apiKey: "test", reasoning: "xhigh" });
    await s.result();

    expect(capturedParams.reasoning).toMatchObject({ effort: "xhigh" });
  });
});
