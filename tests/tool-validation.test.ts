import { describe, expect, it } from "vitest";

import { getModel } from "../src/models.js";
import { stream } from "../src/stream.js";

describe("tool validation", () => {
  it("rejects duplicate tool names", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/duplicate tool name/);
  });

  it("rejects tool names that are not provider-safe", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "not ok",
              parameters: {
                type: "object",
                properties: {},
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/tool\.name/);
  });

  it("rejects required keys not present in properties", () => {
    const model = getModel("openai", "gpt-5.2");

    expect(() =>
      stream(
        model,
        {
          messages: [],
          tools: [
            {
              name: "add",
              parameters: {
                type: "object",
                properties: {
                  a: { type: "number" },
                },
                required: ["a", "b"],
                additionalProperties: false,
              },
            },
          ],
        },
        { apiKey: "x" },
      ),
    ).toThrow(/required/);
  });
});
