import { EventStream } from "./event-stream.js";
import type { AgentResult, AgentStreamEvent } from "./types.js";
import { exhaustive } from "./utils/exhaustive.js";

export class AgentStream extends EventStream<AgentStreamEvent, AgentResult> {
  constructor() {
    super(
      (e) => e.type === "done" || e.type === "error",
      (e) => {
        switch (e.type) {
          case "done":
            return e.result;
          case "error":
            return e.error;
          case "turn_start":
          case "assistant_event":
          case "tool_result":
            throw new Error("Unexpected event type");
          default:
            return exhaustive(e);
        }
      },
    );
  }

  async resultOrThrow(): Promise<AgentResult> {
    const result = await this.result();
    if (result.stopReason === "error" || result.stopReason === "aborted") {
      const error: Error & { agentResult?: AgentResult } = new Error(
        result.errorMessage ?? "Request failed",
      );
      error.agentResult = result;
      throw error;
    }
    return result;
  }
}
