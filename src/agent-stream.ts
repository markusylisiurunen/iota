import { EventStream } from "./event-stream.js";
import type { AgentResult, AgentStreamEvent } from "./types.js";

export class AgentStream extends EventStream<AgentStreamEvent, AgentResult> {
  constructor() {
    super(
      (e) => e.type === "done" || e.type === "error",
      (e) => {
        if (e.type === "done") return e.result;
        if (e.type === "error") return e.error;
        throw new Error("Unexpected event type");
      },
    );
  }

  async resultOrThrow(): Promise<AgentResult> {
    const result = await this.result();
    if (result.stopReason === "error" || result.stopReason === "aborted") {
      const error = new Error(result.errorMessage ?? "Request failed");
      (error as any).agentResult = result;
      throw error;
    }
    return result;
  }
}
