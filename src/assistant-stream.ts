import { EventStream } from "./event-stream.js";
import type { AssistantMessage, AssistantStreamEvent } from "./types.js";
import { exhaustive } from "./utils/exhaustive.js";

export class AssistantStream extends EventStream<AssistantStreamEvent, AssistantMessage> {
  constructor() {
    super(
      (e) => e.type === "done" || e.type === "error",
      (e) => {
        switch (e.type) {
          case "done":
            return e.message;
          case "error":
            return e.error;
          case "start":
          case "part_start":
          case "part_delta":
          case "part_end":
            throw new Error("Unexpected event type");
          default:
            return exhaustive(e);
        }
      },
    );
  }

  async resultOrThrow(): Promise<AssistantMessage> {
    const msg = await this.result();
    if (msg.stopReason === "error" || msg.stopReason === "aborted") {
      const error: Error & { assistantMessage?: AssistantMessage } = new Error(
        msg.errorMessage ?? "Request failed",
      );
      error.assistantMessage = msg;
      throw error;
    }
    return msg;
  }
}
