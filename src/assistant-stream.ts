import { EventStream } from "./event-stream.js";
import type { AssistantMessage, AssistantStreamEvent } from "./types.js";

export class AssistantStream extends EventStream<AssistantStreamEvent, AssistantMessage> {
  constructor() {
    super(
      (e) => e.type === "done" || e.type === "error",
      (e) => {
        if (e.type === "done") return e.message;
        if (e.type === "error") return e.error;
        throw new Error("Unexpected event type");
      },
    );
  }

  async resultOrThrow(): Promise<AssistantMessage> {
    const msg = await this.result();
    if (msg.stopReason === "error" || msg.stopReason === "aborted") {
      const error = new Error(msg.errorMessage ?? "Request failed");
      (error as any).assistantMessage = msg;
      throw error;
    }
    return msg;
  }
}
