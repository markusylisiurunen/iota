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
}
