import { AssistantStream } from "./assistant-stream.js";
import type { Model } from "./models.js";
import { calculateCost } from "./models.js";
import type {
  AssistantMessage,
  AssistantMessageDraft,
  AssistantPart,
  StopReason,
  StreamOptions,
  Usage,
} from "./types.js";
import { emptyUsage } from "./usage.js";

export class StreamController {
  readonly stream = new AssistantStream();
  readonly output: AssistantMessageDraft;

  constructor(
    private model: Model,
    private options: StreamOptions,
  ) {
    this.output = {
      role: "assistant",
      provider: model.provider,
      model: model.id,
      content: [],
      stopReason: "stop",
      usage: emptyUsage(),
    };
  }

  start(): void {
    this.stream.push({ type: "start", partial: this.output });
  }

  addPart(part: AssistantPart): number {
    this.output.content.push(part);
    const index = this.output.content.length - 1;
    this.stream.push({ type: "part_start", index, partial: this.output });
    return index;
  }

  delta(index: number, delta: string): void {
    this.stream.push({ type: "part_delta", index, delta, partial: this.output });
  }

  endPart(index: number): void {
    this.stream.push({ type: "part_end", index, partial: this.output });
  }

  setUsage(usage: Usage): void {
    this.output.usage = {
      ...usage,
      cost: calculateCost(this.model, usage),
    };
  }

  setStopReason(stopReason: StopReason): void {
    this.output.stopReason = stopReason;
  }

  finish(): void {
    if (this.options.signal?.aborted) {
      this.output.stopReason = "aborted";
      this.output.errorMessage = this.output.errorMessage ?? "Request was aborted";
    }

    this.output.usage = this.output.usage ?? emptyUsage();
    this.output.usage = {
      ...this.output.usage,
      cost: calculateCost(this.model, this.output.usage),
    };

    this.output.stopReason = this.output.stopReason ?? "stop";

    if (
      this.output.content.some((p) => p.type === "tool_call") &&
      this.output.stopReason === "stop"
    ) {
      this.output.stopReason = "tool_use";
    }

    const final = this.output as AssistantMessage;

    if (final.stopReason === "error" || final.stopReason === "aborted") {
      if (!final.errorMessage) final.errorMessage = "Request failed";
      this.stream.push({ type: "error", error: final });
    } else {
      this.stream.push({ type: "done", message: final });
    }

    this.stream.end();
  }

  fail(error: unknown): void {
    this.output.stopReason = this.options.signal?.aborted ? "aborted" : "error";
    this.output.usage = this.output.usage ?? emptyUsage();
    this.output.errorMessage = error instanceof Error ? error.message : String(error);

    const final = this.output as AssistantMessage;

    if (!final.errorMessage) {
      final.errorMessage =
        final.stopReason === "aborted" ? "Request was aborted" : "Request failed";
    }

    this.stream.push({ type: "error", error: final });
    this.stream.end();
  }
}
