export class EventStream<TEvent, TResult = TEvent> implements AsyncIterable<TEvent> {
  private queue: TEvent[] = [];
  private waiting: ((value: IteratorResult<TEvent>) => void)[] = [];
  private done = false;
  private finalResultPromise: Promise<TResult>;
  private resolveFinalResult!: (result: TResult) => void;

  constructor(
    private isComplete: (event: TEvent) => boolean,
    private extractResult: (event: TEvent) => TResult,
  ) {
    this.finalResultPromise = new Promise((resolve) => {
      this.resolveFinalResult = resolve;
    });
  }

  push(event: TEvent): void {
    if (this.done) return;

    if (this.isComplete(event)) {
      this.done = true;
      this.resolveFinalResult(this.extractResult(event));
    }

    const waiter = this.waiting.shift();
    if (waiter) waiter({ value: event, done: false });
    else this.queue.push(event);
  }

  end(result?: TResult): void {
    this.done = true;
    if (result !== undefined) this.resolveFinalResult(result);

    while (this.waiting.length > 0) {
      this.waiting.shift()!({ value: undefined as any, done: true });
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterator<TEvent> {
    while (true) {
      if (this.queue.length > 0) {
        yield this.queue.shift()!;
        continue;
      }

      if (this.done) return;

      const result = await new Promise<IteratorResult<TEvent>>((resolve) =>
        this.waiting.push(resolve),
      );
      if (result.done) return;
      yield result.value;
    }
  }

  result(): Promise<TResult> {
    return this.finalResultPromise;
  }
}
