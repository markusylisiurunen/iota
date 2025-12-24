export function exhaustive(value: never, message?: string): never {
  const rendered = (() => {
    try {
      return String(value);
    } catch {
      return "[unserializable]";
    }
  })();

  throw new Error(message ?? `Unhandled case: ${rendered}`);
}
