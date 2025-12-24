export function parseStreamingJson<T = unknown>(json: string | undefined): T {
  if (!json || json.trim() === "") {
    return {} as T;
  }

  try {
    return JSON.parse(json) as T;
  } catch {
    return {} as T;
  }
}
