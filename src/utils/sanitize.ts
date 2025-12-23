export function sanitizeSurrogates(input: string): string {
  let out = "";
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i);
    // High surrogate
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = input.charCodeAt(i + 1);
      // If next is a valid low surrogate, keep both.
      if (next >= 0xdc00 && next <= 0xdfff) {
        out += input.slice(i, i + 2);
        i++;
        continue;
      }
      // Unpaired high surrogate, replace.
      out += "\uFFFD";
      continue;
    }
    // Low surrogate without preceding high surrogate.
    if (code >= 0xdc00 && code <= 0xdfff) {
      out += "\uFFFD";
      continue;
    }
    out += input[i];
  }
  return out;
}
