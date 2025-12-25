import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

type DebugLogger = {
  enabled: boolean;
  logRequest: (payload: unknown) => void;
  logResponseEvent: (event: unknown) => void;
  flushResponse: () => Promise<void>;
};

const debugDir = process.env.IOTA_DEBUG_LOG_DIR?.trim() ?? "";

export function createDebugLogger(provider: string): DebugLogger {
  if (debugDir.length === 0) {
    return {
      enabled: false,
      logRequest: () => {},
      logResponseEvent: () => {},
      flushResponse: async () => {},
    };
  }

  const base = `${provider}-${formatTimestamp()}`;
  const logPath = path.join(debugDir, `${base}.json`);
  const responseEvents: unknown[] = [];
  let requestPayload: unknown;

  let dirReady: Promise<void> | null = null;
  const ensureDir = () => {
    if (!dirReady) {
      dirReady = mkdir(debugDir, { recursive: true })
        .then(() => undefined)
        .catch(() => undefined);
    }
    return dirReady;
  };

  const write = async (filePath: string, data: unknown) => {
    try {
      await ensureDir();
      await writeFile(filePath, stringify(data), "utf8");
    } catch {
      // Debug logging should never break streaming.
    }
  };

  return {
    enabled: true,
    logRequest: (payload) => {
      requestPayload = payload;
    },
    logResponseEvent: (event) => {
      responseEvents.push(event);
    },
    flushResponse: async () => {
      await write(logPath, {
        provider,
        timestamp: new Date().toISOString(),
        request: requestPayload,
        response: responseEvents,
      });
    },
  };
}

function formatTimestamp(): string {
  const iso = new Date().toISOString();
  return iso.replace(/[:.]/g, "-");
}

function stringify(value: unknown): string {
  return JSON.stringify(value, (_key, v) => (typeof v === "bigint" ? v.toString() : v), 2);
}
