import { defineConfig } from "vitest/config";

const tsconfigRaw = {
  compilerOptions: {
    target: "es2022",
  },
};

export default defineConfig({
  test: {
    environment: "node",
  },
  esbuild: {
    target: "es2022",
    tsconfigRaw,
  },
  optimizeDeps: {
    esbuildOptions: {
      target: "es2022",
      tsconfigRaw,
    },
  },
});
