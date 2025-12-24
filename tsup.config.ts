import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/**/*.ts", "!src/**/*.test.ts"],
  bundle: false,
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2024",
});
