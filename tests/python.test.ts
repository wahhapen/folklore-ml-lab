import { describe, expect, it } from "vitest";

import { resolvePythonInterpreter } from "./python.js";

describe("Python interpreter resolution", () => {
  it("honors an explicit executable from PYTHON", () => {
    expect(resolvePythonInterpreter({
      ...process.env,
      PYTHON: process.execPath,
    })).toBe(process.execPath);
  });

  it("falls back after an unavailable PYTHON override", () => {
    expect(resolvePythonInterpreter({
      ...process.env,
      PYTHON: "missing-python-interpreter",
    })).toMatch(/^python3?$/);
  });
});
