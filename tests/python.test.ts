import { describe, expect, it } from "vitest";

import { resolvePythonInterpreter } from "./python.js";

describe("Python interpreter resolution", () => {
  it("honors an explicit Python 3.12 executable", () => {
    const python = resolvePythonInterpreter();
    expect(resolvePythonInterpreter({
      ...process.env,
      PYTHON: python,
    })).toBe(python);
  });

  it("rejects a non-Python override and falls back", () => {
    expect(resolvePythonInterpreter({
      ...process.env,
      PYTHON: process.execPath,
    })).not.toBe(process.execPath);
  });
});
