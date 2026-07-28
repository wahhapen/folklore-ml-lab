import { spawnSync } from "node:child_process";

export function resolvePythonInterpreter(
  environment: NodeJS.ProcessEnv = process.env,
): string {
  const candidates = [
    environment.PYTHON,
    "python3",
    "python",
  ].filter((candidate, index, values): candidate is string =>
    Boolean(candidate) && values.indexOf(candidate) === index
  );
  for (const candidate of candidates) {
    const probe = spawnSync(candidate, [
      "-c",
      "import sys; raise SystemExit(" +
        "0 if sys.version_info[:2] == (3, 12) else 1)",
    ], {
      env: environment,
      stdio: "ignore",
    });
    if (probe.status === 0) return candidate;
  }
  throw new Error(
    `No Python 3.12 interpreter found; tried ${candidates.join(", ")}`,
  );
}

export const PYTHON = resolvePythonInterpreter();
