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
    const probe = spawnSync(candidate, ["--version"], {
      env: environment,
      stdio: "ignore",
    });
    if (probe.status === 0) return candidate;
  }
  throw new Error(
    `No Python interpreter found; tried ${candidates.join(", ")}`,
  );
}

export const PYTHON = resolvePythonInterpreter();
