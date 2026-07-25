import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

describe("ML Lab Corpus Release boundary", () => {
  const temporaryDirectories: string[] = [];

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("verifies every declared corpus artifact before preparing ML data", () => {
    const output = execFileSync(
      "python",
      ["-m", "folklore_ml", "verify-corpus"],
      {
        encoding: "utf8",
        env: {
          ...process.env,
          FOLKLORE_CORPUS_DIR: resolve(
            "data/derived/releases/corpus-v0.1.0",
          ),
        },
      },
    );
    const result = JSON.parse(output);
    expect(result.releaseId).toBe("fa:release:corpus-v0.1.0");
    expect(result.manifestSha256).toBe(
      "1e614c013f4ec9a21e574a17653c8430eee11ae95ba80cc099a7dc52c7f257ca",
    );
    expect(result.artifactCount).toBe(13);
  });

  it("fails closed when a release artifact changes", () => {
    const temporaryRoot = mkdtempSync(
      join(tmpdir(), "folklore-ml-release-"),
    );
    temporaryDirectories.push(temporaryRoot);
    const releaseRoot = join(temporaryRoot, "release");
    cpSync(
      resolve("data/derived/releases/corpus-v0.1.0"),
      releaseRoot,
      { recursive: true },
    );
    const documentsPath = join(releaseRoot, "documents.jsonl");
    writeFileSync(
      documentsPath,
      `${readFileSync(documentsPath, "utf8")}changed`,
    );

    expect(() =>
      execFileSync(
        "python",
        ["-m", "folklore_ml", "verify-corpus"],
        {
          env: { ...process.env, FOLKLORE_CORPUS_DIR: releaseRoot },
          stdio: "pipe",
        },
      ),
    ).toThrow(/Artifact byte length mismatch: documents.jsonl/);
  });
});
