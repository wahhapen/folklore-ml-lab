import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

describe("ML Lab experiment contract", () => {
  const temporaryDirectories: string[] = [];

  afterEach(() => {
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("verifies preserved v0.1 runs without requiring the current pinned cache", () => {
    const root = mkdtempSync(join(tmpdir(), "folklore-ml-legacy-"));
    temporaryDirectories.push(root);
    const lock = join(root, "corpus-release.lock.json");
    writeFileSync(
      lock,
      `${JSON.stringify({
        schemaVersion: "folklore-corpus-lock-v1",
        source: {
          repository: "wahhapen/folklore-corpus",
          tag: "corpus-v0.2.0",
          asset: "folklore-corpus-v0.2.0.tar.gz",
          url: "https://example.invalid/folklore-corpus-v0.2.0.tar.gz",
        },
        archiveSha256: "0".repeat(64),
        manifestSha256: "1".repeat(64),
        releaseId: "fa:release:corpus-v0.2.0",
        version: "0.2.0",
        manifestSchemaVersion: "folklore-release-manifest-v1",
      }, null, 2)}\n`,
    );

    expect(() =>
      execFileSync(
        "python",
        ["-m", "folklore_ml", "verify", "--legacy-only"],
        {
          env: {
            ...process.env,
            FOLKLORE_CORPUS_LOCK: lock,
            FOLKLORE_CACHE_DIR: join(root, "empty-cache"),
          },
          stdio: "pipe",
        },
      ),
    ).not.toThrow();
  });

  it("does not rewrite preserved v0.1 task artifacts during legacy preparation", () => {
    const root = mkdtempSync(join(tmpdir(), "folklore-ml-prepare-"));
    temporaryDirectories.push(root);
    const path = "ml/data/edition-fingerprint-v1/manifest.json";
    const before = createHash("sha256").update(readFileSync(path)).digest("hex");
    execFileSync("python", ["-m", "folklore_ml", "prepare"], {
      env: {
        ...process.env,
        FOLKLORE_ML_DATA_DIR: join(root, "task-data"),
      },
      stdio: "pipe",
    });
    const after = createHash("sha256").update(readFileSync(path)).digest("hex");
    expect(after).toBe(before);
  });

  it("prepares deterministic, release-pinned task data", () => {
    const root = mkdtempSync(join(tmpdir(), "folklore-ml-prepare-"));
    temporaryDirectories.push(root);
    const taskData = join(root, "task-data");
    execFileSync("python", ["-m", "folklore_ml", "prepare"], {
      env: {
        ...process.env,
        FOLKLORE_ML_DATA_DIR: taskData,
      },
      stdio: "pipe",
    });
    const manifest = JSON.parse(
      readFileSync(join(taskData, "manifest.json"), "utf8"),
    );
    expect(manifest.corpusRelease).toBe("fa:release:corpus-v0.2.1");
    expect(manifest.corpusManifestSha256).toBe(
      "d809fe8acf43642217af58c0e8ed9399740a0349a21064e40ae63eb3cd030bbd",
    );
    expect(manifest.counts.train + manifest.counts.validation + manifest.counts.test).toBe(
      170,
    );
    expect(manifest.datasetSha256).toMatch(/^[a-f0-9]{64}$/);
  });


  it("validates v2 records and rejects incomplete lifecycle fields", () => {
    const root = mkdtempSync(join(tmpdir(), "folklore-ml-run-v2-"));
    temporaryDirectories.push(root);
    const recordPath = join(root, "run.json");
    const valid = {
      schemaVersion: "folklore-ml-run-v2",
      experimentId: "test-run-v2",
      question: { kind: "learning", text: "Does the candidate beat the baseline?" },
      hypothesis: "The candidate improves macro-F1.",
      evaluation: {
        frozenIdentity: { id: "evaluation-v1", sha256: "a".repeat(64) },
      },
      baseline: { name: "majority", metrics: { macroF1: 0.08 } },
      candidate: { name: "character-tfidf", metrics: { macroF1: 1.0 } },
      metrics: { primary: ["macroF1"], secondary: ["accuracy"] },
      humanReview: { criteria: ["Reject cultural-origin inference."] },
      provenance: {
        datasetSha256: "b".repeat(64),
        code: { repository: "wahhapen/folklore-ml-lab", revision: "c".repeat(40) },
        command: "python -m folklore_ml classifier",
        source: { kind: "new-run" },
      },
      cost: {
        time: { status: "recorded", value: 12.5, unit: "seconds" },
        compute: { status: "recorded", value: 12.5, unit: "cpu-seconds" },
        money: { status: "recorded", value: 0, currency: "USD" },
      },
      limitations: ["Small frozen evaluation."],
      decision: {
        target: "candidate",
        scope: "Candidate use on the frozen evaluation.",
        outcome: "continue",
        rationale: "Expand evaluation before adoption.",
      },
    };
    const verify = (record: unknown) => {
      writeFileSync(recordPath, `${JSON.stringify(record, null, 2)}\n`);
      return execFileSync(
        "python",
        ["-m", "folklore_ml", "verify-run", recordPath],
        { encoding: "utf8", stdio: "pipe" },
      );
    };

    expect(() => verify(valid)).not.toThrow();
    for (const invalid of [
      { ...valid, decision: { ...valid.decision, outcome: "ship" } },
      { ...valid, provenance: undefined },
      { ...valid, baseline: { name: "majority", metrics: {} } },
      { ...valid, cost: { time: valid.cost.time, compute: valid.cost.compute } },
    ]) {
      expect(() => verify(invalid)).toThrow();
    }
  }, 15_000);

  it("publishes verifiable classifier and tiny-transformer run artifacts", () => {
    for (const run of ["edition-fingerprint-v1", "tiny-byte-transformer-v1"]) {
      const root = `ml/runs/${run}`;
      expect(existsSync(`${root}/metrics.json`)).toBe(true);
      expect(existsSync(`${root}/run.json`)).toBe(true);
      expect(existsSync(`${root}/model-card.md`)).toBe(true);
      const runMetadata = JSON.parse(readFileSync(`${root}/run.json`, "utf8"));
      expect(runMetadata.corpusRelease).toBe("fa:release:corpus-v0.1.0");
      expect(runMetadata.seed).toBe(20260724);
    }
  });
});
