import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

describe("ML Lab experiment contract", () => {
  it("prepares deterministic, release-pinned task data", () => {
    execFileSync("python", ["-m", "folklore_ml", "prepare"], {
      stdio: "pipe",
    });
    const manifest = JSON.parse(
      readFileSync("ml/data/edition-fingerprint-v1/manifest.json", "utf8"),
    );
    expect(manifest.corpusRelease).toBe("fa:release:corpus-v0.1.0");
    expect(manifest.counts.train + manifest.counts.validation + manifest.counts.test).toBe(
      170,
    );
    expect(manifest.datasetSha256).toMatch(/^[a-f0-9]{64}$/);
  });

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
