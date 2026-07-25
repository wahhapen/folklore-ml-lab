import { ChildProcess, execFileSync, spawn } from "node:child_process";
import { createServer, Server } from "node:http";
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

type Fixture = {
  archive: string;
  archiveSha256: string;
  manifestSha256: string;
  releaseId: string;
  version: string;
  manifestSchemaVersion: string;
};

type RunningServer = {
  server: Server;
  url: string;
  requests: () => number;
};

type PausedServer = RunningServer & {
  started: Promise<void>;
};

const LOCK_SCHEMA = "folklore-corpus-lock-v1";
const sourceRelease = resolve("data/derived/releases/corpus-v0.1.0");

function makeFixture(
  root: string,
  options: {
    omit?: string;
    extraPath?: string;
    extraType?: "regular" | "symlink" | "hardlink";
    duplicate?: boolean;
    corrupt?: string;
    releaseId?: string;
    producerRepository?: string;
    dropManifestSchema?: boolean;
    invalidManifestField?: boolean;
    invalidManifestSchema?: boolean;
    unsupportedManifestSchemaDialect?: boolean;
    dropFirstDocumentPassages?: boolean;
  } = {},
): Fixture {
  const releaseRoot = join(root, "release");
  const archive = join(root, `release-${Math.random()}.tar.gz`);
  cpSync(sourceRelease, releaseRoot, { recursive: true });
  const args = [
    "tests/fixtures/build-corpus-archive.py",
    releaseRoot,
    archive,
  ];
  if (options.omit) args.push("--omit", options.omit);
  if (options.extraPath) args.push("--extra-path", options.extraPath);
  if (options.extraType) args.push("--extra-type", options.extraType);
  if (options.duplicate) args.push("--duplicate");
  if (options.corrupt) args.push("--corrupt", options.corrupt);
  if (options.releaseId) args.push("--release-id", options.releaseId);
  if (options.producerRepository) {
    args.push("--producer-repository", options.producerRepository);
  }
  if (options.dropManifestSchema) args.push("--drop-manifest-schema");
  if (options.invalidManifestField) args.push("--invalid-manifest-field");
  if (options.invalidManifestSchema) args.push("--invalid-manifest-schema");
  if (options.unsupportedManifestSchemaDialect) {
    args.push("--unsupported-manifest-schema-dialect");
  }
  if (options.dropFirstDocumentPassages) {
    args.push("--drop-first-document-passages");
  }
  const result = JSON.parse(
    execFileSync("python", args, { encoding: "utf8" }),
  );
  return { archive, ...result };
}

function writeLock(root: string, fixture: Fixture, url: string): string {
  const lockPath = join(root, `lock-${Math.random()}.json`);
  writeFileSync(
    lockPath,
    `${JSON.stringify({
      schemaVersion: LOCK_SCHEMA,
      source: {
        repository: "wahhapen/folklore-corpus",
        tag: `corpus-v${fixture.version}`,
        asset: `folklore-corpus-v${fixture.version}.tar.gz`,
        url,
      },
      archiveSha256: fixture.archiveSha256,
      manifestSha256: fixture.manifestSha256,
      releaseId: fixture.releaseId,
      version: fixture.version,
      manifestSchemaVersion: fixture.manifestSchemaVersion,
    }, null, 2)}\n`,
  );
  return lockPath;
}

async function serveArchive(
  archivePath: string,
  partial = false,
): Promise<RunningServer> {
  let requestCount = 0;
  const archive = readFileSync(archivePath);
  const server = createServer((_, response) => {
    requestCount += 1;
    response.writeHead(200, {
      "content-type": "application/gzip",
      "content-length": archive.length,
    });
    if (partial) {
      response.end(archive.subarray(0, Math.floor(archive.length / 2)));
      return;
    }
    response.end(archive);
  });
  await new Promise<void>((resolvePromise) =>
    server.listen(0, "127.0.0.1", resolvePromise),
  );
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No server port");
  return {
    server,
    url: `http://127.0.0.1:${address.port}/release.tar.gz`,
    requests: () => requestCount,
  };
}

async function servePausedArchive(archivePath: string): Promise<PausedServer> {
  let requestCount = 0;
  let signalStarted: () => void = () => {};
  const started = new Promise<void>((resolvePromise) => {
    signalStarted = resolvePromise;
  });
  const archive = readFileSync(archivePath);
  const server = createServer((_, response) => {
    requestCount += 1;
    response.writeHead(200, {
      "content-type": "application/gzip",
      "content-length": archive.length,
    });
    response.write(archive.subarray(0, 64 * 1024));
    signalStarted();
  });
  await new Promise<void>((resolvePromise) =>
    server.listen(0, "127.0.0.1", resolvePromise),
  );
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No server port");
  return {
    server,
    started,
    url: `http://127.0.0.1:${address.port}/release.tar.gz`,
    requests: () => requestCount,
  };
}

async function runCorpus(
  args: string[],
  lockPath: string,
  cacheRoot: string,
  extraEnv: NodeJS.ProcessEnv = {},
): Promise<string> {
  const child = spawn(
    "python",
    ["-m", "folklore_ml", "corpus", ...args, "--lock", lockPath],
    {
      env: {
        ...process.env,
        FOLKLORE_CACHE_DIR: cacheRoot,
        ...extraEnv,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  const result = await completion(child);
  if (result.code !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
  return result.stdout;
}

async function runPython(
  args: string[],
  environment: NodeJS.ProcessEnv = {},
): Promise<string> {
  const child = spawn("python", args, {
    env: { ...process.env, ...environment },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const result = await completion(child);
  if (result.code !== 0) {
    throw new Error(result.stderr || result.stdout);
  }
  return result.stdout;
}

function spawnCorpus(
  args: string[],
  lockPath: string,
  cacheRoot: string,
): ChildProcess {
  return spawn(
    "python",
    ["-m", "folklore_ml", "corpus", ...args, "--lock", lockPath],
    {
      env: { ...process.env, FOLKLORE_CACHE_DIR: cacheRoot },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
}

async function completion(child: ChildProcess): Promise<{
  code: number | null;
  stdout: string;
  stderr: string;
}> {
  let stdout = "";
  let stderr = "";
  child.stdout?.on("data", (value) => (stdout += value.toString()));
  child.stderr?.on("data", (value) => (stderr += value.toString()));
  const code = await new Promise<number | null>((resolvePromise) =>
    child.on("close", resolvePromise),
  );
  return { code, stdout, stderr };
}

describe("Corpus Release installer CLI", () => {
  const temporaryDirectories: string[] = [];
  const servers: Server[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolvePromise) =>
            server.close(() => resolvePromise()),
          ),
      ),
    );
    for (const directory of temporaryDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  function temporaryRoot(): string {
    const root = mkdtempSync(join(tmpdir(), "folklore-ml-installer-"));
    temporaryDirectories.push(root);
    return root;
  }

  it("installs online once and reuses the verified cache offline", async () => {
    const root = temporaryRoot();
    const fixture = makeFixture(root);
    const remote = await serveArchive(fixture.archive);
    servers.push(remote.server);
    const lock = writeLock(root, fixture, remote.url);
    const cache = join(root, "cache");

    const online = JSON.parse(await runCorpus(["install"], lock, cache));
    expect(online.corpus.releaseId).toBe(fixture.releaseId);
    expect(online.corpus.archiveSha256).toBe(fixture.archiveSha256);
    expect(existsSync(join(online.path, "acquisition.json"))).toBe(true);
    const acquisition = JSON.parse(
      readFileSync(join(online.path, "acquisition.json"), "utf8"),
    );
    expect(acquisition.release).toEqual(online.corpus);
    expect(acquisition.corpus).toBeUndefined();

    const offline = JSON.parse(
      await runCorpus(["install", "--offline"], lock, cache),
    );
    expect(offline.path).toBe(online.path);
    expect(remote.requests()).toBe(1);
  });

  it.each([
    ["archive digest mismatch", "archive"],
    ["manifest digest mismatch", "manifest"],
    ["missing declared artifact", "missing"],
    ["changed declared artifact", "artifact"],
    ["archive traversal entry", "traversal"],
    ["absolute archive entry", "absolute"],
    ["archive symlink", "symlink"],
    ["archive hard link", "hardlink"],
    ["duplicate archive path", "duplicate"],
    ["wrong release identity", "identity"],
    ["wrong producer identity", "producer"],
    ["missing manifest schema contract", "missing-schema"],
    ["invalid manifest schema contract", "invalid-schema"],
    ["unsupported manifest schema dialect", "unsupported-schema"],
    ["manifest schema mismatch", "schema-mismatch"],
  ])("fails closed for %s", async (_, failure) => {
    const root = temporaryRoot();
    const fixture = makeFixture(root, {
      omit: failure === "missing" ? "documents.jsonl" : undefined,
      corrupt: failure === "artifact" ? "documents.jsonl" : undefined,
      extraPath:
        failure === "traversal"
          ? "../escape"
          : failure === "absolute"
            ? "/escape"
            : failure === "symlink" || failure === "hardlink"
              ? "linked-entry"
              : undefined,
      extraType:
        failure === "symlink"
          ? "symlink"
          : failure === "hardlink"
            ? "hardlink"
            : undefined,
      duplicate: failure === "duplicate",
      producerRepository:
        failure === "producer" ? "somebody-else/folklore-corpus" : undefined,
      dropManifestSchema: failure === "missing-schema",
      invalidManifestSchema: failure === "invalid-schema",
      unsupportedManifestSchemaDialect: failure === "unsupported-schema",
      invalidManifestField: failure === "schema-mismatch",
      releaseId:
        failure === "identity" ? "fa:release:corpus-v9.9.9" : undefined,
    });
    const remote = await serveArchive(fixture.archive);
    servers.push(remote.server);
    const lock = writeLock(root, fixture, remote.url);
    if (failure === "archive") {
      const value = JSON.parse(readFileSync(lock, "utf8"));
      value.archiveSha256 = "0".repeat(64);
      writeFileSync(lock, `${JSON.stringify(value, null, 2)}\n`);
    } else if (failure === "manifest") {
      const value = JSON.parse(readFileSync(lock, "utf8"));
      value.manifestSha256 = "0".repeat(64);
      writeFileSync(lock, `${JSON.stringify(value, null, 2)}\n`);
    } else if (failure === "identity") {
      const value = JSON.parse(readFileSync(lock, "utf8"));
      value.releaseId = "fa:release:corpus-v0.1.0";
      writeFileSync(lock, `${JSON.stringify(value, null, 2)}\n`);
    }
    const cache = join(root, "cache");

    await expect(runCorpus(["install"], lock, cache)).rejects.toThrow();
    const cacheParent = join(cache, "folklore-atlas", "corpus", "sha256");
    expect(existsSync(cacheParent) ? readdirSync(cacheParent) : []).toEqual([]);
  });

  it("rejects a partial response without exposing a cache entry", async () => {
    const root = temporaryRoot();
    const fixture = makeFixture(root);
    const remote = await serveArchive(fixture.archive, true);
    servers.push(remote.server);
    const lock = writeLock(root, fixture, remote.url);
    const cache = join(root, "cache");

    await expect(runCorpus(["install"], lock, cache)).rejects.toThrow(
      /Partial Corpus download/,
    );
    const cacheParent = join(cache, "folklore-atlas", "corpus", "sha256");
    expect(existsSync(cacheParent) ? readdirSync(cacheParent) : []).toEqual([]);
  });

  it(
    "an interrupted installer exposes no final entry and a retry succeeds",
    async () => {
      const root = temporaryRoot();
      const fixture = makeFixture(root);
      const paused = await servePausedArchive(fixture.archive);
      servers.push(paused.server);
      const interruptedLock = writeLock(root, fixture, paused.url);
      const cache = join(root, "cache");
      const child = spawnCorpus(["install"], interruptedLock, cache);
      const interrupted = completion(child);
      await paused.started;

      child.kill("SIGKILL");
      expect((await interrupted).code).not.toBe(0);
      const finalPath = join(
        cache,
        "folklore-atlas",
        "corpus",
        "sha256",
        fixture.manifestSha256,
      );
      expect(existsSync(finalPath)).toBe(false);

      const retry = await serveArchive(fixture.archive);
      servers.push(retry.server);
      const retryLock = writeLock(root, fixture, retry.url);
      const installed = JSON.parse(
        await runCorpus(["install"], retryLock, cache),
      );
      expect(installed.path).toBe(finalPath);
      expect(
        JSON.parse(await runCorpus(["verify"], retryLock, cache)).corpus
          .releaseId,
      ).toBe(fixture.releaseId);
    },
    30_000,
  );

  it("concurrent installers expose one complete verified cache", async () => {
    const root = temporaryRoot();
    const fixture = makeFixture(root);
    const remote = await serveArchive(fixture.archive);
    servers.push(remote.server);
    const lock = writeLock(root, fixture, remote.url);
    const cache = join(root, "cache");

    const [first, second] = await Promise.all([
      completion(spawnCorpus(["install"], lock, cache)),
      completion(spawnCorpus(["install"], lock, cache)),
    ]);
    expect([first.code, second.code]).toEqual([0, 0]);
    const firstResult = JSON.parse(first.stdout);
    const secondResult = JSON.parse(second.stdout);
    expect(secondResult.path).toBe(firstResult.path);
    expect(
      JSON.parse(await runCorpus(["verify"], lock, cache)).corpus.releaseId,
    ).toBe(fixture.releaseId);
    expect(readdirSync(join(cache, "folklore-atlas", "corpus", "sha256"))).toEqual([
      fixture.manifestSha256,
    ]);
  });

  it("offline mode rejects missing and corrupt cache entries without networking", async () => {
    const root = temporaryRoot();
    const fixture = makeFixture(root);
    const unreachableUrl = "http://127.0.0.1:1/should-not-connect";
    const lock = writeLock(root, fixture, unreachableUrl);
    const cache = join(root, "cache");

    await expect(runCorpus(["install", "--offline"], lock, cache)).rejects.toThrow(
      /offline/i,
    );

    const remote = await serveArchive(fixture.archive);
    servers.push(remote.server);
    const onlineLock = writeLock(root, fixture, remote.url);
    const installed = JSON.parse(await runCorpus(["install"], onlineLock, cache));
    writeFileSync(join(installed.path, "documents.jsonl"), "corrupt");
    await expect(runCorpus(["install", "--offline"], lock, cache)).rejects.toThrow(
      /mismatch/i,
    );
    expect(remote.requests()).toBe(1);
  });

  it(
    "rejects task preparation when a selected document has no passages",
    async () => {
      const root = temporaryRoot();
      const fixture = makeFixture(root, {
        dropFirstDocumentPassages: true,
      });
      const remote = await serveArchive(fixture.archive);
      servers.push(remote.server);
      const lock = writeLock(root, fixture, remote.url);
      const cache = join(root, "cache");
      await runCorpus(["install"], lock, cache);

      await expect(
        runPython(["-m", "folklore_ml", "prepare"], {
          FOLKLORE_CORPUS_LOCK: lock,
          FOLKLORE_CACHE_DIR: cache,
          FOLKLORE_OFFLINE: "1",
          FOLKLORE_ML_DATA_DIR: join(root, "task"),
        }),
      ).rejects.toThrow(/no relevant Passage IDs/);
    },
    30_000,
  );

  it(
    "prepares provenance-bearing task and run artifacts from the resolved cache",
    async () => {
      const root = temporaryRoot();
      const fixture = makeFixture(root);
      const remote = await serveArchive(fixture.archive);
      servers.push(remote.server);
      const lock = writeLock(root, fixture, remote.url);
      const cache = join(root, "cache");
      const taskRoot = join(root, "task");
      const runRoot = join(root, "run");
      await runCorpus(["install"], lock, cache);
      const environment = {
        FOLKLORE_CORPUS_LOCK: lock,
        FOLKLORE_CACHE_DIR: cache,
        FOLKLORE_OFFLINE: "1",
        FOLKLORE_ML_DATA_DIR: taskRoot,
        FOLKLORE_ML_RUN_DIR: runRoot,
      };

      await runPython(["-m", "folklore_ml", "classifier"], environment);

      const task = JSON.parse(
        readFileSync(join(taskRoot, "manifest.json"), "utf8"),
      );
      const run = JSON.parse(readFileSync(join(runRoot, "run.json"), "utf8"));
      const train = JSON.parse(
        readFileSync(join(taskRoot, "train.jsonl"), "utf8").split("\n")[0],
      );
      for (const artifact of [task, run]) {
        expect(artifact.corpus).toEqual({
          releaseId: fixture.releaseId,
          version: fixture.version,
          manifestSchemaVersion: fixture.manifestSchemaVersion,
          manifestSha256: fixture.manifestSha256,
          archiveSha256: fixture.archiveSha256,
          sourceRepository: "wahhapen/folklore-corpus",
          sourceTag: `corpus-v${fixture.version}`,
          sourceAsset: `folklore-corpus-v${fixture.version}.tar.gz`,
        });
        expect(artifact.passageIds.length).toBeGreaterThan(3_000);
      }
      expect(train.passageIds.length).toBeGreaterThan(0);
      expect(train.passageIds[0]).toMatch(/^fa:passage:/);
      await runPython(["-m", "folklore_ml", "verify"], environment);

      const trainPath = join(taskRoot, "train.jsonl");
      const trainRows = readFileSync(trainPath, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      trainRows[0].passageIds = ["fa:passage:unknown"];
      writeFileSync(
        trainPath,
        `${trainRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
      );
      await expect(
        runPython(["-m", "folklore_ml", "verify"], environment),
      ).rejects.toThrow(/unknown Passage ID/);
      expect(remote.requests()).toBe(1);
    },
    30_000,
  );
});
