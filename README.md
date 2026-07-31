# Folklore ML Lab

Reproducible learning experiments over immutable Folklore Corpus releases.

The preserved v0.1 runs include:

- an edition-fingerprint leakage diagnostic with simple controls;
- a genuine 19,009-parameter byte Transformer trained from scratch for
  600 CPU steps;
- task manifests, metrics, predictions, checkpoint, history, samples,
  memorization probes and model cards.

These are pipeline proofs, not useful folklore models. The next valuable ML
work should target retrieval, entity/motif extraction or variant-family
similarity with human-reviewed evaluation.

## Commands

```bash
npm install
python -m pip install -r ml/requirements-lock.txt
npm run corpus:fetch
npm run corpus:verify
python -m folklore_ml corpus status
python -m folklore_ml prepare
python -m folklore_ml classifier
npm run ml:tiny
python -m folklore_ml verify
python -m folklore_ml verify --legacy-only
npm test
```

Python versions are recorded in `ml/requirements-lock.txt`.

## Corpus release boundary

ML preparation resolves one reviewed `corpus-release.lock.json`, installs its
archive into a content-addressed cache, and verifies the archive, enclosed
manifest, release identity, and every declared artifact before reading any
records. `FOLKLORE_CACHE_DIR` overrides the user cache root and
`FOLKLORE_OFFLINE=1` forbids network access.

`corpus-release.lock.json` pins the published Corpus v0.2.1 archive and enclosed
manifest by SHA-256. It is used by the Corpus install and verification commands
and by newly generated task and run outputs. The preserved checked-in task and
all preserved run records remain pinned to Corpus v0.1.0; no preserved run
consumes the v0.2.1 lock. The lock has this field contract:

```text
corpus-release.lock.json
├── schemaVersion
├── source.repository / tag / asset / url
├── archiveSha256
├── manifestSha256
├── releaseId / version
└── manifestSchemaVersion
```

The exact field contract is executable in
`corpus-release.lock.schema.json`. `npm run corpus:fetch` installs it and
task/run outputs are namespaced by the manifest digest, preserving the v0.1
artifacts. New task rows and task/run manifests also record the relevant
Passage IDs and the full deterministic Corpus provenance block.

`python -m folklore_ml verify --legacy-only` verifies the preserved v0.1 task
and runs without resolving the current lock or requiring its cache. This keeps
the educational history independently auditable after the v0.2 pin lands.

`FOLKLORE_CORPUS_DIR` remains available only for verifying the preserved legacy
release; pinned work does not select releases by versioned directory name.

## Experiment records

New work uses the backward-compatible `folklore-ml-run-v2` lifecycle record.
Preserved v1 runs remain unchanged and readable. Validate either version with:

```bash
python -m folklore_ml verify-run path/to/run.json
```

See [the v2 experiment-record contract](docs/experiment-record-v2.md) and the
[representative v2 migration](ml/runs/edition-fingerprint-v1-record-v2/run.json).
