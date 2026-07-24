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
python -m folklore_ml prepare
python -m folklore_ml classifier
npm run ml:tiny
python -m folklore_ml verify
npm test
```

Python versions are recorded in `ml/requirements-lock.txt`. The vendored Corpus
Release is pinned by ID and manifest digest.
