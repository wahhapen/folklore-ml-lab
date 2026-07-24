# Edition Fingerprint v1

This diagnostic asks whether character 3–5 gram patterns can identify which of
the five seed editions supplied a held-out story. It measures editorial,
translator, orthographic, and formatting leakage—not folklore understanding.

- Corpus: `fa:release:corpus-v0.1.0`
- Split: immutable document-level general-v1 split
- Seed: `20260724`
- Test macro-F1: `1.000`
- Majority macro-F1: `0.080`
- Length-only macro-F1: `0.100`
- Shuffled-label macro-F1: `0.080`

The useful result is evidence of corpus leakage risk. Do not use this model to
infer cultural origin, ethnicity, authenticity, motif, or tale type.
