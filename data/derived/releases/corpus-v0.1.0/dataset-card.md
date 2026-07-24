# Folklore Corpus v0.1.0

## Summary

Folklore Corpus v0.1.0 is a deterministic educational release of 170 readable
documents extracted from five historical English-language Project Gutenberg
editions. It exists to exercise source-aware corpus engineering, passage
retrieval, evaluation, and small reproducible ML experiments.

It is not a representative survey of world folklore, a source-language oral
archive, or a scholarly motif catalogue.

## Contents

- *Grimms' Fairy Tales* — 62 documents
- *Andersen's Fairy Tales* — 18 authored literary tales
- *English Fairy Tales* — 43 documents
- *Celtic Fairy Tales* — 26 documents
- *Japanese Fairy Tales* — 21 documents

All readable witnesses are English historical translations, retellings, or
editions. Collection-wide tradition and region labels are deliberately coarse.

## Provenance

The raw UTF-8 books are GITenberg mirrors of Project Gutenberg items. Each raw
file is pinned by byte length and SHA-256. The release records separate Capture,
Edition, Document, Witness, and Passage identities and emits deterministic
lineage events from raw capture through passage generation.

The compiler removes Project Gutenberg wrapper material only from derived
reading text. Raw files remain unchanged.

## Corrections from the incubation corpus

- `THE WEDDING OF MRS FOX` is one Document containing its `FIRST STORY` and
  `SECOND STORY` section headings, rather than two incorrectly titled records.
- `CONAL YELLOWCLAW` is recovered through an edition-specific exact parser
  alias for the body heading `CONALL YELLOWCLAW`; the TOC spelling remains the
  canonical display title.
- Legacy reader identifiers are retained in `aliases.jsonl`.

## Splits

`splits.jsonl` provides a deterministic general-purpose 80/10/10-style
assignment based on the stable Document ID. It keeps whole Documents together.
The current seed has no exact normalized-text duplicate groups.

These assignments are incubation splits. Retrieval benchmarks and supervised
tasks may publish stricter task-specific splits and must not claim that the
general split controls unknown folklore variants, translation families, or
editorial leakage.

## Suitable uses

- Passage-level lexical retrieval and citation experiments
- Corpus/release engineering practice
- Source-edition fingerprint diagnostics
- Educational tokenization and tiny language-model training
- Building review queues for proposed entities, motifs, and relationships

## Unsuitable or unsupported uses

- Treating regex-derived motif, being, or role suggestions as gold labels
- Cultural, ethnic, national, or geographic classification of people or texts
- Claims about historical transmission or canonical tale identity
- Measuring multilingual or source-language performance
- Training a practically useful foundation model
- Reconstructing oral performance context not present in these editions

## Known limitations

- Only five English-language editions are present.
- Historical editors and translators strongly shape the prose.
- Teller, locality, performance, and source-language metadata is usually absent.
- Collection labels are easy lexical fingerprints and can leak into models.
- Near-duplicate and variant detection is not complete.
- Passage identities begin with this release; future boundary corrections must
  publish explicit replacement or split/merge relations.

## Rule-derived annotations

The reader compatibility bundle contains deterministic regex suggestions for
navigation. These are separate from the canonical release records and are not
scholarly assertions or evaluation labels.

## Reproducibility

Run `npm run build:corpus` followed by `npm run validate:release`. A valid clean
build reproduces byte-identical JSONL artifacts and the hashes in
`manifest.json` without network access.
