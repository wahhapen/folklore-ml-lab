# Identity Stability Policy

Corpus v0.1 IDs are release-stable source-coordinate identifiers, not a claim
that every future parser can preserve every boundary forever.

- Capture IDs include the immutable source item, capture date, and digest
  prefix.
- Edition IDs identify the pinned Project Gutenberg item represented by this
  release.
- Document and Witness IDs use the immutable edition item plus its declared TOC
  ordinal. Correcting titles does not change them.
- Passage IDs use the Witness ID plus passage ordinal. A passage-boundary
  correction may replace, split, or merge them.

For an unchanged pinned edition and compiler, rebuilds reproduce the same IDs
byte-for-byte. A future release that changes Document or Passage boundaries
must publish explicit aliases or replacement/split/merge relations; it must not
silently reuse an old ID for different content.

The incubation repository does not yet have a global mint-once registry across
different institutional sources. That becomes necessary before cross-source
deduplication or curated tale/variant entities are promoted into shared
identity.
