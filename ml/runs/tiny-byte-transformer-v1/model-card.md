# Tiny Byte Transformer v1

This 19,009-parameter, one-block causal Transformer
was trained from scratch on the training split of Folklore Corpus v0.1.0. It is
an instrumented learning experiment, not a useful language model and not a
candidate for deployment.

- Corpus: `fa:release:corpus-v0.1.0`
- Seed: `20260724`
- Context: 64 bytes
- Width / heads / feed-forward: 32 / 4 / 64
- Steps: 600
- Validation bits/byte: 3.436
- Unigram bits/byte: 4.404
- Bigram bits/byte: 3.428
- Improvement over unigram: 22.0%

The byte vocabulary avoids an external tokenizer. The edition mix, historical
English, and small corpus dominate the result. Generated samples are diagnostic
only. The memorization report checks exact copied spans and is not a privacy
guarantee.
