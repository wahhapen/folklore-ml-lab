# First useful source-grounded ML task for v0.2

Status: recommendation for issue #2  
Date: 2026-07-24  
Scope: task selection, benchmark contract, small-model stack, and experiment ladder

## Decision

Build **human-judged query-to-passage retrieval with immutable citations**.

Given a natural-language folklore question, rank passages from a pinned Corpus
Release. Every positive label must name the exact `passageId` that supports the
query. The output is a ranked, cited review surface; it is not an answer
generator and it does not create folklore claims.

This is the first task because it is simultaneously:

- useful to Search and future source-grounded assistants;
- labelable from the source text without pretending that collection metadata,
  regex suggestions, or model output are scholarly truth;
- measurable with standard information-retrieval metrics;
- small enough for exact retrieval over the current 3,291 passages;
- a clean learning ladder from TF-IDF through frozen embeddings, contrastive
  fine-tuning, and optional reranking;
- reusable later for duplicate/variant discovery.

The first neural checkpoint should be
[`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2), used
first frozen and then fine-tuned. Its model card records 33.4M parameters, 384
embedding dimensions, a 512-token limit, English-only scope, and an MIT
licence. It is deliberately modest rather than novel.

## What the repository already proves

The existing experiments are valuable diagnostics, not candidates for
promotion:

- [`edition-fingerprint-v1`](../../ml/runs/edition-fingerprint-v1/model-card.md)
  reaches 1.000 macro-F1 on only 12 test documents. That is direct evidence
  that the five editions are trivially distinguishable by editorial,
  translator, orthographic, and formatting fingerprints.
- [`tiny-byte-transformer-v1`](../../ml/runs/tiny-byte-transformer-v1/model-card.md)
  is a reproducible 19,009-parameter training exercise. After 600 CPU steps it
  reaches 3.436 validation bits/byte, slightly worse than the 3.428 bigram
  baseline. It should remain a pipeline proof, not be scaled into a generator.
- The pinned v0.1 release contains 170 Documents and 3,291 Passages, but
  [`duplicate-candidates.jsonl`](../../data/derived/releases/corpus-v0.1.0/duplicate-candidates.jsonl)
  is empty.
- The
  [`dataset card`](../../data/derived/releases/corpus-v0.1.0/dataset-card.md)
  explicitly says the general document split does not control unknown
  variants, translation families, or editorial leakage. It also says
  regex-derived motif, being, and role suggestions are not gold labels.
- A local audit of the pinned passages found a maximum of 220 whitespace words
  per passage (median 87). That fits comfortably inside E5's 512-token ceiling
  in normal cases, but the trainer must still record truncation counts rather
  than assume zero.

These facts rule out treating edition, region, tradition, or rule suggestions
as a useful supervised target. They also make scaling the from-scratch causal
model the wrong next move.

## Candidate comparison

| Candidate | Defensible labels now | Leakage-safe evaluation | Compute | Immediate value | Decision |
|---|---|---|---|---|---|
| Duplicate / variant retrieval | No current gold families; exact normalized duplicates are absent. Human pair judgments can be built, but “same variant” needs a stricter scholarly rubric than relevance. | Requires family-disjoint splits that cannot exist until families are curated. Edition and title fingerprints are serious shortcuts. | Low to moderate | Very high for corpus consolidation | **Second task.** Use the passage-retrieval stack to propose candidates, then curate family labels. |
| Passage representation / cited retrieval | Yes. A reviewer can decide whether an exact passage directly supports a query and preserve the evidence ID. | Group by Document, known derivation cluster, and later curated variant family. Freeze test queries and qrels. | Low; 33.4M encoder, 3,291 candidate passages | High for Search, RAG, source inspection, and later variant mining | **First task.** |
| Metadata or claim suggestion | Not yet. Current metadata is coarse and current rule suggestions are explicitly non-gold. | The existing 1.000 edition-fingerprint result predicts shortcut learning. | Low to moderate | Potentially useful review queue | **Wait.** First create independently reviewed ontology labels and provenance policy. |
| Constrained generation | There is no defensible target/reference set for correctness, attribution, or cultural adequacy. Format constraints do not solve factual evaluation. | High contamination and memorization risk on five English editions; subjective reference text compounds it. | Highest | Attractive demo, weak foundation | **Wait.** Add only after retrieval has frozen evidence and citation tests. |

“Passage representation learning” is a method, not the product contract. The
contract must remain **retrieve the source passage that supports the query**.
That prevents an embedding score from being misread as a cultural or
historical assertion.

## Benchmark contract: `source-grounded-passage-retrieval-v0.2`

### Unit of work

- **Query:** a natural-language information need, not a copied sentence.
- **Corpus item:** one immutable Passage from a pinned Corpus Release.
- **Judgment:** `2 = directly supports`, `1 = useful context`, or
  `0 = not relevant`.
- **Required evidence:** query ID, Passage ID, Document ID, Witness ID, Corpus
  release ID and manifest digest, annotator/reviewer IDs, rubric version, and
  adjudication status.
- **System output:** ordered Passage IDs with scores. Citation labels are
  joined from the release, never generated by the model.

NIST describes relevance judgments as the “right answers” of a test collection
and uses pooling to judge the union of results produced by multiple systems
([TREC relevance-judgment guidance](https://trec.nist.gov/data/reljudge_eng.html)).
Use the same basic discipline here.

### Annotation plan

1. Select passages across all five editions, document lengths, and query
   intents. Include concrete event, character/action, object, place, and
   comparison-oriented information needs. Do not ask unsupported questions
   about authenticity, ethnicity, oral performance, or historical
   transmission.
2. Have a human write a natural query after reading a source passage. Record
   the seed Passage ID, but do not automatically declare it the only relevant
   passage.
3. Pool the top 10 results from at least word TF-IDF, character TF-IDF, frozen
   E5-small-v2, and one deliberately diverse system such as a small
   cross-encoder over lexical candidates.
4. Judge the deduplicated pool against the written rubric. Validation and test
   judgments require two independent reviewers; disagreements are adjudicated
   and preserved.
5. Permit machine-written query drafts only as disposable annotation aids.
   A human must approve the query and all qrels. Do not use an LLM as the sole
   test judge. NIST's primary-source warning is explicit:
   [“Don't Use LLMs to Make Relevance Judgments”](https://www.nist.gov/publications/dont-use-llms-make-relevance-judgments).
6. Version the query set, qrels, rubric, pools, and annotation decisions as
   immutable artifacts. New judgments publish a new benchmark version.

A useful first release is **500 approved queries** with at least 100 frozen
test queries. If annotation capacity is lower, ship a lexical/frozen-model
benchmark first; do not fine-tune and report a result on a tiny test set like
the current 12-document diagnostic.

### Task-specific splits

The current `general-v1` split is insufficient. Publish a new split manifest
with these grouping rules, in descending precedence:

1. all representations and passages derived from one Witness stay together;
2. every Document stays together;
3. known duplicate, translation, or variant-family members stay together;
4. all paraphrases of one information need stay together.

No group may cross train, validation, or test. Group-based splitting is the
standard mechanism for preventing related samples from appearing in both
training and evaluation; scikit-learn's
[`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
documents this non-overlap guarantee.

Stratify the resulting groups approximately by edition and query intent, but
never break a group to improve balance. Report:

- the main grouped test result;
- per-edition and per-intent slices;
- a leave-one-edition-out stress test;
- results with title, collection, region, and tradition fields unavailable to
  the model.

The test queries, test qrels, pooling depth, primary metric, and seed list must
be frozen before tuning. Test data is evaluated once for a release candidate,
not on every training run.

### Metrics and diagnostics

Primary metric: **nDCG@10**, because relevance is graded and early rank matters.
The original DCG work was designed for graded relevance and position discount
([Järvelin and Kekäläinen, 2002](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf)).

Always report alongside it:

- Recall@20 and Recall@100;
- MRR@10 and Success@1/5;
- query count and judged-pool coverage;
- 95% paired bootstrap intervals against each baseline;
- per-edition, per-intent, and lexical-overlap quartiles;
- index bytes, embedding time, query p50/p95 latency, peak memory, model
  parameters, training device/time, energy if available, and three fixed
  random seeds for fine-tuning;
- a versioned error table with false positives, missed relevant passages,
  unjudged high ranks, truncations, and suspected family leakage.

Sentence Transformers'
[`InformationRetrievalEvaluator`](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html)
supports the required query/corpus/qrels shape and standard MRR, nDCG, and
Recall metrics. Keep the benchmark format independent of that library so a
second implementation can verify the scores.

## Small-model stack

### Data and evaluation

- Python 3.12.
- Keep the current NumPy and scikit-learn path for deterministic baselines.
- Add pinned PyTorch, Transformers, Datasets, and Sentence Transformers
  versions only when the frozen E5 experiment is implemented.
- Store JSONL task data, qrels, run manifests, predictions, metrics, model
  cards, dependency locks, and SHA-256 hashes using the repository's existing
  artifact conventions.
- At 3,291 passages, use normalized NumPy matrix multiplication for exact dense
  search. Do not add a vector database or approximate nearest-neighbour index
  yet.

### Models

1. **Lexical controls:** word `(1, 2)` TF-IDF and character `(3, 5)` TF-IDF
   with cosine similarity using scikit-learn
   [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
   Fit vocabulary/IDF on the pinned candidate corpus, never on query text.
2. **Frozen dense encoder:** `intfloat/e5-small-v2`, normalized embeddings and
   dot product. Follow its required asymmetric `query:` and `passage:`
   prefixes. The
   [E5 paper](https://arxiv.org/abs/2212.03533) trains single-vector
   representations contrastively and evaluates zero-shot and fine-tuned
   retrieval.
3. **Fine-tuned dense encoder:** the same E5 checkpoint, not a new architecture.
   Use `MultipleNegativesRankingLoss` only with a qrel-aware
   no-duplicates sampler so another relevant passage is never treated as an
   in-batch negative. Sentence Transformers documents the objective as making
   the matched query/passage score higher than in-batch alternatives and
   explicitly recommends no-duplicate batches
   ([loss documentation](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiple-negatives-ranking-loss)).
   Add reviewed hard negatives from the lexical/frozen pools after the first
   clean pair-only run.
4. **Optional reranker:** use
   [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
   frozen over only the top 20 candidates. Its card records 22.7M parameters.
   A cross-encoder is slower because it scores each query/passage pair jointly,
   so it belongs after a fast retriever, as the
   [Sentence Transformers retrieve-and-rerank design](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
   explains.

Why a bi-encoder first: SBERT showed that independently encoded text can be
compared efficiently rather than running a full transformer for every possible
pair ([Reimers and Gurevych, 2019](https://arxiv.org/abs/1908.10084)).
Dense Passage Retrieval later demonstrated the practical query/passage
dual-encoder pattern
([Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)).

Why lexical baselines remain mandatory: BEIR found BM25 robust across
heterogeneous retrieval tasks and found rerankers effective but more expensive;
no one neural architecture won consistently
([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)). Historical spelling,
names, and formulaic phrases are also exactly where lexical retrieval may win.

## Experiment ladder

Each rung emits the full run contract and is promoted only if it adds
information over the previous rung.

### 0. Benchmark and leakage tests

- Validate IDs, release digest, grouping invariants, qrel ranges, pool coverage,
  and train/validation/test non-overlap.
- Run random ranking, passage-length ranking, edition-prior ranking, and
  shuffled-query controls.
- Fail the benchmark if edition-only or length-only controls perform
  suspiciously well.

### 1. Classical retrieval

- Word TF-IDF.
- Character TF-IDF.
- A deterministic lexical fusion of both.
- Error analysis by spelling, named entity, long query, and edition.

This rung teaches data contracts, sparse matrices, ranking metrics, controls,
and reproducibility without hiding errors behind a pretrained model.

### 2. Frozen representation learning

- Run E5-small-v2 without training.
- Record prefix handling, truncation, batch size, embedding normalization,
  latency, and exact embedding hashes.
- Compare dense, lexical, and a simple deterministic fusion.

This establishes whether a general English representation model helps this
historical-English corpus before spending GPU time.

### 3. Contrastive fine-tuning

- Fine-tune the same E5 checkpoint on train query/positive pairs.
- Use qrel-aware batches, three fixed seeds, early stopping on validation
  nDCG@10, and a small declared hyperparameter grid.
- Repeat with reviewed hard negatives.
- Ablate prefixes, hard negatives, character baseline, and metadata fields.

Do not claim success from lower training loss. The only success condition is a
frozen grouped retrieval improvement.

### 4. Optional reranking

- Rerank top 20 from the best lexical/dense/fused retriever with the frozen
  MiniLM cross-encoder.
- Fine-tune the reranker only if the frozen reranker improves validation and
  enough independently judged positive/negative pairs exist.
- Report the first-stage Recall@20 ceiling so reranker gains are not confused
  with retrieval gains.

### 5. Transfer to variant discovery

Use the best passage/document representation to generate a cross-edition
duplicate/variant review queue. Humans then label `same witness`, `probable
variant`, `related motif only`, or `unrelated`, with evidence. Once enough
families exist, publish a separate family-disjoint benchmark. Never silently
promote embedding neighbours into canonical tale identities.

## Promotion gates

A model is useful only if all gates pass:

1. The task release is pinned, reproducible, human judged, and group-disjoint.
2. Fine-tuned E5 improves test nDCG@10 by at least **0.03 absolute** over both
   the best lexical system and frozen E5, and its paired 95% interval excludes
   zero.
3. It does not lose more than 0.03 absolute Recall@20 against the best
   competitor or collapse on an edition/intent slice.
4. Shuffled-label/query controls stay near chance and metadata ablations do not
   reveal a shortcut.
5. The model card states English-only, historical-edition, annotation, rights,
   and cultural-claim limitations.
6. A clean checkout reproduces metrics within a declared tolerance.

If the fine-tuned model does not clear these gates, the experiment is still
successful ML engineering: keep the lexical/frozen system, publish the negative
result, expand judgments or corpus diversity, and do not scale the model.

## Concrete v0.2 deliverables

1. `ml/data/source-grounded-passage-retrieval-v0.2/` with pinned queries, qrels,
   pools, group assignments, rubric, manifest, and digests.
2. One evaluator that scores any TREC-style run of `(queryId, passageId,
   rank, score)`.
3. Checked-in word/character TF-IDF, frozen E5, and fine-tuned E5 run artifacts.
4. A leakage report covering document/family groups, edition slices, query
   overlap, and controls.
5. A model card and error analysis for every promoted neural run.
6. A machine-generated variant-candidate queue explicitly marked
   `review-only`, feeding the second task rather than altering Corpus truth.

## Sources

Only primary framework/model documentation and original papers were used for
the external technical claims:

- Wang et al.,
  [Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533),
  2022.
- Reimers and Gurevych,
  [Sentence-BERT](https://arxiv.org/abs/1908.10084), 2019.
- Karpukhin et al.,
  [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906), 2020.
- Thakur et al.,
  [BEIR](https://arxiv.org/abs/2104.08663), 2021.
- Järvelin and Kekäläinen,
  [Cumulated Gain-Based Evaluation of IR Techniques](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf),
  2002.
- [Sentence Transformers documentation](https://sbert.net/).
- [scikit-learn documentation](https://scikit-learn.org/stable/).
- [NIST TREC relevance-judgment guidance](https://trec.nist.gov/data/reljudge_eng.html).
- Soboroff,
  [Don't Use LLMs to Make Relevance Judgments](https://www.nist.gov/publications/dont-use-llms-make-relevance-judgments),
  2025.
