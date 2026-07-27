# Folklore ML experiment records v2

`folklore-ml-run-v2` records the complete decision lifecycle around an experiment without rewriting preserved `folklore-ml-run-v1` evidence.

A v2 record must name its product or learning question, hypothesis, frozen evaluation identity, baseline and candidate results, primary and secondary metrics, human-review criteria, dataset provenance, an immutable 40-character code revision, command and source lineage, explicit time/compute/money cost, limitations, and one decision with an explicit target and scope: `adopt`, `reject`, `continue`, or `inconclusive`.

Unknown historical cost is allowed only when it is fail-visible: use `status: "not-recorded"`, a null value, a unit or currency, and a rationale. Omitting a cost channel is invalid.

## Minimal verification

```bash
python -m folklore_ml verify-run ml/runs/edition-fingerprint-v1-record-v2/run.json
python -m folklore_ml verify --legacy-only
npm test
```

The first command validates one v1 or v2 record. The second proves that existing committed v1 task data, runs, artifacts, and checkpoint remain readable and reproducible. The representative v2 record is a metadata migration around preserved v1 evidence, not a claim that the historical experiment was rerun.
