from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .experiment_record import (
    assert_finite_numbers,
    validate_experiment_record,
    validate_experiment_record_file,
)
from .corpus import (
    InstalledCorpus,
    default_lock_path,
    install_release,
    load_lock,
    verify_cached_release,
    verify_release,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_TASK_DATA = ROOT / "ml/data/edition-fingerprint-v1"
LEGACY_RUN_ROOT = ROOT / "ml/runs/edition-fingerprint-v1"
SEED = 20260724


def _resolve_legacy_release() -> Path:
    configured = os.environ.get("FOLKLORE_CORPUS_DIR")
    if configured:
        return Path(configured).resolve()
    releases_root = ROOT / "data/derived/releases"
    candidates = sorted(
        path for path in releases_root.iterdir()
        if path.is_dir() and path.name.startswith("corpus-v")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one installed Corpus Release, "
            f"found {len(candidates)}. Set FOLKLORE_CORPUS_DIR explicitly."
        )
    return candidates[0]


def resolve_corpus_release(*, install: bool = False) -> InstalledCorpus:
    if os.environ.get("FOLKLORE_CORPUS_DIR"):
        return verify_release(_resolve_legacy_release())
    lock_path = default_lock_path(ROOT)
    if lock_path is not None:
        lock = load_lock(lock_path)
        if install:
            return install_release(lock)
        return verify_cached_release(lock)
    release = _resolve_legacy_release()
    return verify_release(release)


def _task_data(corpus: InstalledCorpus) -> Path:
    configured = os.environ.get("FOLKLORE_ML_DATA_DIR")
    if configured:
        return Path(configured).resolve()
    if corpus.provenance:
        return (
            LEGACY_TASK_DATA
            / "by-corpus"
            / corpus.provenance["manifestSha256"]
        )
    return LEGACY_TASK_DATA


def _run_root(corpus: InstalledCorpus) -> Path:
    configured = os.environ.get("FOLKLORE_ML_RUN_DIR")
    if configured:
        return Path(configured).resolve()
    if corpus.provenance:
        return (
            LEGACY_RUN_ROOT
            / "by-corpus"
            / corpus.provenance["manifestSha256"]
        )
    return LEGACY_RUN_ROOT


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def verify_corpus_release() -> dict:
    return resolve_corpus_release().summary


def prepare() -> dict:
    corpus = resolve_corpus_release(install=True)
    release = corpus.path
    release_manifest_bytes = (release / "manifest.json").read_bytes()
    manifest = json.loads(release_manifest_bytes)
    documents = {row["id"]: row for row in _records(release / "documents.jsonl")}
    witnesses = {
        row["documentId"]: row for row in _records(release / "witnesses.jsonl")
    }
    splits = _records(release / "splits.jsonl")
    passage_ids: dict[str, list[str]] = {}
    if corpus.provenance:
        for passage in _records(release / "passages.jsonl"):
            passage_ids.setdefault(passage["documentId"], []).append(passage["id"])
    grouped: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for assignment in sorted(splits, key=lambda row: row["documentId"]):
        document = documents.get(assignment["documentId"])
        witness = witnesses.get(assignment["documentId"])
        if (
            document is None
            or witness is None
            or not isinstance(document.get("editionId"), str)
            or not isinstance(witness.get("text"), str)
        ):
            continue
        row = {
            "documentId": document["id"],
            "witnessId": witness["id"],
            "label": document["editionId"],
            "title": document["title"],
            "text": witness["text"],
        }
        if corpus.provenance:
            row["passageIds"] = sorted(passage_ids.get(document["id"], []))
            if not row["passageIds"]:
                raise RuntimeError(
                    f"Task document has no relevant Passage IDs: {document['id']}"
                )
        grouped[assignment["split"]].append(row)

    task_data = _task_data(corpus)
    task_data.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for split, rows in grouped.items():
        contents = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        (task_data / f"{split}.jsonl").write_text(contents, encoding="utf-8")
        digest.update(split.encode())
        digest.update(contents.encode())
    task_manifest = {
        "schemaVersion": "folklore-ml-task-v1",
        "task": "edition-fingerprint-v1",
        "purpose": "Diagnostic source-edition attribution; not folklore classification.",
        "corpusRelease": manifest["releaseId"],
        "corpusManifestSha256": hashlib.sha256(release_manifest_bytes).hexdigest(),
        "counts": {split: len(rows) for split, rows in grouped.items()},
        "labels": sorted({row["label"] for rows in grouped.values() for row in rows}),
        "datasetSha256": digest.hexdigest(),
        "seed": SEED,
    }
    if corpus.provenance:
        task_manifest["corpus"] = corpus.provenance
        task_manifest["selection"] = {
            "kind": "documents-with-edition-and-text-witness-v1",
            "note": (
                "Preserves the v0.1 edition-fingerprint diagnostic; "
                "new multilingual retrieval tasks are defined separately."
            ),
        }
        task_manifest["passageIds"] = sorted(
            passage_id
            for rows in grouped.values()
            for row in rows
            for passage_id in row["passageIds"]
        )
    _write_json(task_data / "manifest.json", task_manifest)
    print(json.dumps(task_manifest, indent=2))
    return task_manifest


def _load_task(task_data: Path, split: str) -> list[dict]:
    return _records(task_data / f"{split}.jsonl")


def run_classifier() -> dict:
    task = prepare()
    corpus = resolve_corpus_release()
    task_data = _task_data(corpus)
    run_root = _run_root(corpus)
    train = _load_task(task_data, "train")
    test = _load_task(task_data, "test")
    train_text = [row["text"] for row in train]
    train_labels = np.array([row["label"] for row in train])
    test_text = [row["text"] for row in test]
    test_labels = np.array([row["label"] for row in test])
    labels = task["labels"]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)
    model = LinearSVC(C=1.0, random_state=SEED)
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train), 1)), train_labels)
    majority_predictions = majority.predict(np.zeros((len(test), 1)))

    length_model = make_pipeline(StandardScaler(), LinearSVC(C=0.1, random_state=SEED))
    train_lengths = np.array(
        [[len(row["text"]), len(row["text"].split())] for row in train],
        dtype=float,
    )
    test_lengths = np.array(
        [[len(row["text"]), len(row["text"].split())] for row in test],
        dtype=float,
    )
    length_model.fit(train_lengths, train_labels)
    length_predictions = length_model.predict(test_lengths)

    rng = np.random.default_rng(SEED)
    shuffled_model = LinearSVC(C=1.0, random_state=SEED)
    shuffled_model.fit(train_features, rng.permutation(train_labels))
    shuffled_predictions = shuffled_model.predict(test_features)

    def scores(values: np.ndarray) -> dict:
        return {
            "macroF1": float(
                f1_score(test_labels, values, labels=labels, average="macro")
            ),
            "accuracy": float(accuracy_score(test_labels, values)),
        }

    feature_names = vectorizer.get_feature_names_out()
    top_features = {}
    for label, coefficients in zip(model.classes_, model.coef_):
        indices = np.argsort(coefficients)[-12:][::-1]
        top_features[label] = [
            {"feature": feature_names[index], "weight": float(coefficients[index])}
            for index in indices
        ]

    metrics = {
        "experiment": "edition-fingerprint-v1",
        "interpretation": (
            "Measures how easily editorial and translation fingerprints identify "
            "the five seed editions; it is not a folklore-genre classifier."
        ),
        "testCount": len(test),
        "labels": labels,
        "model": scores(predictions),
        "controls": {
            "majority": scores(majority_predictions),
            "lengthOnly": scores(length_predictions),
            "shuffledLabels": scores(shuffled_predictions),
        },
        "confusionMatrix": {
            "labelOrder": labels,
            "values": confusion_matrix(test_labels, predictions, labels=labels).tolist(),
        },
    }
    predictions_output = []
    for row, actual, predicted in zip(test, test_labels, predictions):
        output = {
            "documentId": row["documentId"],
            "title": row["title"],
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
        }
        if "passageIds" in row:
            output["passageIds"] = row["passageIds"]
        predictions_output.append(output)
    predictions_text = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in predictions_output
    )
    model_card = f"""# Edition Fingerprint v1

This diagnostic asks whether character 3–5 gram patterns can identify which of
the five seed editions supplied a held-out story. It measures editorial,
translator, orthographic, and formatting leakage—not folklore understanding.

- Corpus: `{task["corpusRelease"]}`
- Split: immutable document-level general-v1 split
- Seed: `{SEED}`
- Test macro-F1: `{metrics["model"]["macroF1"]:.3f}`
- Majority macro-F1: `{metrics["controls"]["majority"]["macroF1"]:.3f}`
- Length-only macro-F1: `{metrics["controls"]["lengthOnly"]["macroF1"]:.3f}`
- Shuffled-label macro-F1: `{metrics["controls"]["shuffledLabels"]["macroF1"]:.3f}`

The useful result is evidence of corpus leakage risk. Do not use this model to
infer cultural origin, ethnicity, authenticity, motif, or tale type.
"""
    artifact_contents = {
        "metrics.json": _json_text(metrics),
        "top-features.json": _json_text(top_features),
        "predictions.jsonl": predictions_text,
        "model-card.md": model_card,
    }
    run = {
        "schemaVersion": "folklore-ml-run-v1",
        "experiment": "edition-fingerprint-v1",
        "corpusRelease": task["corpusRelease"],
        "corpusManifestSha256": task["corpusManifestSha256"],
        "datasetSha256": task["datasetSha256"],
        "seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
        },
        "command": "python -m folklore_ml classifier",
        "artifacts": {
            filename: _sha256_text(contents)
            for filename, contents in artifact_contents.items()
        },
    }
    if corpus.provenance:
        run["corpus"] = corpus.provenance
        run["passageIds"] = task["passageIds"]
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "run.json", run)
    for filename, contents in artifact_contents.items():
        (run_root / filename).write_text(contents, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def verify(*, legacy_only: bool = False) -> None:
    legacy_corpus = verify_release(_resolve_legacy_release())
    task = json.loads(
        (LEGACY_TASK_DATA / "manifest.json").read_text(encoding="utf-8")
    )
    release = json.loads(
        (legacy_corpus.path / "manifest.json").read_text(encoding="utf-8")
    )
    if task["corpusRelease"] != release["releaseId"]:
        raise RuntimeError("Task data is not pinned to the current release.")
    release_digest = hashlib.sha256(
        (legacy_corpus.path / "manifest.json").read_bytes()
    ).hexdigest()
    if task["corpusManifestSha256"] != release_digest:
        raise RuntimeError("Task data manifest digest does not match the release.")
    task_digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        contents = (LEGACY_TASK_DATA / f"{split}.jsonl").read_text(encoding="utf-8")
        task_digest.update(split.encode())
        task_digest.update(contents.encode())
    if task["datasetSha256"] != task_digest.hexdigest():
        raise RuntimeError("Prepared task data digest mismatch.")

    for run_name in ("edition-fingerprint-v1", "tiny-byte-transformer-v1"):
        run_root = ROOT / "ml/runs" / run_name
        for filename in ("run.json", "metrics.json", "model-card.md"):
            if not (run_root / filename).is_file():
                raise RuntimeError(f"Missing {run_name}/{filename}")
        run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        validate_experiment_record(run)
        if (
            run["corpusRelease"] != release["releaseId"]
            or run["corpusManifestSha256"] != release_digest
            or run["seed"] != SEED
            or run["datasetSha256"] != task["datasetSha256"]
        ):
            raise RuntimeError(f"Unpinned or unseeded run: {run_name}")
        for filename, expected_digest in run["artifacts"].items():
            artifact = run_root / filename
            if not artifact.is_file():
                raise RuntimeError(f"Missing {run_name}/{filename}")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_digest:
                raise RuntimeError(f"Artifact digest mismatch: {run_name}/{filename}")
        metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
        assert_finite_numbers(metrics, "metrics")

    checkpoint = json.loads(
        (ROOT / "ml/runs/tiny-byte-transformer-v1/checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    parameter_count = 0
    for name, weight in checkpoint["weights"].items():
        expected_size = math.prod(weight["shape"])
        if expected_size != len(weight["values"]):
            raise RuntimeError(f"Checkpoint shape mismatch: {name}")
        if not all(math.isfinite(value) for value in weight["values"]):
            raise RuntimeError(f"Non-finite checkpoint value: {name}")
        parameter_count += expected_size
    tiny_metrics = json.loads(
        (ROOT / "ml/runs/tiny-byte-transformer-v1/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if parameter_count != tiny_metrics["parameterCount"]:
        raise RuntimeError("Checkpoint parameter count does not match metrics.")
    if legacy_only:
        print("Verified preserved v0.1 task, runs, and checkpoint independently.")
        return
    validate_experiment_record_file(
        ROOT / "ml/runs/edition-fingerprint-v1-record-v2/run.json"
    )
    lock_path = default_lock_path(ROOT)
    if lock_path is None:
        print(
            "Verified task digest, ML artifact digests, checkpoint shape, "
            "and release pins."
        )
        return
    current_corpus = verify_cached_release(load_lock(lock_path))
    if current_corpus.provenance:
        current_task_root = _task_data(current_corpus)
        if current_task_root.is_dir():
            current_task = _read_provenance_artifact(
                current_task_root / "manifest.json",
                current_corpus,
                "task",
            )
            current_digest = hashlib.sha256()
            release_passage_ids = {
                row["id"]
                for row in _records(current_corpus.path / "passages.jsonl")
            }
            task_row_passage_ids: set[str] = set()
            for split in ("train", "validation", "test"):
                contents = (current_task_root / f"{split}.jsonl").read_text(
                    encoding="utf-8"
                )
                for row in _records(current_task_root / f"{split}.jsonl"):
                    row_passage_ids = row.get("passageIds")
                    if not isinstance(row_passage_ids, list) or not row_passage_ids:
                        raise RuntimeError(
                            f"Pinned task row has no Passage IDs: "
                            f"{row.get('documentId')}"
                        )
                    if len(row_passage_ids) != len(set(row_passage_ids)):
                        raise RuntimeError("Pinned task row has duplicate Passage IDs.")
                    if not set(row_passage_ids) <= release_passage_ids:
                        raise RuntimeError(
                            "Pinned task row references an unknown Passage ID."
                        )
                    task_row_passage_ids.update(row_passage_ids)
                current_digest.update(split.encode())
                current_digest.update(contents.encode())
            if current_task["datasetSha256"] != current_digest.hexdigest():
                raise RuntimeError("Prepared pinned task data digest mismatch.")
            if task_row_passage_ids != set(current_task["passageIds"]):
                raise RuntimeError(
                    "Pinned task Passage IDs do not match its task rows."
                )
            current_run_root = _run_root(current_corpus)
            if (current_run_root / "run.json").is_file():
                current_run = _read_provenance_artifact(
                    current_run_root / "run.json",
                    current_corpus,
                    "run",
                )
                if current_run["datasetSha256"] != current_task["datasetSha256"]:
                    raise RuntimeError("Pinned run and task data disagree.")
                if current_run["passageIds"] != current_task["passageIds"]:
                    raise RuntimeError("Pinned run and task Passage IDs disagree.")
    print("Verified task digest, ML artifact digests, checkpoint shape, and release pins.")


def _read_provenance_artifact(
    path: Path,
    corpus: InstalledCorpus,
    description: str,
) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("corpus") != corpus.provenance:
        raise RuntimeError(f"Pinned {description} Corpus provenance mismatch.")
    passage_ids = artifact.get("passageIds")
    if (
        not isinstance(passage_ids, list)
        or not passage_ids
        or len(passage_ids) != len(set(passage_ids))
        or not all(
            isinstance(passage_id, str) and passage_id.startswith("fa:passage:")
            for passage_id in passage_ids
        )
    ):
        raise RuntimeError(f"Pinned {description} Passage IDs are invalid.")
    return artifact
