from __future__ import annotations

import hashlib
import json
import math
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

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/derived/releases/corpus-v0.1.0"
TASK_DATA = ROOT / "ml/data/edition-fingerprint-v1"
RUN_ROOT = ROOT / "ml/runs/edition-fingerprint-v1"
SEED = 20260724


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


def prepare() -> dict:
    release_manifest_bytes = (RELEASE / "manifest.json").read_bytes()
    manifest = json.loads(release_manifest_bytes)
    documents = {row["id"]: row for row in _records(RELEASE / "documents.jsonl")}
    witnesses = {row["documentId"]: row for row in _records(RELEASE / "witnesses.jsonl")}
    splits = _records(RELEASE / "splits.jsonl")
    grouped: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for assignment in sorted(splits, key=lambda row: row["documentId"]):
        document = documents[assignment["documentId"]]
        witness = witnesses[assignment["documentId"]]
        grouped[assignment["split"]].append(
            {
                "documentId": document["id"],
                "witnessId": witness["id"],
                "label": document["editionId"],
                "title": document["title"],
                "text": witness["text"],
            }
        )

    TASK_DATA.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for split, rows in grouped.items():
        contents = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        (TASK_DATA / f"{split}.jsonl").write_text(contents, encoding="utf-8")
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
    _write_json(TASK_DATA / "manifest.json", task_manifest)
    print(json.dumps(task_manifest, indent=2))
    return task_manifest


def _load_task(split: str) -> list[dict]:
    return _records(TASK_DATA / f"{split}.jsonl")


def run_classifier() -> dict:
    task = prepare()
    train = _load_task("train")
    test = _load_task("test")
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
    predictions_output = [
        {
            "documentId": row["documentId"],
            "title": row["title"],
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
        }
        for row, actual, predicted in zip(test, test_labels, predictions)
    ]
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
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_ROOT / "run.json", run)
    for filename, contents in artifact_contents.items():
        (RUN_ROOT / filename).write_text(contents, encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def verify() -> None:
    task = json.loads((TASK_DATA / "manifest.json").read_text(encoding="utf-8"))
    release = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    if task["corpusRelease"] != release["releaseId"]:
        raise RuntimeError("Task data is not pinned to the current release.")
    release_digest = hashlib.sha256((RELEASE / "manifest.json").read_bytes()).hexdigest()
    if task["corpusManifestSha256"] != release_digest:
        raise RuntimeError("Task data manifest digest does not match the release.")
    task_digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        contents = (TASK_DATA / f"{split}.jsonl").read_text(encoding="utf-8")
        task_digest.update(split.encode())
        task_digest.update(contents.encode())
    if task["datasetSha256"] != task_digest.hexdigest():
        raise RuntimeError("Prepared task data digest mismatch.")

    def assert_finite(value: object, path: str = "metrics") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert_finite(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                assert_finite(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite value at {path}")

    for run_name in ("edition-fingerprint-v1", "tiny-byte-transformer-v1"):
        run_root = ROOT / "ml/runs" / run_name
        for filename in ("run.json", "metrics.json", "model-card.md"):
            if not (run_root / filename).is_file():
                raise RuntimeError(f"Missing {run_name}/{filename}")
        run = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
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
        assert_finite(metrics)

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
    print("Verified task digest, ML artifact digests, checkpoint shape, and release pins.")
