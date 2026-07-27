from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

RUN_V1 = "folklore-ml-run-v1"
RUN_V2 = "folklore-ml-run-v2"
DECISIONS = {"adopt", "reject", "continue", "inconclusive"}
SHA256_LENGTH = 64


def _mapping(value: object, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path} must be an object.")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{path} must be a non-empty string.")
    return value


def _sha256(value: object, path: str) -> str:
    text = _text(value, path)
    if len(text) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{path} must be a lowercase SHA-256 digest.")
    return text


def _finite_numbers(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{path} contains a non-finite number.")


def _string_list(value: object, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{path} must be a non-empty list.")
    for index, item in enumerate(value):
        _text(item, f"{path}[{index}]")
    return value


def _cost_channel(value: object, path: str, *, money: bool = False) -> None:
    channel = _mapping(value, path)
    status = channel.get("status")
    if status not in {"recorded", "not-recorded"}:
        raise RuntimeError(f"{path}.status must be recorded or not-recorded.")
    required = {"status", "value", "currency" if money else "unit"}
    missing = required - channel.keys()
    if missing:
        raise RuntimeError(f"{path} is incomplete; missing {', '.join(sorted(missing))}.")
    if status == "recorded":
        number = channel["value"]
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number) or number < 0:
            raise RuntimeError(f"{path}.value must be a finite non-negative number.")
    else:
        if channel["value"] is not None:
            raise RuntimeError(f"{path}.value must be null when status is not-recorded.")
        _text(channel.get("rationale"), f"{path}.rationale")
    _text(channel["currency" if money else "unit"], f"{path}.{'currency' if money else 'unit'}")


def validate_experiment_record(record: object) -> dict:
    run = dict(_mapping(record, "record"))
    schema_version = run.get("schemaVersion")
    if schema_version == RUN_V1:
        for field in (
            "experiment",
            "corpusRelease",
            "corpusManifestSha256",
            "datasetSha256",
            "seed",
            "command",
            "artifacts",
        ):
            if field not in run:
                raise RuntimeError(f"v1 record is missing {field}.")
        _text(run["experiment"], "experiment")
        _text(run["corpusRelease"], "corpusRelease")
        _sha256(run["corpusManifestSha256"], "corpusManifestSha256")
        _sha256(run["datasetSha256"], "datasetSha256")
        if not isinstance(run["seed"], int) or isinstance(run["seed"], bool):
            raise RuntimeError("seed must be an integer.")
        _text(run["command"], "command")
        _mapping(run["artifacts"], "artifacts")
        return run
    if schema_version != RUN_V2:
        raise RuntimeError(f"Unsupported experiment record schema: {schema_version!r}")

    _text(run.get("experimentId"), "experimentId")
    question = _mapping(run.get("question"), "question")
    if question.get("kind") not in {"product", "learning"}:
        raise RuntimeError("question.kind must be product or learning.")
    _text(question.get("text"), "question.text")
    _text(run.get("hypothesis"), "hypothesis")

    evaluation = _mapping(run.get("evaluation"), "evaluation")
    frozen = _mapping(evaluation.get("frozenIdentity"), "evaluation.frozenIdentity")
    _text(frozen.get("id"), "evaluation.frozenIdentity.id")
    _sha256(frozen.get("sha256"), "evaluation.frozenIdentity.sha256")

    for field in ("baseline", "candidate"):
        method = _mapping(run.get(field), field)
        _text(method.get("name"), f"{field}.name")
        _mapping(method.get("metrics"), f"{field}.metrics")
        _finite_numbers(method["metrics"], f"{field}.metrics")

    metrics = _mapping(run.get("metrics"), "metrics")
    _string_list(metrics.get("primary"), "metrics.primary")
    _string_list(metrics.get("secondary"), "metrics.secondary")

    human_review = _mapping(run.get("humanReview"), "humanReview")
    _string_list(human_review.get("criteria"), "humanReview.criteria")

    provenance = _mapping(run.get("provenance"), "provenance")
    _sha256(provenance.get("datasetSha256"), "provenance.datasetSha256")
    code = _mapping(provenance.get("code"), "provenance.code")
    _text(code.get("repository"), "provenance.code.repository")
    _text(code.get("revision"), "provenance.code.revision")
    _text(provenance.get("command"), "provenance.command")

    cost = _mapping(run.get("cost"), "cost")
    _cost_channel(cost.get("time"), "cost.time")
    _cost_channel(cost.get("compute"), "cost.compute")
    _cost_channel(cost.get("money"), "cost.money", money=True)

    _string_list(run.get("limitations"), "limitations")
    decision = _mapping(run.get("decision"), "decision")
    if decision.get("outcome") not in DECISIONS:
        raise RuntimeError(
            "decision.outcome must be adopt, reject, continue, or inconclusive."
        )
    _text(decision.get("rationale"), "decision.rationale")
    return run


def validate_experiment_record_file(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    return validate_experiment_record(record)
