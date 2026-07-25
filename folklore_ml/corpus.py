from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, urlunparse

from jsonschema import SchemaError
from jsonschema.validators import validator_for

LOCK_SCHEMA_VERSION = "folklore-corpus-lock-v1"
SUPPORTED_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
REQUIRED_ML_ARTIFACTS = frozenset(
    {
        "manifest.schema.json",
        "schema.json",
        "documents.jsonl",
        "witnesses.jsonl",
        "passages.jsonl",
        "splits.jsonl",
        "dataset-card.md",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CorpusLock:
    path: Path
    repository: str
    tag: str
    asset: str
    url: str
    archive_sha256: str
    manifest_sha256: str
    release_id: str
    version: str
    manifest_schema_version: str

    @property
    def provenance(self) -> dict:
        return {
            "releaseId": self.release_id,
            "version": self.version,
            "manifestSchemaVersion": self.manifest_schema_version,
            "manifestSha256": self.manifest_sha256,
            "archiveSha256": self.archive_sha256,
            "sourceRepository": self.repository,
            "sourceTag": self.tag,
            "sourceAsset": self.asset,
        }


@dataclass(frozen=True)
class InstalledCorpus:
    path: Path
    summary: dict
    provenance: dict | None


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing {description}: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description.capitalize()} must be a JSON object: {path}")
    return value


def _safe_relative_path(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"Unsafe {description}: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"Unsafe {description}: {value}")
    if path.as_posix() != value:
        raise RuntimeError(f"Unsafe {description}: {value}")
    return value


def load_lock(path: Path) -> CorpusLock:
    value = _read_json(path, "Corpus lock")
    expected_fields = {
        "schemaVersion",
        "source",
        "archiveSha256",
        "manifestSha256",
        "releaseId",
        "version",
        "manifestSchemaVersion",
    }
    if set(value) != expected_fields:
        raise RuntimeError("Corpus lock fields do not match folklore-corpus-lock-v1.")
    if value["schemaVersion"] != LOCK_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported Corpus lock schema: {value['schemaVersion']}")
    source = value["source"]
    if not isinstance(source, dict) or set(source) != {
        "repository",
        "tag",
        "asset",
        "url",
    }:
        raise RuntimeError("Corpus lock source fields are invalid.")
    if not all(isinstance(source[field], str) for field in source):
        raise RuntimeError("Corpus lock source values must be strings.")
    for field in ("archiveSha256", "manifestSha256"):
        if not isinstance(value[field], str) or not SHA256_PATTERN.fullmatch(value[field]):
            raise RuntimeError(f"Corpus lock {field} must be a lowercase SHA-256.")
    version = value["version"]
    release_id = value["releaseId"]
    if not isinstance(version, str) or not version:
        raise RuntimeError("Corpus lock version is invalid.")
    if release_id != f"fa:release:corpus-v{version}":
        raise RuntimeError("Corpus lock release ID and version disagree.")
    if source["tag"] != f"corpus-v{version}":
        raise RuntimeError("Corpus lock source tag and version disagree.")
    if source["asset"] != f"folklore-corpus-v{version}.tar.gz":
        raise RuntimeError("Corpus lock asset and version disagree.")
    if not isinstance(source["repository"], str) or not source["repository"]:
        raise RuntimeError("Corpus lock source repository is invalid.")
    parsed_url = urlparse(source["url"])
    loopback_http = (
        parsed_url.scheme == "http"
        and parsed_url.hostname in {"127.0.0.1", "::1", "localhost"}
    )
    if parsed_url.scheme != "https" and not loopback_http:
        raise RuntimeError("Corpus lock URL must use HTTPS.")
    if parsed_url.username or parsed_url.password:
        raise RuntimeError("Corpus lock URL must not contain credentials.")
    if Path(parsed_url.path).name != source["asset"] and not loopback_http:
        raise RuntimeError("Corpus lock URL and source asset disagree.")
    if value["manifestSchemaVersion"] != "folklore-release-manifest-v1":
        raise RuntimeError(
            "Unsupported Corpus manifest schema: "
            f"{value['manifestSchemaVersion']}"
        )
    return CorpusLock(
        path=path.resolve(),
        repository=source["repository"],
        tag=source["tag"],
        asset=source["asset"],
        url=source["url"],
        archive_sha256=value["archiveSha256"],
        manifest_sha256=value["manifestSha256"],
        release_id=release_id,
        version=version,
        manifest_schema_version=value["manifestSchemaVersion"],
    )


def default_lock_path(project_root: Path) -> Path | None:
    configured = os.environ.get("FOLKLORE_CORPUS_LOCK")
    if configured:
        return Path(configured).resolve()
    committed = project_root / "corpus-release.lock.json"
    return committed if committed.is_file() else None


def _platform_cache_root() -> Path:
    configured = os.environ.get("FOLKLORE_CACHE_DIR")
    if configured:
        return Path(configured).resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg).resolve() if xdg else Path.home() / ".cache"


def cache_path(lock: CorpusLock) -> Path:
    return (
        _platform_cache_root()
        / "folklore-atlas"
        / "corpus"
        / "sha256"
        / lock.manifest_sha256
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(
    manifest: dict,
    manifest_digest: str,
    release_root: Path,
    lock: CorpusLock | None,
) -> dict:
    if manifest.get("schemaVersion") != "folklore-release-manifest-v1":
        raise RuntimeError(
            f"Unsupported Corpus Release manifest: {manifest.get('schemaVersion')}"
        )
    version = manifest.get("version")
    release_id = manifest.get("releaseId")
    if not isinstance(version, str) or release_id != f"fa:release:corpus-v{version}":
        raise RuntimeError("Corpus Release ID and version disagree.")
    if lock and (
        release_id != lock.release_id
        or version != lock.version
        or manifest["schemaVersion"] != lock.manifest_schema_version
    ):
        raise RuntimeError("Corpus Release identity does not match the lock.")
    producer = manifest.get("producer")
    if (
        not isinstance(producer, dict)
        or not isinstance(producer.get("repository"), str)
        or not producer["repository"]
        or not isinstance(producer.get("commit"), str)
        or not producer["commit"]
    ):
        raise RuntimeError("Corpus Release producer identity is invalid.")
    if lock and producer["repository"] != lock.repository:
        raise RuntimeError("Corpus Release producer repository does not match the lock.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Corpus Release has no declared artifacts.")
    paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("Corpus Release artifact declaration is invalid.")
        relative = _safe_relative_path(artifact.get("path"), "artifact path")
        if relative in paths:
            raise RuntimeError(f"Duplicate artifact path: {relative}")
        paths.add(relative)
        byte_length = artifact.get("byteLength")
        expected_digest = artifact.get("sha256")
        if not isinstance(byte_length, int) or byte_length < 0:
            raise RuntimeError(f"Invalid artifact byte length: {relative}")
        if not isinstance(expected_digest, str) or not SHA256_PATTERN.fullmatch(
            expected_digest
        ):
            raise RuntimeError(f"Invalid artifact digest: {relative}")
        artifact_path = release_root / relative
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise RuntimeError(f"Missing Corpus artifact: {relative}")
        if artifact_path.stat().st_size != byte_length:
            raise RuntimeError(f"Artifact byte length mismatch: {relative}")
        if _sha256(artifact_path) != expected_digest:
            raise RuntimeError(f"Artifact digest mismatch: {relative}")
    missing = sorted(REQUIRED_ML_ARTIFACTS - paths)
    if missing:
        raise RuntimeError(f"Corpus Release is missing required artifact: {missing[0]}")
    manifest_schema = _read_json(
        release_root / "manifest.schema.json",
        "Corpus Release manifest schema",
    )
    schema_dialect = manifest_schema.get("$schema")
    if schema_dialect not in (None, SUPPORTED_JSON_SCHEMA_DIALECT):
        raise RuntimeError(
            f"Unsupported Corpus Release manifest schema dialect: {schema_dialect}"
        )
    try:
        validator_class = validator_for(manifest_schema)
        validator_class.check_schema(manifest_schema)
        validator = validator_class(manifest_schema)
    except SchemaError as error:
        raise RuntimeError(
            f"Corpus Release manifest schema is not valid: {error.message}"
        ) from error
    schema_errors = sorted(
        validator.iter_errors(manifest),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    if schema_errors:
        error = schema_errors[0]
        raise RuntimeError(
            f"Corpus Release manifest schema mismatch at {error.json_path}: "
            f"{error.message}"
        )
    expected_files = paths | {"manifest.json"}
    if (release_root / "acquisition.json").exists():
        expected_files.add("acquisition.json")
    actual_files: set[str] = set()
    for path in release_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(
                f"Corpus cache contains a symbolic link: "
                f"{path.relative_to(release_root).as_posix()}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(release_root).as_posix())
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise RuntimeError(f"Corpus Release contains undeclared file: {unexpected[0]}")
    return {
        "releaseId": release_id,
        "version": version,
        "manifestSchemaVersion": manifest["schemaVersion"],
        "manifestSha256": manifest_digest,
        "artifactCount": len(artifacts),
    }


def verify_release(
    release_root: Path,
    lock: CorpusLock | None = None,
    *,
    require_acquisition: bool = False,
) -> InstalledCorpus:
    release_root = release_root.resolve()
    manifest_path = release_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing Corpus Release manifest: {manifest_path}") from error
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if lock and manifest_digest != lock.manifest_sha256:
        raise RuntimeError("Corpus manifest digest mismatch.")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Invalid Corpus Release manifest.") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("Corpus Release manifest must be a JSON object.")
    summary = _validate_manifest(manifest, manifest_digest, release_root, lock)
    provenance = lock.provenance if lock else None
    if require_acquisition:
        acquisition_path = release_root / "acquisition.json"
        if acquisition_path.is_symlink():
            raise RuntimeError("Corpus acquisition record must not be a symbolic link.")
        acquisition = _read_json(acquisition_path, "acquisition record")
        expected = {
            "schemaVersion": "folklore-corpus-acquisition-v1",
            "release": provenance,
        }
        if acquisition.get("schemaVersion") != expected["schemaVersion"]:
            raise RuntimeError("Corpus acquisition record schema mismatch.")
        if acquisition.get("release") != provenance:
            raise RuntimeError("Corpus acquisition record does not match the lock.")
    return InstalledCorpus(release_root, summary, provenance)


def _download(lock: CorpusLock, destination: Path) -> str:
    request = urllib.request.Request(
        lock.url,
        headers={"User-Agent": "folklore-ml-lab-corpus-installer/1"},
    )
    digest = hashlib.sha256()
    byte_count = 0
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", 200)
        if status < 200 or status >= 300:
            raise RuntimeError(f"Corpus download failed with HTTP {status}.")
        expected_length = response.headers.get("Content-Length")
        with destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        if expected_length is not None and byte_count != int(expected_length):
            raise RuntimeError(
                f"Partial Corpus download: expected {expected_length} bytes, "
                f"received {byte_count}."
            )
        resolved_url = response.geturl()
    if digest.hexdigest() != lock.archive_sha256:
        raise RuntimeError("Corpus archive digest mismatch.")
    return resolved_url


def _public_url(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, host, parsed.path, "", "", ""))


def _archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        relative = _safe_relative_path(member.name, "archive entry")
        if not member.isreg():
            raise RuntimeError(f"Corpus archive entry is not a regular file: {relative}")
        if relative in members:
            raise RuntimeError(f"Duplicate Corpus archive entry: {relative}")
        members[relative] = member
    return members


def _extract_verified_archive(
    archive_path: Path,
    destination: Path,
    lock: CorpusLock,
) -> None:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _archive_members(archive)
            manifest_member = members.get("manifest.json")
            if manifest_member is None:
                raise RuntimeError("Corpus archive is missing manifest.json.")
            manifest_stream = archive.extractfile(manifest_member)
            if manifest_stream is None:
                raise RuntimeError("Corpus archive manifest is unreadable.")
            manifest_bytes = manifest_stream.read()
            if hashlib.sha256(manifest_bytes).hexdigest() != lock.manifest_sha256:
                raise RuntimeError("Corpus manifest digest mismatch.")
            try:
                manifest = json.loads(manifest_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("Invalid Corpus Release manifest.") from error
            if not isinstance(manifest, dict):
                raise RuntimeError("Corpus Release manifest must be a JSON object.")
            declared = {
                _safe_relative_path(artifact.get("path"), "artifact path")
                for artifact in manifest.get("artifacts", [])
                if isinstance(artifact, dict)
            }
            expected = declared | {"manifest.json"}
            actual = set(members)
            if actual != expected:
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                if missing:
                    raise RuntimeError(
                        f"Corpus archive is missing declared entry: {missing[0]}"
                    )
                raise RuntimeError(
                    f"Corpus archive contains undeclared entry: {unexpected[0]}"
                )
            destination.mkdir()
            for relative in sorted(expected):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(members[relative])
                if source is None:
                    raise RuntimeError(f"Corpus archive entry is unreadable: {relative}")
                with target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (tarfile.TarError, EOFError) as error:
        raise RuntimeError("Invalid or partial Corpus archive.") from error
    verify_release(destination, lock)


def install_release(lock: CorpusLock, *, offline: bool = False) -> InstalledCorpus:
    final = cache_path(lock)
    if final.exists():
        return verify_release(final, lock, require_acquisition=True)
    if offline or os.environ.get("FOLKLORE_OFFLINE") == "1":
        raise RuntimeError(
            "Corpus Release is not available in verified offline cache. "
            f"Expected {lock.release_id} ({lock.manifest_sha256}) at {final}. "
            "Run `npm run corpus:fetch` while online."
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{lock.manifest_sha256}.staging-",
            dir=final.parent,
        )
    )
    archive_path = staging / f"{lock.asset}.part"
    extracted = staging / "release"
    try:
        resolved_url = _download(lock, archive_path)
        _extract_verified_archive(archive_path, extracted, lock)
        acquisition = {
            "schemaVersion": "folklore-corpus-acquisition-v1",
            "release": lock.provenance,
            "fetchedAt": datetime.now(timezone.utc).isoformat(),
            "requestedUrl": _public_url(lock.url),
            "resolvedUrl": _public_url(resolved_url),
        }
        (extracted / "acquisition.json").write_text(
            json.dumps(acquisition, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        try:
            extracted.rename(final)
        except OSError:
            if not final.exists():
                raise
        return verify_release(final, lock, require_acquisition=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_cached_release(lock: CorpusLock) -> InstalledCorpus:
    final = cache_path(lock)
    if not final.exists():
        raise RuntimeError(
            f"Corpus Release cache is missing: {final}. "
            "Run `npm run corpus:fetch` while online."
        )
    return verify_release(final, lock, require_acquisition=True)


def result_json(installed: InstalledCorpus) -> dict:
    corpus = installed.provenance or installed.summary
    return {"path": str(installed.path), "corpus": corpus}
