from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--omit")
    parser.add_argument("--extra-path")
    parser.add_argument(
        "--extra-type",
        choices=("regular", "symlink", "hardlink"),
        default="regular",
    )
    parser.add_argument("--duplicate", action="store_true")
    parser.add_argument("--corrupt")
    parser.add_argument("--release-id")
    parser.add_argument("--producer-repository")
    parser.add_argument("--drop-manifest-schema", action="store_true")
    parser.add_argument("--invalid-manifest-field", action="store_true")
    parser.add_argument("--invalid-manifest-schema", action="store_true")
    parser.add_argument(
        "--unsupported-manifest-schema-dialect",
        action="store_true",
    )
    parser.add_argument("--drop-first-document-passages", action="store_true")
    args = parser.parse_args()

    manifest_path = args.release_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.release_id:
        manifest["releaseId"] = args.release_id
    if args.producer_repository:
        manifest["producer"]["repository"] = args.producer_repository
    if args.invalid_manifest_field:
        manifest["unexpectedField"] = True
    artifact_contents = {
        artifact["path"]: (args.release_root / artifact["path"]).read_bytes()
        for artifact in manifest["artifacts"]
    }
    if args.drop_manifest_schema:
        manifest["artifacts"] = [
            artifact
            for artifact in manifest["artifacts"]
            if artifact["path"] != "manifest.schema.json"
        ]
    if args.invalid_manifest_schema or args.unsupported_manifest_schema_dialect:
        schema = json.loads(artifact_contents["manifest.schema.json"])
        if args.invalid_manifest_schema:
            schema["type"] = 7
        if args.unsupported_manifest_schema_dialect:
            schema["$schema"] = "https://example.invalid/unsupported-schema"
        contents = (
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        artifact_contents["manifest.schema.json"] = contents
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "manifest.schema.json":
                artifact["byteLength"] = len(contents)
                artifact["sha256"] = hashlib.sha256(contents).hexdigest()
    if args.drop_first_document_passages:
        passage_rows = [
            json.loads(line)
            for line in artifact_contents["passages.jsonl"].splitlines()
            if line
        ]
        document_id = passage_rows[0]["documentId"]
        artifact_contents["passages.jsonl"] = b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for row in passage_rows
            if row["documentId"] != document_id
        )
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "passages.jsonl":
                contents = artifact_contents["passages.jsonl"]
                artifact["byteLength"] = len(contents)
                artifact["sha256"] = hashlib.sha256(contents).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        files = [("manifest.json", manifest_bytes)]
        files.extend(
            (
                artifact["path"],
                artifact_contents[artifact["path"]]
                + (b"changed" if artifact["path"] == args.corrupt else b""),
            )
            for artifact in manifest["artifacts"]
            if artifact["path"] != args.omit
        )
        for relative, contents in files:
            info = tarfile.TarInfo(relative)
            info.size = len(contents)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(contents))
        if args.extra_path:
            info = tarfile.TarInfo(args.extra_path)
            info.mode = 0o644
            info.mtime = 0
            if args.extra_type == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "manifest.json"
                archive.addfile(info)
            elif args.extra_type == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = "manifest.json"
                archive.addfile(info)
            else:
                info.size = len(b"escape")
                archive.addfile(info, io.BytesIO(b"escape"))
        if args.duplicate:
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest_bytes))

    result = {
        "archiveSha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "releaseId": manifest["releaseId"],
        "version": manifest["version"],
        "manifestSchemaVersion": manifest["schemaVersion"],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
