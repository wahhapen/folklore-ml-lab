import argparse
import json
from pathlib import Path

from .corpus import (
    default_lock_path,
    install_release,
    load_lock,
    result_json,
    verify_cached_release,
)
from .lab import (
    prepare,
    resolve_corpus_release,
    run_classifier,
    verify,
    verify_corpus_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m folklore_ml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("classifier")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--legacy-only", action="store_true")
    subparsers.add_parser("verify-corpus")
    corpus_parser = subparsers.add_parser("corpus")
    corpus_commands = corpus_parser.add_subparsers(
        dest="corpus_command",
        required=True,
    )
    for command in ("install", "verify", "status"):
        command_parser = corpus_commands.add_parser(command)
        command_parser.add_argument("--lock")
        if command == "install":
            command_parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "classifier":
        run_classifier()
    elif args.command == "verify-corpus":
        print(json.dumps(verify_corpus_release(), indent=2))
    elif args.command == "corpus":
        lock_path = (
            Path(args.lock).resolve()
            if args.lock
            else default_lock_path(Path(__file__).resolve().parents[1])
        )
        if lock_path is None:
            if args.corpus_command == "install":
                raise RuntimeError(
                    "Corpus v0.2 is not published or pinned yet. "
                    "Commit corpus-release.lock.json after Corpus issue #9."
                )
            installed = resolve_corpus_release()
        else:
            lock = load_lock(lock_path)
            if args.corpus_command == "install":
                installed = install_release(lock, offline=args.offline)
            else:
                installed = verify_cached_release(lock)
        output = result_json(installed)
        if args.corpus_command == "status":
            output["status"] = "verified"
        print(json.dumps(output, indent=2))
    else:
        verify(legacy_only=args.legacy_only)


if __name__ == "__main__":
    main()
