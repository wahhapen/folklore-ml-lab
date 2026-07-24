import argparse
import json

from .lab import prepare, run_classifier, verify, verify_corpus_release


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m folklore_ml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("classifier")
    subparsers.add_parser("verify")
    subparsers.add_parser("verify-corpus")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "classifier":
        run_classifier()
    elif args.command == "verify-corpus":
        print(json.dumps(verify_corpus_release(), indent=2))
    else:
        verify()


if __name__ == "__main__":
    main()
