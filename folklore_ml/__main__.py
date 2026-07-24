import argparse

from .lab import prepare, run_classifier, verify


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m folklore_ml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("classifier")
    subparsers.add_parser("verify")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "classifier":
        run_classifier()
    else:
        verify()


if __name__ == "__main__":
    main()
