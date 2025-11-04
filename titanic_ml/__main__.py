"""Command line interface for titanic_ml package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from titanic_ml import config
from titanic_ml.models.training import train_models


def _train_command(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the training command."""
    result = train_models(
        data_path=Path(args.data_path),
        use_mlflow=args.use_mlflow,
        fast_mode=args.fast_mode,
    )
    report = result.to_report()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Titanic ML command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train ensemble models on Titanic dataset")
    train_parser.add_argument("--data-path", default=str(config.DATA_PATH), help="Path to titanic.csv dataset")
    train_parser.add_argument("--use-mlflow", action="store_true", help="Log runs to MLflow if configured")
    train_parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Reduce hyperparameter search space for quicker iterations",
    )
    train_parser.add_argument(
        "--output",
        help="Optional path to write a JSON training report",
    )
    train_parser.set_defaults(func=_train_command)

    args = parser.parse_args()
    report = args.func(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
