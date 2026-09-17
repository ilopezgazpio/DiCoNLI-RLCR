"""Register task-scoring commands without importing model or dataset libraries."""
from .scorer_pin import DEFAULT_SCORER_DIR


def add_scoring_commands(commands):
    fetch = commands.add_parser(
        "fetch-scorer", help="Explicitly download/verify the pinned official scorer"
    )
    fetch.add_argument("--scorer-dir", default=DEFAULT_SCORER_DIR)
    export = commands.add_parser(
        "export-submission", help="Export complete valid predictions without gold"
    )
    export.add_argument("--predictions", required=True, help="Local saved prediction Dataset/Dict")
    export.add_argument(
        "--instances", required=True, help="Independent local dataset of expected IDs"
    )
    export.add_argument("--split", default="dev")
    export.add_argument(
        "--prediction-column", required=True, help="One generated completion column"
    )
    export.add_argument(
        "--output-dir", required=True, help="New directory for submission and diagnostics"
    )
    score = commands.add_parser("score", help="Run the pinned official DiCo-NLI scorer")
    reference = score.add_mutually_exclusive_group(required=True)
    reference.add_argument("--gold", help="Official reference CSV/TSV")
    reference.add_argument(
        "--reference-dataset", help="Canonical prepared dataset with known gold pairing"
    )
    score.add_argument("--split", default="dev", help="Split for --reference-dataset")
    score.add_argument("--predictions", required=True, help="Two-column submission CSV/TSV")
    score.add_argument("--scorer-dir", default=DEFAULT_SCORER_DIR)
    score.add_argument(
        "--output-dir", required=True, help="New directory for official reports and provenance"
    )


def run_scoring_command(args):
    if args.command == "fetch-scorer":
        from .scorer_source import fetch_scorer

        return fetch_scorer(args.scorer_dir)
    if args.command == "export-submission":
        from .export import export_submission

        return export_submission(
            args.predictions,
            args.instances,
            split=args.split,
            prediction_column=args.prediction_column,
            output_dir=args.output_dir,
        )
    from .scoring import score_submission

    return score_submission(
        args.predictions,
        gold=args.gold,
        reference_dataset=args.reference_dataset,
        split=args.split,
        scorer_dir=args.scorer_dir,
        output_dir=args.output_dir,
    )
