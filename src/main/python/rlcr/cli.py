"""One application CLI; distributed process launching remains external."""
import argparse
import json

from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rlcr", description="Train and evaluate uncertainty-aware language models."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train",
        help="Run full-model, LoRA, or QLoRA GRPO training",
        description="Run a training YAML. Extra arguments override its Transformers/GRPO/model settings.",
        epilog="Example: rlcr train --config /path/to/experiment.yaml --max_steps 10",
    )
    train.add_argument("--config", required=True, help="Training YAML path")
    evaluate = commands.add_parser(
        "evaluate", help="Generate batch predictions (task scoring is not implemented yet)"
    )
    evaluate.add_argument("--config", required=True, help="Evaluation YAML path")
    evaluate.add_argument("--output-dir", help="Override the local evaluation run directory")
    evaluate.add_argument("--dataset", help="Override the Hub dataset ID or local directory")
    evaluate.add_argument("--split", help="Override the dataset split")
    evaluate.add_argument("--sample-size", type=int, help="Override the number of evaluated rows")
    evaluate.add_argument("--model", help="Override the model in a single-model recipe")
    evaluate.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Regenerate selected models, or --no-fresh to reuse existing predictions",
    )
    infer = commands.add_parser("infer", help="Generate completions for one or more prompts")
    infer.add_argument("--model", required=True, help="Model ID or local model/adapter directory")
    infer.add_argument(
        "--prompt", action="append", required=True, help="Input text; repeat for multiple inputs"
    )
    infer.add_argument("--system-prompt", help="Optional literal system message, not a preset name")
    infer.add_argument(
        "--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16"
    )
    infer.add_argument("--load-in-4bit", action="store_true")
    infer.add_argument("--n", type=int, default=1)
    infer.add_argument("--temperature", type=float, default=0)
    infer.add_argument("--max-tokens", type=int, default=4096)
    infer.add_argument("--hf-batch-size", type=int, default=1)
    infer.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    parser = build_parser()
    args, overrides = parser.parse_known_args(argv)
    if overrides and args.command != "train":
        parser.error("unrecognized arguments: " + " ".join(overrides))
    # Lazy imports keep --help usable without initializing ML libraries/devices.
    if args.command == "train":
        from .training.configuration import load_training_config
        from .training.runner import run_training

        try:
            parsed = load_training_config(args.config, overrides)
        except ValueError as error:
            parser.error(str(error))
        run_training(*parsed)
    elif args.command == "evaluate":
        from .evaluation.configuration import load_evaluation_config
        from .evaluation.runner import run_evaluation

        try:
            parsed = load_evaluation_config(
                args.config,
                output_dir=args.output_dir,
                dataset_name=args.dataset,
                split=args.split,
                sample_size=args.sample_size,
                model=args.model,
                fresh=args.fresh,
            )
        except ValueError as error:
            parser.error(str(error))
        run_evaluation(*parsed)
    else:
        from .inference.runner import run_inference

        outputs = run_inference(args, args.prompt, args.system_prompt)
        for prompt, result in zip(args.prompt, outputs):
            print(
                json.dumps(
                    {"prompt": prompt, "completions": [item.text for item in result.outputs]}
                )
            )
    return 0
