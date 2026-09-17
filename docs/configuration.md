# Configuration and run storage

There are three distinct responsibilities:

- `configs/`: human-authored YAML recipes.
- `data/`: input datasets saved with Hugging Face Datasets.
- `outputs/`: generated models, checkpoints, predictions, metrics, and run settings.

## Training recipes

`configs/train/` contains complete experiments, not the model's architecture JSON.
Filenames describe the dataset, model, objective, and training method. For example:

```bash
python -m rlcr train \
  --config configs/train/hotpot-qwen3b-rlcr-qlora.yaml \
  --dataset_name /datasets/hotpot \
  --output_dir /scratch/my-training-run \
  --max_steps 10
```

Training keeps its flat YAML format and Transformers-style underscore CLI overrides.
Unknown YAML keys and unsupported options now raise an error instead of being
silently ignored. Full fine-tuning, LoRA, and QLoRA still use the same training workflow.
All 11 historical recipes are retained; `1000steps`/`2000steps` in new filenames
describe optimizer steps, not dataset size. The `sft-rlcr` recipe starts GRPO from
an SFT checkpoint; it does not run a separate SFT stage.

The final model or adapter is saved directly in `output_dir`, alongside its
tokenizer, `config.json`/`adapter_config.json`, and trainer state. Intermediate
`checkpoint-*` directories are created when checkpointing is enabled. Do not
move these metadata files away from their weights or merge them with recipes.

Existing run names are preserved under `outputs/train/`. To avoid replacing a
previous run's final weights, supply a new `--output_dir`. The existing training
behavior of automatically resuming the last checkpoint, when present, is retained.

`configs/accelerate/zero2.yaml` describes distributed execution. Its gradient
accumulation and clipping values are `auto`: the experiment owns
`gradient_accumulation_steps` and `max_grad_norm`. The launcher still controls
process counts and the ZeRO strategy.

### Supported options and removed leftovers

The training parser derives accepted keys from the application dataclasses. It
preserves CLI precedence and TRL's optional `env` mapping, but does not load implicit
`*.args` files. The inert `num_processes` key was removed from all training recipes;
set worker counts in Accelerate instead.

These previously accepted but inactive options have been removed:

| Removed option | Supported behavior |
|---|---|
| Training `callbacks` | Python callers can still pass callback objects directly to `GRPOTrainer(callbacks=...)`. |
| Training `system_prompt` | Select a named prompt with `sys_prompt_name`. Inference's `--system-prompt` remains supported. |
| `completion_logging_steps` | Completion tables use the regular Trainer logging cadence (`logging_steps`/`logging_strategy`). |
| `eval_log_keys` | The current evaluation logger has no configurable column selection. |
| `set_pad_token` | Preserve the tokenizer's pad token, or fall back to its EOS token when absent. |
| `gradient_checkpointing_use_reentrant` | Use `gradient_checkpointing_kwargs: {use_reentrant: false}`. |
| `ignore_bias_buffers` | No application implementation; the inherited no-op setting was removed. |
| `orm_key` | No ORM training workflow is exposed. |
| Evaluation `correctness_fn` | Select the evaluator with `check_fn`. |

Both training and evaluation support only `task_spec: gen`. Unsupported SFT/ORM
task values now fail explicitly. Their disconnected preprocessing helpers were
removed; using an existing SFT checkpoint as the initial GRPO policy still works.

The default `sys_prompt_name` is now the valid `gen` prompt. All supplied recipes
retain their explicit prompt choices. Supported rewards are `accuracy`, `format`,
`brier`, `mean_confidence`, and `confidence_one_or_zero`; unknown names fail before
dataset/model loading. LoRA settings and reference synchronization options remain
supported, including those consumed inside TRL rather than the local code.

### Extra model-loading settings

Additional Transformers loader kwargs are preserved instead of overwritten:

```yaml
model_init_kwargs:
  local_files_only: true
  cache_dir: /scratch/huggingface
```

Dedicated model settings have a single owner. Do not duplicate `revision`,
`torch_dtype`, `trust_remote_code`, `attn_implementation`, `load_in_4bit`, or
`load_in_8bit` inside `model_init_kwargs`; use their top-level fields instead
(`model_revision` for the revision). `use_cache`, `quantization_config`, and
`device_map` are managed by the training workflow and are also rejected as extras.

The training tokenizer receives the same revision and shared Hub-loading options
as the model, including cache/offline settings. Model-only arguments are not sent
to the tokenizer. Credentials (`token`/`use_auth_token`) are rejected in these
serialized extras; use `HF_TOKEN` or Hub login instead. `local_files_only` applies
to model/tokenizer loading, not dataset downloads.

## Evaluation recipes

Evaluation now uses a YAML mapping rather than a positional JSON list:

```yaml
dataset:
  name: mehuldamani/hotpot_qa  # Or a local dataset directory
  split: test
  hash_key: problem
  sample_size: 32

models:
  - name: my-adapter          # Unique label used in result columns and metrics
    model: outputs/train/RLCR-hotpot-qwen-3b-peft
    sys_prompt_name: tabc_long
    check_fn: confidence_verifier
    tasks: [confidence_at_end, ans_at_end]
    load_in_4bit: true
    max_tokens: 512

output_dir: outputs/eval/my-adapter-hotpot
fresh: false
```

Dataset options are `name`, optional Hub subset `config`, `split`, `hash_key`,
and `sample_size`. Model entries retain the previous generation, postprocessing,
and scoring options. Multiple models can still be evaluated against the same
dataset. Names must be unique. Unknown configuration keys are rejected.

```bash
python -m rlcr evaluate \
  --config configs/eval/hotpot-peft-3b-smoke.yaml \
  --dataset /datasets/hotpot \
  --model /scratch/my-training-run \
  --output-dir /scratch/my-evaluation-run \
  --sample-size 16
```

The optional `--model` override only applies to a single-model recipe. Other
overrides are `--split`, `--fresh`, and `--no-fresh`. The last one disables the
global fresh flag; a model's own `fresh: true` still forces that model to regenerate.

Each output directory contains:

```text
my-evaluation-run/
├── predictions/          # Saved Arrow dataset: answers and per-example scores
├── metrics.json          # Aggregate metrics, keyed by model label
└── resolved-config.yaml  # Defaults and CLI overrides included
```

`store_name` and `log_path` no longer exist. Evaluation outputs are always local
paths, never Hub dataset identifiers. The predictions keep their existing saved
dataset format and may contain columns for several models.

Existing predictions are reused by model label unless fresh generation is
requested. Use a **new output directory** when changing the dataset, split,
sample count, model, or evaluation protocol. Alternatively, `--fresh` regenerates
selected models against the same input rows. A different input row set/order is
rejected to prevent combining unrelated predictions. Skipped models retain their
previous metrics. Cached model labels are not a content-addressed model cache:
changing a model under the same label requires a new directory or `--fresh`.

## Paths, snapshots, and deployment

Relative paths retain their previous meaning: relative to the process working
directory, **not** the recipe's directory. Run supplied examples from the checkout
root. For Slurm or containers, use explicit absolute paths for mounted datasets,
models, recipes, and writable output directories. No repository-specific storage
root or additional environment configuration file is required.

Training uses `--output_dir`/`--dataset_name`; evaluation uses
`--output-dir`/`--dataset`. This preserves the training parser's existing interface.
The Slurm script forwards training overrides unchanged.

New invocations save resolved settings in `resolved-config.yaml`. When settings
change within the same output directory, the previous snapshot is preserved under
`config-history/`; an identical snapshot is not duplicated. Training credentials
are excluded. Training snapshots describe application settings, not a complete
record of launcher flags or hardware; retain the job submission script as well.
Historical runs moved during cleanup have no reconstructed configuration snapshot.

`outputs/` is ignored by Git. Back up valuable runs independently; configuration
files in `configs/` remain version-control inputs.

## Migration

| Previous location | Current location |
|---|---|
| `configs/Qwen-*/<dataset>/*.yaml` | `configs/train/<descriptive-experiment>.yaml` |
| `eval_configs/Hotpot-models/<dataset>.json` | `configs/eval/hotpot-models-<dataset>.yaml` |
| `eval_configs/Math-models/<dataset>.json` | `configs/eval/math-models-<dataset>.yaml` |
| `eval_configs/PEFT/<name>.json` | `configs/eval/<name>.yaml` |
| `data/RLCR-*` | `outputs/train/RLCR-*` |
| `eval_outputs/Hotpot-models/<dataset>/` | `outputs/eval/hotpot-models-<dataset>/predictions/` |
| `results/Hotpot-models/<dataset>/metrics.json` | `outputs/eval/hotpot-models-<dataset>/metrics.json` |
| `eval_outputs/Math-models/<dataset>/` | `outputs/eval/math-models-<dataset>/predictions/` |
| `results/Math-models/<dataset>/metrics.json` | `outputs/eval/math-models-<dataset>/metrics.json` |
| `eval_outputs/PEFT/<name>/` | `outputs/eval/<name>/predictions/` |
| `results/PEFT/<name>/metrics.json` | `outputs/eval/<name>/metrics.json` |

The paper-model recipes already targeted `Hotpot-models-fresh` and
`Math-models-fresh`, while bundled historical artifacts lived under names without
`-fresh`. This distinction is preserved: current recipes target
`outputs/eval/hotpot-models-fresh-<dataset>` and
`outputs/eval/math-models-fresh-<dataset>`. No historical run is implicitly relabeled
as a new evaluation.

The `RLCR-hotpot-qwen-7b-peft-g8-1k` training directory was empty before migration
and remains empty; its evaluation recipe needs a completed training run first.
All existing weights, dataset files, saved predictions, and metrics were moved
without modifying their contents. Empty legacy configuration/output folders were
removed after the move. No model or dataset was deleted.
