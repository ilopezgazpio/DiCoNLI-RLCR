# Configuration and run storage

`configs/` holds human-authored configuration, `data/` input datasets, and
`outputs/` generated model/prediction artifacts. `configs/accelerate/zero2.yaml`
and `configs/data/dico-nli-en.yaml` are bundled; DiCo-NLI training/evaluation recipes
are pending. Do not use old benchmark recipes against the cleaned workflow.

## Data preparation

`python -m rlcr prepare-data --config configs/data/dico-nli-en.yaml` imports local
official CSVs into canonical task records, with strict validation, source-group
sampling, hashes, and audit reports. Its data-only YAML is independent of model
settings. See [data.md](data.md) for all fields, acquisition steps, and limitations.
The output has no prompt column yet; it is not directly training-ready.

## Training

Training uses flat YAML with Transformers-style underscore CLI overrides:

```bash
python -m rlcr train --config /path/to/training.yaml \
  --dataset_name /datasets/prepared --output_dir /scratch/new-run --max_steps 10
```

A minimal **illustrative** single-GPU QLoRA configuration follows. Paths are
placeholders; it is not a validated DiCo-NLI preset:

```yaml
model_name_or_path: /models/base-instruct
dataset_name: /datasets/prepared
dataset_train_split: train
output_dir: outputs/train/new-run
use_peft: true
load_in_4bit: true
load_in_8bit: false
lora_target_modules: [q_proj, v_proj]
torch_dtype: bfloat16
bf16: true
attn_implementation: sdpa
gradient_checkpointing: true
gradient_checkpointing_kwargs: {use_reentrant: false}
reward_funcs: [accuracy, brier]
reward_weights: [1.0, 0.5]
beta: 0.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
num_generations: 2
max_prompt_length: 512
max_completion_length: 64
max_steps: 10
eval_strategy: "no"
save_strategy: "no"
report_to: []
push_to_hub: false
```

Use compatible LoRA target modules for your selected architecture.
Full-weight training sets both `use_peft` and quantized loading to false;
unquantized LoRA keeps `use_peft: true` but disables quantized loading.

Prepared training data needs a `prompt` column (text or chat messages), plus
`label` (nonempty string) when using accuracy/Brier. Labels remain reward
metadata; the caller supplies the complete task instructions in the prompt.
Optional split names default to `train`/`test`; evaluation splits are required
only when the training evaluation strategy is enabled.
Subset sizes select the first rows; no task-aware pair sampling exists yet.

Supported rewards are `accuracy`, `format`, and `brier`.
All parse `<answer>LABEL</answer><confidence>NUMBER</confidence>`, with finite
confidence in [0, 1]. Invalid structure gets -1 from each reward.
Accuracy gives 1/0 for exact label correctness; Brier gives
`1 - (confidence - correctness)^2`; format gives 1 for valid structure.
No official label vocabulary is enforced yet. Default selected rewards are
accuracy and Brier; absent explicit weights, the trainer averages them.
The roadmap's experiment weights therefore require explicit `reward_weights`.
This is a foundation, not the planned knowledge/pair-aware reward.

Unknown YAML keys and unused CLI overrides fail. The parser derives keys from
application dataclasses, preserves override precedence and an optional `env`
mapping, and does not load implicit `*.args` files.
Process counts belong to Accelerate, not the training YAML.

### Model loading

Additional loader kwargs are preserved:

```yaml
model_init_kwargs:
  local_files_only: true
  cache_dir: /scratch/huggingface
```

Dedicated settings have one owner. Do not duplicate revision, dtype, remote-code
trust, attention, or quantization inside these extras; use top-level fields
(`model_revision` for revision). `use_cache`, `device_map`, and
`quantization_config` are managed internally and rejected as extras.
Shared Hub-loading options are forwarded to the tokenizer. Credentials are
rejected in serialized kwargs: use `HF_TOKEN` or Hub login.
`local_files_only` here governs model/tokenizer loading, not datasets.

## Batch generation

`evaluate` temporarily means raw generation, not official task scoring:

```yaml
dataset:
  name: /datasets/prepared
  split: test
  id_column: instance_id
  sample_size: 32
models:
  - name: candidate
    model: /models/base-or-local-adapter
    tokenize_key: prompt
    load_in_4bit: true
    torch_dtype: bfloat16
    n: 1
    temperature: 0
    max_tokens: 64
    hf_batch_size: 1
    seed: 42
output_dir: outputs/eval/new-run
fresh: false
```

Input can be a local saved Dataset/DatasetDict or Hub dataset. It must already
have prepared prompts and unique nonempty string identifiers. Duplicate IDs are
rejected, even outside a requested subset. Labels are not required for generation.

Dataset keys: `name`, optional `config`, `split`, `id_column`, `sample_size`.
Model keys: `name`, `model`, `tokenize_key`, `n`, `temperature`,
`max_tokens`, `seed`, `fresh`, `load_in_4bit`, `torch_dtype`, `hf_batch_size`.
Model names must be unique. Defaults include n=1, temperature=0, max_tokens=4096,
batch size=1. Keep a short explicit limit for small-card experiments.

CLI overrides use hyphens: `--dataset`, `--split`, `--sample-size`,
`--output-dir`, `--model` (single-model config only), `--fresh`/`--no-fresh`.
Disabling global fresh does not override a model's own `fresh: true`.

Outputs:

```text
predictions/          # Saved Arrow dataset, original columns + NAME-output_N
metrics.json          # Per-model examples/completions counts, NOT task scores
resolved-config.yaml  # Defaults and overrides resolved
```

Raw malformed completions remain raw; no repair generation or confidence
replacement occurs. Generic Brier/ECE helpers are not called as official scoring.
The task adapter/scorer will be a separate, explicit addition.

## Removed options

These are intentional breaking changes, not silently ignored settings:

| Removed | Current boundary |
|---|---|
| Training/evaluation `sys_prompt_name`, `task_spec` | Supply prepared prompts; no implicit task preprocessing |
| Training `format_pattern` | One strict exact-label/confidence response contract |
| `mean_confidence`, `confidence_one_or_zero` rewards | No direct incentive to report high/extreme confidence |
| Evaluation `check_fn`, `check_fn_args`, `correctness_fn`, `pass_k_vals` | Legacy answer scoring removed; official task scoring pending |
| Evaluation `tasks`, `class_model`, `split_at_confidence` | Raw generation only |
| Dataset `hash_key` | Preserve unique `id_column` identifiers |
| `prepare-data --recipe ...` | Use the DiCo-NLI data-only `prepare-data --config ...` command |
| Training `callbacks`, `system_prompt` | Python callback objects still work; system messages belong in prepared prompts |
| `completion_logging_steps`, `eval_log_keys` | Standard logging cadence, no configurable completion columns |
| `set_pad_token` | Tokenizer pad token, falling back to EOS |
| `gradient_checkpointing_use_reentrant` | `gradient_checkpointing_kwargs: {use_reentrant: false}` |
| `ignore_bias_buffers`, `orm_key` | No implementation exposed |
| Training `num_processes` | Launcher setting |

Ad-hoc inference's `--system-prompt` remains available, now as **literal text**.
There is no SFT/ORM training command. Starting GRPO from a previously trained
checkpoint remains possible.

## Storage and deployment cautions

Relative paths resolve against the current working directory, not the YAML file.
For Slurm/containers use explicit mounted paths or set the working directory.
Configs/data are not embedded in the package wheel.

Training saves final weights/adapters, tokenizer, state, and model card directly
in `output_dir`; periodic `checkpoint-*` directories need checkpointing enabled.
Existing checkpoints auto-resume; without one, reusing a directory can replace
final weights. Prefer new run directories.

Batch generation reuses output columns by model name. Changing the model/settings
under the same name needs `--fresh` or a new directory. Changed input data/order
is rejected. Counts for skipped models are preserved. Results are saved only
after all models finish.

Resolved settings are snapshotted; changed snapshots go into `config-history/`.
They do not record the launcher/hardware automatically. Keep job logs and back up
outputs separately. Previous benchmark artifacts were archived outside this
checkout; their historical settings were not reconstructed.
