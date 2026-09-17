# Architecture

This repository retains one modular application, under `src/main/python/rlcr`.
The old benchmark layer has been removed. DiCo-NLI data, prompting, ordinary GRPO
task rewards, export, and scoring are implemented;
the remaining steps are tracked in [short-term.md](short-term.md).

## Responsibilities

| Package | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | One CLI: train, evaluate (raw generation), infer, prepare-data, prepare-prompts, fetch-scorer, export-submission, score |
| `arguments/` | One configuration dataclass per file |
| `configuration/` | Resolved run-settings snapshots |
| `data/validation.py` | Prepared prompt/label validation, no task transformations |
| `data/dico_nli/` | CSV records, labels, reference joins, pair/split validation, sampling, audit/storage, versioned prompt preparation |
| `text/prediction.py` | Strict exact-label and scalar-confidence parsing |
| `models/` | Policy, reference, tokenizer, inference, quantization, device memory |
| `rewards/` | Exact accuracy, scalar Brier, format, explicit DiCo vocabulary bindings, registry |
| `training/` | Config parsing, split/gold validation, prompt token-budget preflight, lifecycle, logging, model cards |
| `training/grpo/` | Sampling, rollouts, advantages, loss, metrics |
| `inference/` | Prepared-prompt rendering, generation, optional token log-probabilities |
| `evaluation/` | Batch-generation configs, raw outputs, local persistence; generic calibration helpers |
| `evaluation/dico_nli/` | Gold-free export, local dataset boundaries, external scorer pin/acquisition/execution, score artifacts |

Stateless operations are functions. Stateful components have separate files;
there is no service framework or mixin hierarchy.

## Model loading and the training loop

```text
CLI -> strict training configuration -> training/runner.py
    -> training/datasets.py: load and validate prepared inputs
    -> DiCo rewards: load tokenizer and check complete prompt token lengths
    -> training/grpo/trainer.py: initialize GRPOTrainer
        -> models/training_kwargs.py: resolve loading/quantization settings
        -> models/policy.py: base model, adapters, optional reference
    -> inherited Transformers Trainer.train()
        -> rollout -> reward gather -> group advantages
        -> policy forward -> GRPO loss -> backward -> optimizer update
    -> final model/adapter, tokenizer, state, model card
```

The runner resolves model-loading kwargs before dataset/model loading.
`models/tokenizer.py` handles padding and adapter/base tokenizer fallback;
`models/inference.py` loads full models or local PEFT adapters for generation.

Architecture JSON is stored with base-model weights, not in experiment YAML.
Full fine-tuning updates policy weights. LoRA/QLoRA freezes the base and trains
adapters. Generation uses the current local policy with gradients disabled;
there is no separate RL layer or rollout-model copy. Nonzero `beta` creates a
KL reference copy. Native SDPA avoids an external attention extension.

## GRPO components

- `dataloader.py`, `repeat_sampler.py`: expanded rollout batches and repeated inputs.
- `rollout.py`: chat formatting, local generation, completion masks, reward calls.
- `reward_evaluator.py`: callable/model rewards and gathering across workers.
- `advantages.py`: weighted rewards, group centering, optional normalization.
- `rollout_buffer.py`: shuffle/reuse across accumulation steps and iterations.
- `log_probs.py`, `loss.py`: policy forwards, clipped objective, optional KL.
- `training_metrics.py`: aggregation and completion logging.

Only the prompt is sent to TRL's chat formatter; labels and other metadata go
to reward callables. This avoids TRL treating `label` as its own preference-data
schema. No label is injected into the model prompt by the training workflow.

A group currently means multiple completions of one prompt. Retaining
`pair_id` metadata does **not** activate pair-aware training. Future paired
sampling and advantage construction must coordinate both directions before
generic shuffling/buffering. The standard trainer is still ordinary GRPO.

## Data and generation boundaries

Training uses prepared text/chat prompts and exact string labels.
Batch generation additionally requires stable unique identifiers.
There are no dataset-name switches, implicit prompt injection, answer normalization,
symbolic verification, answer-repair generations, or external judge calls.

`evaluate` renders each prepared prompt, generates raw completions, and saves
columns plus counts. Generic Brier/ECE helpers are available but not an official
DiCo-NLI scorer. Input-label and source/pair validation belong to the data adapter.
`export-submission` enforces output labels/confidence and exact ID coverage without
using gold. `score` explicitly combines a submission with a reference and invokes
the unchanged pinned upstream scorer. The source cache is external, not a second
application package or entry point. `dico_accuracy`, `dico_brier`, and `dico_format`
fix the training parser vocabulary without duplicating reward formulas.

Within `evaluation/dico_nli/`, `submission.py` owns pure validation/serialization;
`datasets.py` owns local Arrow boundaries; `export.py` owns export artifacts.
`scorer_pin.py` and `scorer_source.py` own version/setup verification;
`official_scorer.py` owns isolated execution; `scoring.py` owns explicit gold and
score provenance. `commands.py` only registers/dispatches shared CLI subcommands.
No custom implementation of the official metrics is added. See
[evaluation.md](evaluation.md) for the contracts and licensing boundary.

The data adapter saves canonical records without prompts. Its immutable record
keeps texts, labels, source IDs, and reference availability explicit.
`prompts.py` is the pure text/language-only builder; `prompt_preparation.py` validates
canonical inputs, adds prompts offline, and records hashes. Both training and
inference consume those same messages through the model's chat template.
See [data.md](data.md) and [training.md](training.md) for contracts and audit results.

## Launching and storage

`python -m rlcr` and the console alias `rlcr` invoke the same `cli.main`.
Accelerate starts copies of this application; each worker calls `train()`.
No Python application code launches another Accelerate process.

`configs/accelerate/zero2.yaml` retains ZeRO-2: policy weights are replicated,
optimizer state and gradients partitioned. The global rollout batch is the
local batch times worker count. The local batch is
`per_device_train_batch_size × steps_per_generation`, with
`steps_per_generation` defaulting to accumulation steps.
The global batch must be divisible by `num_generations`.

Data, recipes, and outputs stay separate; data and starter train/eval recipes are bundled.
Resolved snapshots describe application arguments, not launcher flags/hardware.
Input metadata is preserved, and batch reruns reject changed input rows.
Model-label caching is not content-addressed: change the output directory or
request fresh generation after changing the model or generation configuration.
Outputs are persisted after the entire batch-generation model list completes.

## Slurm and containers

The single-node template `scripts/slurm/train.sbatch` launches Accelerate from
one Slurm task. It defaults to the checkout's venv; set `RLCR_PYTHON` to override.
Adjust site settings and submit from the repository root. Multi-node rendezvous
needs separate setup and validation.

A container can install the package and use
`ENTRYPOINT ["python", "-m", "rlcr"]`, with explicit mounted input/cache/output
paths. There is no Dockerfile or validated multi-node/container deployment yet.

See [configuration.md](configuration.md) for current schemas and removed options.
