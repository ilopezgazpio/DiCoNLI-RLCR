# Architecture

This repository retains one modular application, under `src/main/python/rlcr`.
The old benchmark layer has been removed. DiCo-NLI data integration is implemented;
the remaining steps are tracked in [short-term.md](short-term.md).

## Responsibilities

| Package | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | One CLI: train, evaluate (raw generation), infer, prepare-data |
| `arguments/` | One configuration dataclass per file |
| `configuration/` | Resolved run-settings snapshots |
| `data/validation.py` | Prepared prompt/label validation, no task transformations |
| `data/dico_nli/` | CSV records, input labels, reference joins, pair/split validation, sampling, audit/storage |
| `text/prediction.py` | Strict exact-label and scalar-confidence parsing |
| `models/` | Policy, reference, tokenizer, inference, quantization, device memory |
| `rewards/` | Exact accuracy, scalar Brier, format, registry |
| `training/` | Config parsing, split loading, lifecycle, logging, model cards |
| `training/grpo/` | Sampling, rollouts, advantages, loss, metrics |
| `inference/` | Prepared-prompt rendering, generation, optional token log-probabilities |
| `evaluation/` | Batch-generation configs, raw outputs, local persistence; generic calibration helpers |

Stateless operations are functions. Stateful components have separate files;
there is no service framework or mixin hierarchy.

## Model loading and the training loop

```text
CLI -> strict training configuration -> training/runner.py
    -> training/datasets.py: load and validate prepared inputs
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
There are no dataset-name switches, named task prompts, answer normalization,
symbolic verification, answer-repair generations, or external judge calls.

`evaluate` renders each prepared prompt, generates raw completions, and saves
columns plus counts. Generic Brier/ECE helpers are available but not an official
DiCo-NLI scorer. Input-label and source/pair validation are implemented in the
data adapter; output-label checks, submission export, and official scoring remain pending.

The data adapter saves canonical records without prompts. Its immutable record
keeps texts, labels, source IDs, and reference availability explicit. Future prompt
construction must whitelist model inputs, not stringify metadata. See [data.md](data.md)
for the module contract, sampling decisions, and audit results.

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

Data, recipes, and outputs stay separate; the data recipe is bundled, model recipes are not.
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
