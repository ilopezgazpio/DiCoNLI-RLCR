# Application structure

All application Python code is an installable `rlcr` package under
`src/main/python`. There are no root-level Python launch scripts.

```text
src/main/python/rlcr/
├── __main__.py          # python -m rlcr -> the shared CLI
├── cli.py               # train / evaluate / infer / prepare-data dispatch
├── arguments/           # one configuration dataclass per file
├── configuration/       # shared resolved-configuration snapshots
├── data/                # conversation transforms, dataset IDs, creation recipes
├── text/                # system prompts and answer normalization
├── models/              # policy/reference/inference loading and quantization
├── rewards/             # format, accuracy, Brier, confidence, and registry
├── training/
│   ├── runner.py        # experiment orchestration and checkpoint lifecycle
│   ├── configuration.py # strict YAML keys and CLI overrides
│   ├── datasets.py      # training dataset splits and subsets
│   └── grpo/
│       ├── trainer.py          # Transformers integration, not a new train loop
│       ├── dataloader.py       # expanded generation batches
│       ├── repeat_sampler.py  # repeated questions for GRPO groups
│       ├── rollout_buffer.py  # reuse across accumulation steps/iterations
│       ├── rollout.py         # local policy generation and scoring
│       ├── reward_evaluator.py # callable/model rewards and reward gather
│       ├── advantages.py      # per-group reward baseline/normalization
│       ├── log_probs.py       # bounded-size policy forward passes
│       ├── loss.py            # clipped objective and optional KL
│       └── training_metrics.py # metric aggregation and completion logging
├── evaluation/
│   ├── runner.py        # stage orchestration
│   ├── configuration.py # named YAML sections, validation, and CLI overrides
│   ├── storage.py      # input/result datasets and incremental output columns
│   ├── generation.py   # inference plus postprocessing stages
│   ├── postprocessing/ # answers, confidence, and classifier scores
│   └── verifiers/      # symbolic checking, LLM judging, shared reports
└── inference/           # generation, selected-token scores, and response types
```

Use functions for stateless transformations and numerical operations. Stateful
objects (the trainer, reward evaluator, rollout buffer, and metrics collector)
have separate files. There is no mixin hierarchy or generic service framework.

## One entry point, different launchers

`rlcr ...` and `python -m rlcr ...` invoke the same `cli.main` function. The
console command is only a packaging alias, not a second implementation.

Training proceeds as follows:

```text
shell / container / Slurm
    -> optional Accelerate or torchrun launcher
        -> python -m rlcr train --config EXPERIMENT.yaml [overrides]
            -> training.runner.run_training
                -> GRPOTrainer.train() inherited from Transformers
                    -> buffered rollout -> reward gather -> advantages
                    -> GRPO loss -> distributed backward/optimizer update
```

Accelerate starts workers and supplies their distributed environment. It does
not replace the application's training workflow. Each worker executes the same
CLI and calls `trainer.train()`. The application never recursively launches
Accelerate, `torchrun`, or Slurm.

Evaluation uses `python -m rlcr evaluate --config EVALUATION.yaml`. Ad-hoc
generation uses `python -m rlcr infer --model MODEL --prompt QUESTION`.
These commands do not start training or initialize a distributed launcher.

Dataset creation uses `python -m rlcr prepare-data --recipe NAME --output DIRECTORY`.
The four former `data/creation_scripts` modules now live under `data/recipes`.
Imports do not download data. Only this explicit command loads the source
datasets, and it writes locally without publishing or overwriting existing
outputs. Preparation randomness is seeded; regenerated datasets are not
claimed to match the originally published random samples exactly.

## Model ownership and memory

`models/training_kwargs.py` maps model settings to Transformers loading kwargs,
including the quantization configuration and per-worker `LOCAL_RANK` device map.
`models/policy.py` loads the policy, prepares a quantized base for adapter
training, and creates a reference policy only when `beta` is nonzero.

Full fine-tuning updates policy weights. LoRA/QLoRA updates adapters while the
base is frozen. Generation uses the same local policy with gradients disabled;
there is no separate RL layer or dedicated inference-model copy.

The provided Accelerate configuration (`configs/accelerate/zero2.yaml`) uses
ZeRO-2: forward-pass weights remain replicated, while optimizer state and
gradients are partitioned. ZeRO-3/FSDP would require separate validation,
especially around generation and reference models.

Generation batches are larger than training microbatches:

```text
local generation batch = per_device_train_batch_size × steps_per_generation
global generation batch = local generation batch × number of workers
```

By default, `steps_per_generation` equals `gradient_accumulation_steps`.
Changing the launcher process count can therefore change the effective batch
size; distributed execution does not automatically reduce local generation
memory.

## Configuration and packaging

- `requirements.txt`: the single runtime/test dependency list.
- `pyproject.toml`: package/build/CLI metadata; reads dependencies from that list.
- `configs/train/`: training YAML recipes; unsupported/unknown options are rejected.
- `configs/accelerate/zero2.yaml`: process/distributed configuration.
- `configs/eval/`: evaluation YAML with named `dataset`, `models`, and `output_dir` sections.
- `data/`: input datasets only, never trained models or evaluation outputs.
- `outputs/train/<run>/`: standard Transformers/PEFT model directory and checkpoints.
- `outputs/eval/<run>/`: predictions and aggregate metrics from the same evaluation.

The launcher delegates gradient accumulation and clipping to the experiment's
training arguments. New runs persist resolved configuration snapshots, including
CLI overrides. Only the global main training process writes the snapshot.
Evaluation uses local output directories; it no longer attempts Hub dataset loads
when an output directory does not exist. Incremental runs retain previously saved
metrics when skipping a model and reject incompatible input row sets.

Training script arguments are application-owned rather than inheriting inactive
TRL script flags. The GRPO config still inherits Transformers training arguments,
and LoRA/reference-sync options are still consumed by TRL. Extra model loader kwargs
are merged with managed settings without allowing conflicting overrides. Shared
Hub-loading settings are forwarded to the training tokenizer as well.

See [configuration and migration details](configuration.md). All historical
experiment variants remain available; multi-dataset evaluation suites and recipe
deduplication are not part of this layout/schema change.

Install editable for development (`python -m pip install -e .`) or normally for
deployment (`python -m pip install .`), after the documented PyTorch bootstrap.
Imports work outside the checkout's current working directory after installation.
Config, dataset, and output paths are interpreted relative to the process's
working directory; use absolute paths or set the working directory explicitly.
Configs and datasets are external inputs, not embedded inside the package wheel.

## Slurm and containers

`scripts/slurm/train.sbatch` is a single-node, four-GPU example. It runs one
Slurm task that starts the Accelerate workers. Adjust site-specific resource,
partition, account, and time settings before submitting. It uses the repository
venv by default; `RLCR_PYTHON` can select another installed environment.

For multiple nodes, use one launcher per node and supply the correct Accelerate
machine count/rank and rendezvous address/port. The included Slurm template is
not a validated multi-node deployment.

A future container can install the same package and use
`ENTRYPOINT ["python", "-m", "rlcr"]`, with commands such as
`train --config /configs/experiment.yaml`. Mount model/data caches and outputs
explicitly. The Python entry point has no Slurm- or container-specific logic.

## Migration from the flat layout

| Previous location | New location |
|---|---|
| `rl_runner.py` | `cli.py` + `training/runner.py` + `models/training_kwargs.py` |
| `GRPO_Trainer.py` | `training/grpo/` |
| `arguments.py`, `eval/eval_args.py` | `arguments/` |
| `dataset_processing.py` | `data/` |
| `data/creation_scripts/*.py` | `data/recipes/` + the `prepare-data` command |
| `reward_fns.py` | `rewards/` |
| `trainer_utils.py` | `training/grpo/` + `training/profiling.py` |
| `evaluation.py`, `eval/` helpers | `evaluation/` |
| `inference_utils.py`, `inference_example.py` | `models/` + `inference/` + the `infer` command |
| `system_prompts.py` | `text/prompts.py` |

Paths in the right column are relative to `src/main/python/rlcr`.
The former `CustomTrainer` class is now `GRPOTrainer`.
