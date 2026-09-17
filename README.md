# DiCoNLI-RLCR

An adaptation of **Beyond Binary Rewards: Training LMs to Reason About Their
Uncertainty** for SemEval 2027 Task 2: DiCo-NLI.

The reusable GRPO/full-model/LoRA/QLoRA infrastructure is retained. Unrelated
benchmark datasets, recipes, prompts, answer verifiers, judge/classifier workflows,
and example suites have been removed. Previous local data and model/output
artifacts were moved to a recovery directory outside the checkout.

**Current status:** this is a cleaned foundation, not a complete DiCo-NLI system.
Prepared-prompt training, raw batch generation, and ad-hoc inference work.
DiCo-NLI CSV ingestion, label-set validation, official scoring/submission export,
WordNet evidence, and pair-aware GRPO are next; none is implemented yet.
See [short term](docs/short-term.md), [medium term](docs/medium-term.md), and
[long term](docs/long-term.md) for the research plan and results log.

## Installation

Use Python 3.10 with venv support, Git, and Linux x86_64. GPU training requires a
CUDA 12.4-compatible NVIDIA driver; BF16 settings require compatible hardware.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip check
```

`requirements.txt` is the single dependency list; `pyproject.toml` reads it.
Install PyTorch before DeepSpeed. DeepSpeed's build/runtime extensions may need
a compatible CUDA toolkit (`nvcc`, preferably 12.4), `CUDA_HOME`, and a C++
compiler. The PyTorch wheel supplies runtime libraries, not a driver or compiler
toolkit. The packaging pins accommodate the existing DeepSpeed build.
Use `python -m pip install .` for a non-editable deployment.

Full fine-tuning and QLoRA use this same environment. No Conda, vLLM, flash-attn,
or xformers is required. Attention defaults to native PyTorch SDPA through
Transformers. Accelerate orchestrates training; it is not an attention backend.

Public assets need network access or a populated cache. Private/gated models
may need approval and `HF_TOKEN`. Keep credentials out of YAML. Model loaders
allow remote repository code by default: use trusted sources. Set
`report_to: []` and `push_to_hub: false` for local-only experiments; WandB
reporting otherwise needs authentication.

## Entry point and layout

```bash
python -m rlcr --help
python -m rlcr train --help
python -m rlcr evaluate --help
python -m rlcr infer --help
```

The installed `rlcr` command aliases the same CLI. Accelerate/Slurm launches this
application; each training worker calls `GRPOTrainer.train()`.
The old dataset-specific `prepare-data` command is removed.

```text
src/main/python/rlcr/  # Modular application and reusable GRPO infrastructure
configs/accelerate/   # Retained distributed launcher configuration
data/                 # Future prepared DiCo-NLI datasets
outputs/train/        # Runtime model/adapter checkpoints
outputs/eval/         # Runtime raw predictions and generation counts
docs/                 # Architecture, configuration, and research roadmap
scripts/slurm/        # Single-node launcher template
tests/                # Offline synthetic CPU regression fixtures
```

There are no bundled training/evaluation recipes or prepared task datasets yet.
Future task recipes can live in `configs/train/` and `configs/eval/`.
Paths in YAML are relative to the process working directory, not the YAML file.
Data and configs are external inputs, not packaged resources.

Model architecture `config.json` comes from the selected Hub model or local model
directory. It is distinct from an experiment YAML. Adapter directories contain
`adapter_config.json` and still need their referenced base weights.

See [architecture](docs/architecture.md) and [configuration](docs/configuration.md).

## Prepared-data contract

Training expects a saved Hugging Face `DatasetDict` (or compatible Hub dataset)
with named splits, a `prompt` column containing text or chat-message lists, and
a string `label` column for accuracy/Brier rewards. Metadata columns are preserved.
No task-specific system message or input-column conversion is injected.

Responses use one strict format:

```text
<answer>LABEL</answer><confidence>0.75</confidence>
```

The label is matched exactly, including case, after trimming tag-boundary
whitespace. Confidence must be finite and in [0, 1]; it estimates correctness of
the emitted label. Invalid structure receives a negative reward, not a repair
generation. Only `accuracy`, `format`, and `brier` rewards remain. Brier reward
is `1 - (confidence - correctness)^2`; accuracy and Brier are the defaults.
This generic parser does not yet enforce the official DiCo-NLI label vocabulary.

Batch generation also requires unique, nonempty string `instance_id` values
(or a configured `id_column`). Original identifiers are retained, not hashed.
It accepts prepared prompts, **not** raw official CSV files.

## Training and generation

Supply your own prepared dataset and YAML; the following paths are placeholders:

```bash
python -m rlcr train --config /path/to/training.yaml \
  --output_dir outputs/train/new-run --report_to none --push_to_hub false
python -m rlcr evaluate --config /path/to/generation.yaml \
  --output-dir outputs/eval/new-run
python -m rlcr infer --model /path/to/model-or-adapter \
  --system-prompt "Return the requested label and confidence." \
  --prompt "Your prepared premise, hypothesis, and label instructions."
```

Inference's optional `--system-prompt` is literal text, not a preset name.
Repeat `--prompt` for multiple inputs. Each yields a JSON record of completions.
Use `--load-in-4bit`, `--n`, `--temperature`, `--max-tokens`, and
`--hf-batch-size` as appropriate.

Training selects full fine-tuning with `use_peft: false`; LoRA with
`use_peft: true`; QLoRA additionally sets `load_in_4bit: true`.
Do not enable 4-bit/8-bit loading for full-weight training.
GRPO is an objective, not an added neural layer. The policy generates without
gradients, then trains on its sampled completions. QLoRA freezes the quantized
base and updates adapters. A nonzero `beta` creates a separate KL reference
model; use `beta: 0` to avoid this extra copy (the default is 0.04).

`evaluate` currently saves unmodified completions and generation counts only.
Its `metrics.json` is **not** an official task score report. There is no LLM
judge, answer-repair stage, classifier, or official submission exporter.

Use a new output directory per experiment. Training auto-resumes the latest
checkpoint if one exists; otherwise it can replace final weights in a reused
directory. Enable periodic checkpointing for longer jobs. A saved final adapter
alone is not an optimizer-resume checkpoint. Resolved settings are saved in
`resolved-config.yaml`, with earlier versions under `config-history/`.

Batch generation caches by model label, not model/config content. Use a fresh
directory or `--fresh` after changing a model or generation settings.
Mismatched input rows/order are rejected. Results are saved after the whole
model list completes, so interruption may lose new results. Back up valuable
outputs independently; `outputs/` is ignored by Git.

## Distributed execution and deployment

```bash
accelerate launch --num_processes 4 \
  --config_file configs/accelerate/zero2.yaml \
  --module rlcr train --config /path/to/training.yaml
sbatch scripts/slurm/train.sbatch /path/to/training.yaml
```

ZeRO-2 partitions gradients and optimizer state, **not base weights**. Workers
generate local responses, gather group rewards, and synchronize updates.
The global generation batch must divide evenly into `num_generations` groups.
Local generation batch size is `per_device_train_batch_size × steps_per_generation`;
the latter defaults to `gradient_accumulation_steps`. More workers do not
automatically reduce local rollout memory.

The Slurm template starts one launcher task with one worker per GPU. Adjust site
resources/account/partition/time and submit from the checkout root. It uses
`.venv/bin/python`, overridable through `RLCR_PYTHON`. Multi-node launch,
GPU DeepSpeed, and distributed QLoRA require separate validation. There is no
Dockerfile; a future container can install this package and use
`python -m rlcr` with mounted inputs/caches/outputs.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest -q tests
```

Tests construct tiny local models and synthetic data, without downloading any
benchmark. They exercise exact-label parsing/rewards, prepared data, strict YAML,
model loading, generation, full-model/LoRA updates and reloads, storage, and CLI.

Opt-in two-worker CPU DDP checks require local inter-process networking:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m pytest -q tests/test_distributed.py
```

CPU checks do not validate GPU 4-bit kernels, task quality, or official scoring.
Successful execution does not establish calibration or the WordNet hypothesis.

## Attribution

The RLCR method is from Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld,
Leshem Choshen, Yoon Kim, and Jacob Andreas, *Beyond Binary Rewards: Training LMs
to Reason About Their Uncertainty* (ICLR 2026).
[Paper](https://arxiv.org/abs/2507.16806) ·
[Local method reference](docs/RLCR_paper_2507.16806v2.pdf).
This fork builds on [TRL](https://github.com/huggingface/trl) and
[Open-R1](https://github.com/huggingface/open-r1); retained source notices preserve
their attribution. The paper PDF is a method reference, not a bundled benchmark.

Task resources: [SemEval 2027 Task 2: DiCo-NLI](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI).
The roadmap records the inspected task contract; pin and verify the official
data/scorer revision when implementing the adapter.
