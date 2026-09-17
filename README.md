# Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty

This repository is a modified, refactored version of the code accompanying:

> **Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty**  
> Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, Jacob Andreas  
> ICLR 2026 · [arXiv:2507.16806](https://arxiv.org/abs/2507.16806) · [Local paper PDF](docs/RLCR_paper_2507.16806v2.pdf)

RLCR adds calibration rewards to reinforcement learning: the language model learns to generate answers and confidence estimates together. This adaptation supports full-model GRPO training and LoRA/QLoRA in one Python environment, with Transformers generation, one application CLI, and modular code under `src/main/python/rlcr`. It is not an unchanged copy of the upstream implementation.

This repository builds on top of [TRL](https://github.com/huggingface/trl) and [Open-R1](https://github.com/huggingface/open-r1). We thank the authors and maintainers of these projects.

## Installation

Prerequisites: Python 3.10 with `venv` support, Git, and Linux x86_64. GPU training requires a CUDA 12.4-compatible NVIDIA driver. The supplied recipes use BF16; check that your GPU supports it. No Conda environment is needed.

From this repository's root directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip check
```

`requirements.txt` is the single dependency list for full model fine-tuning, PEFT/QLoRA, inference, evaluation, and tests. `pyproject.toml` reads that list and installs the `rlcr` package and console command. It includes PyTorch, Transformers, Accelerate, DeepSpeed, PEFT, and bitsandbytes. The core libraries and required TRL Git commit are pinned; pip resolves their supporting dependencies. No separate TRL checkout is needed. Use `python -m pip install .` instead of `-e .` for a non-editable deployment installation.

Keep the installation order: install the [PyTorch 2.5.1 CUDA 12.4 wheel](https://pytorch.org/get-started/previous-versions/#v251) before building DeepSpeed. The packaging-tool pins preserve compatibility with DeepSpeed 0.17.0's build setup. The PyTorch wheel installs CUDA runtime libraries, not the NVIDIA driver or a full compiler toolkit.

DeepSpeed CUDA extensions need a compatible CUDA toolkit (`nvcc`, preferably 12.4) and a C++ compiler. Its [0.17.0 source setup](https://github.com/deepspeedai/DeepSpeed/blob/v0.17.0/setup.py) also probes the toolkit during installation when it detects a CUDA GPU. If installation reports a missing `CUDA_HOME` or `nvcc`, install the toolkit and point `CUDA_HOME` to its installation directory; see [DeepSpeed's installation notes](https://www.deepspeed.ai/tutorials/advanced-install/). This is separate from the prebuilt PyTorch and bitsandbytes runtimes used by single-GPU QLoRA.

In subsequent shells, run `source .venv/bin/activate` before training or evaluation. Run `deactivate` to leave the environment. The `.venv` directory is already ignored by Git.

Select the training method through the experiment YAML:

| Method | Model settings |
|---|---|
| Full fine-tuning | `use_peft: false`, `load_in_4bit: false`, `load_in_8bit: false` |
| LoRA | `use_peft: true`, `load_in_4bit: false`, `load_in_8bit: false` |
| QLoRA | `use_peft: true`, `load_in_4bit: true`, `load_in_8bit: false` |

Training and inference use Transformers. During training, the same model generates responses and computes the GRPO loss. QLoRA keeps the base weights quantized and frozen while training the LoRA adapters.

There is no vLLM backend or dedicated rollout-model copy. Training uses PyTorch's native scaled dot-product attention through Transformers (`attn_implementation: sdpa` in all supplied recipes); neither `flash-attn` nor `xformers` is required. Accelerate is still used for training orchestration, not as an attention implementation. A nonzero GRPO `beta` creates a separate KL reference model and increases memory usage; all supplied training recipes set `beta: 0.0`, but the dataclass default is `0.04`.

### Authentication and external services

Public model/data downloads need network access unless cached. Gated or private Hub repositories may also require access approval and Hugging Face authentication, for example through `HF_TOKEN`. Keep credentials out of committed YAML files. Inference/evaluation loaders currently allow model repository code (`trust_remote_code=True`), so use trusted model sources.

For experiments with `report_to: [wandb]`, authenticate with:

```bash
wandb login
```

The full-model recipes retain `push_to_hub: true` and WandB reporting. The examples below explicitly disable both. The QLoRA recipes already disable them. Local training does not require publishing a model or enabling external tracking.

## Application structure and entry point

All application Python code lives in `src/main/python/rlcr`, organized into `arguments`, `configuration`, `data`, `text`, `models`, `rewards`, `training`, `evaluation`, and `inference` packages. See [the architecture guide](docs/architecture.md) for the complete module map and migration from the old root-level files.

There is one CLI: `python -m rlcr`, also available as the equivalent `rlcr` console command after installation.

```bash
python -m rlcr --help
python -m rlcr train --help
python -m rlcr evaluate --help
python -m rlcr infer --help
python -m rlcr prepare-data --help
```

Accelerate launches copies of this same application. Inside each worker, the training workflow creates `GRPOTrainer` and calls its inherited `train()` loop. There is no separate launcher embedded in the Python application.

The main training path is:

```text
cli.py -> training/configuration.py -> training/runner.py
    -> training/grpo/trainer.py -> models/policy.py loads the policy/adapters
    -> Trainer.train(): generate -> rewards -> group advantages -> GRPO loss -> update
```

`training/grpo/` separates sampling, rollout buffering, reward gathering, advantages, loss, and logging. GRPO is an objective and training procedure, not an extra neural-network layer trained alongside the LM. Full fine-tuning updates the policy weights; LoRA/QLoRA updates its adapters. Generation uses that same current policy without gradients, followed by gradient-bearing training forwards.

## Configuration, data, and outputs

```text
configs/train/       # Training YAML recipes: model, data, rewards, optimization
configs/eval/        # Evaluation YAML recipes: dataset, models, output_dir
configs/accelerate/  # Distributed execution settings
data/               # Local input datasets only
outputs/train/      # Saved models/adapters, tokenizer files, checkpoints
outputs/eval/       # Per-run predictions/ and metrics.json together
```

Training recipes select a base model with `model_name_or_path`; evaluation model entries use `model`. The base architecture's `config.json` comes from the model's Hub repository or local model directory, not `configs/train/`. Downloaded Hub assets use the Hugging Face cache unless configured otherwise. Local adapter directories retain `adapter_config.json` and still require access to their referenced base model.

Run the examples from the repository root. Relative paths are resolved against the current working directory, **not** the YAML file's directory. Configs and datasets are external inputs, not bundled in the installed Python package.

Each new training/evaluation invocation records its resolved settings in
`output_dir/resolved-config.yaml`. Changed settings archive the previous snapshot
under `config-history/`. Existing historical artifacts are preserved; their exact
resolved configurations have not been reconstructed. Generated outputs are ignored
by Git, so back up valuable runs separately. Snapshots describe application settings, not the full launcher command, hardware, or a guarantee of successful completion; retain job scripts/logs too.

See [the configuration guide](docs/configuration.md) for the YAML schema, overrides,
path conventions, and old-to-new locations. No configuration inheritance or extra
configuration dependency is required.

### Models and datasets

Published paper assets are linked from the [RLCR Hugging Face collection](https://huggingface.co/collections/mehuldamani/rlcr-68912f9731b0bce30e4cc8c0). Local adapters and historical evaluation outputs are separate artifacts; do not assume they exist in a fresh clone.

The former `data/creation_scripts/` recipes are now import-safe modules under `rlcr.data.recipes`, executed through the same CLI:

```bash
python -m rlcr prepare-data --recipe trivia --output data/trivia-prepared --seed 42
```

Available recipes are `hotpotqa`, `big-math-digits`, `gpqa`, and `trivia`. This command downloads source datasets and saves a new local dataset directory; it never uploads to the Hub or overwrites an existing output. Randomized preparation is seeded. Use the published datasets above when reproducing the exact paper datasets.

Training accepts a Hub dataset ID or a local Hugging Face `DatasetDict` saved with `.save_to_disk()`, with the configured splits (defaults: `train`/`test`). Examples need a `question` or `problem` column; correctness/calibration rewards also need `answer` and compatible task-specific metadata. In particular, the training accuracy reward selects Hotpot exact matching using `source: hotpot`; otherwise it uses math verification. Arbitrary CSV/JSON schemas are not automatically adapted or fully validated in advance.

## Training

Start with the 1.5B single-GPU QLoRA smoke recipe (10 optimizer steps). Choose a **new output directory** for each independent experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr train \
  --config configs/train/hotpot-qwen1.5b-rlcr-qlora-smoke.yaml \
  --output_dir outputs/train/quickstart-qlora \
  --report_to none --push_to_hub false
```

Training YAML fields can be overridden with Transformers-style underscore flags, for example `--max_steps 10 --dataset_name /datasets/hotpot`. Evaluation/inference CLI flags instead use hyphens, such as `--output-dir` and `--load-in-4bit`.

Training rejects unknown YAML keys. Process counts belong to Accelerate, not the
training recipe. Inactive legacy options have been removed; see the
[option migration guide](docs/configuration.md#supported-options-and-removed-leftovers).
Extra model-loading kwargs such as `model_init_kwargs: {local_files_only: true}`
are honored, while duplicates of managed model/quantization settings are rejected.

Use `sys_prompt_name` for training/evaluation prompts and `check_fn` for evaluation scoring. Prompt names are case-sensitive; the training/evaluation default is `gen`. Supported rewards are `accuracy`, `format`, `brier`, `mean_confidence`, and `confidence_one_or_zero`. Only `task_spec: gen` is supported: the `sft-rlcr` recipe starts GRPO from an existing SFT checkpoint, not a separate SFT training stage. Classifier-based evaluation remains available, but there is no ORM training command.

### Checkpoints and safe reruns

The final model or adapter, tokenizer, and trainer state are saved directly in `output_dir`. Intermediate checkpoints are separate `checkpoint-*` directories.

- The supplied QLoRA recipes use `save_strategy: "no"`: they save at successful completion, not periodically. For longer jobs, enable intermediate saves, for example `--save_strategy steps --save_steps 100 --save_total_limit 2`.
- The runner automatically resumes the latest checkpoint in an existing output directory unless `--resume_from_checkpoint /path/to/checkpoint-N` selects another one. A final adapter/model directory is not a complete optimizer-resume checkpoint.
- If an existing output directory has no checkpoint, training can start again and replace its final weights. There is no protective overwrite preflight. A new directory is the safest way to start a new experiment.

### Distributed training

For full-model training on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 --config_file configs/accelerate/zero2.yaml \
  --module rlcr train --config configs/train/hotpot-qwen7b-rlcr-full.yaml \
  --output_dir outputs/train/hotpot-full-example \
  --report_to none --push_to_hub false
```

`zero2.yaml` is a launcher configuration, not an environment file, and is not needed for the direct single-GPU command. It uses DeepSpeed ZeRO-2: each GPU holds policy weights, while optimizer state and gradients are partitioned. It does **not** shard the base weights to fit a model too large for one GPU. Gradient accumulation and clipping are taken from the training recipe (`auto` in the launcher); process counts belong to Accelerate.

Workers generate local completions and gather rewards so GRPO groups can span workers. Weight updates are synchronized. Generation memory depends on the local rollout batch, not just the training microbatch:

```text
local generation batch  = per_device_train_batch_size × steps_per_generation
global generation batch = local generation batch × number of workers
```

`steps_per_generation` defaults to `gradient_accumulation_steps`; alternatively set a global `generation_batch_size`, not both. These batches count completion sequences, including the `num_generations` repetitions per question. The global generation batch must be divisible by `num_generations`. Increasing worker count does not automatically reduce local rollout memory. The full-model recipes have large rollout batches and are not small-GPU presets; start from a QLoRA recipe when memory is limited.

### Slurm and containers

For Slurm, a single-node template launches the same application:

```bash
sbatch scripts/slurm/train.sbatch configs/train/hotpot-qwen7b-rlcr-full.yaml \
  --output_dir outputs/train/hotpot-slurm-example \
  --report_to none --push_to_hub false
```

Adjust the template's resource/account/partition/time settings for your cluster before submission. Submit from the repository root. It uses `.venv/bin/python` by default (`RLCR_PYTHON` can override this) and launches one Accelerate worker per allocated GPU from a **single Slurm task**. Do not also request one Slurm task per GPU.

The template is single-node; multi-node launch/rendezvous settings need separate setup and validation. There is no Dockerfile yet. A container can install the package and use `python -m rlcr` as its entry point, mounting configs, caches, data, and writable outputs. See [deployment notes](docs/architecture.md#slurm-and-containers).

Additional research presets are in [scripts/train_examples.sh](scripts/train_examples.sh); review their resources, output paths, and publication settings before running. [Original research WandB logs](https://wandb.ai/mehuldamani/RLCR?nw=nwusermehuldamani) are historical results, not validation of this refactor.

## Inference

After the quickstart training run, generate from its local adapter on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr infer \
  --model outputs/train/quickstart-qlora \
  --load-in-4bit --system-prompt tabc_long \
  --max-tokens 512 \
  --prompt "Which popular dessert was invented at the Hungry Monk in Alfriston, Sussex?"
```

The loader recognizes a local adapter through `adapter_config.json` and loads its base plus adapters; no merge step is required. `--model` also accepts a full-model directory or Hub model ID, such as `mehuldamani/hotpot-v2-brier-7b-no-split`. The 10-step smoke run tests the workflow, not model quality.

Repeat `--prompt` for multiple questions. Inference prints one JSON record per question, containing its generated completions. `--n`, `--temperature`, `--max-tokens`, and `--hf-batch-size` control generation; inference's default prompt is `tabc_long`.

### Published models

| Name              | Training Dataset                                         | Model Path                                             | System Prompt   |
|:------------------|:------------------------------|:--------------------------------------------------------------|:-------------|
| RLCR-hotpot       | HotpotQA-Modified                          | mehuldamani/hotpot-v2-brier-7b-no-split | `tabc_long` |
| RLVR-hotpot       | HotpotQA-Modified (RLVR)        | mehuldamani/hotpot-v2-correctness-7b    | `gen` |
| Classifier-hotpot | HotpotQA-Modified (Classifier)  | mehuldamani/orm-hotpot-v2-final-correctness  | Classifier stage |
| RLCR-math         | Big-Math-Digits                            | mehuldamani/big-math-digits-v2-brier-base-tabc | `tabc` |
| SFT-RLCR-math     | Big-Math-Digits (SFT Warmup)                    | mehuldamani/big-math-digits-v2-brier | `tabc` |
| RLVR-math         | Big-Math-Digits (RLVR)          | mehuldamani/big-math-digits-v2-correctness    | `gen` |
| Classifier-math   | Big-Math-Digits (Classifier) | mehuldamani/orm-big-math-digits-v2-correctness  | Classifier stage |

Prompt names are lowercase and case-sensitive. Classifiers are loaded through evaluation's `gen_then_classify` task and `class_model` setting, not the generative `infer` command.

## Evaluation

Evaluate the quickstart adapter on 32 Hotpot examples:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate \
  --config configs/eval/hotpot-peft-smoke.yaml \
  --model outputs/train/quickstart-qlora \
  --output-dir outputs/eval/quickstart-hotpot \
  --sample-size 32
```

Evaluation recipes are named YAML mappings, not the old positional JSON lists. For example:

```yaml
dataset:
  name: mehuldamani/hotpot_qa  # Hub ID or local saved dataset directory
  split: test
  hash_key: problem
  sample_size: 32
models:
  - name: my-adapter          # Unique label for prediction columns and metrics
    model: outputs/train/quickstart-qlora
    sys_prompt_name: tabc_long
    check_fn: confidence_verifier
    tasks: [confidence_at_end, ans_at_end]
    load_in_4bit: true
    max_tokens: 512
output_dir: outputs/eval/my-adapter-hotpot
fresh: false
```

`--dataset`, `--split`, `--sample-size`, and `--output-dir` override the recipe. `--model` is supported only for a single-model recipe. Unknown YAML keys are rejected. See [the configuration guide](docs/configuration.md#evaluation-recipes) for all dataset options and migration details.

### Scoring and memory

Evaluation uses Transformers for generation, classification, and LLM judging. `hf_batch_size` defaults to 1; generation defaults are `temperature: 0` and `max_tokens: 4096`, with shorter limits in local-adapter recipes. `tasks` selects postprocessing such as `ans_at_end`, `confidence_at_end`, or `confidence_prob`. The last option collects selected-token probabilities, not full vocabulary scores for every step.

`check_fn: confidence_verifier` tries math verification, then normalized exact matching as a fallback; it is used for Hotpot and math datasets. `llm_confidence_verifier` scores answers with a separate judge. Supplied trivia, SimpleQA, CommonsenseQA, and GPQA recipes use the latter. Their default judge is `meta-llama/Llama-3.1-8B-Instruct`; ensure you have access and enough memory for it, or configure an accessible alternative through `judge_model` inside the `check_fn_args` mapping.

`load_in_4bit: true` quantizes the **generator**, including local adapters. To also quantize the judge, set `check_fn_args: {judge_load_in_4bit: true}`. The generator is released before loading the judge. Classifier postprocessing has its own loader and does not inherit the generator's 4-bit flag.

### Outputs and reuse

Each run's `output_dir` contains `predictions/` (a saved Arrow dataset), `metrics.json`, and `resolved-config.yaml`.

- Results are reused by model `name`, not by a fingerprint of weights or settings. For a changed model or scoring protocol, use a new output directory or `--fresh` to regenerate against the same input rows.
- Changed dataset rows, order, split, or sample count require a **new output directory** when they no longer match saved predictions. `--fresh` does not bypass the input-row compatibility check.
- `--no-fresh` disables the global fresh flag; a model-level `fresh: true` still forces regeneration. The supplied local-adapter recipes set global `fresh: true`.
- Predictions and metrics are saved after the entire configured model list finishes, not after every model. An interrupted multi-model run can lose new results from that invocation.

### Paper evaluation suites

To evaluate the paper's Hotpot model group on TriviaQA:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-trivia.yaml
```

To launch all 16 paper-group/dataset evaluations sequentially on GPU 0:

```bash
bash scripts/evaluate_examples.sh
```

These commands launch real model/data downloads and evaluation, not a report viewer. The paper presets include multiple models, classifiers, and sometimes an LLM judge; they are not small-GPU smoke tests and do not enable 4-bit loading by default. Some datasets use local `data/` paths; obtain those inputs before launching the suite.

Paper-model recipes retain `outputs/eval/*-models-fresh-*` destinations. Migrated historical runs, when present, use names without `-fresh` and are not automatically reused. Historical artifacts were preserved during cleanup; they are not evidence that the refactored code has reproduced all paper results.

## Tests

With `.venv` activated:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest -q tests
```

The tests construct tiny Qwen models locally and run on CPU without model downloads. They cover strict configuration parsing and removed options, loader kwargs, generation, full-model and adapter training/reloading, evaluation storage and postprocessing, judge integration, and the CLI. The normal suite skips the opt-in distributed cases.

The two-worker CPU regression is opt-in because it needs local inter-process networking:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m pytest -q tests/test_distributed.py
```

It checks groups split across workers, synchronized full-model/LoRA updates, and frozen adapter-base weights. It validates CPU DDP, not GPU DeepSpeed or 4-bit execution.

### Validation limits

CPU regressions do not validate actual GPU QLoRA kernels, multi-GPU DeepSpeed, or a complete paper-scale experiment. GPU quantization, ZeRO-3/FSDP, multi-node Slurm, and container deployment require separate hardware/deployment checks. There is no automated CI workflow yet. The checkpoint-overwrite, evaluation-cache, and end-of-run persistence limitations described above remain current behavior, not completed fixes.

Research caveat: successful execution does not guarantee calibration. Monitor answer accuracy and confidence distributions; a policy can collapse to a narrow range of confidence values.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{damani2026beyond,
  title={Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty},
  author={Damani, Mehul and Puri, Isha and Slocum, Stewart and Shenfeld, Idan and Choshen, Leshem and Kim, Yoon and Andreas, Jacob},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2507.16806}
}
```
