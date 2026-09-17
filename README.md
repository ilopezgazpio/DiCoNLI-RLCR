# Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty

This repository contains the official code for the paper:

> **Beyond Binary Rewards: Training LMs to Reason about Their Uncertainty**  
> Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Yoon Kim, Jacob Andreas  
> *[arXiv:2507.16806](https://arxiv.org/abs/2507.16806)*

This repository builds on top of [TRL](https://github.com/huggingface/trl) and [Open-R1](https://github.com/huggingface/open-r1). We thank the authors and maintainers of these projects.

---

## 🛠 Installation

### Environment Setup

Prerequisites: Python 3.10 with `venv` support, Git, and Linux x86_64. GPU training requires a CUDA 12.4-compatible NVIDIA driver.

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
| Full fine-tuning | `use_peft: false`, `load_in_4bit: false` |
| LoRA | `use_peft: true`, `load_in_4bit: false` |
| QLoRA | `use_peft: true`, `load_in_4bit: true` |

Training and inference use Transformers. During training, the same model generates responses and computes the GRPO loss. QLoRA keeps the base weights quantized and frozen while training the LoRA adapters.

Training defaults to PyTorch's native scaled dot-product attention (`attn_implementation: sdpa`), which is selected in all supplied training configs and requires no separate attention extension.

### Login to wandb 

For experiments with `report_to: [wandb]`:

```bash
wandb login
```

### Deepspeed & Accelerate Setup

`configs/accelerate/zero2.yaml` configures distributed execution using Accelerate and DeepSpeed ZeRO-2. The launch command selects the number of GPU processes. Gradient accumulation and clipping are taken from the training recipe (`auto` in the launcher configuration), avoiding competing values. This is an execution configuration, separate from experiment recipes and environment dependencies.

### Application Structure and Entry Point

All application Python code lives in `src/main/python/rlcr`, organized into `arguments`, `data`, `models`, `rewards`, `training/grpo`, `evaluation`, and `inference` packages. The GRPO coordinator delegates generation, reward evaluation, loss calculation, buffering, and metrics to focused modules. See [the architecture guide](docs/architecture.md) for the module map and migration from the old files.

There is one CLI: `python -m rlcr`, also available as the equivalent `rlcr` console command after installation.

```bash
python -m rlcr --help
python -m rlcr train --help
python -m rlcr evaluate --help
python -m rlcr infer --help
python -m rlcr prepare-data --help
```

Accelerate launches copies of this same application. Inside each worker, the training workflow creates `GRPOTrainer` and calls its inherited `train()` loop. There is no separate launcher embedded in the Python application.

### Configuration, Data, and Outputs

```text
configs/train/       # Training YAML recipes: model, data, rewards, optimization
configs/eval/        # Evaluation YAML recipes: dataset, models, output_dir
configs/accelerate/  # Distributed execution settings
data/               # Local input datasets only
outputs/train/      # Saved models/adapters, tokenizer files, checkpoints
outputs/eval/       # Per-run predictions/ and metrics.json together
```

Each new training/evaluation invocation records its resolved settings in
`output_dir/resolved-config.yaml`. Changed settings archive the previous snapshot
under `config-history/`. Existing historical artifacts are preserved; their exact
resolved configurations have not been reconstructed. Generated outputs are ignored
by Git, so back up valuable runs separately.

See [the configuration guide](docs/configuration.md) for the YAML schema, overrides,
path conventions, and old-to-new locations. No configuration inheritance or extra
configuration dependency is required.

### HuggingFace Models and Data
All models and datasets are available at this [RLCR HuggingFace Collection.](https://huggingface.co/collections/mehuldamani/rlcr-68912f9731b0bce30e4cc8c0)

The former `data/creation_scripts/` recipes are now import-safe modules under `rlcr.data.recipes`, executed through the same CLI:

```bash
python -m rlcr prepare-data --recipe trivia --output data/trivia --seed 42
```

Available recipes are `hotpotqa`, `big-math-digits`, `gpqa`, and `trivia`. This command downloads source datasets and saves a new local dataset directory; it never uploads to the Hub or overwrites an existing output. Randomized preparation is seeded. Use the published datasets above when reproducing the exact paper datasets.

---

## 🚀 Training

To run RLCR on hotpot:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file configs/accelerate/zero2.yaml --module rlcr train --config configs/train/hotpot-qwen7b-rlcr-full.yaml
```

For a small single-GPU QLoRA run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr train --config configs/train/hotpot-qwen1.5b-rlcr-qlora-smoke.yaml
```

Training YAML fields can be overridden on the same command line, for example `--max_steps 10 --report_to none --push_to_hub false`. Disabling hub upload and external tracking is useful for local smoke runs.

Training rejects unknown YAML keys. Process counts belong to Accelerate, not the
training recipe. Inactive legacy options have been removed; see the
[option migration guide](docs/configuration.md#supported-options-and-removed-leftovers).
Extra model-loading kwargs such as `model_init_kwargs: {local_files_only: true}`
are honored, while duplicates of managed model/quantization settings are rejected.

For Slurm, a single-node template launches the same application:

```bash
sbatch scripts/slurm/train.sbatch configs/train/hotpot-qwen7b-rlcr-full.yaml
```

Adjust the template's resource/account/partition/time settings for your cluster before submission. It launches one Accelerate process per allocated GPU from a single Slurm task. See [deployment notes](docs/architecture.md#slurm-and-containers) for working directories, multi-node considerations, and a future container entry point. No Slurm job or container build is needed for local execution.

### 📝 Notes

- Wandb logs from reproduced runs available [here](https://wandb.ai/mehuldamani/RLCR?nw=nwusermehuldamani). Intermediate generations are also logged and are useful for debugging. 
- Additional training examples are available in `scripts/train_examples.sh`.
- Training experiments are defined under `configs/train/`; distributed launcher settings live under `configs/accelerate/`.
- **Compute details**:
  - We ran HotpotQA experiments on **4×A100 GPUs**
  - Math experiments were run on **6×A100 GPUs**
  - The **generation batch size** is computed as:
    ```
    generation_batch_size = num_processes × per_device_train_batch_size × gradient_accumulation_steps
    ```
    It should be kept **constant or increased** if more compute is available. Lowering it may lead to instability during training.
- **Limitations**:
  - This field is evolving rapidly. We believe that both the **base RL implementation** and **hyperparameter settings** can be further improved. Doing so may reduce some training instabilities we encountered and enhance model calibration and reasoning quality.
  - Learning well-calibrated policies requires exploration over a range of verbalized confidences. If training problems have similar difficulty, the policy may collapse to outputting a narrow band—or even a single—confidence value, hindering calibration. If this behavior is encountered, incentivizing more exploration in verbalized confidence scores through higher temperature/modifications to system prompt can be effective.

We welcome suggestions and contributions!

---

## 📊 Evaluation

To run inference with our trained RLCR model on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr infer \
  --model mehuldamani/hotpot-v2-brier-7b-no-split \
  --load-in-4bit --system-prompt tabc_long \
  --prompt "Which popular dessert was invented at the Hungry Monk in Alfriston, Sussex?"
```

The example loads the model in 4-bit using the same `.venv` environment. Repeat `--prompt` for multiple questions. Inference prints one JSON record per question, containing its generated completions.

### 📚 Available Models

| Name              | Training Dataset                                         | Model Path                                             | System Prompt   |
|:------------------|:------------------------------|:--------------------------------------------------------------|:-------------|
| RLCR-hotpot       | HotpotQA-Modified                          | mehuldamani/hotpot-v2-brier-7b-no-split |   TABC_Long     |
| RLVR-hotpot       | HotpotQA-Modified (RLVR)        | mehuldamani/hotpot-v2-correctness-7b    |    GEN   |
| Classifier-hotpot | HotpotQA-Modified (Classifier)  | mehuldamani/orm-hotpot-v2-final-correctness  |    Gen             |
| RLCR-math         | Big-Math-Digits                            | mehuldamani/big-math-digits-v2-brier-base-tabc |       TABC          |
| SFT-RLCR-math     | Big-Math-Digits (SFT Warmup)                    | mehuldamani/big-math-digits-v2-brier |       TABC          |
| RLVR-math         | Big-Math-Digits (RLVR)          | mehuldamani/big-math-digits-v2-correctness    |      Gen           |
| Classifier-math   | Big-Math-Digits (Classifier) | mehuldamani/orm-big-math-digits-v2-correctness  |      Gen           |

### 🧪 Sample Evaluation Run

Run evaluation on a dataset using a config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m rlcr evaluate --config configs/eval/hotpot-models-trivia.yaml
```

For a full eval suite on a single GPU (We already provide the outputs/results from this):

```bash
bash scripts/evaluate_examples.sh
```

### 📝 Notes

- Each evaluation's `output_dir` contains `predictions/`, `metrics.json`, and `resolved-config.yaml`.
- To evaluate new datasets/models, edit or add YAML recipes inside `configs/eval/`.
- Override run locations and sample counts with `--output-dir /scratch/eval-run --sample-size 32`. Input locations can be overridden with `--dataset /datasets/example` and, for single-model recipes, `--model /models/adapter`.
- Existing predictions are reused by model `name` unless `fresh: true` or `--fresh` is selected. When changing a model or its generation/scoring settings, use a new output directory or regenerate with `--fresh`. `--no-fresh` disables the global fresh flag; individual model-level `fresh` flags still apply.
- Paper-model recipes preserve their separate `*-fresh-*` output destinations. Migrated historical results are under the corresponding names without `-fresh`; they are not automatically reused by those recipes.
- Default evaluation uses `temperature = 0` and `max_tokens = 4096`.
- Evaluation uses Transformers for generation, classification, and LLM judging. `hf_batch_size` controls the inference batch size and defaults to 1. `load_in_4bit` enables quantized generation, including local adapter checkpoints.
- Evaluation configs select post-processing through `tasks` (for example, `ans_at_end`, `confidence_at_end`, or `confidence_prob`). Token probabilities are collected only for `confidence_prob`, retaining only the selected token's score for each step.
- For LLM judging on a small GPU, set `"check_fn_args": {"judge_load_in_4bit": true}`. The generator is released before loading the judge.
- Currently supported evaluation functions:
  - **Exact Match** (Used for hotpotqa)
  - **Math Verify** (Used for all math datasets)
  - **LLM-as-a-Judge** (Used for trivia, simpleqa, commonsenseqa, gpqa)

---

## Tests

With `.venv` activated:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q tests
```

The tests construct tiny Qwen models locally and run on CPU. They cover config parsing, generation, full-model and adapter training and reloading, evaluation post-processing, judge integration, and the application CLI. GPU quantization and multi-GPU DeepSpeed require separate hardware checks.

The two-worker CPU regression is opt-in because it needs local inter-process networking:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m pytest -q tests/test_distributed.py
```

It checks groups split across workers, synchronized full-model/LoRA updates, and frozen adapter-base weights. It validates CPU DDP, not GPU DeepSpeed or 4-bit execution.

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{damani2025beyond,
  title={Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty},
  author={Damani, Mehul and Puri, Isha and Slocum, Stewart and Shenfeld, Idan and Choshen, Leshem and Kim, Yoon and Andreas, Jacob},
  journal={arXiv preprint arXiv:2507.16806},
  year={2025}
}
```
