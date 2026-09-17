# No-knowledge DiCo-NLI training — Task 3

The ordinary one-policy GRPO path now supports the task's prompt and four-label
output contract. WordNet and pair-aware advantages are **not** active yet.
One GRPO group still contains several responses to one ordered prompt.

## Versioned prompt preparation

After [canonical data preparation](data.md):

```bash
python -m rlcr prepare-prompts \
  --dataset data/dico-nli/prepared/en \
  --output-dir data/dico-nli/prompted/en
```

The command requires a local, nonempty DatasetDict of canonical records with
named splits. It validates IDs, gold labels when present, known reciprocal pairs,
and split isolation. It preserves every record/column and adds `prompt`; existing
destinations and inputs already containing a prompt are rejected. To select a
small run, first set `max_source_pairs` per split in the data recipe and prepare
that subset, then prepare its prompts. Nothing samples or changes splits here.

`data/dico_nli/prompts.py` defines `dico-nli-no-knowledge-v1`. The builder accepts
only `text1`, `text2`, `text1_lang`, and `text2_lang`. It creates one user message
containing fixed instructions and a JSON-escaped input object. A user-only turn
also works with models whose chat templates do not support a separate system role.
Texts are preserved, not normalized or concatenated with bookkeeping metadata.
JSON escaping separates the input structure; it is not a general prompt-injection defense.

The instructions define forward entailment as Text 1 implying Text 2 but not the
converse, backward as the opposite direction, equivalence as mutual implication,
and negative/other as outside those relations. They follow the
[pinned task description](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/blob/588968e610197ddc4c440314701cbc587afc4c1b/README.md).
No rationale, worked gold example, evidence, other-direction prediction, or
orientation marker is included. Confidence means correctness of the chosen label,
not ontology agreement. Each direction is built independently from its own texts.
Language codes en/es/eu are accepted; released-data validation remains English-only.

Unlabeled records use the identical builder. Gold, instance/source IDs,
`reverse_pair_id`, `pairing_available`, and extra metadata cannot affect prompt
text. Tests change/remove these fields and assert identical prompts. Labels stay
available to training rewards and explicit scoring, not to the model.

Artifacts include Arrow splits, `audit.json`, a source-manifest snapshot when
available, and `manifest.json` published last. The manifest records prompt version,
instruction hash, permitted input fields, source path/manifest hash, and per-split
input-record/prompt hashes. Content hashes are independent of row order; stored
rows keep their original order. Source manifests are optional for synthetic or
externally prepared canonical records, but missing provenance is recorded as null.
An absent output manifest means incomplete storage; use a new output directory.
Changing instructions or serialization requires a new version and prompted dataset.

## Task rewards and invalid responses

The generic strict parser accepts an optional label vocabulary. The DiCo bindings
in `rewards/dico_nli.py` fix it to the four official labels and reuse the generic
formulas. There is no second implementation of accuracy or Brier:

| Reward name | Valid response reward | Invalid response reward |
|---|---|---|
| `dico_accuracy` | 1 if label matches gold, otherwise 0 | -1 |
| `dico_brier` | `1 - (confidence - correctness)^2` | -1 |
| `dico_format` | 1 | -1 |

Unknown/case-changed labels, malformed/duplicated tags, missing/nonfinite/out-of-
range confidence, and extra prose are invalid. An unknown label at confidence 0
must not earn a Brier reward of 1. A **valid but wrong** label at confidence 0 does
earn Brier reward 1, but accuracy 0: this is why correctness and calibration have
separate terms. With accuracy/Brier training, unsupported gold labels fail before
tokenizer/model loading.

The starter recipe uses `reward_funcs: [dico_accuracy, dico_brier]` and explicit
weights `[1.0, 0.5]`; an invalid response therefore totals -1.5. Accuracy-only uses
`[dico_accuracy]` with weight `[1.0]`. The unrestricted primitives remain for
programmatic/generic workflows, with legacy defaults unchanged. Mixing generic
and DiCo reward families is rejected so one component cannot relax the contract.

## Training and matched generation

```bash
python -m rlcr train --config configs/train/dico-nli-rlcr.yaml \
  --model_name_or_path /models/your-base-instruct \
  --output_dir outputs/train/rlcr-smoke

python -m rlcr train --config configs/train/dico-nli-rlcr.yaml \
  --model_name_or_path /models/your-base-instruct \
  --reward_funcs dico_accuracy --reward_weights 1.0 \
  --output_dir outputs/train/accuracy-smoke

python -m rlcr evaluate --config configs/eval/dico-nli.yaml \
  --model /models/your-base-instruct --output-dir outputs/eval/base
python -m rlcr evaluate --config configs/eval/dico-nli.yaml \
  --model outputs/train/rlcr-smoke --output-dir outputs/eval/rlcr-smoke
```

These paths require a user-selected model. No model is downloaded or chosen by
prompt preparation. Training defaults to QLoRA/BF16 with q_proj/v_proj adapters;
verify that these modules, dtype, context length, and memory fit your model/GPU.
For unquantized LoRA override `--load_in_4bit false`; for full training also use
`--use_peft false`. A CPU toy run additionally needs `--use_cpu true --bf16 false
--torch_dtype float32`. Generation quantization/dtype live in its separate model
section; change them there when GPU quantization is unavailable.

The recipe has one trainable policy, `beta: 0`, four generations per prompt,
`scale_rewards: false`, 64 completion tokens, and ten optimizer steps. These are
starting smoke settings, not optimized hyperparameters or a GPU-tested budget.
Train/dev evaluation during optimization is disabled; standalone generation and
[official scoring](evaluation.md) remain explicit. Use a source-grouped training
holdout for repeated tuning, not dev gradients. Preserve the documented overlap audit.

Before loading policy weights, DiCo CLI training validates gold and checks every
rendered prompt against `max_prompt_length`. Over-budget inputs raise an error,
not silent left truncation. The same tokenizer instance is passed to GRPO; tests
check training/inference chat-template parity. Ensure prompt plus completion fits
the model's actual context window too; the preflight does not infer that limit.
Direct Python use of the generic trainer retains its original truncation behavior.

DiCo training rejects first-N subset flags; sample source groups during data
preparation. Batch generation's generic `sample_size` option still exists: do not
use it for official paired scoring. Use the complete prepared subset and matching
reference for export/scoring instead. No post-hoc reverse-label forcing occurs.

## Diagnostics and validation limits

Trainer logs include each reward component, `dico/invalid_rate`, and
`zero_reward_std_fraction`. With group-centered GRPO, identical rewards give zero
advantage. In particular, **an all-invalid group has no relative learning signal**.
First inspect frozen-model output validity; do not assume a longer RL run will
teach a model the schema when every response is invalid. A future supervised
warm-up would need a separate implementation/decision; none is silently introduced.

CPU tests exercise actual sampling, gradients, optimizer updates, frozen LoRA base
weights, save/reload logit equality, greedy inference, and CLI generation. They
use a synthetic tiny Qwen2 model whose vocabulary contains whole response strings
as single tokens. This avoids asking a random model to learn XML before checking
the plumbing. It is **not** a pretrained baseline, realistic tokenization, a
learning-quality result, or a constrained decoder installed in production.

Prompt preparation has also run on all 3,042 train and 660 dev English records.
No working NVIDIA driver was accessible in this session: GPU 4-bit kernels,
memory use, and QLoRA remain unvalidated. No B0, calibration gain, or task-quality
comparison has been completed. Results and remaining gates are in
[short-term.md](short-term.md); the next implementation task is the conservative
WordNet provider and training-only coverage/error audit.
