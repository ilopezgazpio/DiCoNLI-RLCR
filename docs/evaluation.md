# DiCo-NLI export and official scoring

Task 2 is implemented under `rlcr/evaluation/dico_nli/`. These commands are
subcommands of the existing application, not additional entry points. Generation,
gold-free export, and gold-based scoring have explicit boundaries:

```text
prepared prompts -> evaluate -> raw completions
                                + independent expected IDs -> export-submission
                                                               -> submission.csv
explicit reference + submission.csv -> score -> official reports + provenance
```

NLI prompt construction/training integration is the next task. These commands
already work with synthetic responses or externally supplied predictions;
they do not establish a trained DiCo-NLI model.

## Official scorer setup and provenance

The authoritative implementation is the task's
[evaluation_functions package](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/588968e610197ddc4c440314701cbc587afc4c1b/evaluation_functions),
not our generic calibration helpers. Revision and all 14 source/documentation/
license SHA-256 hashes are recorded in `evaluation/dico_nli/scorer_pin.py`.

```bash
python -m rlcr fetch-scorer
```

This explicitly downloads to `.cache/rlcr/dico-nli/588968e610197ddc4c440314701cbc587afc4c1b/`.
An existing cache is verified and reused, never overwritten. Source acquisition
does not need an LLM, Torch, GPU, Git checkout, or additional dependency installation.
Downloads and hashes are checked before creating the destination. An incomplete
cache after a storage failure is rejected on reuse; choose a new directory rather
than expecting automatic repair/deletion.

For an offline worker/container, fetch on a connected machine, copy/mount the
cache, then pass `--scorer-dir /absolute/path/to/cache` to `score`. The default
cache is relative to the current working directory. Scoring never downloads.
Each invocation verifies the pinned bytes, copies only those files into a
temporary directory, and runs the official CLI in a separate Python process.
Ambient `PYTHONPATH`, site packages, and cached bytecode do not select a different
scorer. Execution has a 60-second timeout. File inputs are limited to 64 MiB,
matching the upstream default; current released files are well below this.

The upstream repository uses [GPL-3.0](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/blob/588968e610197ddc4c440314701cbc587afc4c1b/LICENSE).
Its source, copyright notices, README, and license remain unchanged in the external
cache; they are not bundled into this application's MIT-licensed source/package.
Neither project's license is changed. Preserve upstream notices when distributing
the cached scorer, including in container images, and review its licensing obligations.

To update the pin, inspect the new upstream contract/source/license, update the
revision and each file hash, fetch into a new cache, and rerun scorer parity and
data compatibility checks. Do not replace cached source without changing the pin.
Recheck the task's current rules/scorer before any official submission.

## Gold-free submission export

```bash
python -m rlcr export-submission \
  --predictions outputs/eval/example/predictions \
  --instances data/dico-nli/prepared/en --split dev \
  --prediction-column candidate-output_0 \
  --output-dir outputs/eval/example/export
```

The example prediction path/column require an existing generation result. Both
inputs must be local saved Hugging Face Datasets or DatasetDicts, not CSVs/Hub
identifiers. `--split` selects the same named split from any DatasetDict; a single
Dataset is used directly. `--instances` needs only `instance_id`, so an unlabeled
prepared test dataset works. The two input paths must be independent: deriving
expected IDs from predictions themselves would hide dropped examples.

Export selects only IDs and the explicitly requested response column. It does
not read gold labels, texts, reverse links, or choose the best response by gold.
Canonical task metadata and prompt columns cannot be selected as predictions.
When generation saved several `NAME-output_N` columns, choose exactly one; there
is no implicit best-of-N, aggregation, or retry policy.

Each completion must parse as:

```text
<answer>FORWARD_ENTAILMENT</answer><confidence>0.75</confidence>
```

Allowed labels are `EQUIVALENCE`, `FORWARD_ENTAILMENT`, `BACKWARD_ENTAILMENT`, and
`NEGATIVE_OTHER`. Confidence must be finite in [0, 1]. Whitespace around tag
contents is accepted; case repair, duplicated tags, free-form explanations,
missing confidence, NaN, and out-of-range values are not. A singleton assistant
message list is accepted by the shared parser too.

IDs must be unique, nonempty strings without whitespace/NUL, matching the
independent expected set exactly. Missing/extra/duplicate IDs fail before any
output is created. Row order does not affect the resulting submission, which is
sorted by instance ID.

Artifacts in a new export directory:

| File | Contents |
|---|---|
| `submission.csv` | Exactly `instance_id,label`; created only if every response is valid |
| `diagnostics.jsonl` | Raw responses, parsed labels/confidence, per-instance parsing errors |
| `export-report.json` | Counts, invalid rate, ready/blocked status |
| `manifest.json` | Selected column/split, input paths, expected-ID and output hashes |

Malformed responses leave diagnostics and a `blocked` manifest, exit nonzero,
and produce **no partial submission**. Nothing is silently converted to a negative
label or repaired. Reruns need a new output directory. Confidence is retained for
future calibration analysis; it is not an official submission field or task metric.

## Explicit gold-based scoring

Against a full official reference:

```bash
python -m rlcr score \
  --gold /path/to/official-reference.csv \
  --predictions /path/to/submission.csv \
  --output-dir outputs/eval/example/scores
```

Or against the canonical prepared split used for generation:

```bash
python -m rlcr score \
  --reference-dataset data/dico-nli/prepared/en --split dev \
  --predictions /path/to/submission.csv \
  --output-dir outputs/eval/example/scores
```

Exactly one reference source is required. Raw reference and prediction files may
be CSV or TSV and are checked by upstream, without our rewriting its validation
or metrics. Predictions from another system can go directly to `score` if they
already satisfy the two-column contract; they need not have RLCR confidence.

Prepared references require all canonical record fields, valid labels, known
pairing, and complete reciprocal pairs. This deliberately retains the stricter
Task 1 preparation contract. For a source-grouped subset, use that same subset
as `--instances` for export and `--reference-dataset` for scoring. No command
silently expands a subset to full dev or drops unmatched reference IDs.
Do not use arbitrary first-N-row subsets that break pairs.

The official scorer computes weighted F1 over all instances and SoftCons/HardCons
over eligible reciprocal pairs. Negative instances contribute to classification,
not to consistency. A reference with **zero eligible pairs is rejected** by this
pinned scorer; the wrapper preserves that behavior instead of inventing zero scores.

Successful scoring writes exact upstream `scores.json` and `scores.txt`, byte
snapshots `submission.csv` and `reference.csv`, and a manifest recording the
scorer revision/source hashes and all artifact hashes. JSON includes per-label
scores, confusion matrix, pair counts, and pair errors. The manifest is published
last; its absence means incomplete storage. Existing destinations are rejected.
Invalid input or scorer failure produces no result directory.

**Only upload `submission.csv`.** A scoring directory contains gold references
for local reproducibility, and export diagnostics contain non-submission fields.
No command uploads artifacts. Confidence calibration and evidence diagnostics
remain separate future work; `evaluate/metrics.json` still holds generation counts.

## Validation

Run `fetch-scorer` once, then tests are offline:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m pytest -q tests/dico_nli
```

Set `RLCR_DICO_SCORER_DIR=/absolute/path/to/cache` for a cache outside the checkout.
That environment variable is a **test fixture** setting; the application uses
`--scorer-dir`. Without a cache, integration tests explicitly skip. A missing or
modified file inside an existing cache fails validation, not a skip.

Tests cover hand-calculated perfect/all-equivalence/consistent-but-wrong/
inconsistent cases, negative exclusion, row ordering, duplicate/missing/extra IDs,
invalid outputs and references, direct upstream parity, offline source integrity,
CLI use outside the checkout, and scoring complete-pair subsets. These checks
validate plumbing and metric semantics, not NLI quality or GPU training.

The working checkout also records a 660-instance English-dev mechanical check in
`outputs/eval/task2-scorer-validation/`: synthetic oracle predictions score 1.0 on
all three metrics, while all-equivalence remains soft-consistent but often wrong.
These are **gold-derived/dummy pipeline checks, not model predictions or B0**.
See [short-term.md](short-term.md) for the results log.
