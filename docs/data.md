# DiCo-NLI data preparation

Task 1 imports and audits local official CSVs. It does not build model prompts,
run inference/training, export submissions, or compute official scores.
Use the same application entry point:

```bash
python -m rlcr prepare-data --config configs/data/dico-nli-en.yaml
```

The English recipe pins revision
`588968e610197ddc4c440314701cbc587afc4c1b` of the
[task repository](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/tree/588968e610197ddc4c440314701cbc587afc4c1b).
It verifies a SHA-256 digest for each input. Source files and generated datasets
are local artifacts ignored by Git; the small recipe is versioned.

## Obtain the pinned English files

The files are already downloaded in the working checkout where task 1 was
implemented. For a fresh checkout, run this explicit acquisition step from the
repository root. No imports or training loop perform network downloads.

```bash
dico_revision=588968e610197ddc4c440314701cbc587afc4c1b
dico_raw="data/dico-nli/raw/$dico_revision"
dico_url="https://raw.githubusercontent.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI/$dico_revision"
for dico_split in train dev; do
  mkdir -p "$dico_raw/$dico_split"
  for dico_kind in participant_labeled reference; do
    dico_file="dico_nli_${dico_split}_track1_${dico_kind}.csv"
    if [ ! -e "$dico_raw/$dico_split/$dico_file" ]; then
      curl -fL "$dico_url/final_data/$dico_split/$dico_file" \
        -o "$dico_raw/$dico_split/$dico_file" || break 2
    fi
  done
done
curl -fL "$dico_url/LICENSE" -o "$dico_raw/UPSTREAM-LICENSE"
curl -fL "$dico_url/final_data/README.md" -o "$dico_raw/UPSTREAM-README.md"
```

Preparation detects incomplete or changed downloads through their hashes.
Use the upstream data terms and attribution; the recipe does not grant new
rights to redistribute the data. The recorded revision identifies the source
chosen by the recipe. Offline preparation checks bytes against its digests,
not membership in a remote Git commit.

## Why the data layer is separate

An immutable `NLIRecord` represents an ordered phrase pair. CSV loading,
reference joining, pair validation, source-group sampling, reporting, and storage
are separate small modules under `rlcr/data/dico_nli/`.

The canonical record contains:

| Field | Meaning |
|---|---|
| `instance_id` | Official ordered-instance identifier, never rehashed or parsed for a label |
| `pair_id` | Underlying source identity used for grouping and split checks |
| `text1`, `text2` | Original ordered texts, preserved exactly |
| `text1_lang`, `text2_lang` | Language codes, currently en/es/eu |
| `label` | One official label, or null for unlabeled inference |
| `reverse_pair_id` | Official **instance ID** of the reverse, despite the upstream name; nullable |
| `pairing_available` | Whether a matching public reference was supplied |

`pairing_available=false` means pairing is unknown. It must not be interpreted
as a negative gold label. With a validated reference, `NEGATIVE_OTHER` has no
reverse link; other labels require a reciprocal link. The importer never guesses
links from ID suffixes, duplicate texts, or label predictions.

Inputs, IDs, labels, and link metadata remain together as auditable data, but the
prompt builder whitelists only texts/languages. It never stringifies the entire
record. `prepare-data` creates no `prompt` column; run the separate
`prepare-prompts --dataset INPUT --output-dir NEW` transformation before
training/evaluation. See [training.md](training.md) for its contract and provenance.

## Recipe and validation policy

`configs/data/dico-nli-en.yaml` separates data choices from model/optimizer choices.
Paths resolve against the process working directory, as elsewhere in this project.
`--output-dir NEW_DIRECTORY` overrides the destination.

Top-level settings are `source`, `output_dir`, optional `seed` (42), and `splits`.
Every split requires:

- `labeled`: an explicit boolean.
- `participant`: a mapping with `path` and required `sha256`.
- Optional `reference`: another path/digest mapping, only for labeled input.
- Optional `max_source_pairs`: a positive source-group limit, not a row limit.

Unknown/duplicate YAML keys, aliases, bad split names, branch-name revisions, and
missing hashes fail. There are no implicit YAML includes or inherited configs.

Participant CSVs require `instance_id,pair_id,text1_lang,text2_lang,text1,text2`,
plus `label` exactly when `labeled: true`. For unlabeled inference use
`labeled: false` and omit the reference; gold columns/joins are rejected.
Labeled data without a reference is accepted with unknown pairing.

Reference CSVs require `instance_id,pair_id,reverse_pair_id,label`; optional
text/language columns are checked against participant values when present.
Other columns fail explicitly, making upstream schema changes visible.
UTF-8 with an optional BOM and quoted commas/newlines is supported. Empty or
malformed rows, invalid labels/languages, duplicate IDs, whitespace in IDs,
unknown links, self-links, nonreciprocal links, wrong labels, and unswapped
text/language pairs fail. Text whitespace is preserved; labels are not repaired.

This importer deliberately requires complete reversible pairs **when references
are supplied**. This is a stricter preparation contract, not a reimplementation
of the scorer's validation policy. The separate [scoring commands](evaluation.md)
now execute the official scorer without replacing its validation policy.

All full input splits are checked for shared `instance_id` and `pair_id` before
sampling. Either is a hard error, including across different language variants.
Exact unordered text/language overlaps under different source IDs generate
warnings with up to 20 examples. They do not trigger implicit deduplication or
changes to official splits. Near-duplicate detection is not implemented.

## Sampling decisions

A source group includes every loaded direction/language variant sharing a
`pair_id`. Sampling never fabricates a reversed negative example.

For labeled inputs, groups are stratified by their set of labels: typically
equivalence, forward/backward entailment, and negative. Each observed stratum
receives at least one group, then allocations track its source-level proportion.
A requested limit too small to preserve the strata, or larger than the input,
raises an error. Very small samples inevitably distort proportions; both original
and selected distributions are recorded. This is not exact instance-level balance.

Within a stratum, source IDs are ranked by a hash of the seed and ID.
Selection therefore survives CSV row reordering and preserves global RNG state.
Selected records are sorted by instance ID. With no limit, all records are kept.
For unlabeled data there is one sampling stratum; no label balance is inferred.

Train and dev budgets can differ by setting `max_source_pairs` under each split.
For example, 100 train source pairs are suitable for the proposed evidence audit.
Do not later apply the generic training/evaluation first-N-row limits to this data:
they can break pairs. Source-group-aware subsets belong in this preparation step.
An automatic inner training/validation split is not introduced in task 1.

## Artifacts and safe reruns

```text
data/dico-nli/prepared/en/
  dataset_dict.json
  train/                  # Arrow records, no prompts
  dev/
  resolved-config.yaml
  audit.json              # Full-input and selected counts/checks/warnings
  manifest.json           # Schema, provenance, hashes, selected IDs, sampling version
```

Load records with `datasets.load_from_disk("data/dico-nli/prepared/en")`.
The manifest marks `stage: validated_task_records` and `prompt_ready: false`.
Raw inputs remain unchanged.

Any existing destination, including an empty directory or symlink, is rejected.
Validation and sampling complete before creating the output. An exclusive mkdir
protects against concurrent overwrites. Storage failures can leave an incomplete
directory; successful completion publishes the manifest last via rename.
Treat an absent manifest as an incomplete preparation and use a new destination.
No automatic deletion or overwrite is performed.

## First released-data audit

The pinned English release produced:

| Split | Instances | Source pairs | Reciprocal pairs | Negative instances |
|---|---:|---:|---:|---:|
| train | 3,042 | 1,760 | 1,282 | 478 |
| dev | 660 | 383 | 277 | 106 |

All CSV hashes, reference joins, reciprocal links, reverse labels, and exact text/
language swaps passed. No instance IDs or source-pair IDs overlap across splits.

Four unordered text pairs recur under different source IDs, affecting seven dev
instances: “a small baby”/“a baby”, “are playing”/“play”, “into the camera”/
“at the camera”, and “driving down”/“driving”. Their directional labels agree
across the observed copies. The audit retains IDs for review. Official splits
remain unchanged; before quality experiments, decide and document whether to
also run a training-only decontamination comparison. This is not a model result.

Only the English official release was audited. Synthetic tests cover other
language combinations and grouping; they are not validation of the other tracks'
released files. Official scoring/export, NLI prompts, and training-label integration
are implemented. The current unfinished task is Task 4: ordinary A/C GPU QLoRA
training/save/reload/evaluation. WordNet and its training-only audit are Task 5,
after the ordinary training acceptance gate passes.

The full CPU regression suite passed 248 tests (2 opt-in distributed tests skipped).
A real-data 100-source-pair sample with seed 42 contained 173 instances: 73
reciprocal pairs and 27 negatives. Reordering the input rows produced the same
selection. This validates preparation mechanics, not WordNet coverage or model quality.
