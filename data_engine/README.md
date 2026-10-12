# TrajCond-VLA Data Engine

Configurable NAVSIM data engine adapted from [DriveTeach-VLA](https://github.com/ShivaTeam/DriveTeach-VLA) for two-stage TrajCond-VLA training.

## Overview

The engine extracts front-view images, driving intent, and ego motion from [NAVSIM](https://github.com/autonomousvision/navsim), merges external CoT pseudo-labels, encodes action tokens, and renders LLaMA-Factory-compatible training samples.

- **Stage 1: TrajCond semantics learning** — Driving reasoning followed by discrete actions (`CoT → AT`)
- **Stage 2: TrajCond-conditioned planning** — Discrete actions, driving reasoning, and continuous planning (`AT → CoT → PT`)

Both stages encode NAVSIM future poses into action tokens and reuse the same three CoT fields: `critical_objects`, `meta_behaviour`, and `explanation`. The future horizon is eight steps over four seconds at 2 Hz. Time labels use frame offsets: `t = (frame_index - current_frame_index) * 0.5`.

### Six Built-in Pipelines

| Representation | Stage 1 | Stage 2 | Tokens per future step |
|----------------|---------|---------|------------------------|
| KMeans | [navsim_kmeans_stage1.yaml](configs/pipelines/navsim_kmeans_stage1.yaml) | [navsim_kmeans_stage2.yaml](configs/pipelines/navsim_kmeans_stage2.yaml) | One joint token |
| KDisks | [navsim_kdisks_stage1.yaml](configs/pipelines/navsim_kdisks_stage1.yaml) | [navsim_kdisks_stage2.yaml](configs/pipelines/navsim_kdisks_stage2.yaml) | One joint token |
| Brohan | [navsim_brohan_stage1.yaml](configs/pipelines/navsim_brohan_stage1.yaml) | [navsim_brohan_stage2.yaml](configs/pipelines/navsim_brohan_stage2.yaml) | Three per-axis tokens |

## Quick Start

Use Python 3.9 for the data engine, matching [NAVSIM's default environment](https://github.com/autonomousvision/navsim/blob/main/environment.yml). Run from the repository root:

```bash
pip install -r data_engine/requirements.txt

# Install NAVSIM v2 and its dependencies following the official instructions.
pip install -e /path/to/navsim/

# Update NAVSIM paths and the selected pipeline configuration before running.
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --check-config
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --num-workers 8
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage2.yaml --num-workers 8
```

Select the corresponding KMeans or KDisks YAMLs to build the other datasets. `--check-config` validates the processing chain, templates, resource paths, codebook, and normalization settings. It does not extract NAVSIM samples or check every CoT record.

## Architecture

```text
NAVSIM → Adapter → SampleIR → Enricher chain → Stage-specific PromptBuilder → QA JSON
```

**Adapter** — Reads NAVSIM scenes and extracts the raw sample fields.

**SampleIR** — Pydantic intermediate representation shared by all processing steps.

**Enrichers** — Configurable processing units for input validation, image path rewriting, CoT merging, action encoding, reconstruction-error filtering, and normalization. Each declares its required and produced fields; the runner validates dependencies before execution.

**PromptBuilders** — `TrajcondStage1Builder` and `TrajcondStage2Builder` render the two training formats using the release TXT templates and their shared format contract.

## Configuration

### NAVSIM paths

Update [configs/datasets/navsim.yaml](configs/datasets/navsim.yaml) with the training logs, sensor blobs, and navtrain SceneFilter paths:

```yaml
name: navsim
splits:
  train:
    data_path: /path/to/datasets/navsim/trainval_logs/trainval
    sensor_blobs_path: /path/to/datasets/navsim/trainval_sensor_blobs/trainval
    log_names_file: /path/to/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml
```

The six pipelines use `path_base: project_root`, so relative resource and output paths resolve from the repository root. Set `replace_image_base.old_base/new_base` to produce image paths readable in the training environment. Local image checks occur before path rewriting.

### CoT pseudo-labels

All six pipelines read `data_engine/annotations/cot.jsonl` by default. Each line is a JSON object with an `id` matching the NAVSIM sample token and a `predict` string containing the three CoT fields.

- `critical_objects`: all twelve classes listed in the [format contract](src/prompts/templates/navsim_trajcond_format_release.json), each labeled `yes` or `no`
- `meta_behaviour`: one `speed` label and one `command` label from the prescribed categories
- `explanation`: nonempty natural-language driving reasoning

The pipeline import configuration is:

```yaml
- name: merge_pseudo_labels
  params:
    source_path: data_engine/annotations/cot.jsonl
    parser: poutine
    key_field: id
    json_repair: true
    strict_ids: true
```

The importer matches exact IDs and writes the three fields to `SampleIR.pseudo_label`. Duplicate IDs stop the run. Missing or invalid CoT records are excluded by `filter_incomplete_pseudo`. Both stages reuse the same annotation content for each ID.

### Action encoding and filtering

Set `encode_trajcond.method`, `codebook_path`, and `variances_path` in the selected YAML. All methods use `action_variances.json` beside the codebook. Codebook size means the total action vocabulary: the number of joint centers for KMeans/KDisks, or the sum of the three axis sizes for Brohan.

Search settings use `candidate_top_k` for KMeans/KDisks and `per_axis_top_k` for Brohan. KMeans/KDisks default to a beam size of 4, retaining paths by cumulative vehicle corner error; Brohan keeps a beam size of 1. See the [codebook guide](../codebook/README.md) for construction commands.

Stage 2 sets `require_existing_stats: false` and computes normalization statistics in memory after filtering. Its configs omit `stats_file`.

After all samples have been searched, the engine prints reconstruction error statistics before error filtering: the searched sample count, mean and median corner error at each future step, and overall mean and median across all sample-step errors. Distances are reported in meters; no statistics file is written.

Each pipeline independently removes the worst `floor(N * drop_fraction)` reconstructions among its valid samples, ranked by mean vehicle-corner error:

```yaml
- name: filter_action_error
  params:
    drop_fraction: 0.01
```

Set `drop_fraction: 0` to disable this filter. The six pipelines calculate their own removal lists. Final output is sorted by sample ID.

### Prompt templates

The two release templates are [Stage 1](src/prompts/templates/navsim_trajcond_stage1_release.txt) and [Stage 2](src/prompts/templates/navsim_trajcond_stage2_release.txt), paired with [navsim_trajcond_format_release.json](src/prompts/templates/navsim_trajcond_format_release.json).

TXT placeholders describe the human prompt: intent, ego history, task instructions, class lists, and output schema. Actual CoT labels are inserted into the assistant answer. Both stages wrap explanation text in one pair of `<thinking>...</thinking>` tags and action/planning strings in `<answer>...</answer>`.

## Output Format

The default outputs are ShareGPT JSON arrays saved as `data_engine/output/QA_navsim_<method>_stage<1|2>.json`. Each record contains `id`, `image`, an empty `system`, and two `conversations` entries (`human` and `gpt`). The GPT `value` is a serialized JSON object with these ordered fields:

| Stage | Assistant fields, in order |
|-------|----------------------------|
| Stage 1 | `critical_objects`, `explanation`, `meta_behaviour`, `future_trajectory` |
| Stage 2 | `discrete_action_trajectory`, `critical_objects`, `meta_behaviour`, `explanation`, `future_trajectory` |

Stage 1 stores AT in `future_trajectory`. Stage 2 stores AT in `discrete_action_trajectory` and PT in `future_trajectory`. KMeans/KDisks actions use eight adjacent `<action_i>` tokens; Brohan uses eight groups of three tokens. PT contains eight `(x, y, heading)` points formatted to two decimal places.

The writer uses a temporary file and replaces the output after successful completion. The terminal reports retained sample counts and filtering reasons.

## Cache and Validation

Base NAVSIM IR is cached under `data_engine/cache/`. CoT merging and action encoding run again when using the cache. Use `--rebuild-cache` after replacing raw logs in place:

```bash
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --rebuild-cache --num-workers 8
```

Run the search, statistics, and synthetic pipeline checks:

```bash
python -m unittest discover -s data_engine/tests -p test_action_search.py -v
python -m unittest discover -s data_engine/tests -p test_action_error_stats.py -v
python -m unittest discover -s data_engine/tests -p test_trajcond_pipelines.py -v
```

See the [configuration guide](docs/CONFIG.md) and [path reference](docs/PATHS.md) for pipeline parameters and deployment settings.
