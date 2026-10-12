# Configuration Guide

Run commands from the repository root. This guide covers the six TrajCond NAVSIM pipelines and the inherited DriveTeach configuration interfaces. See [PATHS.md](PATHS.md) for resource locations and path resolution.

## File Structure

```
configs/
  defaults.yaml             # Global defaults (workers, paths, cache)
  datasets/
    navsim.yaml              # Dataset registration (train/test splits)
  pipelines/
    navsim_kmeans_stage1.yaml
    navsim_kmeans_stage2.yaml
    navsim_kdisks_stage1.yaml
    navsim_kdisks_stage2.yaml
    navsim_brohan_stage1.yaml
    navsim_brohan_stage2.yaml
    poutine_label.yaml       # Pipeline: VLM pseudo-labeling input
    prompter.yaml            # Pipeline: DVD pretraining (TGP-Prompter)
    planner.yaml             # Pipeline: CoT-SFT training (TGP-Planner)
```

## Loading and Overrides

The [CLI](../main.py) loads `configs/defaults.yaml`, merges the selected pipeline, applies CLI overrides, then loads the dataset configuration and resolves resource paths.

- Mappings merge recursively; a pipeline's `enrichers` list replaces the entire default list.
- `dataset.name: navsim` loads [configs/datasets/navsim.yaml](../configs/datasets/navsim.yaml) when the pipeline does not already provide `splits` or `data_path`. Pipeline dataset fields override fields from that file. `dataset.config_path` can select another dataset YAML.
- Top-level `num_workers` takes precedence over `runtime.num_workers`; `--num-workers` overrides both. The six TrajCond YAMLs explicitly use `num_workers: 1`.
- `--rebuild-cache` sets `cache.rebuild: true` for the current run.
- `path_base: project_root` resolves resource paths from the repository root. The path passed to `--config` is still opened relative to the current working directory.

| Common field | TrajCond default | Behavior |
| --- | --- | --- |
| `dataset.name` / `dataset.split` | `navsim` / `train` | Select the adapter and dataset split |
| `path_base` | `project_root` | Resolve resource and output paths from the repository root |
| `num_workers` | `1` | Positive integer; workers reuse a process-local action searcher |
| `render_errors` | `raise` | Stop on rendering failure and preserve the previous output file |
| `cache.enabled` | `true` | Reuse the base IR cache when available |
| `cache.rebuild` | Omitted (`false`) | Re-extract base IR when enabled |
| `paths.cache_dir` | `data_engine/cache` | Store the NAVSIM IR before enrichment |
| `output.format` | `json` | Accept `json` or `jsonl` |
| `output.chunk_size` | Omitted | Write one file; a positive integer enables numbered shards |

`paths.stats_dir` and `paths.output_dir` remain inherited defaults. They do not automatically set `normalize_trajectory.stats_file` or `output.path`; configure those fields directly when needed.

## Dataset Configuration

`configs/datasets/navsim.yaml` defines train/test splits. SceneFilter params (frame_interval, num_history_frames, log_names, has_route) are read from the referenced YAML files via `log_names_file`.

```yaml
name: navsim

splits:
  train:
    data_path: "/path/to/datasets/navsim/trainval_logs/trainval"
    sensor_blobs_path: "/path/to/datasets/navsim/trainval_sensor_blobs/trainval"
    log_names_file: "/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"

  test:
    data_path: "/path/to/datasets/navsim/test_navsim_logs/test"
    sensor_blobs_path: "/path/to/datasets/navsim/test_sensor_blobs/test"
    log_names_file: "/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
```

The adapter tries Hydra/OmegaConf instantiation first. If those packages are unavailable, it extracts the SceneFilter parameters manually. Use the original NAVSIM scene filters; synthetic scenes are not supported by these training pipelines.

TrajCond selects four ego history poses at `-1.5, -1.0, -0.5, 0` seconds and eight future poses at `0.5, ..., 4.0` seconds. Time labels come from frame offsets at 2 Hz, rather than capture timestamps. The current front-view image must exist before image paths are rewritten.

## Pipeline Configuration

Each pipeline YAML defines: **dataset**, **enrichers** (ordered chain), **prompt** (registered `@register_prompt` name), **output** (format + path).

### TrajCond Pipelines

| Representation | Stage 1 configuration | Stage 2 configuration |
| --- | --- | --- |
| KMeans | [navsim_kmeans_stage1.yaml](../configs/pipelines/navsim_kmeans_stage1.yaml) | [navsim_kmeans_stage2.yaml](../configs/pipelines/navsim_kmeans_stage2.yaml) |
| KDisks | [navsim_kdisks_stage1.yaml](../configs/pipelines/navsim_kdisks_stage1.yaml) | [navsim_kdisks_stage2.yaml](../configs/pipelines/navsim_kdisks_stage2.yaml) |
| Brohan | [navsim_brohan_stage1.yaml](../configs/pipelines/navsim_brohan_stage1.yaml) | [navsim_brohan_stage2.yaml](../configs/pipelines/navsim_brohan_stage2.yaml) |

Both stages encode action tokens (AT) from NAVSIM future poses and reuse the same three CoT fields for each sample ID. Stage 1 renders `CoT → AT`. Stage 2 renders `AT → CoT → PT`.

The shared enrichment steps and second-stage normalization are:

```text
Stage 1:
  validate_trajcond_input → replace_image_base
  → merge_pseudo_labels → filter_incomplete_pseudo
  → encode_trajcond → filter_action_error

Stage 2:
  validate_trajcond_input → replace_image_base
  → merge_pseudo_labels → filter_incomplete_pseudo
  → encode_trajcond → filter_action_error → normalize_trajectory
```

### Complete Stage 1 Example

This example matches the shipped Brohan Stage 1 configuration:

```yaml
dataset:
  name: navsim
  split: train
path_base: project_root
render_errors: raise
num_workers: 1
paths:
  cache_dir: data_engine/cache
cache:
  enabled: true
enrichers:
- name: validate_trajcond_input
  params:
    missing_policy: drop
- name: replace_image_base
  params:
    old_base: /path/to/datasets/navsim
    new_base: navsim
- name: merge_pseudo_labels
  params:
    source_path: data_engine/annotations/cot.jsonl
    parser: poutine
    key_field: id
    json_repair: true
    strict_ids: true
- name: filter_incomplete_pseudo
  params:
    require_meta_behaviour: true
    validate_classes: true
    require_image: true
- name: encode_trajcond
  params:
    method: brohan
    codebook_path: codebook/brohan_codebook_1d_size_768.json
    variances_path: codebook/action_variances.json
    search:
      beam_size: 1
      per_axis_top_k: 4
      vehicle:
        width: 2.297
        front_bumper_to_axle: 4.049
        rear_bumper_to_axle: -1.127
- name: filter_action_error
  params:
    drop_fraction: 0.01
prompt:
  name: trajcond_stage1
  params:
    template_path: data_engine/src/prompts/templates/navsim_trajcond_stage1_release.txt
    format_contract_path: data_engine/src/prompts/templates/navsim_trajcond_format_release.json
    decimals: 2
output:
  format: json
  path: data_engine/output/QA_navsim_brohan_stage1.json
```

### Stage 2 Normalization

Use the corresponding Stage 2 YAML from the table above. Its `normalize_trajectory.params` uses `num_timesteps: 8` and `require_existing_stats: false`, with no `stats_file` by default.

After action-error filtering, each pipeline computes per-timestep `(8,3)` mean and population standard deviation across its retained samples. Standard deviation is clamped to `1e-6`, and normalization uses `(value - mean) / std`. Each algorithm can retain different samples and therefore produce different statistics.

`stats_file` is optional. If supplied and already present, it is reused even when `require_existing_stats: false`; otherwise it is created from the current samples. Setting `require_existing_stats: true` requires an existing file with finite `(8,3)` mean and strictly positive finite std.

### CoT Import and Input Filtering

`merge_pseudo_labels.source_path` points to one JSONL shared by all six pipelines. Each outer record has `id` and a `predict` string containing a JSON object with `critical_objects`, `meta_behaviour`, and `explanation`.

`parser: poutine` selects the existing three-field parser. TrajCond requires `strict_ids: true` and all three `filter_incomplete_pseudo` flags to be true. Missing or invalid CoT is filtered before search. `critical_objects` must contain the twelve classes in the [release format contract](../src/prompts/templates/navsim_trajcond_format_release.json), `meta_behaviour` must contain valid `speed` and `command` labels, and explanation text must be nonempty with valid thinking tags. Both stages render one pair of `<thinking>...</thinking>` tags.

`validate_trajcond_input` checks sample IDs, current front-view image, driving intent, and the required history/future poses. `missing_policy` accepts `drop` or `raise`; duplicate NAVSIM IDs always stop the run. These are sample checks during extraction/enrichment, rather than checks performed by `--check-config`.

### Search Parameters

Configure these fields under `encode_trajcond.params.search`:

| Parameter | KMeans / KDisks | Brohan | Meaning |
| --- | --- | --- | --- |
| `beam_size` | `4` | `1` only | Number of reconstructed action paths retained after each step |
| `candidate_top_k` | `80` | Omit | Nearest joint centers considered per beam path |
| `per_axis_top_k` | Omit | `4` | Nearest values considered on each axis before forming combinations |
| `vehicle.width` | `2.297` | `2.297` | Vehicle width in meters |
| `vehicle.front_bumper_to_axle` | `4.049` | `4.049` | Front extent from the pose origin in meters |
| `vehicle.rear_bumper_to_axle` | `-1.127` | `-1.127` | Rear extent from the pose origin in meters |

Candidate counts and beam size must be positive integers. Shortlists are limited to the available codebook entries. Brohan forms up to `4**3 = 64` combinations per step with the default axis settings. Joint beam paths are ranked by cumulative vehicle-corner error.

`codebook_path` selects the codebook; its actual entries determine vocabulary size. Joint codebooks use integer `k` and a `(k,3)` array. Brohan uses three axis arrays and `k=[nx,ny,nyaw]`, with total token size equal to their sum. `variances_path` contains three positive finite local-action variances. Joint search uses them to weight candidate shortlisting; Brohan preserves its unweighted axis selection. These variances do not supply PT normalization mean/std.

### Error Filtering and Reports

Each of the six pipelines removes `floor(N * drop_fraction)` samples with the highest mean corner reconstruction error from its own valid sample set. `drop_fraction` defaults to `0.01`, accepts finite values in `[0,1)`, and can be set to `0` to disable removal. Ties at the removal boundary use sample IDs in ascending order.

After all valid samples have been searched, and before this filter, the parent process prints the searched sample count, the mean and median error at each of eight future steps, and the overall mean and median over all `(sample, step)` errors. Distances use meters and six decimal places. Serial and multiprocess runs report once; no error-statistics file is produced.

### Prompts and Output Metadata

The two TrajCond prompt registrations are `trajcond_stage1` and `trajcond_stage2`. Their `params` select the matching release TXT and `navsim_trajcond_format_release.json`; `decimals` must be `2`. Only the release format is supported. Templates must match the canonical release files, use `${name}` placeholders and LF newlines, and contain exactly one final newline. The format contract controls class lists, answer ordering, tag depth, and action formatting as well as validation.

Output is sorted by sample ID. Joint AT uses eight adjacent action tokens; Brohan AT uses eight groups of three tokens. Stage 2 PT contains eight tuples with two decimal places. The Stage 2 writer adds `trajectory_stats` as the first top-level field of the first complete QA sample, containing the statistics actually used for normalization. Default computed statistics contain full-precision `mean` and `std`. Subsequent samples omit that field; each numbered shard starts with the same dataset-wide statistics. Keep this field outside conversations and outside LLaMA-Factory's column mapping.

The writer preserves the existing output file if rendering or serialization fails. An empty valid TrajCond sample set stops before replacing output. See [PATHS.md](PATHS.md) for output filenames, cache behavior, and deployment settings.

### Legacy Poutine Label (`poutine_label`)

VLM pseudo-labeling input. Includes expert future trajectory and velocity for the VLM to label.

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}

prompt: poutine_label

output:
  format: json
  path: "./output/QA_navsim_train_poutine_label.json"
```

### Legacy Prompter (`prompter`)

DVD pretraining — TGP-Prompter 2D-TGP regression.

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: inject_bbox_path
    params: {old_base: "/path/to/datasets/navsim", new_base: "boxed_navsim"}
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}
  - name: normalize_trajectory
    params: {stats_file: "./stats/navsim_train_stats.json"}
  - name: project_2d_tgp
    params: {image_height: 1080, image_width: 1920, target_height: 448, target_width: 784}

prompt: prompter

output:
  format: json
  path: "./output/QA_navsim_train_prompter.json"
```

### Legacy Planner (`planner`)

CoT-SFT — TGP-Planner with CoT reasoning and normalized BEV trajectory.

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}
  - name: normalize_trajectory
    params: {stats_file: "./stats/navsim_train_stats.json"}
  - name: project_2d_tgp
    params: {image_height: 1080, image_width: 1920, target_height: 448, target_width: 784}
  - name: merge_pseudo_labels
    params: {source_path: "/path/to/vlm_output.jsonl", parser: poutine, key_field: id, json_repair: true}
  - filter_incomplete_pseudo

prompt: planner

output:
  format: json
  path: "./output/QA_navsim_train_planner.json"
```

## Enricher Reference

| Enricher | reads | writes | Key params |
|----------|-------|--------|------------|
| `replace_image_base` | images | images | `old_base`, `new_base` |
| `validate_trajcond_input` | metadata, frames, images, trajectory | — | `missing_policy: drop` |
| `encode_trajcond` | trajectory | action_plan | `method`, `codebook_path`, `variances_path`, `search` |
| `filter_action_error` | action_plan | — | `drop_fraction: 0.01` |
| `normalize_trajectory` | Configured `source_field` | Configured `output_field` | `stats_file` (optional), `num_timesteps: 8`, `require_existing_stats: false` |
| `inject_bbox_path` | images | bbox_image | `old_base`, `new_base` |
| `project_2d_tgp` | trajectory, normalized_trajectory, frames | trajectory_2d_tgp | `image_height`, `image_width`, `target_height`, `target_width` |
| `merge_pseudo_labels` | — | pseudo_label | `source_path`, `parser`, `key_field`, `json_repair`, `strict_ids` |
| `filter_incomplete_pseudo` | pseudo_label | — | `require_meta_behaviour`, `validate_classes`, `require_image` |

The runner checks declared reads/writes and required TrajCond ordering before extraction. Filtering enrichers remove samples without introducing a new field.

## PromptBuilder Reference

| Builder | Registered name | Purpose |
|---------|----------------|---------|
| `TrajcondStage1Builder` | `trajcond_stage1` | CoT followed by NAVSIM AT |
| `TrajcondStage2Builder` | `trajcond_stage2` | NAVSIM AT followed by CoT and continuous PT |
| `PoutineLabelBuilder` | `poutine_label` | VLM pseudo-labeling input |
| `PrompterBuilder` | `prompter` | DVD pretraining (2D-TGP regression) |
| `PlannerBuilder` | `planner` | CoT-SFT (CoT + normalized BEV) |

All builders use `@register_prompt(name)` decorator and follow the same pattern:

```python
@register_prompt("my_prompt")
class MyPromptBuilder(PromptBuilder):
    def build(self, sample: SampleIR) -> dict:
        return {"id": ..., "image": ..., "system": ..., "conversations": [...]}
```

## CLI

```bash
# Check configuration and required resources without extracting NAVSIM
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --check-config

# Build the two stages
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --num-workers 8
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage2.yaml --num-workers 8

# Rebuild base IR after replacing raw logs
python data_engine/main.py --config data_engine/configs/pipelines/navsim_kdisks_stage2.yaml --rebuild-cache --num-workers 8
```

`--check-config` validates the processing chain, prompt profile, required paths, codebook/variance contents, and fixed normalization statistics when explicitly required. It does not load NAVSIM samples, parse every CoT record, search actions, or compute new normalization statistics.
