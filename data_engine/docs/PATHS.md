# Configurable Paths Reference

This guide describes resource paths for the six TrajCond pipelines. Keys identify their location in YAML; enricher paths are listed by registered name rather than list index. See [CONFIG.md](CONFIG.md) for parameter meanings and processing order.

## Path Resolution

All six TrajCond YAMLs set `path_base: project_root`. The [configuration loader](../src/pipelines/config.py) derives the repository root from its own location and resolves recognized resource fields after loading the complete dataset configuration.

| Value | Resolution |
| --- | --- |
| Absolute resource path, such as `/data/navsim/trainval_logs/trainval` | Keep the absolute location |
| Relative resource path, such as `codebook/action_variances.json` | Resolve from the repository root |
| CLI `--config data_engine/configs/pipelines/navsim_brohan_stage1.yaml` | Open relative to the current working directory |
| `replace_image_base.params.old_base` / `new_base` | Keep literal strings for image-path replacement |
| Configuration without `path_base` | Keep inherited path behavior; use absolute resource paths when deploying |

Recognized fields include `data_path`, `sensor_blobs_path`, `log_names_file`, `config_path`, `source_path`, `codebook_path`, `variances_path`, `stats_file`, `template_path`, `format_contract_path`, `cache_dir`, and `output_dir`. `output.path` is resolved separately by the same rule. Relative `dataset.config_path` also uses the repository root when `path_base: project_root` is enabled.

Only `project_root` is accepted when `path_base` is supplied. Paths are normalized by the local operating system; leading whitespace, `~`, and environment-variable references are not expanded or stripped by the loader. Use the actual path string for the machine running the engine.

## NAVSIM Dataset Paths

[configs/datasets/navsim.yaml](../configs/datasets/navsim.yaml) supplies the default dataset configuration. `dataset.name: navsim` and `dataset.split: train` select its training split. Set the three training resources before running any of the six pipelines.

| Key | Shipped placeholder | Resource |
| --- | --- | --- |
| `splits.train.data_path` | `/path/to/datasets/navsim/trainval_logs/trainval` | Training raw-log directory |
| `splits.train.sensor_blobs_path` | `/path/to/datasets/navsim/trainval_sensor_blobs/trainval` | Training sensor blobs containing camera images |
| `splits.train.log_names_file` | `/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml` | Training SceneFilter YAML |
| `splits.test.data_path` | `/path/to/datasets/navsim/test_navsim_logs/test` | Test raw-log directory |
| `splits.test.sensor_blobs_path` | `/path/to/datasets/navsim/test_sensor_blobs/test` | Test sensor blobs |
| `splits.test.log_names_file` | `/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml` | Test SceneFilter YAML |

Only the selected split's resources are required by preflight. The shipped TrajCond configurations select `train`. SceneFilter settings are read from the referenced YAML; synthetic scenes must remain disabled.

The adapter combines `sensor_blobs_path` with camera `data_path` from the raw frame. For example, a camera path `log/frame.jpg` under sensor root `/data/navsim/trainval_sensor_blobs/trainval` becomes `/data/navsim/trainval_sensor_blobs/trainval/log/frame.jpg`. The current front-view file must exist at this extraction-time location.

To use another dataset YAML, set `dataset.config_path`, for example:

```yaml
dataset:
  name: navsim
  split: train
  config_path: data_engine/configs/datasets/navsim.yaml
path_base: project_root
```

## TrajCond Pipeline Resources

The following values are shared by all six shipped pipeline files unless a stage restriction is shown. `name.params.key` denotes the enricher with that registered name, wherever it appears in the ordered `enrichers` list.

| Configuration field | Default value | Used by |
| --- | --- | --- |
| `merge_pseudo_labels.params.source_path` | `data_engine/annotations/cot.jsonl` | Both stages; one JSONL containing ID-matched CoT |
| `encode_trajcond.params.codebook_path` | Representation-specific; see below | Both stages; discrete action vocabulary |
| `encode_trajcond.params.variances_path` | `codebook/action_variances.json` | Both stages; local-action variances for search |
| `prompt.params.format_contract_path` | `data_engine/src/prompts/templates/navsim_trajcond_format_release.json` | Both stages; canonical release format |
| `prompt.params.template_path` | Stage-specific; see below | Human prompt TXT |
| `paths.cache_dir` | `data_engine/cache` | Shared base NAVSIM IR cache |
| `output.path` | Algorithm- and stage-specific; see below | Final training JSON |
| `normalize_trajectory.params.stats_file` | Omitted | Optional Stage 2 statistics file; not needed by default |

CoT IDs match NAVSIM tokens as exact strings; duplicate annotation IDs stop the run.

### Codebooks

| Representation | `codebook_path` | Default vocabulary |
| --- | --- | --- |
| KMeans | `codebook/kmeans_codebook_768.json` | 768 joint tokens |
| KDisks | `codebook/kdisks_codebook_768.json` | 768 joint tokens |
| Brohan | `codebook/brohan_codebook_1d_size_768.json` | 256 tokens per axis; 768 in total |

Use the same method, codebook, variance file, and search settings for a representation's two stages. If a codebook is stored elsewhere, update `codebook_path` and `variances_path` in both selected YAMLs. The filename does not determine vocabulary size; the loader verifies `k` against the actual entries. See [the codebook guide](../../codebook/README.md) for construction commands.

### Prompt Templates

| Stage | `prompt.name` | `template_path` |
| --- | --- | --- |
| 1 | `trajcond_stage1` | `data_engine/src/prompts/templates/navsim_trajcond_stage1_release.txt` |
| 2 | `trajcond_stage2` | `data_engine/src/prompts/templates/navsim_trajcond_stage2_release.txt` |

Both pair with the release format contract listed above. The builder supports only the release format and checks canonical template bytes and contract contents. A different path is accepted only when its contents match the corresponding release file.

### Output Files

| Pipeline | `output.path` |
| --- | --- |
| [navsim_kmeans_stage1.yaml](../configs/pipelines/navsim_kmeans_stage1.yaml) | `data_engine/output/QA_navsim_kmeans_stage1.json` |
| [navsim_kmeans_stage2.yaml](../configs/pipelines/navsim_kmeans_stage2.yaml) | `data_engine/output/QA_navsim_kmeans_stage2.json` |
| [navsim_kdisks_stage1.yaml](../configs/pipelines/navsim_kdisks_stage1.yaml) | `data_engine/output/QA_navsim_kdisks_stage1.json` |
| [navsim_kdisks_stage2.yaml](../configs/pipelines/navsim_kdisks_stage2.yaml) | `data_engine/output/QA_navsim_kdisks_stage2.json` |
| [navsim_brohan_stage1.yaml](../configs/pipelines/navsim_brohan_stage1.yaml) | `data_engine/output/QA_navsim_brohan_stage1.json` |
| [navsim_brohan_stage2.yaml](../configs/pipelines/navsim_brohan_stage2.yaml) | `data_engine/output/QA_navsim_brohan_stage2.json` |

The writer creates output directories as needed. `output.format` accepts `json` or `jsonl`; choose a matching filename extension when changing formats. Without `output.chunk_size`, one file is written atomically through a temporary file in the same directory. With a positive chunk size, files use `_1`, `_2`, and subsequent suffixes before the extension, and each shard is replaced independently.

Each Stage 2 file stores the actual normalization statistics in its first complete QA sample's first top-level field, `trajectory_stats`. Computed statistics contain full-precision `(8,3)` `mean` and `std`. Numbered shards carry the same dataset-wide statistics in each first record. Default pipelines do not write an independent statistics file; reports of action reconstruction errors are printed only.

If `output.path` or sharding changes, update the corresponding `file_name` entries in [sft/datasets/dataset_info.json](../../sft/datasets/dataset_info.json) before training.

### Optional Normalization Statistics File

The three shipped Stage 2 YAMLs omit `stats_file` and use `require_existing_stats: false`. Statistics are computed across each pipeline's retained samples and saved inside its QA output.

To reuse an existing statistics file, set these fields in the selected Stage 2 normalizer's `params`:

```yaml
num_timesteps: 8
stats_file: data_engine/stats/custom_stage2.json
require_existing_stats: true
```

This is an optional example, not a default resource. With `require_existing_stats: true`, the file must already exist and contain valid `(8,3)` mean/std. With `false` and an explicit `stats_file`, an existing file is still reused; a missing file is computed and written. The output metadata stores the statistics actually used in either case.

## Image Paths for Training

`replace_image_base` applies `path.replace(old_base, new_base)` to each entry in `SampleIR.images`. Both parameters are literal strings, independent of `path_base`. Set `old_base` to text matching the extraction-time sensor location, and `new_base` to the location where training can read those images.

For example:

```yaml
- name: replace_image_base
  params:
    old_base: /data/navsim
    new_base: /mnt/training/navsim
```

This transforms `/data/navsim/trainval_sensor_blobs/trainval/log/frame.jpg` into `/mnt/training/navsim/trainval_sensor_blobs/trainval/log/frame.jpg`. It rewrites the JSON image path without copying image files. Input validation checks the original image before this replacement; the replacement location must be available in the training environment.

The shipped `new_base: navsim` produces relative image paths. The SFT configuration has no `media_dir`; relative paths are interpreted from LLaMA-Factory's `dataset_dir` by default. Use an absolute `new_base` when the image root is elsewhere. An absolute target also avoids depending on the training command's working directory.

## Defaults and Cache Paths

[configs/defaults.yaml](../configs/defaults.yaml) contains `paths.cache_dir: ./cache`, `paths.stats_dir: ./stats`, `paths.output_dir: ./output`, `cache.enabled: true`, and `runtime.num_workers: 1`. The TrajCond YAMLs override the cache directory with `data_engine/cache` and define `output.path` explicitly. `paths.output_dir` does not construct the output filename, and `paths.stats_dir` does not set or create a normalizer statistics file.

Cache keys include dataset settings, the SceneFilter content, strict mode, and relevant IR/adapter/trajectory source hashes. The cache contains base IR before enrichment. Pipelines using the same NAVSIM configuration can share it; their enrichment still runs independently.

Use `--rebuild-cache` after replacing raw logs in place, because the key does not hash every raw log or image file. Changing a CoT file or codebook does not require rebuilding base IR.

## Deployment and Preflight

Before running a TrajCond pipeline, configure these resources:

| File | Field | Deployment setting |
| --- | --- | --- |
| `configs/datasets/navsim.yaml` | Selected split's `data_path`, `sensor_blobs_path`, `log_names_file` | Actual NAVSIM directories and SceneFilter YAML |
| Selected pipeline YAML | `merge_pseudo_labels.params.source_path` | Existing three-field CoT JSONL |
| Selected pipeline YAML | `encode_trajcond.params.codebook_path` / `variances_path` | Existing codebook and local-action variance JSON |
| Selected pipeline YAML | `replace_image_base.params.old_base` / `new_base` | Extraction-time text and training image location |
| Selected pipeline YAML | `paths.cache_dir` / `output.path` | Writable cache directory and final output location |

Run from the repository root:

```bash
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --check-config
python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage2.yaml --check-config
```

Preflight checks selected NAVSIM directories, SceneFilter and required resource paths, codebook/variance contents, processing order, prompt files, output options, and required fixed statistics. It does not extract NAVSIM samples or validate every CoT entry. Image existence and exact CoT ID matching are checked during the actual run; output writability is exercised when writing.

## Inherited Pipeline Paths

The original configurations remain available with their inherited path behavior. Use the enricher names below rather than positional list indices.

| Pipeline | Resource fields | Shipped values |
| --- | --- | --- |
| `poutine_label.yaml` | `replace_image_base.params.old_base` / `new_base`; `output.path` | `/path/to/datasets/navsim` / `navsim`; `./output/QA_navsim_train_poutine_label.json` |
| `prompter.yaml` | `inject_bbox_path.params.old_base` / `new_base`; `replace_image_base.params.old_base` / `new_base` | `/path/to/datasets/navsim` / `boxed_navsim`; `/path/to/datasets/navsim` / `navsim` |
| `prompter.yaml` | `normalize_trajectory.params.stats_file`; `output.path` | `./stats/navsim_train_stats.json`; `./output/QA_navsim_train_prompter.json` |
| `planner.yaml` | `replace_image_base.params.old_base` / `new_base`; `normalize_trajectory.params.stats_file` | `/path/to/datasets/navsim` / `navsim`; `./stats/navsim_train_stats.json` |
| `planner.yaml` | `merge_pseudo_labels.params.source_path`; `output.path` | `/path/to/VLA_train/parallel_infer/output/poutine_labels_infer_output.jsonl`; `./output/QA_navsim_train_planner.json` |

Legacy projection dimensions are numeric parameters rather than resource paths; see [CONFIG.md](CONFIG.md) for their role. The inherited pipelines do not enable TrajCond's `project_root` path mode. Absolute resource paths avoid differences between working-directory resolution and the legacy SceneFilter lookup relative to the engine directory.
