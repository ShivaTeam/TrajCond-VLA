"""Extract base IR, run validated enrichment, then stream QA output."""

import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import tempfile
import time
import sys
from tqdm import tqdm
from ir.builder import build_sample_ir
from datasets.registry import get_adapter_class
from enrichers.base import get_enricher
from enrichers.encode_trajcond import EncodeTrajcond, init_search_worker
from utils.action_search_adapter import load_searcher
from utils.trajectory import compute_sample_trajectory
from writers.json_writer import write_output
from .config import prepare_config

for module in ("normalize_trajectory", "project_2d_tgp", "inject_bbox_path", "merge_pseudo_labels",
               "filter_incomplete", "replace_image_base", "validate_trajcond_input",
               "merge_gt_trajectories", "encode_trajcond", "filter_action_error"):
    importlib.import_module(f"enrichers.{module}")


def instantiate_enrichers(configs):
    return [get_enricher(ec, {}) if isinstance(ec, str) else get_enricher(ec["name"], ec.get("params", {})) for ec in configs]


def validate_enricher_chain(enricher_configs, base_fields, instances=None):
    instances = instances if instances is not None else instantiate_enrichers(enricher_configs)
    produced = set(base_fields)
    for enricher in instances:
        missing = set(enricher.reads) - produced
        if missing:
            raise ValueError(f"Enricher '{enricher.name}' requires missing fields: {sorted(missing)}")
        produced.update(enricher.writes)
    return produced


def base_cache_key(ds_config, strict):
    split = ds_config.get("split", "train")
    split_cfg = ds_config.get("splits", {}).get(split, ds_config)
    payload = dict(schema_version=2, dataset=ds_config, strict=strict,
                   window=[-2.0, 4.0, 0.5])
    source_files = [Path(__file__).parents[1] / p for p in
                    ("ir/schema.py", "ir/builder.py", "utils/trajectory.py")]
    adapter = Path(__file__).parents[1] / "datasets" / ds_config["name"] / "adapter.py"
    if adapter.exists():
        source_files.append(adapter)
    payload["sources"] = {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    scene_filter = split_cfg.get("log_names_file")
    if scene_filter:
        filter_path = Path(scene_filter)
        if not filter_path.is_absolute() and not strict:
            filter_path = Path(__file__).parents[2] / filter_path
        payload["scene_filter_sha256"] = hashlib.sha256(filter_path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def preflight(config, builder, enrichers):
    strict = getattr(builder, "stage", None) in {1, 2}
    produced = validate_enricher_chain(config.get("enrichers", []),
                                     {"sample_id", "metadata", "frames", "images", "trajectory"}, enrichers)
    if strict:
        required = {"pseudo_label", "action_plan"}
        if builder.stage == 2:
            required.add("normalized_target_trajectory")
        if not required <= produced:
            raise ValueError(f"Prompt requires missing IR fields: {required - produced}")
        names = [e.name for e in enrichers]
        required_chain = ["validate_trajcond_input", "merge_pseudo_labels", "filter_incomplete_pseudo",
                          "encode_trajcond", "filter_action_error"]
        if any(names.count(name) != 1 for name in required_chain):
            raise ValueError("TrajCond requires one input check, CoT join/filter, encoder, and error filter")
        if [names.index(name) for name in required_chain] != sorted(names.index(name) for name in required_chain):
            raise ValueError("TrajCond input and CoT checks must precede search and ranking")
        if "replace_image_base" in names and names.index("replace_image_base") < names.index("validate_trajcond_input"):
            raise ValueError("Local input images must be checked before path rewriting")
        if names.count("encode_trajcond") != 1 or names.count("filter_action_error") != 1:
            raise ValueError("TrajCond requires one action encoder and one error filter")
        if names.index("filter_action_error") < names.index("encode_trajcond"):
            raise ValueError("Error filtering must follow action search")
        if builder.stage == 2 and names.index("merge_gt_trajectories") > names.index("encode_trajcond"):
            raise ValueError("PKL validity must be checked before action search")
        if builder.stage == 1 and "merge_gt_trajectories" in names:
            raise ValueError("Stage 1 must not depend on PKL targets")
        for enricher in enrichers:
            if enricher.name == "merge_pseudo_labels" and not enricher.strict_ids:
                raise ValueError("TrajCond requires strict CoT IDs")
            if enricher.name == "filter_incomplete_pseudo" and not all(
                [enricher.require_meta_behaviour, enricher.validate_classes, enricher.require_image]
            ):
                raise ValueError("TrajCond requires complete CoT and image checks")
            if builder.stage == 2 and enricher.name == "normalize_trajectory":
                if enricher.num_timesteps != 8 or enricher.source_field != "target_trajectory":
                    raise ValueError("Stage 2 requires eight-step target trajectory normalization")
            for key in ("source_path", "stats_file"):
                if key == "stats_file" and not getattr(enricher, "require_existing_stats", False):
                    continue
                path = getattr(enricher, key, None)
                if path and not Path(path).is_file():
                    raise FileNotFoundError(f"{enricher.name}: missing {key}: {path}")
            for key in ("codebook_path", "variances_path"):
                path = getattr(enricher, "params", {}).get(key)
                if path and not Path(path).is_file():
                    raise FileNotFoundError(f"{enricher.name}: missing {key}: {path}")
            if isinstance(enricher, EncodeTrajcond):
                load_searcher(**enricher.params)
        if config["dataset"]["name"] == "navsim":
            ds = config["dataset"]
            split_cfg = ds.get("splits", {}).get(ds.get("split", "train"), ds)
            for key in ("data_path", "sensor_blobs_path"):
                if not Path(split_cfg[key]).is_dir():
                    raise FileNotFoundError(f"NAVSIM {key}: {split_cfg[key]}")
            if split_cfg.get("log_names_file") and not Path(split_cfg["log_names_file"]).is_file():
                raise FileNotFoundError(f"NAVSIM SceneFilter: {split_cfg['log_names_file']}")
    output = config["output"]
    if output.get("format", "json") not in {"json", "jsonl"}:
        raise ValueError("Unknown output format")
    chunk_size = output.get("chunk_size")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    return strict


def _cache_write(samples, cache_path):
    target = Path(cache_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ir-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(samples, stream)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_pipeline(pipeline_cfg, prompt_builder):
    config = prepare_config(pipeline_cfg)
    enrichers = instantiate_enrichers(config.get("enrichers", []))
    strict = preflight(config, prompt_builder, enrichers)
    ds_config = config["dataset"]
    workers = config.get("num_workers", 1)
    if not strict and isinstance(workers, int) and workers <= 0:
        workers = 1
    if type(workers) is not int or workers < 1:
        raise ValueError("num_workers must be a positive integer")
    cache_cfg = config.get("cache", {})
    cache_path = None
    if cache_cfg.get("enabled", True):
        cache_dir = config.get("paths", {}).get("cache_dir", "./cache")
        cache_path = str(Path(cache_dir) / f"{ds_config['name']}_{ds_config.get('split', 'train')}_{base_cache_key(ds_config, strict)}.pkl")
    encoder = next((e for e in enrichers if isinstance(e, EncodeTrajcond)), None)
    context = mp.get_context("spawn" if os.name == "nt" else "fork")
    pool = context.Pool(workers, initializer=init_search_worker if encoder else None,
                        initargs=(encoder.params,) if encoder else ()) if workers > 1 else None
    started = time.time()
    try:
        if cache_path and Path(cache_path).is_file() and not cache_cfg.get("rebuild", False):
            with open(cache_path, "rb") as stream:
                samples = pickle.load(stream)
            print(f"Loaded {len(samples)} base IR records from {cache_path}")
        else:
            if ds_config["name"] == "navsim":
                importlib.import_module("datasets.navsim.adapter")
            adapter = get_adapter_class(ds_config["name"])(ds_config)
            samples = [build_sample_ir(adapter.extract_sample(i)) for i in tqdm(range(adapter.get_sample_count()), desc="Extract")]
            processed = (pool.imap(compute_sample_trajectory, samples, chunksize=50)
                         if pool else map(compute_sample_trajectory, samples))
            samples = [s for s in processed if s is not None]
            if cache_path:
                _cache_write(samples, cache_path)
        for enricher in enrichers:
            samples = [s for s in enricher.enrich_all(samples, pool) if s is not None]
            print(f"{enricher.name}: {len(samples)} retained")
        if strict and not samples:
            raise ValueError("No valid samples remain; output was not replaced")
        if strict:
            samples.sort(key=lambda sample: sample.sample_id)

        first_record_fields = None
        if getattr(prompt_builder, "stage", None) == 2:
            normalizer = next((e for e in enrichers if e.name == "normalize_trajectory"
                               and e.output_field == "normalized_target_trajectory"), None)
            if normalizer is not None:
                first_record_fields = {"trajectory_stats": normalizer.trajectory_stats}

        def render():
            for sample in tqdm(samples, desc="Render"):
                try:
                    record = prompt_builder.build(sample)
                    if record is None and strict:
                        raise ValueError("Builder returned no QA record")
                    if record is not None:
                        yield record
                except Exception as exc:
                    if config.get("render_errors", "raise" if strict else "skip") == "raise":
                        raise ValueError(f"Render failed for {sample.sample_id}: {exc}") from exc
                    print(f"Skipped render {sample.sample_id}: {exc}")
        count = write_output(render(), config["output"], first_record_fields=first_record_fields)
        print(f"Output: {count} QA records -> {config['output']['path']} ({time.time()-started:.1f}s)")
        return count
    finally:
        if pool:
            if sys.exc_info()[0] is not None:
                pool.terminate()
            else:
                pool.close()
            pool.join()
