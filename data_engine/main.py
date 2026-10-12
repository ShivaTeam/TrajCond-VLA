"""CLI for configurable DriveTeach and TrajCond NAVSIM data pipelines."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipelines.config import load_yaml, merge_config, prepare_config
from pipelines.runner import run_pipeline, instantiate_enrichers, preflight
from prompts.base import get_prompt
import prompts.poutine_label
import prompts.prompter
import prompts.planner
import prompts.trajcond_stage1
import prompts.trajcond_stage2


def main():
    parser = argparse.ArgumentParser(description="TrajCond-VLA data engine")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--check-config", action="store_true", help="Validate configuration/resources without extracting NAVSIM")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    defaults = load_yaml(root / "configs" / "defaults.yaml")
    config = merge_config(defaults, load_yaml(args.config))
    config.setdefault("num_workers", defaults.get("runtime", {}).get("num_workers", 1))
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.rebuild_cache:
        config.setdefault("cache", {})["rebuild"] = True
    config = prepare_config(config)
    prompt = config["prompt"]
    builder = get_prompt(prompt) if isinstance(prompt, str) else get_prompt(prompt["name"], prompt.get("params"))
    if args.check_config:
        preflight(config, builder, instantiate_enrichers(config.get("enrichers", [])))
        print("Configuration and required resource paths are valid; NAVSIM data was not extracted.")
    else:
        run_pipeline(config, builder)


if __name__ == "__main__":
    main()
