"""Filter incomplete NAVSIM inputs before path rewriting and expensive search."""

from collections import Counter
from pathlib import Path
from .base import Enricher, register_enricher
from utils.trajcond_contract import select_points


@register_enricher("validate_trajcond_input")
class ValidateTrajcondInput(Enricher):
    reads = ["metadata", "frames", "images", "trajectory"]
    writes = []

    def __init__(self, missing_policy="drop"):
        if missing_policy not in {"drop", "raise"}:
            raise ValueError("missing_policy must be drop or raise")
        self.missing_policy = missing_policy

    def enrich_all(self, samples, pool=None):
        result, counts = [], Counter()
        seen = set()
        for sample in samples:
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate NAVSIM ID: {sample.sample_id}")
            seen.add(sample.sample_id)
            try:
                idx = sample.metadata.current_frame_index
                if idx is None or not 0 <= idx < len(sample.frames):
                    raise ValueError("invalid_current_frame")
                current = sample.frames[idx]
                if sample.metadata.command not in {"turn left", "go straight", "turn right"}:
                    raise ValueError("unknown_intent")
                if len(sample.images) != 1 or sample.images[0] != current.images.get("cam_f0") or not Path(sample.images[0]).is_file():
                    raise ValueError("missing_front_image")
                expected = [i * 0.5 for i in range(-3, 9)]
                sample.trajectory = select_points(sample.trajectory, expected)
                result.append(sample)
            except ValueError as exc:
                if self.missing_policy == "raise":
                    raise ValueError(f"{sample.sample_id}: {exc}") from exc
                counts[str(exc)] += 1
        print(f"NAVSIM input: kept={len(result)}, dropped={dict(counts)}")
        return result
