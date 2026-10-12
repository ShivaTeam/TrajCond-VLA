"""Join physical PKL targets by exact ID; never substitute NAVSIM targets."""

from collections import Counter
import pickle
import numpy as np
from ir.schema import EgoPoint
from .base import Enricher, register_enricher


@register_enricher("merge_gt_trajectories")
class MergeGtTrajectories(Enricher):
    reads = ["sample_id"]
    writes = ["target_trajectory"]

    def __init__(self, source_path, key_field="id", trajectory_key="gt", missing_policy="drop"):
        if missing_policy not in {"drop", "raise"}:
            raise ValueError("missing_policy must be drop or raise")
        self.source_path, self.key_field, self.trajectory_key = source_path, key_field, trajectory_key
        self.missing_policy = missing_policy
        self.index = None

    def _load(self):
        with open(self.source_path, "rb") as stream:
            records = pickle.load(stream)
        if not isinstance(records, list):
            raise ValueError(f"{self.source_path}: PKL must contain a list of records")
        self.index = {}
        for i, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get(self.key_field), str) or not record[self.key_field]:
                raise ValueError(f"{self.source_path}: malformed record {i}")
            key = record[self.key_field]
            if key in self.index:
                raise ValueError(f"{self.source_path}: duplicate ID {key}")
            self.index[key] = record.get(self.trajectory_key)

    def enrich_all(self, samples, pool=None):
        if self.index is None:
            self._load()
        result, counts = [], Counter()
        for sample in samples:
            try:
                if sample.sample_id not in self.index:
                    raise ValueError("missing_pkl_id")
                try:
                    poses = np.asarray(self.index[sample.sample_id], dtype=float)
                except (ValueError, TypeError) as exc:
                    raise ValueError("invalid_pkl_gt") from exc
                if poses.shape != (8, 3) or not np.isfinite(poses).all():
                    raise ValueError("invalid_pkl_gt")
                sample.target_trajectory = [EgoPoint(t=(i+1)*0.5, x=p[0], y=p[1], heading=p[2]) for i, p in enumerate(poses)]
                result.append(sample)
            except ValueError as exc:
                if self.missing_policy == "raise":
                    raise ValueError(f"{sample.sample_id}: {exc}") from exc
                counts[str(exc)] += 1
        print(f"PKL join: kept={len(result)}, dropped={dict(counts)}")
        return result
