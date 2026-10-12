"""normalize_trajectory enricher: two-pass z-score normalization."""

import json
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from ir.schema import SampleIR, EgoPoint
from .base import Enricher, register_enricher


@register_enricher("normalize_trajectory")
class NormalizeTrajectoryEnricher(Enricher):
    """Normalize trajectory (x,y,heading) using z-score normalization.

    Two-pass: collect stats across all samples, then apply.
    If stats_file exists, skip collection pass (cached).
    Without stats_file, compute statistics in memory on every run.

    reads:  trajectory
    writes: normalized_trajectory
    """

    reads = ["trajectory"]
    writes = ["normalized_trajectory"]

    def __init__(self, stats_file: Optional[str] = None, num_timesteps: int = 8,
                 source_field="trajectory", output_field="normalized_trajectory",
                 require_existing_stats=False):
        if (source_field, output_field) not in {
            ("trajectory", "normalized_trajectory"),
            ("target_trajectory", "normalized_target_trajectory")
        }:
            raise ValueError("Unsupported trajectory source/output field pair")
        if type(num_timesteps) is not int or num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer")
        self.stats_file = stats_file
        self.num_timesteps = num_timesteps
        self.source_field, self.output_field = source_field, output_field
        self.require_existing_stats = require_existing_stats
        self.reads, self.writes = [source_field], [output_field]
        if require_existing_stats:
            if not stats_file:
                raise ValueError("stats_file is required when require_existing_stats is true")
            self._load_fixed_stats()

    def _load_fixed_stats(self):
        with open(self.stats_file, encoding="utf-8") as stream:
            stats = json.load(stream)
        mean, std = np.asarray(stats["mean"], dtype=float), np.asarray(stats["std"], dtype=float)
        if mean.shape != (self.num_timesteps, 3) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Trajectory stats require finite (T,3) mean and positive std")
        return stats

    def enrich_all(self, samples: list[SampleIR],
                   pool=None) -> list[SampleIR]:
        # Load or compute stats
        if self.require_existing_stats:
            stats = self._load_fixed_stats()
        elif self.stats_file and os.path.exists(self.stats_file):
            with open(self.stats_file) as f:
                stats = json.load(f)
        else:
            stats = self._collect_stats(samples)
            if self.stats_file:
                os.makedirs(os.path.dirname(self.stats_file) or '.', exist_ok=True)
                with open(self.stats_file, 'w') as f:
                    json.dump(stats, f)

        mean = np.array(stats["mean"])   # (T, 3)
        std = np.array(stats["std"])

        # Pass 2: apply normalization (single-threaded — light scalar math)
        self._mean = mean
        self._std = std
        self.trajectory_stats = stats
        for s in tqdm(samples, desc="normalize_trajectory"):
            self._normalize_one(s)
        return samples

    def _normalize_one(self, s: SampleIR) -> SampleIR:
        source = getattr(s, self.source_field)
        if source is None:
            return s
        future_pts = self._future_points(source)
        normalized = []
        for i, pt in enumerate(future_pts):
            if i >= self.num_timesteps:
                break
            nx = (pt.x - self._mean[i][0]) / self._std[i][0]
            ny = (pt.y - self._mean[i][1]) / self._std[i][1]
            nh = (pt.heading - self._mean[i][2]) / self._std[i][2]
            normalized.append(EgoPoint(t=pt.t, x=nx, y=ny, heading=nh))
        setattr(s, self.output_field, normalized)
        return s

    def _future_points(self, source):
        if self.require_existing_stats or self.source_field == "target_trajectory":
            from utils.trajcond_contract import future_points
            return future_points(source)
        return [p for p in source if p.t > 1e-9]

    def _collect_stats(self, samples: list[SampleIR]) -> dict:
        """Collect all future trajectory points for per-timestep stats."""
        all_vals = [[] for _ in range(self.num_timesteps)]
        for s in tqdm(samples, desc="collecting trajectory stats"):
            source = getattr(s, self.source_field)
            if source is None:
                continue
            future_pts = self._future_points(source)
            for i, pt in enumerate(future_pts):
                if i >= self.num_timesteps:
                    break
                all_vals[i].append([pt.x, pt.y, pt.heading])

        mean = []
        std = []
        for vals in all_vals:
            if not vals:
                mean.append([0.0, 0.0, 0.0])
                std.append([1e-6, 1e-6, 1e-6])
                continue
            arr = np.array(vals)
            m = arr.mean(axis=0)
            s = arr.std(axis=0)
            s[s < 1e-6] = 1e-6
            mean.append(m.tolist())
            std.append(s.tolist())

        return {"mean": mean, "std": std}
