"""Encode NAVSIM future poses using a process-local reusable searcher."""

import numpy as np

from .base import Enricher, register_enricher
from utils.action_search import build_searcher
from utils.action_search_adapter import load_searcher, validate_result
from utils.trajcond_contract import future_points

_SEARCHER = None


def report_reconstruction_errors(samples, method):
    """Report validated vehicle corner distances before error filtering."""
    print(f"\n--- Reconstruction error statistics ({method}, before error filtering) ---")
    print(f"Valid searched samples: {len(samples)}")
    if not samples:
        print("No samples available for reconstruction error statistics.")
        return

    errors = np.asarray([s.action_plan.step_errors for s in samples], dtype=np.float64)
    mean_per_step = errors.mean(axis=0)
    median_per_step = np.median(errors, axis=0)
    print(f"{'Time / step':<18} | {'Mean corner error (m)':<22} | {'Median corner error (m)':<24}")
    print("-" * 70)
    for i in range(errors.shape[1]):
        label = f"{(i + 1) * 0.5:.1f}s (Step {i + 1})"
        print(f"{label:<18} | {mean_per_step[i]:<22.6f} | {median_per_step[i]:<24.6f}")
    print("-" * 70)
    # Overall statistics pool all sample-step errors, as in the reference script.
    print(f"Overall mean (all sample-step errors): {errors.mean():.6f} m")
    print(f"Overall median (all sample-step errors): {np.median(errors):.6f} m")


def init_search_worker(params):
    global _SEARCHER
    _SEARCHER = load_searcher(**params)


def encode_one(sample):
    if _SEARCHER is None:
        raise RuntimeError("Search worker was not initialized")
    poses = [[p.x, p.y, p.heading] for p in future_points(sample.trajectory)]
    result = _SEARCHER.encode(poses)
    sample.action_plan = validate_result(result, _SEARCHER.method, _SEARCHER.sizes, sample.sample_id)
    return sample


@register_enricher("encode_trajcond")
class EncodeTrajcond(Enricher):
    reads = ["trajectory"]
    writes = ["action_plan"]

    def __init__(self, method, codebook_path, variances_path, search=None):
        # Validate branch-specific options without loading large resources.
        dummy = [[0], [0], [0]] if method == "brohan" else [[0, 0, 0]]
        build_searcher(method=method, codebook=dummy, variances=[1, 1, 1], **(search or {}))
        self.params = dict(method=method, codebook_path=codebook_path, variances_path=variances_path, search=search)

    def enrich_all(self, samples, pool=None):
        if pool is None:
            init_search_worker(self.params)
            result = [encode_one(s) for s in samples]
        else:
            result = list(pool.imap(encode_one, samples, chunksize=32))
        report_reconstruction_errors(result, self.params["method"])
        return result
