"""Load validated codebooks and enforce the search output contract."""

import json
import numpy as np
from ir.schema import ActionPlanIR
from .action_search import build_searcher


def load_searcher(method, codebook_path, variances_path, search=None):
    with open(codebook_path, encoding="utf-8") as stream:
        data = json.load(stream)
    with open(variances_path, encoding="utf-8") as stream:
        variances = json.load(stream)["variances"]
    searcher = build_searcher(method=method, codebook=data["codebook"], variances=variances, **(search or {}))
    expected = searcher.sizes if method == "brohan" else searcher.sizes[0]
    if data.get("k") != expected:
        raise ValueError(f"Codebook k does not match its actual rows: {codebook_path}")
    return searcher


def validate_result(result, method, sizes, sample_id):
    """Reject invalid IDs before a model can coerce floats into integers."""
    if method not in {"kmeans", "kdisks", "brohan"} or len(sizes) != (3 if method == "brohan" else 1) or any(type(k) is not int or k <= 0 for k in sizes):
        raise ValueError(f"{sample_id}: invalid method/codebook sizes")
    tokens = np.asarray(result.token_ids)
    columns = 3 if method == "brohan" else 1
    if tokens.shape != (8, columns) or tokens.dtype.kind not in "iu":
        raise ValueError(f"{sample_id}: invalid integer token matrix")
    if result.index_space not in {"global", "per_axis"}:
        raise ValueError(f"{sample_id}: unknown token index space")
    tokens = tokens.copy()
    offset = 0
    for d, size in enumerate(sizes):
        lower = offset if result.index_space == "global" else 0
        if np.any(tokens[:, d] < lower) or np.any(tokens[:, d] >= lower + size):
            raise ValueError(f"{sample_id}: token outside codebook axis {d}")
        if result.index_space == "per_axis":
            tokens[:, d] += offset
        offset += size
    errors = np.asarray(result.step_errors, dtype=float)
    if errors.shape != (8,) or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError(f"{sample_id}: invalid step errors")
    reconstruction = result.reconstructed_trajectory
    if reconstruction is not None:
        reconstruction = np.asarray(reconstruction, dtype=float)
        if reconstruction.shape != (8, 3) or not np.isfinite(reconstruction).all():
            raise ValueError(f"{sample_id}: invalid reconstructed trajectory")
    return ActionPlanIR(method=method, vocabulary_size=sum(sizes), codebook_sizes=sizes,
                        token_ids=tokens.tolist(), step_errors=errors.tolist(),
                        reconstructed_trajectory=None if reconstruction is None else reconstruction.tolist())
