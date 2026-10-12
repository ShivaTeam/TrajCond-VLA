"""Independently remove the worst floor(N*r) action reconstructions."""

import math
import numpy as np
from .base import Enricher, register_enricher


@register_enricher("filter_action_error")
class FilterActionError(Enricher):
    reads = ["action_plan"]
    writes = []

    def __init__(self, drop_fraction=0.01):
        if isinstance(drop_fraction, bool) or not isinstance(drop_fraction, (float, int)) or not math.isfinite(drop_fraction) or not 0 <= drop_fraction < 1:
            raise ValueError("drop_fraction must be finite and satisfy 0 <= r < 1")
        self.drop_fraction = drop_fraction

    def enrich_all(self, samples, pool=None):
        scores = []
        for sample in samples:
            errors = np.asarray(sample.action_plan.step_errors if sample.action_plan else [], dtype=float)
            if errors.shape != (8,) or not np.isfinite(errors).all() or np.any(errors < 0):
                raise ValueError(f"{sample.sample_id}: missing or invalid action errors")
            scores.append((float(errors.mean()), sample.sample_id))
        count = math.floor(len(samples) * self.drop_fraction)
        removed = {key for _, key in sorted(scores, key=lambda item: (-item[0], item[1]))[:count]}
        result = [s for s in samples if s.sample_id not in removed]
        print(f"Action error filter: valid={len(samples)}, removed={count}, retained={len(result)}, drop_fraction={self.drop_fraction}")
        return result
