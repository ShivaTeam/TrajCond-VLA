"""filter_incomplete enricher: drop samples with missing pseudo_label."""

from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from ir.schema import SampleIR
from .base import Enricher, register_enricher


@register_enricher("filter_incomplete_pseudo")
class FilterIncompletePseudoEnricher(Enricher):
    """Remove samples whose pseudo_label is missing or has empty critical_objects.

    reads:  pseudo_label
    writes: - (no new field, just filters)
    """

    reads = ["pseudo_label"]
    writes = []

    def __init__(self, require_meta_behaviour=False, validate_classes=False, require_image=False):
        self.require_meta_behaviour = require_meta_behaviour
        self.validate_classes = validate_classes
        self.require_image = require_image

    def enrich_all(self, samples: list[SampleIR],
                   pool: Optional[Pool] = None) -> list[SampleIR]:
        input_count = len(samples)
        from collections import Counter
        from utils.trajcond_contract import validate_cot, explanation_body
        result, counts = [], Counter()
        for s in samples:
            try:
                if s.pseudo_label is None or s.pseudo_label.critical_objects is None or s.pseudo_label.explanation is None:
                    raise ValueError("missing_cot")
                if self.require_meta_behaviour:
                    if s.pseudo_label.meta_behaviour is None:
                        raise ValueError("missing_meta_behaviour")
                    explanation_body(s.pseudo_label.explanation)
                if self.validate_classes:
                    validate_cot(s.pseudo_label)
                if self.require_image and len(s.images) != 1:
                    raise ValueError("missing_image")
                result.append(s)
            except ValueError as exc:
                counts[str(exc)] += 1
        if input_count > len(result):
            print(f"filter_incomplete: {input_count - len(result)}/{input_count} "
                  f"samples removed: {dict(counts)}")
        return result
