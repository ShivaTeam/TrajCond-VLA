"""merge_pseudo_labels enricher: read JSONL and parse VLM outputs."""

import json

from tqdm import tqdm

from ir.schema import SampleIR, PseudoLabelIR
from utils.json_repair import extract_clean_json
from .base import Enricher, register_enricher


_PARSERS: dict[str, type["PseudoLabelParser"]] = {}


def register_parser(name: str):
    def wrapper(cls):
        cls.name = name
        _PARSERS[name] = cls
        return cls
    return wrapper


class PseudoLabelParser:
    name: str
    def parse(self, data: dict, sample: SampleIR) -> dict:
        raise NotImplementedError


@register_parser("poutine")
class PoutineParser(PseudoLabelParser):
    name = "poutine"

    def __init__(self, repair=True):
        self.repair = repair

    def parse(self, data: dict, sample: SampleIR) -> dict:
        raw = data.get("predict", "")
        obj = extract_clean_json(raw, repair=self.repair) if isinstance(raw, str) else None
        if not isinstance(obj, dict):
            return {}
        return {
            "critical_objects": obj.get("critical_objects"),
            "meta_behaviour": obj.get("meta_behaviour"),
            "explanation": obj.get("explanation"),
        }


@register_enricher("merge_pseudo_labels")
class MergePseudoLabelsEnricher(Enricher):
    reads = []
    writes = ["pseudo_label"]

    def __init__(self, source_path: str, parser: str = "poutine",
                 key_field: str = "id", json_repair: bool = True, strict_ids: bool = False):
        self.source_path = source_path
        self.key_field = key_field
        self.json_repair = json_repair
        self.strict_ids = strict_ids
        self.index = None
        parser_cls = _PARSERS.get(parser)
        if parser_cls is None:
            raise KeyError(f"Unknown parser '{parser}'. Registered: {list(_PARSERS.keys())}")
        self.parser = parser_cls()
        if isinstance(self.parser, PoutineParser):
            self.parser.repair = json_repair

    def enrich_all(self, samples: list[SampleIR], pool=None) -> list[SampleIR]:
        if self.index is None:
            self.index = self._load_index()
        for s in tqdm(samples, desc="merge_pseudo_labels"):
            parsed = self.index.get(s.sample_id)
            if parsed:
                try:
                    s.pseudo_label = PseudoLabelIR(**parsed)
                except (ValueError, TypeError):
                    s.pseudo_label = None
        return samples

    def _load_index(self):
        index, seen = {}, set()
        with open(self.source_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(tqdm(f, desc="Loading pseudo labels"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{self.source_path}:{line_no}: malformed JSONL") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{self.source_path}:{line_no}: expected a JSON object")
                pid = record.get(self.key_field, record.get("id", ""))
                if self.strict_ids:
                    if not isinstance(pid, str) or not pid:
                        raise ValueError(f"{self.source_path}:{line_no}: invalid ID")
                    if pid in seen:
                        raise ValueError(f"{self.source_path}:{line_no}: duplicate ID {pid}")
                    seen.add(pid)
                if not pid:
                    continue
                parsed = self.parser.parse(record, None)
                if parsed:
                    index[pid] = parsed

        return index
