"""Strict TXT rendering and QA serialization for two-stage training datasets."""

import hashlib
import json
from pathlib import Path
from string import Template
import numpy as np
from .base import PromptBuilder
from utils.coord import format_point
from utils.trajcond_contract import TEMPLATES, select_points, future_points, validate_cot, explanation_body
from utils.action_search import SearchResult
from utils.action_search_adapter import validate_result


class TrajcondBuilder(PromptBuilder):
    stage = None

    def __init__(self, template_path=None, format_contract_path=None, decimals=2):
        if decimals != 2:
            raise ValueError("The public format uses exactly two decimals")
        self.decimals = decimals
        canonical_contract_path = TEMPLATES / "navsim_trajcond_format_release.json"
        contract_path = Path(format_contract_path or canonical_contract_path)
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        profile = self.contract.get("profile")
        if profile != "release":
            raise ValueError(f"Only the release format is supported: {profile}")
        canonical_path = TEMPLATES / f"navsim_trajcond_stage{self.stage}_release.txt"
        self.template_path = Path(template_path or canonical_path)
        raw = self.template_path.read_bytes()
        if hashlib.sha256(raw).digest() != hashlib.sha256(canonical_path.read_bytes()).digest():
            raise ValueError("TXT does not match the canonical release template")
        canonical_contract = json.loads(canonical_contract_path.read_text(encoding="utf-8"))
        if self.contract != canonical_contract:
            raise ValueError("Format contract differs from the canonical release format")
        text = raw.decode("utf-8")
        if "\r" in text or not text.endswith("\n") or text.endswith("\n\n"):
            raise ValueError("TXT must use LF and exactly one final newline")
        self.template = Template(text)
        found = set()
        for match in Template.pattern.finditer(text):
            if match.group("braced"):
                found.add(match.group("braced"))
            else:
                raise ValueError("Only ${name} placeholders are supported")
        self.spec = self.contract["stages"][str(self.stage)]
        if found != set(self.spec["required_placeholders"]):
            raise ValueError("TXT placeholder set does not match the contract")

    def question(self, sample, method):
        if method not in {"kmeans", "kdisks", "brohan"}:
            raise ValueError("Unknown trajectory method")
        history = select_points(sample.trajectory, [-1.5, -1.0, -0.5, 0.0])
        timeline = " ".join(f"   - t-{3-i}: {format_point(p)}" for i, p in enumerate(history))
        branch = "brohan" if method == "brohan" else "joint"
        context = dict(command=sample.metadata.command, history_seconds="1.5", history_steps="3",
                       sample_rate_hz="2", past_timeline=timeline, future_seconds="4", future_steps="8",
                       critical_object_list="\n".join("     - " + key for key in self.contract["critical_objects"]),
                       speed_labels=", ".join(self.contract["speed_labels"]["prompt"]),
                       command_labels=", ".join(self.contract["command_labels"]["prompt"]),
                       action_prediction_instruction=self.spec["action_prediction_instructions"][branch],
                       output_schema=self.spec["output_schema_text"])
        if self.stage == 2:
            context["continuous_prediction_instruction"] = self.spec["continuous_prediction_instruction"]
        question = self.template.substitute({key: context[key] for key in self.spec["required_placeholders"]})
        if question.count("<image>") != 1:
            raise ValueError("Prompt must contain one image placeholder")
        return question

    @staticmethod
    def action_text(plan, sample_id):
        validated = validate_result(SearchResult(np.asarray(plan.token_ids), plan.reconstructed_trajectory,
                                                np.asarray(plan.step_errors)), plan.method, plan.codebook_sizes, sample_id)
        if validated.vocabulary_size != plan.vocabulary_size or plan.trajectory_source != "navsim":
            raise ValueError(f"{sample_id}: inconsistent action plan")
        if plan.method == "brohan":
            body = ", ".join("(" + ",".join(f"<action_{t}>" for t in row) + ")" for row in plan.token_ids)
        else:
            body = "".join(f"<action_{row[0]}>" for row in plan.token_ids)
        return "<answer>[AT, " + body + "]</answer>"

    def build(self, sample):
        if len(sample.images) != 1 or sample.action_plan is None:
            raise ValueError(f"{sample.sample_id}: missing image/action plan")
        validate_cot(sample.pseudo_label)
        label, plan = sample.pseudo_label, sample.action_plan
        body = explanation_body(label.explanation)
        depth = self.spec["thinking_depth"]
        action = self.action_text(plan, sample.sample_id)
        values = dict(critical_objects={k: label.critical_objects[k] for k in self.contract["critical_objects"]},
                      meta_behaviour={k: label.meta_behaviour[k] for k in ("speed", "command")},
                      explanation="<thinking>" * depth + body + "</thinking>" * depth,
                      future_trajectory=action, discrete_action_trajectory=action)
        if self.stage == 2:
            points = future_points(sample.normalized_target_trajectory)
            values["future_trajectory"] = "<answer>[PT, " + ", ".join(format_point(p) for p in points) + "]</answer>"
        answer = {key: values[key] for key in self.spec["answer_fields"]}
        return dict(id=sample.sample_id, image=list(sample.images), system="", conversations=[
            {"from": "human", "value": self.question(sample, plan.method)},
            {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False)}])
