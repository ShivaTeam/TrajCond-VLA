"""NAVSIM action-conditioned continuous trajectory QA builder."""
from .base import register_prompt
from .trajcond_common import TrajcondBuilder


@register_prompt("trajcond_stage2")
class TrajcondStage2Builder(TrajcondBuilder):
    stage = 2
