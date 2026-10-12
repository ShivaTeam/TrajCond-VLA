"""NAVSIM action vocabulary alignment QA builder."""
from .base import register_prompt
from .trajcond_common import TrajcondBuilder


@register_prompt("trajcond_stage1")
class TrajcondStage1Builder(TrajcondBuilder):
    stage = 1
