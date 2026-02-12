"""Prompt templates and few-shot examples."""

from .orchestration import get_orchestration_prompt
from .synthesis import get_synthesis_prompt
from .few_shot_examples import get_few_shot_examples

__all__ = ["get_orchestration_prompt", "get_synthesis_prompt", "get_few_shot_examples"]
