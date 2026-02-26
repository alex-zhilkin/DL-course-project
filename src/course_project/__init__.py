"""Lean self-contained training/evaluation package for course submission."""

from .config import ExperimentConfig
from .runner import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
