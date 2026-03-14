"""Lean self-contained training/evaluation package for course submission."""

from .config import ExperimentConfig
from .runner import run_graph_cv_experiment, run_graph_experiment

__all__ = ["ExperimentConfig", "run_graph_experiment", "run_graph_cv_experiment"]
