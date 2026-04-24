"""Typed message payloads for pubsub communication."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class SimulationParams:
    body_part: str
    proj_axis: int
    exposure_time: float
    n_photons: int
    motion_type: str
    velocity: float
    amplitude: float
    frequency: float
    motion_axis: int
    n_steps: int
    noise_type: str
    mitigation: str


@dataclass(frozen=True)
class SimulationProgressMessage:
    value: int


@dataclass(frozen=True)
class SimulationDoneMessage:
    static: np.ndarray
    motion: np.ndarray
    mitigated: np.ndarray
    params: SimulationParams
    metrics: Dict[str, float]


@dataclass(frozen=True)
class SimulationErrorMessage:
    traceback_text: str
