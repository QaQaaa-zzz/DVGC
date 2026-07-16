"""Frozen candidate continuous C_D membership models for later audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LocalTrustRegions:
    centers: np.ndarray
    scales: np.ndarray
    radii: np.ndarray

    def __post_init__(self):
        centers=np.asarray(self.centers);scales=np.asarray(self.scales);radii=np.asarray(self.radii)
        if scales.shape != centers.shape or radii.shape != (len(centers),):raise ValueError("Local trust-region shapes mismatch")
        if (scales<=0).any() or (radii<0).any():raise ValueError("Local scales must be positive and radii non-negative")

    def contains(self, feature):
        distance=np.linalg.norm((np.asarray(feature)[None,:]-self.centers)/self.scales,axis=1)
        return bool(np.any(distance<=self.radii))


@dataclass(frozen=True)
class FrozenViabilityThreshold:
    probability_threshold: float
    uncertainty_ceiling: float
    calibration_hash: str

    def contains(self, probability, uncertainty, *, calibration_hash):
        if str(calibration_hash)!=self.calibration_hash:raise ValueError("Viability threshold calibration changed after freeze")
        return bool(float(probability)>=self.probability_threshold and float(uncertainty)<=self.uncertainty_ceiling)
