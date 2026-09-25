from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumBins:
    first_bin_center: float = 1.0
    bin_width: float = 1.0
    bin_number: int = 512

    def __post_init__(self) -> None:
        if not np.isfinite(self.first_bin_center):
            raise ValueError("first_bin_center must be finite")
        if not np.isfinite(self.bin_width) or self.bin_width <= 0:
            raise ValueError("bin_width must be positive and finite")
        if self.bin_number < 1:
            raise ValueError("bin_number must be positive")

    @property
    def centers(self) -> np.ndarray:
        return self.first_bin_center + self.bin_width * np.arange(self.bin_number)

    @property
    def edges(self) -> np.ndarray:
        return self.first_bin_center + self.bin_width * (
            np.arange(self.bin_number + 1) - 0.5
        )

    @property
    def bin_centers(self) -> np.ndarray:
        return self.centers

    @property
    def bin_edges(self) -> np.ndarray:
        return self.edges

    def mass_to_index(self, mass: np.ndarray | float) -> np.ndarray:

        values = np.asarray(mass, dtype=np.float64)
        raw = np.floor((values - self.first_bin_center) / self.bin_width + 0.5)
        valid = np.isfinite(raw) & (raw >= 0) & (raw < self.bin_number)
        index = np.full(values.shape, -1, dtype=np.int64)
        index[valid] = raw[valid].astype(np.int64)
        return index

    def peaks_to_dense(
        self, masses: np.ndarray, intensities: np.ndarray | None = None
    ) -> np.ndarray:
        if intensities is None:
            peaks = np.asarray(masses, dtype=np.float64)
            if peaks.ndim != 2 or peaks.shape[1] != 2:
                raise ValueError("Expected peaks with shape [P,2]")
            masses, intensities = peaks[:, 0], peaks[:, 1]
        masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
        if masses.shape != intensities.shape:
            raise ValueError("mass and intensity arrays must match")
        if not np.isfinite(masses).all() or not np.isfinite(intensities).all():
            raise ValueError("spectrum peaks must be finite")
        if np.any(intensities < 0):
            raise ValueError("spectrum intensity cannot be negative")
        index = self.mass_to_index(masses)
        valid = index >= 0
        dense = np.zeros(self.bin_number, dtype=np.float64)
        np.add.at(dense, index[valid], intensities[valid])
        return dense

    def histogram(
        self, masses: np.ndarray, intensities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        return self.edges, self.centers, self.peaks_to_dense(masses, intensities)


def create_spectrum_bins(
    first_bin_center: float = 1.0,
    bin_width: float = 1.0,
    bin_number: int = 512,
) -> SpectrumBins:
    return SpectrumBins(
        first_bin_center=float(first_bin_center),
        bin_width=float(bin_width),
        bin_number=int(bin_number),
    )
