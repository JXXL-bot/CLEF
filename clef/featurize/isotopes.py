from __future__ import annotations

from functools import lru_cache

import numpy as np

from .atom_features import ELEMENTS


ISOTOPES = {
    "H": ((1, 0.999885), (2, 0.000115)),
    "C": ((12, 0.9893), (13, 0.0107)),
    "N": ((14, 0.99636), (15, 0.00364)),
    "O": ((16, 0.99757), (17, 0.00038), (18, 0.00205)),
    "F": ((19, 1.0),),
    "P": ((31, 1.0),),
    "S": ((32, 0.9499), (33, 0.0075), (34, 0.0425), (36, 0.0001)),
    "Cl": ((35, 0.7576), (37, 0.2424)),
}
MONOISOTOPIC_MASS = np.asarray(
    [1.007825032, 12.0, 14.003074004, 15.994914620,
     18.998403163, 30.973761998, 31.972071174, 34.968852682],
    dtype=np.float64,
)


@lru_cache(maxsize=None)
def _element_distribution(symbol: str, count: int) -> np.ndarray:

    if count < 0:
        raise ValueError("Element count cannot be negative")
    base = ISOTOPES[symbol][0][0]
    atom = np.zeros(max(mass - base for mass, _ in ISOTOPES[symbol]) + 1)
    for mass_number, abundance in ISOTOPES[symbol]:
        atom[mass_number - base] = abundance
    distribution = np.asarray([1.0], dtype=np.float64)
    for _ in range(count):
        distribution = np.convolve(distribution, atom)
    return distribution


def formula_exact_mass(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (8,) or np.any(counts < 0):
        raise ValueError("Expected eight nonnegative element counts")
    return float(counts @ MONOISOTOPIC_MASS)


def isotope_template(
    counts: np.ndarray,
    *,
    first_bin_center: float = 1.0,
    bin_width: float = 1.0,
    bin_number: int = 512,
    max_peaks: int = 12,
) -> tuple[np.ndarray, np.ndarray]:


    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape != (8,) or np.any(counts < 0):
        raise ValueError("Expected eight nonnegative element counts")
    indices = np.full(max_peaks, -1, dtype=np.int64)
    intensities = np.zeros(max_peaks, dtype=np.float32)
    if not np.any(counts):
        return indices, intensities
    base_mass = 0
    distribution = np.asarray([1.0], dtype=np.float64)
    for symbol, count in zip(ELEMENTS, counts):
        if count:
            base_mass += int(count) * ISOTOPES[symbol][0][0]
            distribution = np.convolve(
                distribution, _element_distribution(symbol, int(count))
            )
    nominal_masses = base_mass + np.arange(len(distribution), dtype=np.float64)
    raw_indices = np.floor(
        (nominal_masses - first_bin_center) / bin_width + 0.5
    ).astype(np.int64)
    valid = (raw_indices >= 0) & (raw_indices < bin_number) & (distribution > 0)
    candidate = np.flatnonzero(valid)
    if len(candidate) == 0:
        return indices, intensities
    ranked = sorted(candidate, key=lambda i: (-distribution[i], raw_indices[i]))
    chosen = sorted(ranked[:max_peaks], key=lambda i: raw_indices[i])
    weight = distribution[chosen]
    weight /= weight.sum()
    indices[:len(chosen)] = raw_indices[chosen]
    intensities[:len(chosen)] = weight.astype(np.float32)
    return indices, intensities
