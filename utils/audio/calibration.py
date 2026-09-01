"""Calibration-curve loading, derivation, and interpolation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional

from .analysis_utils import smooth_moving_average
from .levels import db20, undb20
from .signal_utils import interpolate_correction


@dataclass
class CorrectionProfile:
    frequencies: np.ndarray
    factors: np.ndarray
    path: Path


def derive_inverse_correction(
            frequencies: np.ndarray,
            measured_db: np.ndarray,
            *,
            smoothing_window: int = 5,
        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a linear inverse-magnitude correction curve.

    Overall gain is intentionally ignored: the smoothed response is referenced to
    its median, and only relative frequency coloration is inverted.
    """
    f = np.asarray(frequencies, dtype=float)
    db = np.asarray(measured_db, dtype=float)
    valid = np.isfinite(f) & np.isfinite(db) & (f > 0)
    if np.sum(valid) < 2:
        raise ValueError("at least two valid frequency measurements are required")
    smooth = smooth_moving_average(db, smoothing_window)
    reference_db = float(np.nanmedian(smooth[valid]))
    factors = np.ones_like(db, dtype=float)
    factors[valid] = undb20(reference_db - smooth[valid])
    return factors, smooth, reference_db


def _load_npz_profile(path: Path) -> CorrectionProfile:
    data = np.load(str(path), allow_pickle=True)
    freq_keys = ("frequencies", "freqs")
    factor_keys = ("correction_factor", "correction_factors", "correction_W")
    f = next((np.asarray(data[k], dtype=float) for k in freq_keys if k in data), None)
    factors = next((np.asarray(data[k], dtype=float) for k in factor_keys if k in data), None)
    if f is None or factors is None:
        raise ValueError(f"{path} does not contain frequencies + correction_factor")
    n = min(len(f), len(factors))
    return CorrectionProfile(f[:n], factors[:n], path)


def load_first_existing(paths: list[Path]) -> Optional[CorrectionProfile]:
    for p in paths:
        if p.exists():
            try:
                return _load_npz_profile(p)
            except Exception:
                continue
    return None


def load_send_correction(data_dir: Path) -> Optional[CorrectionProfile]:
    data_dir = Path(data_dir)
    return load_first_existing([
        data_dir / "cal_send_corrections.npz",
        data_dir / "cal_send_correction.npz",
    ])


def load_receive_correction(data_dir: Path, recv_path: str, *, prefer_corrected_send: bool = False) -> Optional[CorrectionProfile]:
    """Load a receive-path curve for ``dir`` or ``iso``.

    Both base and send-corrected receive calibrations are supported.  The GUI can
    choose the profile matching how the user calibrated the system; correction is
    never automatically enabled merely because a file exists.
    """
    data_dir = Path(data_dir)
    recv_path = recv_path.lower()
    if recv_path not in {"dir", "iso"}:
        raise ValueError("recv_path must be 'dir' or 'iso'")
    variants = ["corr", "base"] if prefer_corrected_send else ["base", "corr"]
    paths = [data_dir / f"cal_recv_{recv_path}_{variant}_corrections.npz" for variant in variants]
    paths += [data_dir / f"cal_recv_{recv_path}_corrections.npz"]
    return load_first_existing(paths)


def correction_at(profile: Optional[CorrectionProfile], frequencies: np.ndarray) -> np.ndarray:
    if profile is None:
        return np.ones_like(np.asarray(frequencies, dtype=float))
    return interpolate_correction(np.asarray(frequencies, dtype=float), profile.frequencies, profile.factors)


def corrected_db(amplitude_db: np.ndarray, frequencies: np.ndarray, profile: Optional[CorrectionProfile]) -> np.ndarray:
    values = np.asarray(amplitude_db, dtype=float)
    if profile is None:
        return values.copy()
    factors = correction_at(profile, frequencies)
    return values + db20(factors)
