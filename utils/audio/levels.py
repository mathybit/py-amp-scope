"""Signal-level helpers used throughout PyAmpScope.

Digital samples are normalized floating-point PCM where |sample| == 1.0 is full
scale.  Consequently a full-scale sine has a peak level of 0 dBFS and an RMS
level of -3.0103 dBFS.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Iterable


_EPS = 1e-30


def db20(value):
    """20*log10(value), suitable for amplitude/RMS ratios.

    Scalars return ``float``; arrays return ``ndarray``.  Non-positive inputs are
    floored to a tiny positive value instead of producing -inf, which keeps saved
    metrics JSON-safe and charting predictable.
    """
    arr = np.asarray(value, dtype=np.float64)
    out = 20.0 * np.log10(np.maximum(np.abs(arr), _EPS))
    if np.ndim(value) == 0:
        return float(out)
    return out


def db10(value):
    """10*log10(value), suitable for power ratios such as PSD ratios."""
    arr = np.asarray(value, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, _EPS))
    if np.ndim(value) == 0:
        return float(out)
    return out


def undb20(db):
    """Convert an amplitude value in dB to a linear ratio."""
    arr = np.asarray(db, dtype=np.float64)
    out = np.power(10.0, arr / 20.0)
    if np.ndim(db) == 0:
        return float(out)
    return out


def undb10(db):
    """Convert a power value in dB to a linear ratio."""
    arr = np.asarray(db, dtype=np.float64)
    out = np.power(10.0, arr / 10.0)
    if np.ndim(db) == 0:
        return float(out)
    return out


def rms(signal: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def peak(signal: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.max(np.abs(arr)))


def rms_dbfs(signal: Iterable[float] | np.ndarray) -> float:
    return db20(rms(signal))


def peak_dbfs(signal: Iterable[float] | np.ndarray) -> float:
    return db20(peak(signal))


def sine_rms_from_peak(peak_amplitude: float) -> float:
    return float(abs(peak_amplitude) / math.sqrt(2.0))


def requested_sine_levels(tone_amplitude: float, send_gain: float) -> dict:
    """Return the requested digital sine peak/RMS after Send Gain."""
    p = float(tone_amplitude) * float(send_gain) / 100.0
    r = sine_rms_from_peak(p)
    return {
        "tone_amplitude": float(tone_amplitude),
        "send_gain_pct": float(send_gain),
        "requested_peak": p,
        "requested_peak_dbfs": db20(p),
        "requested_rms": r,
        "requested_rms_dbfs": db20(r),
    }


def global_peak_scale(signal: np.ndarray, limit: float = 0.95) -> tuple[np.ndarray, float]:
    """Uniformly scale *signal* only when its absolute peak exceeds *limit*.

    Returns ``(scaled_signal, scale_factor)``.  Uniform scaling is important for
    correction signals because it preserves spectral shape.
    """
    arr = np.asarray(signal, dtype=np.float64)
    p = peak(arr)
    if p <= 0.0 or p <= limit:
        return arr.copy(), 1.0
    factor = float(limit / p)
    return arr * factor, factor
