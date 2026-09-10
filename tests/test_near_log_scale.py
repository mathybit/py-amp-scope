#!/usr/bin/env python
"""Visual test of a near-logarithmic (power) frequency axis.

Compares linear, log, and power-scale x-axes side by side so you can judge
which skew works best for your amplifier sweep range.

Each subplot now uses FREQUENCY-AWARE tick placement: ticks are placed at
standard audio frequencies (100, 200, 500, 1k, 2k, 5k, 10k) and the axis
labels show the actual Hz value.
"""

# ── Parameters ───────────────────────────────────────────────────────────────
FREQ_MIN = 40          # Hz — lower bound
FREQ_MAX = 20_000      # Hz — upper bound
NUM_FREQS = 200        # number of frequency points (matches geometric_frequencies)
POWER_ALPHA = 0.5      # < 1 compresses the high end; 1 = linear, log = log scale
# ── End parameters ───────────────────────────────────────────────────────────

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.audio.signal_utils import geometric_frequencies


def power_to_frac(freqs: np.ndarray, alpha: float) -> np.ndarray:
    """Normalize frequencies to [0, 1] using ``f^alpha`` mapping."""
    fmin = float(np.min(freqs))
    fmax = float(np.max(freqs))
    if abs(fmax**alpha - fmin**alpha) < 1e-30:
        return np.full_like(freqs, 0.5)
    return (freqs**alpha - fmin**alpha) / (fmax**alpha - fmin**alpha)


def frac_to_power_freq(display_frac: float, alpha: float) -> float:
    """Convert a [0,1] display position back to an Hz frequency value."""
    if abs(FREQ_MAX - FREQ_MIN) < 1e-30:
        return (FREQ_MIN + FREQ_MAX) / 2
    base = display_frac * (FREQ_MAX**alpha - FREQ_MIN**alpha) + FREQ_MIN**alpha
    return base ** (1.0 / alpha)


def frac_from_power_freq(freq: float, alpha: float) -> float:
    """Convert an Hz frequency back to [0,1] display position (power scale)."""
    denom = FREQ_MAX**alpha - FREQ_MIN**alpha
    if abs(denom) < 1e-30:
        return 0.5
    return (freq**alpha - FREQ_MIN**alpha) / denom


def _valid_pow_ticks(frequencies: list[float], alpha: float) -> list[float]:
    """Return given Hz frequencies as [0,1] display positions.

    The denominator ``FREQ_MAX**alpha - FREQ_MIN**alpha`` can be negative
    when alpha < 0 (not our case here), but for alpha in (0,1) it is always
    positive so the valid range is strictly ``(0, 1)``.  Guard anyway for
    safety with any alpha value.
    """
    denom = FREQ_MAX**alpha - FREQ_MIN**alpha
    if abs(denom) < 1e-30:
        return []
    lo, hi = (0.0, 1.0) if denom > 0 else (1.0, 0.0)
    result: list[float] = []
    for f in frequencies:
        frac = frac_from_power_freq(f, alpha)
        if lo < frac < hi:
            result.append(frac)
    return sorted(result)


def frac_to_log_freq(display_frac: float) -> float:
    """Convert a [0,1] display position back to an Hz frequency value (log scale)."""
    return FREQ_MIN * (FREQ_MAX / FREQ_MIN) ** display_frac





def format_hz(value, _pos):
    """Format any x-axis tick as an Hz label (power scale)."""
    if value <= 0 or value >= 1:
        return ""
    hz = frac_to_power_freq(value, POWER_ALPHA)
    if hz < 100:
        return f"{hz:.0f}"
    elif hz < 1_000:
        return f"{hz:.0f}"
    elif hz < 10_000:
        return f"{hz/1_000:.1f}k"
    else:
        return f"{hz/1_000:.0f}k"


def format_log_hz(value, _pos):
    """Format any x-axis tick as an Hz label (log scale)."""
    if value <= 0 or value >= 1:
        return ""
    hz = frac_to_log_freq(value)
    if hz < 100:
        return f"{hz:.0f}"
    elif hz < 1_000:
        return f"{hz:.0f}"
    elif hz < 10_000:
        return f"{hz/1_000:.1f}k"
    else:
        return f"{hz/1_000:.0f}k"


def format_linear_hz(value, _pos):
    """Format any x-axis tick as an Hz label (linear scale)."""
    if value <= 0 or value >= 1:
        return ""
    hz = FREQ_MIN + value * (FREQ_MAX - FREQ_MIN)
    if hz < 100:
        return f"{hz:.0f}"
    elif hz < 1_000:
        return f"{hz:.0f}"
    elif hz < 10_000:
        return f"{hz/1_000:.1f}k"
    else:
        return f"{hz/1_000:.0f}k"


def main():
    freqs = geometric_frequencies(FREQ_MIN, FREQ_MAX, NUM_FREQS)

    # Make up some plausible response curve for display
    y = (50 * np.sin(2 * np.pi * np.log10(freqs / 1_000))
         + 20 * np.cos(np.log10(freqs / 800) ** 2)
         + 0.3 * (freqs - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) * 30
         + np.random.randn(len(freqs)) * 2)

    # ── Per-chart tick frequencies (edit freely without touching module-level) ─
    _linear_tick_freqs = [500, 2000, 4000, 8000, 12000, 16000, 19999.9]
    _log_tick_freqs     = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 19999.9]
    _power_tick_freqs   = [40.1, 200, 800, 2000, 4000, 7000, 11000, 15000, 19999.9]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=True)

    # ── Subplot 1: Linear (ticks at audio-standard Hz frequencies) ─────────
    x_lin = (freqs - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
    axes[0].plot(x_lin, y, linewidth=0.8)
    # Force independent x-axis to guarantee no residual sharing
    axes[0].set_xscale("linear")
    two_third_frac = 2 / 3
    if POWER_ALPHA > 0:
        freq_at_2_3 = (FREQ_MIN**POWER_ALPHA + two_third_frac *
                       (FREQ_MAX**POWER_ALPHA - FREQ_MIN**POWER_ALPHA)) ** (1/POWER_ALPHA)
    else:
        freq_at_2_3 = math.exp(two_third_frac * math.log(FREQ_MAX / FREQ_MIN)) * FREQ_MIN
    x_10k = np.log10(10_000 / FREQ_MIN) / np.log10(FREQ_MAX / FREQ_MIN)
    axes[0].axvline(x=two_third_frac, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].axvline(x=x_10k, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].set_title(f"Linear\n(2/3 marker at {freq_at_2_3:,.0f} Hz)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-60, 80)
    # Compute tick positions in [0,1] display coords from Hz markers
    _linear_ticks = sorted([(f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) for f in _linear_tick_freqs])
    _linear_ticks = [t for t in _linear_ticks if 0 < t < 1]
    axes[0].xaxis.set_major_locator(FixedLocator(_linear_ticks))
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_linear_hz))

    # ── Subplot 2: Log (ticks at audio-standard Hz frequencies) ────────────
    x_log = np.log10(freqs / FREQ_MIN) / np.log10(FREQ_MAX / FREQ_MIN)
    axes[1].plot(x_log, y, linewidth=0.8)
    axes[1].axvline(x=two_third_frac, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].axvline(x=x_10k, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
    log_2_3_freq = frac_to_log_freq(two_third_frac)
    axes[1].set_title(f"Log (base 10)\n(2/3 marker at {log_2_3_freq:,.0f} Hz)")
    axes[1].set_xlim(0, 1)
    # Map each Hz tick into log-normalized [0,1] display space
    _log_ticks = sorted([np.log10(f / FREQ_MIN) / np.log10(FREQ_MAX / FREQ_MIN)
                         for f in _log_tick_freqs])
    _log_ticks = [t for t in _log_ticks if 0 < t < 1]
    axes[1].xaxis.set_major_locator(FixedLocator(_log_ticks))
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_log_hz))

    # ── Subplot 3: Power (with real Hz labels at audio-standard frequencies) ─
    x_pow = power_to_frac(freqs, POWER_ALPHA)
    axes[2].plot(x_pow, y, linewidth=0.8)
    axes[2].axvline(x=two_third_frac, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].axvline(x=x_10k, color="orange", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[2].set_title(f"Power (alpha={POWER_ALPHA})\n(2/3 marker at {freq_at_2_3:,.0f} Hz)")
    axes[2].set_xlim(0, 1)
    # Power-scale tick positions (uses helper that handles negative denom)
    _power_ticks = _valid_pow_ticks(_power_tick_freqs, POWER_ALPHA)
    axes[2].xaxis.set_major_locator(FixedLocator(_power_ticks))
    axes[2].xaxis.set_major_formatter(FuncFormatter(format_hz))

    print(f"Linear ticks: {axes[0].get_xticks()}")
    print(f"Log ticks: {axes[1].get_xticks()}")
    print(f"Power ticks: {axes[2].get_xticks()}")

    # ── Shared axis labels and grid ────────────────────────────────────────
    axes[0].set_ylabel("Magnitude (arbitrary dB)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Normalized Frequency [0 → 1]")

    fig.suptitle(
        f"Frequency Axis Scales: {FREQ_MIN} – {FREQ_MAX} Hz\n"
        f"{NUM_FREQS} points via geometric_frequencies",
        fontsize=12, y=1.02
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
