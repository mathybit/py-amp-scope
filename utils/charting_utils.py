"""Chart-building utilities for PyAmpScope calibration scripts.

Each function returns raw PNG bytes via an in-memory buffer.
All charts use a log-frequency x-axis and headless matplotlib backend.
"""

import io

import numpy as np


def _smooth_moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving average over *window_size* total points (including center).

    Edge bins use whatever neighbors are available (no padding assumption).

    Example: window_size=5 means +/-2 from the current bin. If fewer than 2 exist on
    one side, only the available points are averaged.
    """
    half = window_size // 2
    result = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(i - half, 0)
        hi = min(i + half + 1, len(arr))
        result[i] = np.mean(arr[lo:hi])
    return result


def build_multichart_png(
    freqs: np.ndarray,
    H_db: np.ndarray,
    num_neighbors: int = 5,
    title: str = "Calibration Response",
) -> bytes:
    """Build a 3-panel PNG chart: response (raw + smoothed), deviation (sigma), correction factor.

    Parameters
    ----------
    freqs : log-spaced frequency bins in Hz.
    H_db : measured amplitude in dBFS at each bin (may be negative).
    num_neighbors : total points in moving average window (default 5 = +/-2).
    title : chart title.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Clamp to valid dB range and compute statistics from unsmoothed data
    H_mag_db = np.maximum(H_db, -200)
    mean_db = float(np.mean(H_mag_db))
    std_db = float(np.std(H_mag_db))

    # Deviation (sigma) from the charting perspective
    deviation_db = H_mag_db - mean_db
    deviation_sigma = deviation_db / max(std_db, 1e-10)

    # Smoothing for the correction factor — moving average over nearest neighbors
    H_smoothed_db = _smooth_moving_average(H_mag_db, window_size=num_neighbors)

    # Correction factor in linear space:
    #   correction = H_mean_linear / H_smoothed_linear
    # = 10^((H_mean_dB - H_smoothed_dB) / 20)
    h_diff_db = mean_db - H_smoothed_db
    correction_factor = 10 ** (h_diff_db / 20.0)

    dev_ylabel = f"Deviation (sigma={std_db:.2f}dB)"

    # 3 panels stacked
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=120, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Frequency response (raw + smoothed trend)
    ax = axes[0]
    ax.set_ylabel("Magnitude (dB)")
    ax.plot(freqs, H_mag_db, "r-", linewidth=1.2, alpha=0.8, label="Measured (raw)")
    ax.plot(freqs, H_smoothed_db, "g--", linewidth=1.5, alpha=0.7, label="Smoothed trend")
    ax.axhline(mean_db, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label=f"Mean ({mean_db:.1f} dB)")
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_min = min(H_mag_db.min(), H_smoothed_db.min())
    y_max = max(H_mag_db.max(), H_smoothed_db.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: Deviation in standard deviations
    ax = axes[1]
    ax.set_ylabel(dev_ylabel)
    ax.plot(freqs, deviation_sigma, "b-", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.axhline(2, color="orange", linewidth=0.5, linestyle=":", alpha=0.5, label="+/- 2 sigma")
    ax.axhline(-2, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_max_s = max(np.max(deviation_sigma), -np.min(deviation_sigma))
    ax.set_ylim(-y_max_s * 1.05, y_max_s * 1.05)

    # Panel 3: Correction factor
    ax = axes[2]
    ax.set_ylabel("Correction factor")
    ax.plot(freqs, correction_factor, "m-", linewidth=1.2, alpha=0.8)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_min_c = min(correction_factor.min(), 0.95)
    y_max_c = max(correction_factor.max(), 1.05)
    y_range_c = y_max_c - y_min_c
    ax.set_ylim(y_min_c - y_range_c * 0.05, y_max_c + y_range_c * 0.05)

    axes[2].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(max(freqs[0], 20), freqs[-1])

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read(), correction_factor, H_smoothed_db


def build_validate_chart_png(
    freqs: np.ndarray,
    deviation_db: np.ndarray,
    pct_dev: np.ndarray,
    title: str = "Validation Result",
) -> bytes:
    """Build a 2-panel PNG chart: deviation (dB) vs deviation (%)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=120)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Deviation from mean in dB
    ax = axes[0]
    ax.plot(freqs, deviation_db, "b-", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Deviation (dB)")
    ax.set_xscale("log")
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_max = max(np.max(deviation_db), -np.min(deviation_db))
    ax.set_ylim(-y_max * 1.1, y_max * 1.1)

    # Panel 2: Absolute percentage deviation from arithmetic mean (linear)
    ax = axes[1]
    ax.plot(freqs, pct_dev, "g-", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Abs % Deviation")
    ax.set_xscale("log")
    ax.grid(True, which="major", axis="x", alpha=0.3)

    axes[1].set_xlabel("Frequency (Hz)")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
