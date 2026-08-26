"""Chart-building utilities for PyAmpScope calibration scripts.

Each function returns raw PNG bytes via an in-memory buffer.
All charts use a log-frequency x-axis and headless matplotlib backend.
"""

import io
from typing import Optional

import numpy as np


def build_multichart_png(
    freqs: np.ndarray,
    H_db: np.ndarray,
    correction_filter: Optional[np.ndarray] = None,
    title: str = "Calibration Response",
) -> bytes:
    """Build a 2-panel PNG chart: response, deviation (sigma with std label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H_mag_db = np.maximum(H_db, -200)
    mean_db = float(np.mean(H_mag_db))
    std_db = float(np.std(H_mag_db))
    deviation_db = H_mag_db - mean_db
    deviation_sigma = deviation_db / max(std_db, 1e-10)

    # Label the single deviation panel with actual std magnitude
    dev_ylabel = f"Deviation (sigma={std_db:.2f}dB)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=120, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: Frequency response
    ax = axes[0]
    ax.set_ylabel("Magnitude (dB)")
    ax.plot(freqs, H_mag_db, "r-", linewidth=1.2, alpha=0.8, label="Measured")
    if correction_filter is not None:
        corr_db = 20 * np.log10(np.maximum(np.abs(correction_filter), 1e-10))
        ax.plot(freqs, corr_db, "b--", linewidth=1.0, alpha=0.6, label="Correction")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_min_db = min(  min(corr_db.min() if correction_filter is not None else H_mag_db.min(), H_mag_db.min()), -1) * 1.05
    y_max_db = max(  max(  np.abs(corr_db).max() if correction_filter is not None else 1, np.abs(H_mag_db).max()  ), 1  ) * 1.05
    ax.set_ylim(y_min_db, y_max_db)
    ax.legend(loc="center right", fontsize=9)
    print(corr_db.shape, np.min(corr_db), np.max(corr_db), np.mean(corr_db), np.std(corr_db), np.std(H_mag_db))

    # Panel 2: Deviation in standard deviations (with std label on y-axis)
    ax = axes[1]
    ax.set_ylabel(dev_ylabel)
    ax.plot(freqs, deviation_sigma, "g-", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.axhline(2, color="orange", linewidth=0.5, linestyle=":", alpha=0.5, label="+/- 2 sigma")
    ax.axhline(-2, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.grid(True, which="major", axis="x", alpha=0.3)
    y_max_s = max(np.max(deviation_sigma), -np.min(deviation_sigma))
    ax.set_ylim(-y_max_s * 1.05, y_max_s * 1.05)

    axes[1].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(max(freqs[0], 20), freqs[-1])

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
