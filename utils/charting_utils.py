"""Chart construction helpers for PyAmpScope."""
from __future__ import annotations

import io
import numpy as np

from .audio.analysis_utils import smooth_moving_average
from .audio.levels import undb20


def _safe_span(values, minimum=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -minimum, minimum
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < minimum:
        mid = (hi + lo) / 2.0
        return mid - minimum/2, mid + minimum/2
    margin = (hi-lo)*0.08
    return lo-margin, hi+margin


def build_multichart_png(freqs, H_db, num_neighbors=5, title="Calibration Response"):
    """Calibration chart: measured response, deviation, inverse correction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.asarray(freqs, dtype=float)
    H_db = np.asarray(H_db, dtype=float)
    smooth = smooth_moving_average(H_db, num_neighbors)
    reference = float(np.nanmedian(smooth))
    deviation = smooth - reference
    correction = undb20(-deviation)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=120, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes[0].plot(freqs, H_db, label="Measured")
    axes[0].plot(freqs, smooth, label="Smoothed")
    axes[0].axhline(reference, linestyle="--", label=f"Median ({reference:.2f} dB)")
    axes[0].set_ylabel("Response (dB)")
    axes[0].legend(fontsize=9)

    axes[1].plot(freqs, deviation)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("Deviation (dB)")

    axes[2].plot(freqs, correction)
    axes[2].axhline(1.0, linestyle="--")
    axes[2].set_ylabel("Correction factor")
    axes[2].set_xlabel("Frequency (Hz)")

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(max(1.0, float(np.nanmin(freqs))), float(np.nanmax(freqs)))
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return buf.getvalue(), correction, smooth


def build_validate_chart_png(freqs, deviation_db, pct_dev, title="Validation Result"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freqs=np.asarray(freqs,float); deviation_db=np.asarray(deviation_db,float); pct_dev=np.asarray(pct_dev,float)
    fig, axes=plt.subplots(1,2,figsize=(16,5),dpi=120)
    fig.suptitle(title,fontsize=14,fontweight="bold")
    axes[0].plot(freqs,deviation_db); axes[0].axhline(0,linestyle="--"); axes[0].set_ylabel("Deviation (dB)")
    axes[1].plot(freqs,pct_dev); axes[1].set_ylabel("Absolute deviation (%)")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("Frequency (Hz)"); ax.grid(True,which="both",alpha=.25)
    fig.tight_layout(); buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def build_noise_chart_png(freqs, H_db, smoothed_db, expected_db, deviation_db, corr_factors,
                          num_neighbors=5, title="Noise Calibration Response"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freqs=np.asarray(freqs,float)
    fig,axes=plt.subplots(3,1,figsize=(12,10),dpi=120,sharex=True)
    fig.suptitle(title,fontsize=14,fontweight="bold")
    axes[0].plot(freqs,H_db,label="Measured"); axes[0].plot(freqs,smoothed_db,label="Smoothed"); axes[0].plot(freqs,expected_db,linestyle=":",label="Expected")
    axes[0].set_ylabel("Level (dB)"); axes[0].legend(fontsize=9)
    axes[1].plot(freqs,deviation_db); axes[1].axhline(0,linestyle="--"); axes[1].set_ylabel("Deviation")
    axes[2].plot(freqs,corr_factors); axes[2].axhline(1,linestyle="--"); axes[2].set_ylabel("Correction factor"); axes[2].set_xlabel("Frequency (Hz)")
    for ax in axes: ax.set_xscale("log"); ax.grid(True,which="both",alpha=.25)
    fig.tight_layout(); buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
    return buf.getvalue(), np.asarray(corr_factors,float), np.asarray(smoothed_db,float)


def create_gui_response_figure(freq_min: float, freq_max: float):
    """Create the GUI's two vertically stacked response axes.

    Top: received level in dBFS.  Bottom: normalized frequency response where a
    frequency-independent gain/attenuation appears as a flat 0 dB line.
    """
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7.2, 5.2), dpi=100)
    ax_level = fig.add_subplot(211)
    ax_response = fig.add_subplot(212, sharex=ax_level)
    ax_level.set_ylabel("Received level (dBFS)")
    ax_response.set_ylabel("Relative response (dB)")
    ax_response.set_xlabel("Frequency (Hz)")
    for ax in (ax_level, ax_response):
        ax.set_xscale("log")
        ax.set_xlim(freq_min, freq_max)
        ax.grid(True, which="both", alpha=0.25)
    level_line, = ax_level.plot([], [], label="Received")
    level_smooth, = ax_level.plot([], [], linestyle="--", label="Smoothed")
    response_line, = ax_response.plot([], [], label="Relative response")
    ax_response.axhline(0.0, linestyle=":")
    ax_level.legend(loc="upper right", fontsize=8)
    ax_response.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax_level, ax_response, level_line, level_smooth, response_line


def update_gui_response_figure(
            ax_level,
            ax_response,
            level_line,
            level_smooth_line,
            response_line,
            freqs,
            received_dbfs,
            smoothed_dbfs,
            relative_response_db,
            freq_min,
            freq_max,
        ):
    """Update GUI response artists and axes from one analysis payload."""
    freqs = np.asarray(freqs, dtype=float)
    level = np.asarray(received_dbfs, dtype=float)
    smooth = np.asarray(smoothed_dbfs, dtype=float)
    response = np.asarray(relative_response_db, dtype=float)

    valid_level = np.isfinite(freqs) & np.isfinite(level)
    level_line.set_data(freqs[valid_level], level[valid_level])
    valid_smooth = np.isfinite(freqs) & np.isfinite(smooth)
    level_smooth_line.set_data(freqs[valid_smooth], smooth[valid_smooth])
    valid_response = np.isfinite(freqs) & np.isfinite(response)
    response_line.set_data(freqs[valid_response], response[valid_response])

    ax_level.set_xlim(float(freq_min), float(freq_max))
    if np.any(valid_level):
        lo = float(np.min(level[valid_level]))
        hi = float(np.max(level[valid_level]))
        margin = max((hi - lo) * 0.12, 2.0)
        ax_level.set_ylim(lo - margin, hi + margin)
    if np.any(valid_response):
        mag = max(float(np.max(np.abs(response[valid_response]))) * 1.2, 1.0)
        ax_response.set_ylim(-mag, mag)
