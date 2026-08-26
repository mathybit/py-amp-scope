#!/usr/bin/env python
"""Validate send-path calibration by measuring frequency deviation with/without correction.

Loads the send calibration profile (cal_send_profile.npz), sends a uniform tone at each
bin frequency, and measures how far each returned amplitude deviates from the mean across all
bins. Two modes:

  Default (no --correct): sends uncorrected signal; shows raw deviation of the calibration
                           profile (i.e. what we measured during the last run).

  --correct: loads per-bin correction factors from the saved NPZ file and applies them to the
             sent tone amplitudes, then measures how much flatter the returned signal is.

The script answers: is the send circuitry uniform enough that calibration matters?

Usage:
    python validate_send_calibration.py              # baseline measurement (no correction)
    python validate_send_calibration.py --correct      # with inverse correction applied
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import config as cfg  # noqa: E402
from utils.charting_utils import build_validate_chart_png  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate send-path calibration frequency response.",
        epilog=(
            "Error metric: absolute percent deviation of each bin's amplitude from the "
            "arithmetic mean across all bins. A lower stddev means the correction is working.\n\n"
            "With --correct, an inverse filter is computed on-the-fly from the calibration "
            "profile and applied to the sent signal before each tone."
        ),
    )
    parser.add_argument(
        "--cal-file", type=str, default=None,
        help="Path to cal_send_profile.npz (default: data/cal_send_profile.npz)",
    )
    parser.add_argument(
        "--correct", action="store_true",
        help="Apply inverse correction filter computed from the calibration profile.",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="sequential",
        help="Capture mode (default: sequential)",
    )
    parser.add_argument("--tone-duration", type=float, default=None,
                        help=f"Tone duration per frequency in seconds (config: {cfg.tone_duration})")
    parser.add_argument("--gap", type=float, default=None,
                        help=f"Gap between tones in seconds (config: {cfg.tone_gap})")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory for WAV files and chart (default: logs)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# ToneSwitcher — re-used from calibrate_send_v2.py
# ---------------------------------------------------------------------------
class _ToneSwitcher:
    """Manages per-frequency tone segments for a single OutputStream."""

    def __init__(self, freqs, duration_s, fs, gap_s, amp_factors):
        self.fs = fs
        self.tone_duration_s = duration_s
        self.gap_samples = int(gap_s * fs)
        offset = 0
        self.tone_starts = []
        self.tone_arrays = []
        for i, freq in enumerate(freqs):
            tone_samples = int(duration_s * fs)
            t = np.arange(tone_samples) / fs
            self.tone_arrays.append(
                (np.sin(2 * np.pi * freq * t) * amp_factors[i]).astype(np.float64)
            )
            self.tone_starts.append(offset)
            offset += tone_samples + self.gap_samples
        self.total_out_samples = offset


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _compute_correction(H_lin):
    """Compute regularized inverse correction filter from linear magnitude response.

    Returns W (complex frequency-domain, unit-magnitude phase-flipped) — not used for
    direct waveform scaling here but available if the user wants to inspect it later.
    """
    tol = 1e-3
    H_mag = np.maximum(np.abs(H_lin), tol)
    return np.conj(H_lin) / H_mag


def _deviation_report(measured, freqs, label, correction_applied=False):
    """Print frequency deviation report for a single measurement set."""
    valid = measured[~np.isnan(measured)]
    if len(valid) < 2:
        print(f"\n{label} -- insufficient data")
        return

    mean_db = float(np.mean(valid))
    std_db = float(np.std(valid))
    min_db = float(np.min(valid))
    max_db = float(np.max(valid))

    # Convert dBFS to linear magnitude for percent-deviation calculation
    lin = 10 ** (valid / 20.0)
    arith_mean = float(np.mean(lin))

    abs_pct_dev = np.abs((lin - arith_mean) / max(arith_mean, 1e-30) * 100.0)

    print(f"\n{'=' * 56}")
    print(f"  {label} -- Frequency Deviation Report")
    print(f"{'=' * 56}")
    print(f"  Correction applied  : {'Yes' if correction_applied else 'No'}")
    print(f"  Valid bins          : {len(valid)}/{len(measured)}")
    print()
    print(f"  Amplitude stats:")
    print(f"    Mean (dBFS)       : {mean_db:.2f} dBFS")
    print(f"    Std deviation     : {std_db:.3f} dB")
    print(f"    Range             : {min_db:.2f} - {max_db:.2f} dB (span={max_db-min_db:.2f} dB)")
    print()
    print(f"  Deviation from arithmetic mean (linear magnitude):")
    print(f"    Mean abs % dev   : {float(np.mean(abs_pct_dev)):.3f}%")
    print(f"    Median abs % dev : {float(np.median(abs_pct_dev)):.3f}%")
    print(f"    Max abs % dev    : {float(np.max(abs_pct_dev)):.3f}%")
    print(f"    Std of abs % dev : {float(np.std(abs_pct_dev)):.3f}%")

    # Octave-band breakdown
    print(f"\n  Octave band std deviation (linear %):")
    octaves = [(20, 100, "sub-bass"), (100, 300, "bass"), (300, 800, "low-mid"),
               (800, 2000, "mid"), (2000, 5000, "upper-mid"), (5000, 10000, "presence"),
               (10000, 20000, "brilliance")]
    for lo, hi, name in octaves:
        mask = (freqs >= lo) & (freqs < hi)
        if np.sum(mask) > 0:
            seg_lin = lin[mask]
            s_std_pct = float(np.std(seg_lin / arith_mean * 100.0))
            print(f"    {name:>14} {lo:>5}-{hi:>6} Hz: std_dev%={s_std_pct:.3f}% bins={np.sum(mask)}")

    # Worst offenders
    sorted_idx = np.argsort(-abs_pct_dev)
    print(f"\n  Top 5 worst bins:")
    for rank in range(min(5, len(sorted_idx))):
        j = sorted_idx[rank]
        if np.isnan(measured[j]):
            continue
        print(f"    #{rank + 1}  {freqs[j]:>8.0f} Hz  =>  "
              f"{measured[j]:>7.2f} dBFS  abs_dev={abs_pct_dev[j]:.3f}%")


# ---------------------------------------------------------------------------
# Hardware measurement — single-capture mode (like calibrate_send_v2.py)
# ---------------------------------------------------------------------------
def _measure_all_single_capture(freqs, tone_duration, gap_s, fs, send_gain, amp_factors, verbose=False):
    """Send tones for all frequencies via single OutputStream + InputStream capture.

    Returns array of measured dBFS at each target frequency.
    """
    switcher = _ToneSwitcher(freqs, tone_duration, fs, gap_s, amp_factors)
    total_out_samples = switcher.total_out_samples
    total_s = total_out_samples / fs
    in_offset = [0]
    out_offset = [0]
    capture_data = np.empty(total_out_samples, dtype="float32")

    def _in_cb(indata, frame_count, time_flag, status):
        n = min(frame_count, len(capture_data) - in_offset[0])
        capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
        in_offset[0] += n

    def _out_cb(outdata, frame_count, time_flag, status):
        start = out_offset[0]
        end = min(start + frame_count, total_out_samples)
        n = min(frame_count, end - start)
        buf = np.zeros(n, dtype=np.float64)
        for i in range(len(switcher.tone_arrays)):
            t_start = switcher.tone_starts[i]
            t_end = t_start + len(switcher.tone_arrays[i])
            lo = max(start, t_start)
            hi = min(end, t_end)
            if lo < hi:
                t_lo = lo - t_start
                t_hi = hi - t_start
                buf[lo - start : hi - start] = switcher.tone_arrays[i][t_lo:t_hi]

        if outdata.ndim == 1:
            outdata[:n] = buf
        else:
            outdata[:n, 0] = buf
        out_offset[0] = end
        return (outdata, "continue")

    out_stream = sd.OutputStream(
        device=cfg.send_device, samplerate=fs, channels=1,
        callback=_out_cb, blocksize=512, latency="low",
    )
    in_stream = sd.InputStream(
        device=cfg.recv_device, samplerate=fs, channels=1,
        callback=_in_cb, blocksize=512, latency="low",
    )
    out_stream.start()
    in_stream.start()

    elapsed = 0
    last_reported_s = -1.0
    while in_offset[0] < total_out_samples and elapsed < total_s * 3:
        if verbose and (elapsed - last_reported_s) >= 0.5:
            pct = in_offset[0] / total_out_samples * 100
            bar_len = 20
            filled = int(bar_len * in_offset[0] / total_out_samples)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"\r  Capturing [{bar}] {pct:5.1f}% ({elapsed:.0f}/{total_s:.0f}s)", end="", flush=True)
            last_reported_s = elapsed
        sd.sleep(100)
        elapsed += 0.1

    if verbose:
        print(f"\r  Capturing [{'' + '=' * 20}] 100.0% ({total_s:.0f}/{total_s:.0f}s)\n", flush=True)

    out_stream.stop()
    in_stream.stop()
    out_stream.close()
    in_stream.close()

    rec = capture_data[:in_offset[0]].astype(float)

    # Analyze each tone segment
    hop = int((tone_duration + gap_s) * fs)
    measured = np.full(len(freqs), float("nan"))
    n_bins = len(freqs)
    for i, target_freq in enumerate(freqs):
        if verbose:
            pct = (i + 1) / n_bins * 100
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"\r  Analyzing [{bar}] {pct:5.1f}% ({i + 1}/{n_bins})", end="", flush=True)
        seg_start = i * hop
        seg_end = min(seg_start + int(tone_duration * fs), len(rec))
        seg = rec[seg_start:seg_end]
        if len(seg) < 64:
            continue
        fft_vals = np.abs(np.fft.rfft(seg.astype(float)))
        freq_bins_arr = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        idx = np.argmin(np.abs(freq_bins_arr - target_freq))
        measured[i] = 20 * np.log10(max(fft_vals[idx], 1e-10))

    # Clear analysis progress line
    if verbose:
        print(f"\r  Analyzing [" + "=" * 30 + f"] 100.0% ({n_bins}/{n_bins})\n", flush=True)

    return rec, measured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Corrected signal (no correction) or pre-corrected signal (with inverse correction).
# With --correct, loads per-bin correction factors from the calibration profile if available.

def _load_corrections(data_dir, logs_dir):
    """Load pre-computed correction factors from the calibration output directory.

    Returns a tuple of (correction_factors, loaded_from_file) or (None, False) if not found.
    """
    # Try standard paths for corrections NPZ (in priority order)
    correction_paths = [
        data_dir / "cal_send_corrections.npz",
        logs_dir / "cal_send_corrections.npz",
        _REPO_ROOT / "data" / "cal_send_corrections.npz",
        _REPO_ROOT / "logs" / "cal_send_corrections.npz",
    ]
    for path in correction_paths:
        if path.exists():
            data = np.load(str(path))
            factors = data["correction_factor"]
            return factors, str(path)
    return None, False


def main():
    args = parse_args()

    # Locate calibration profile
    if args.cal_file:
        cal_path = Path(args.cal_file)
    else:
        cal_path = _REPO_ROOT / "data" / "cal_send_profile.npz"

    if not cal_path.exists():
        print(f"ERROR: Calibration profile not found: {cal_path}")
        print("Run calibrate_send_v2.py first to generate one.")
        sys.exit(1)

    data = np.load(cal_path, allow_pickle=True)
    freqs_cal = data["frequencies"]  # target frequencies from calibration
    H_db_raw = data["response_H"]     # measured dBFS at each target
    fs = int(cfg.fs)

    n_bins = len(freqs_cal)

    # Tone params
    tone_duration = float(args.tone_duration) if args.tone_duration is not None else float(cfg.tone_duration)
    gap_s = float(args.gap) if args.gap is not None else float(cfg.tone_gap)

    correction_applied = args.correct

    print(f"Calibration profile : {cal_path}")
    print(f"  Bins              : {n_bins}")
    print(f"  Freq range        : {freqs_cal[0]:.1f} - {freqs_cal[-1]:.1f} Hz")
    print(f"  Correction applied: {'Yes' if correction_applied else 'No'}")
    print(f"  Tone duration     : {tone_duration}s")
    print(f"  Mode              : {args.mode}")

    # Compute per-tone amplitude factors
    # The correction factors from the profile are dimensionless multipliers (e.g. 1.2 means "boost by 20%").
    # We multiply them by the base tone amplitude (tone_amplitude * send_gain/100) to get the actual DAC output.
    # This ensures validate sends signals at the same absolute level as calibrate_send_v2.
    base_tone = float(cfg.tone_amplitude) * float(cfg.send_gain) / 100.0  # e.g. 0.2 * 0.70 = 0.14

    if correction_applied:
        # Try loading pre-computed correction factors from profile first
        corr_factors, path = _load_corrections(Path(cfg.data_dir), Path(cfg.logs_dir))
        if corr_factors is not None and len(corr_factors) == n_bins:
            print(f"  Loaded corrections from : {path}")
            amp_factors = base_tone * np.array(corr_factors)
        else:
            # Fallback: compute on-the-fly from raw profile (H_mean / H_smoothed via pct_diff)
            print("  Note: no pre-computed corrections found; computing on-the-fly from profile.")
            H_linear = 10 ** (H_db_raw / 20.0)
            H_mean_linear = float(np.mean(H_linear))
            amp_factors = base_tone * np.where(
                H_linear > 1e-6,
                H_mean_linear / H_linear * np.ones(n_bins),
                np.ones(n_bins),
            )
    else:
        # Baseline: send uniform tone at the same amplitude used in calibration
        amp_factors = np.full(n_bins, base_tone, dtype=np.float64)

    # Measure via hardware
    print(f"\n[{args.mode} mode: sending {n_bins} tones...]\n", flush=True)
    rec, measured = _measure_all_single_capture(
        freqs=freqs_cal, tone_duration=tone_duration, gap_s=gap_s,
        fs=fs, send_gain=cfg.send_gain, amp_factors=amp_factors, verbose=True,
    )

    # Save WAV
    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)

    label_suffix = "corrected" if correction_applied else "baseline"
    wav_path = output_dir / f"validate_{label_suffix}_captured.wav"
    wavfile.write(str(wav_path), fs, rec.astype(np.float32))
    print(f"\nWAV saved: {wav_path}")

    # Build deviation metrics for chart
    valid_mask = ~np.isnan(measured)
    mean_db = float(np.mean(measured[valid_mask]))
    std_db = float(np.std(measured[valid_mask]))
    deviation_db = measured - mean_db

    lin_values = 10 ** (measured / 20.0)
    arith_mean_lin = float(np.mean(lin_values[valid_mask])) if np.any(valid_mask) else 1.0
    pct_dev = np.abs((lin_values - arith_mean_lin) / max(arith_mean_lin, 1e-30) * 100.0)

    # Chart: 2-panel (deviation dB vs absolute % deviation)
    chart_png = build_validate_chart_png(
        freqs=freqs_cal[valid_mask],
        deviation_db=deviation_db[valid_mask],
        pct_dev=pct_dev[valid_mask],
        title=f"Validation {'(Corrected)' if correction_applied else '(Baseline)'}",
    )

    # Save chart PNG
    chart_path = output_dir / f"validate_{label_suffix}_chart.png"
    chart_path.write_bytes(chart_png)
    print(f"Chart saved : {chart_path}")

    # Report
    _deviation_report(measured, freqs_cal, "Validation Result", correction_applied)

    return measured


if __name__ == "__main__":
    main()
