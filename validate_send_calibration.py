#!/usr/bin/env python
"""Validate send-path calibration by measuring frequency deviation with/without correction.

Loads the send calibration profile (cal_send_profile.npz), sends a uniform tone at each
bin frequency, and measures how far each returned amplitude deviates from the mean across all
bins. Two modes:

  Default (no --correct-send): sends uncorrected signal; shows raw deviation of the calibration
                               profile.

  --correct-send: loads per-bin correction factors from the saved NPZ file and applies them to the
                  sent tone amplitudes, then measures how much flatter the returned signal is.

The script answers: is the send circuitry uniform enough that calibration matters?

Usage:
    python validate_send_calibration.py              # baseline measurement (no correction)
    python validate_send_calibration.py --correct-send      # with send-correction applied
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
from utils.audio.analysis_utils import extract_tone_measurements, deviation_report  # noqa: E402
from utils.audio.signal_utils import play_one_freq_single  # noqa: E402
from utils.charting_utils import build_validate_chart_png  # noqa: E402
from utils.file_utils import load_send_corrections


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate send-path calibration frequency response.",
        epilog=(
            "Error metric: absolute percent deviation of each bin's amplitude from the "
            "arithmetic mean across all bins. A lower stddev means the correction is working.\n\n"
            "With --correct-send, an inverse filter is computed on-the-fly from the calibration "
            "profile and applied to the sent signal before each tone."
        ),
    )
    parser.add_argument(
        "--cal-file", type=str, default=None,
        help="Path to cal_send_profile.npz (default: data/cal_send_profile.npz)",
    )
    parser.add_argument(
        "--correct-send", action="store_true",
        help="Apply send-correction factors (from cal_send_corrections.npz) to sent tones.",
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
# Hardware measurement — single-capture mode (delegates to play_one_freq_single)
# ---------------------------------------------------------------------------
def _measure_all_single_capture(freqs, tone_duration, gap_s, fs, corr_factors, verbose=False):
    """Send tones via shared play function and analyze captured signal.

    Returns ``(rec_flat, measured_dBFS)`` where *measured_dBFS* has NaN for
    short/missing segments.
    """
    n_samples = int((tone_duration + gap_s) * fs * len(freqs))
    capture_data = np.empty(n_samples, dtype="float32")

    rec_flat = play_one_freq_single(
        freqs=freqs, duration_s=tone_duration, fs=fs, gap_s=gap_s,
        send_device=cfg.send_device, recv_device=cfg.recv_device,
        send_gain=cfg.send_gain, tone_amplitude=float(cfg.tone_amplitude),
        corr_factors=corr_factors, capture_data=capture_data, verbose=verbose,
    )

    measured, _ = extract_tone_measurements(rec_flat, freqs, tone_duration, gap_s, fs)
    return rec_flat, measured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Corrected signal (no correction) or pre-corrected signal (with inverse correction).
# With --correct-send, loads per-bin correction factors from the calibration profile if available.

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

    correct_send_applied = args.correct_send

    print(f"Calibration profile : {cal_path}")
    print(f"  Bins              : {n_bins}")
    print(f"  Freq range        : {freqs_cal[0]:.1f} - {freqs_cal[-1]:.1f} Hz")
    print(f"  Send correction   : {'Yes' if correct_send_applied else 'No'}")
    print(f"  Tone duration     : {tone_duration}s")
    print(f"  Mode              : {args.mode}")

    # Determine per-tone corr_factors for play_one_freq_single (ToneSwitcher handles amp + gain internally)
    if correct_send_applied:
        amp_factors = load_send_corrections(Path(cfg.data_dir))[:n_bins]
    else:
        # Baseline: uniform tone (corr_factors = 1.0)
        amp_factors = np.ones(n_bins, dtype=np.float64)

    # Measure via hardware
    print(f"\n[{args.mode} mode: sending {n_bins} tones...]\n", flush=True)
    rec, measured = _measure_all_single_capture(
        freqs=freqs_cal, tone_duration=tone_duration, gap_s=gap_s,
        fs=fs, corr_factors=amp_factors, verbose=True,
    )

    # Save WAV
    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)

    label_suffix = "corr_send" if correct_send_applied else "base"
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
        title=f"Send Validation {'(Send-Corrected)' if correct_send_applied else '(Baseline)'}",
    )

    # Save chart PNG
    chart_path = output_dir / f"validate_{label_suffix}_chart.png"
    chart_path.write_bytes(chart_png)
    print(f"Chart saved : {chart_path}")

    # Report
    deviation_report(measured, freqs_cal, "Validation Result", correct_send_applied)

    return measured


if __name__ == "__main__":
    main()
