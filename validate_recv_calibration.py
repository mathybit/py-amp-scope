#!/usr/bin/env python
"""Validate receive-path calibration by measuring frequency deviation across 4 configurations.

Tests combinations of send-correction and receive-correction to answer:
does applying receive calibration corrections actually lower std deviation?

Four configurations (labelled cc/cb/bc/bb):
  --correct-send   --correct-recv   |  label   | behavior
  no               no               |  bb      | uniform tone, raw measurement
  yes              no               |  cb      | send-corrected tone, raw measurement
  no               yes              |  bc      | uniform tone, receive corrections applied
  yes              yes              |  cc      | send-corrected tone, receive corrections applied

Usage:
    python validate_recv_calibration.py                          # bb (baseline)
    python validate_recv_calibration.py --correct-send           # cb (send corrected)
    python validate_recv_calibration.py --correct-recv           # bc (recv corrected)
    python validate_recv_calibration.py --correct-send --correct-recv  # cc (both)
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
from utils.audio.analysis_utils import deviation_report, extract_tone_measurements  # noqa: E402
from utils.audio.signal_utils import play_one_freq_single  # noqa: E402
from utils.charting_utils import build_validate_chart_png  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate receive-path calibration frequency response across 4 config combinations.",
        epilog=(
            "Error metric: absolute percent deviation of each bin's amplitude from the "
            "arithmetic mean across all bins. The cc vs cb comparison answers whether "
            "receive corrections lower std deviation.\n\n"
            "With --correct-send, send-correction factors are applied to sent tones.\n"
            "With --correct-recv, receive correction factors (from the cal_recv_*_corr_profile.npz) "
            "are loaded and applied to measured results."
        ),
    )
    parser.add_argument(
        "--cal-file", type=str, default=None,
        help="Path to a specific profile NPZ (default: auto-select based on --correct-recv)",
    )
    parser.add_argument(
        "--correct-send", action="store_true",
        help="Apply send-correction factors (from cal_send_corrections.npz) to sent tones.",
    )
    parser.add_argument(
        "--correct-recv", action="store_true",
        help="Load receive calibration corrections and apply them to measured results. "
             "Requires a receive profile with corrections: cal_recv_{path}_corr_profile.npz.",
    )
    parser.add_argument("--recv-path", choices=["dir", "iso"], default=None,
                        help="Receive path variant (default: config value 'dir')")
    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="single-capture",
        help="Capture mode (default: single-capture)",
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
# Analysis helpers
# ---------------------------------------------------------------------------


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
def _load_corrections(data_dir: Path, logs_dir: Path, recv_path: str):
    """Load per-bin receive correction factors from pre-computed calibration output.

    Returns (correction_factors, loaded_from_file) or (None, False) if not found.
    Tries cal_recv_{path}_corr_corrections.npz first, then _corrections.npz fallback.
    """
    # Try "corr" variant first (explicit receive corrections file)
    prefix = f"cal_recv_{recv_path}_corr_corrections"
    search_paths = [
        data_dir / f"{prefix}.npz",
        logs_dir / f"{prefix}.npz",
        _REPO_ROOT / "data" / f"{prefix}.npz",
        _REPO_ROOT / "logs" / f"{prefix}.npz",
    ]
    for path in search_paths:
        if path.exists():
            d = np.load(str(path))
            factors = d["correction_factor"]
            return factors, str(path)

    # Fallback: try _corrections.npz (old naming, before variant suffix was added)
    prefix_old = f"cal_recv_{recv_path}_corrections"
    search_paths = [
        data_dir / f"{prefix_old}.npz",
        logs_dir / f"{prefix_old}.npz",
        _REPO_ROOT / "data" / f"{prefix_old}.npz",
        _REPO_ROOT / "logs" / f"{prefix_old}.npz",
    ]
    for path in search_paths:
        if path.exists():
            d = np.load(str(path))
            factors = d["correction_factor"]
            return factors, str(path)
    return None, False


def main():
    args = parse_args()

    # Resolve receive path
    recv_path = args.recv_path if args.recv_path else cfg.recv_path

    # Determine config label (cc/cb/bc/bb)
    send_corr = args.correct_send
    recv_corr = args.correct_recv
    label = "bb"  # baseline-send, baseline-recv
    if send_corr:
        label = "cb"  # corrected-send, baseline-recv
    if recv_corr:
        label = ("bc" if not send_corr else "cc")  # bc or cc

    # Determine which profile to load for --correct-recv
    profile_for_recv_corr = None
    if args.cal_file and recv_corr:
        profile_for_recv_corr = Path(args.cal_file)
    elif recv_corr:
        # Default: use the corr variant of the receive profile
        profile_for_recv_corr = _REPO_ROOT / "data" / f"cal_recv_{recv_path}_corr_profile.npz"

    # Validate: --correct-recv requires a profile with corrections to exist
    if recv_corr and not profile_for_recv_corr.exists():
        print(f"WARNING: Receive calibration profile not found at:\n  {profile_for_recv_corr}")
        print("Falling back to on-the-fly correction from raw data (may be inaccurate).")
        recv_corr = False

    # Load config calibration profile for measurement
    cal_dir = Path(cfg.data_dir)
    if args.cal_file and not recv_corr:
        cal_path = Path(args.cal_file)
    elif not recv_corr:
        # Default for baseline: use corr variant (the "gold standard" calibrated profile)
        # TODO: This loads the 'corr' variant even when no corrections are requested.
        # It should default to 'base_profile.npz' instead, with 'corr' only loaded via --cal-file override.
        # Not critical now since we use corr as reference for validation, but worth cleaning up later.
        cal_path = _REPO_ROOT / "data" / f"cal_recv_{recv_path}_corr_profile.npz"
    else:
        cal_path = profile_for_recv_corr

    if not cal_path.exists():
        print(f"ERROR: Receive calibration profile not found: {cal_path}")
        print("Run calibrate_recv.py first to generate one.")
        sys.exit(1)

    data = np.load(cal_path, allow_pickle=True)
    freqs_cal = data["frequencies"]     # target frequencies from calibration
    H_db_raw = data["response_H"]       # measured dBFS at each target
    fs = int(cfg.fs)

    n_bins = len(freqs_cal)

    # Tone params
    tone_duration = float(args.tone_duration) if args.tone_duration is not None else float(cfg.tone_duration)
    gap_s = float(args.gap) if args.gap is not None else float(cfg.tone_gap)

    print(f"Receive profile     : {cal_path}")
    print(f"  Path variant      : {recv_path}")
    print(f"  Config            : {label} (send-correct={send_corr}, recv-correct={recv_corr})")
    print(f"  Bins              : {n_bins}")
    print(f"  Freq range        : {freqs_cal[0]:.1f} - {freqs_cal[-1]:.1f} Hz")
    print(f"  Tone duration     : {tone_duration}s")
    print(f"  Mode              : {args.mode}")

    # Determine per-tone corr_factors for play_one_freq_single (ToneSwitcher handles amp + gain internally)
    if send_corr:
        send_corr_path = _REPO_ROOT / "data" / "cal_send_corrections.npz"
        if not send_corr_path.exists():
            print(f"ERROR: Send corrections file not found: {send_corr_path}")
            sys.exit(1)
        send_data = np.load(str(send_corr_path))
        amp_factors = send_data["correction_factor"][:n_bins]
        print(f"\n  Sent tones: send-corrected (corr range {amp_factors.min():.3f} - {amp_factors.max():.3f})")
    else:
        # Baseline: uniform tone (corr_factors = 1.0)
        amp_factors = np.ones(n_bins, dtype=np.float64)
        print(f"\n  Sent tones: uniform (corr_factor = 1.0)")

    # Measure via hardware
    print(f"\n[{args.mode} mode: sending {n_bins} tones...]\n", flush=True)
    rec, measured = _measure_all_single_capture(
        freqs=freqs_cal, tone_duration=tone_duration, gap_s=gap_s,
        fs=fs, corr_factors=amp_factors, verbose=True,
    )

    # Apply receive corrections to measured results (post-correction)
    recv_corr_applied = False
    if recv_corr:
        corr_factors, loaded_from = _load_corrections(Path(cfg.data_dir), Path(cfg.logs_dir), recv_path)
        if corr_factors is not None and len(corr_factors) == n_bins:
            print(f"\n  Applying receive corrections from: {loaded_from}")
            # Correction factors are linear multipliers; convert dBFS → linear → correct → back to dBFS
            H_lin = 10 ** (measured / 20.0)
            H_corr_lin = H_lin * corr_factors
            measured = 20 * np.log10(H_corr_lin)
            recv_corr_applied = True
        else:
            # Fallback: compute on-the-fly from raw profile (H_mean_linear / H_linear)
            print("\n  Note: no pre-computed corrections found; computing on-the-fly from profile.")
            H_linear = 10 ** (H_db_raw / 20.0)
            H_mean_linear = float(np.mean(H_linear))
            corr_factors = np.where(H_linear > 1e-6, H_mean_linear / H_linear, np.ones(n_bins))
            H_lin = 10 ** (measured / 20.0)
            measured = 20 * np.log10(H_lin * corr_factors)
            recv_corr_applied = True

    # Save WAV
    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)

    label_suffix = f"{recv_path}_{label}"
    wav_path = output_dir / f"validate_recv_{label_suffix}_captured.wav"
    wavfile.write(str(wav_path), fs, rec.astype(np.float32))
    print(f"\nWAV saved: {wav_path}")

    # Build deviation metrics for chart (from measured results, post receive-correction if applied)
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
        title=f"Receive Validation ({label.upper()}) -- {recv_path}",
    )

    # Save chart PNG
    chart_path = output_dir / f"validate_recv_{label_suffix}_chart.png"
    chart_path.write_bytes(chart_png)
    print(f"Chart saved : {chart_path}")

    # Report
    deviation_report(measured, freqs_cal, f"Receive Validation ({label})", send_correction_applied=send_corr, receive_correction_applied=recv_corr_applied)

    return measured


if __name__ == "__main__":
    main()
