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
import math
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import config as cfg  # noqa: E402
from utils.audio.analysis_utils import (  # noqa: E402
    compare_noise_spectral_shape,
    deviation_report,
    extract_tone_measurements,
    print_noise_shape_report,
    smooth_moving_average,
)
from utils.audio.signal_utils import generate_noise_signal, play_one_freq_single  # noqa: E402
from utils.charting_utils import build_noise_chart_png, build_validate_chart_png  # noqa: E402


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
        "--method", choices=["sweep", "pink", "white", "brown"], default="sweep",
        help="Calibration method (default: sweep / tone-based)",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="single-capture",
        help="Capture mode for tone-based validation (default: single-capture)",
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

    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)

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

    fs = int(cfg.fs)
    n_samples_out = int(cfg.noise_calibration_time * fs)

    # ------------------------------------------------------------------
    # Noise capture path (pink / white / brown)
    # ------------------------------------------------------------------
    if args.method in ("pink", "white", "brown"):
        print(f"\n[Noise validation -- receive path ({args.method}, method={args.method})]")

        # Load calibration profile for reference freqs
        cal_path = Path(args.cal_file) if args.cal_file else _REPO_ROOT / "data" / f"cal_recv_{recv_path}_corr_profile.npz"
        if not cal_path.exists():
            print(f"WARNING: Calibration profile not found at {cal_path}")
            freq_array = np.logspace(math.log10(cfg.freq_min), math.log10(cfg.freq_max), cfg.num_freqs_default)
        else:
            data_ref = np.load(cal_path, allow_pickle=True)
            freq_array = data_ref["frequencies"]

        n_bins = len(freq_array)

        # Load receive corrections (for post-correction comparison)
        recv_corr_factors = None
        if recv_corr:
            profile_for_recv_corr = Path(args.cal_file) if args.cal_file else _REPO_ROOT / "data" / f"cal_recv_{recv_path}_corr_profile.npz"
            if not profile_for_recv_corr.exists():
                print(f"WARNING: Receive calibration profile not found at:\n  {profile_for_recv_corr}")
                print("Falling back to on-the-fly correction from raw data (may be inaccurate).")
                recv_corr = False
            else:
                corr_factors_loaded, loaded_from = _load_corrections(Path(cfg.data_dir), Path(cfg.logs_dir), recv_path)
                if corr_factors_loaded is not None and len(corr_factors_loaded) == n_bins:
                    recv_corr_factors = corr_factors_loaded
                    print(f"  Receive corrections : from {loaded_from}")
                else:
                    # On-the-fly from profile
                    prof_data = np.load(profile_for_recv_corr, allow_pickle=True)
                    H_linear = 10 ** (prof_data["response_H"] / 20.0)
                    recv_corr_factors = np.where(H_linear > 1e-6, np.mean(H_linear) / H_linear, np.ones(n_bins))
                    print("  Receive corrections : computed on-the-fly from profile")

        if send_corr:
            send_corr_path = _REPO_ROOT / "data" / "cal_send_corrections.npz"
            if not send_corr_path.exists():
                print(f"ERROR: Send corrections file not found: {send_corr_path}")
                sys.exit(1)
            amp_factors = np.load(str(send_corr_path))["correction_factor"][:n_bins]
        else:
            amp_factors = np.ones(n_bins, dtype=np.float64)

        print(f"  Cal ref           : {cal_path}")
        print(f"  Method            : {args.method}")
        print(f"  Path variant      : {recv_path}")
        print(f"  Config            : {label.upper()}")
        print(f"  Freq range        : {freq_array[0]:.1f} - {freq_array[-1]:.1f} Hz ({n_bins} bins)")
        print(f"  Send correction   : {'Yes' if send_corr else 'No'}")
        print(f"  Recv correction   : {'Yes' if recv_corr else 'No'}")
        print(f"  Capture duration  : {cfg.noise_calibration_time}s")

        # Generate noise signal (optionally corrected by send factors)
        if send_corr:
            # Apply correction envelope to noise (modulate amplitude per-tone-window)
            tone_dur_s = cfg.tone_duration
            gap_s_val = cfg.tone_gap
            hop = int((tone_dur_s + gap_s_val) * fs)
            corr_env = np.zeros(n_samples_out, dtype=np.float64)
            for i, amp_f in enumerate(amp_factors):
                seg_start = i * hop
                seg_end = min(seg_start + int(tone_dur_s * fs), n_samples_out)
                corr_env[seg_start:seg_end] += amp_f
            corr_env = np.maximum(corr_env, 1e-6)
            base_noise = generate_noise_signal(args.method, n_samples_out, fs, cfg.tone_amplitude, cfg.send_gain)
            noise_sig = base_noise * corr_env / max(np.max(corr_env), 1e-6)
        else:
            noise_sig = generate_noise_signal(args.method, n_samples_out, fs, cfg.tone_amplitude, cfg.send_gain)

        # Stream via callbacks
        capture_data = np.empty(n_samples_out, dtype="float32")
        in_offset = [0]
        out_sent = [0]

        def _out_cb(outdata, frame_count, time_flag, status):
            if status:
                print(f"    [Output status: {status}]", flush=True)
            start = out_sent[0]
            end = min(start + frame_count, n_samples_out)
            n = min(frame_count, end - start)
            buf = noise_sig[start:end].astype(np.float64) * (cfg.send_gain / 100.0)
            if outdata.ndim == 1:
                outdata[:n] = buf[:n]
            else:
                outdata[:n, 0] = buf[:n]
            if end < n_samples_out:
                outdata[n:] = 0.0
                out_sent[0] = end
                return (outdata, "continue")
            outdata[n:] = 0.0
            return outdata

        def _in_cb(indata, frame_count, time_flag, status):
            if status:
                print(f"    [Input status: {status}]", flush=True)
            n = min(frame_count, len(capture_data) - in_offset[0])
            capture_data[in_offset[0]:in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
            in_offset[0] += n

        out_stream = sd.OutputStream(device=cfg.send_device, samplerate=fs, channels=1, callback=_out_cb, blocksize=512, latency="low")
        in_stream = sd.InputStream(device=cfg.recv_device, samplerate=fs, channels=1, callback=_in_cb, blocksize=512, latency="low")
        out_stream.start()
        in_stream.start()

        elapsed = 0
        while in_offset[0] < n_samples_out and elapsed < cfg.noise_calibration_time * 3:
            sd.sleep(100)
            elapsed += 0.1
        pct = min(in_offset[0] / n_samples_out * 100, 100)
        print(f"\r  Capturing [{'#' * int(pct // 2):-50s}] {pct:5.1f}% ({elapsed:.0f}/{cfg.noise_calibration_time:.0f}s)\n", end="", flush=True)

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()
        rec = capture_data[:in_offset[0]].astype(float)
        print("  Capture complete.")

        # Noise shape analysis
        shape_result = compare_noise_spectral_shape(rec, args.method, freq_array, fs)
        print_noise_shape_report(shape_result, args.method)

        if "error" in shape_result:
            sys.exit(1)

        smoothed_db = smooth_moving_average(shape_result["smoothed_db"], window_size=cfg.smoothing_neighbors)

        # Correction derivation (consistent with calibrate_recv pattern)
        if args.method in ("pink", "brown"):
            measured_to_expected = 10 ** (shape_result["smoothed_db"] / 20.0) / max(10 ** (shape_result["expected_shifted_db"] / 20.0), 1e-6)
            corr_factors_for_npz = np.where(measured_to_expected > 1e-6, 1.0 / measured_to_expected, np.ones_like(measured_to_expected))
        else:
            valid_sm = ~np.isnan(shape_result["smoothed_db"])
            mean_linear = float(np.mean(10 ** (shape_result["smoothed_db"][valid_sm] / 20.0)))
            corr_factors_for_npz = np.where(smoothed_db > 1e-6, mean_linear / np.maximum(10 ** (smoothed_db / 20.0), 1e-30), np.ones_like(smoothed_db))

        # Apply receive corrections post-capture (same as tone mode)
        measured_for_chart = shape_result["smoothed_db"].copy()
        recv_corr_applied = False
        if recv_corr_factors is not None:
            H_lin_post = 10 ** (measured_for_chart / 20.0)
            measured_for_chart = 20 * np.log10(H_lin_post * recv_corr_factors)
            recv_corr_applied = True

        # Noise chart (shifted reference + percent deviation for pink/brown)
        png_bytes, corr_factors_npz, _ = build_noise_chart_png(
            freqs=freq_array,
            H_db=shape_result["measured_db"],
            smoothed_db=smoothed_db,
            expected_db=shape_result.get("expected_shifted_db", shape_result["expected_db"]),
            deviation_db=shape_result.get("deviation_pct", shape_result["deviation_db"]),
            corr_factors=corr_factors_for_npz,
            num_neighbors=cfg.smoothing_neighbors,
            title=f"Receive Validation ({label.upper()}) -- Noise {args.method.capitalize()}",
        )

        # Save outputs
        label_suffix = f"{recv_path}_{label}"
        wav_path = output_dir / f"validate_recv_noise_{label_suffix}_captured.wav"
        wavfile.write(str(wav_path), fs, rec.astype(np.float32))
        chart_path = output_dir / f"validate_recv_noise_{label_suffix}_chart.png"
        chart_path.write_bytes(png_bytes)
        np.savez(
            str(output_dir / f"validate_recv_noise_{label_suffix}_profile.npz"),
            frequencies=freq_array,
            response_H=shape_result["measured_db"],
            correction_factor=corr_factors_npz,
            smoothed_response_db=smoothed_db,
            expected_db=shape_result.get("expected_shifted_db", shape_result["expected_db"]),
            deviation_db=shape_result.get("deviation_pct", shape_result["deviation_db"]),
            shift_db=np.array([shape_result.get("shift_db", np.nan)]),
            shape_std_pct=shape_result.get("shape_std_pct", np.nan),
            global_offset_db=np.array([shape_result.get("global_offset_db", np.nan)]),
        )
        print(f"\nWAV saved    : {wav_path}")
        print(f"Chart saved  : {chart_path}")
        print(f"Profile NPZ  : {output_dir / f'validate_recv_noise_{label_suffix}_profile.npz'}")

        # Also build tone-style validate chart for comparison
        valid_mask = ~np.isnan(measured_for_chart)
        mean_db_n = float(np.mean(measured_for_chart[valid_mask]))
        lin_n = 10 ** (measured_for_chart[valid_mask] / 20.0)
        arith_mean_lin = float(np.mean(lin_n))
        pct_dev_n = np.abs((lin_n - arith_mean_lin) / max(arith_mean_lin, 1e-30) * 100.0)
        chart_png = build_validate_chart_png(
            freqs=freq_array[valid_mask],
            deviation_db=measured_for_chart[valid_mask] - mean_db_n,
            pct_dev=pct_dev_n,
            title=f"Receive Validation ({label.upper()}) -- Noise {args.method.capitalize()} -- dB Deviation",
        )
        chart_path2 = output_dir / f"validate_recv_noise_{label_suffix}_dev.png"
        chart_path2.write_bytes(chart_png)

        return shape_result

    # ------------------------------------------------------------------
    # Tone-based path (existing behavior — sweep/white)
    # ------------------------------------------------------------------
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
    n_bins = len(freqs_cal)

    # Tone params
    tone_duration = float(args.tone_duration) if args.tone_duration is not None else float(cfg.tone_duration)
    gap_s = float(args.gap) if args.gap is not None else float(cfg.tone_gap)

    print(f"Receive profile     : {cal_path}")
    print(f"  Method            : {args.method}")
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
