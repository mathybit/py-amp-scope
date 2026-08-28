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
        "--method", choices=["sweep", "pink", "white", "brown"], default="sweep",
        help="Calibration method for send path validation (default: sweep / tone-based)",
    )
    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="sequential",
        help="Capture mode for tone-based validation (default: sequential)",
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

    output_dir = Path(args.output_dir) if args.output_dir else _REPO_ROOT / "logs"
    output_dir.mkdir(exist_ok=True)

    fs = int(cfg.fs)
    n_samples_out = int(cfg.noise_calibration_time * fs)

    # ------------------------------------------------------------------
    # Noise capture path
    # ------------------------------------------------------------------
    if args.method in ("pink", "white", "brown"):
        print(f"\n[Noise validation (method={args.method})]")

        # Load calibration profile for reference freqs / amplitude
        cal_path = Path(args.cal_file) if args.cal_file else _REPO_ROOT / "data" / "cal_send_profile.npz"
        if not cal_path.exists():
            print(f"WARNING: Calibration profile not found at {cal_path}")
            freq_array = np.logspace(math.log10(cfg.freq_min), math.log10(cfg.freq_max), cfg.num_freqs_default)
        else:
            data_ref = np.load(cal_path, allow_pickle=True)
            freq_array = data_ref["frequencies"]

        n_bins = len(freq_array)
        correct_send_applied = args.correct_send

        if correct_send_applied:
            amp_factors = load_send_corrections(Path(cfg.data_dir))[:n_bins]
        else:
            amp_factors = np.ones(n_bins, dtype=np.float64)

        print(f"  Cal ref           : {cal_path}")
        print(f"  Method            : {args.method}")
        print(f"  Freq range        : {freq_array[0]:.1f} - {freq_array[-1]:.1f} Hz ({n_bins} bins)")
        print(f"  Send correction   : {'Yes' if correct_send_applied else 'No'}")
        print(f"  Capture duration  : {cfg.noise_calibration_time}s")

        # Generate noise signal (corrected or uncorrected)
        if correct_send_applied:
            # Apply correction as amplitude modulation across the noise signal
            corr_signal = np.zeros(n_samples_out, dtype=np.float64)
            tone_dur_s = cfg.tone_duration
            gap_s_val = cfg.tone_gap
            hop = int((tone_dur_s + gap_s_val) * fs)
            for i, (freq, amp_f) in enumerate(zip(freq_array, amp_factors)):
                seg_start = i * hop
                seg_end = min(seg_start + int(tone_dur_s * fs), n_samples_out)
                t = np.arange(seg_end - seg_start) / fs
                corr_signal[seg_start:seg_end] += (np.sin(2 * np.pi * freq * t) * amp_f).astype(np.float64)
            noise_sig = generate_noise_signal(args.method, n_samples_out, fs, cfg.tone_amplitude, cfg.send_gain)
            # Scale by correction envelope
            corr_env = np.zeros(n_samples_out, dtype=np.float64)
            for i, amp_f in enumerate(amp_factors):
                seg_start = i * hop
                seg_end = min(seg_start + int(tone_dur_s * fs), n_samples_out)
                corr_env[seg_start:seg_end] += amp_f
            corr_env = np.maximum(corr_env, 1e-6)
            noise_sig = noise_sig * corr_env / max(np.max(corr_env), 1e-6)
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

        # Correction derivation (consistent with calibrate_send pattern)
        if args.method in ("pink", "brown"):
            measured_to_expected = 10 ** (shape_result["smoothed_db"] / 20.0) / max(10 ** (shape_result["expected_shifted_db"] / 20.0), 1e-6)
            corr_factors = np.where(measured_to_expected > 1e-6, 1.0 / measured_to_expected, np.ones_like(measured_to_expected))
        else:
            valid_sm = ~np.isnan(shape_result["smoothed_db"])
            mean_linear = float(np.mean(10 ** (shape_result["smoothed_db"][valid_sm] / 20.0)))
            corr_factors = np.where(smoothed_db > 1e-6, mean_linear / np.maximum(10 ** (smoothed_db / 20.0), 1e-30), np.ones_like(smoothed_db))

        # Noise chart (shifted reference + percent deviation for pink/brown)
        png_bytes, corr_factors_npz, _ = build_noise_chart_png(
            freqs=freq_array,
            H_db=shape_result["measured_db"],
            smoothed_db=smoothed_db,
            expected_db=shape_result.get("expected_shifted_db", shape_result["expected_db"]),
            deviation_db=shape_result.get("deviation_pct", shape_result["deviation_db"]),
            corr_factors=corr_factors,
            num_neighbors=cfg.smoothing_neighbors,
            title=f"Send Validation (Noise {args.method.capitalize()})",
        )

        # Save outputs
        label = "noise" + ("_corr" if correct_send_applied else "")
        wav_path = output_dir / f"validate_{label}_captured.wav"
        wavfile.write(str(wav_path), fs, rec.astype(np.float32))
        chart_path = output_dir / f"validate_{label}_chart.png"
        chart_path.write_bytes(png_bytes)
        np.savez(
            str(output_dir / f"validate_{label}_profile.npz"),
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
        print(f"\nWAV saved  : {wav_path}")
        print(f"Chart saved: {chart_path}")
        print(f"Profile NPZ saved: {output_dir / f'validate_{label}_profile.npz'}")

        # Also build tone-style validate chart for comparison
        valid_mask = ~np.isnan(shape_result["smoothed_db"])
        mean_db_n = float(np.mean(shape_result["smoothed_db"][valid_mask]))
        lin_n = 10 ** (shape_result["smoothed_db"][valid_mask] / 20.0)
        arith_mean_lin = float(np.mean(lin_n))
        pct_dev_n = np.abs((lin_n - arith_mean_lin) / max(arith_mean_lin, 1e-30) * 100.0)
        chart_png = build_validate_chart_png(
            freqs=freq_array[valid_mask],
            deviation_db=shape_result["smoothed_db"][valid_mask] - mean_db_n,
            pct_dev=pct_dev_n,
            title=f"Send Validation (Noise {args.method.capitalize()}) -- dB Deviation",
        )
        chart_path2 = output_dir / f"validate_{label}_dev_chart.png"
        chart_path2.write_bytes(chart_png)

        return shape_result

    # ------------------------------------------------------------------
    # Tone-based path (existing behavior — sweep/white)
    # ------------------------------------------------------------------
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
    n_bins = len(freqs_cal)

    correct_send_applied = args.correct_send
    tone_duration = float(args.tone_duration) if args.tone_duration is not None else float(cfg.tone_duration)
    gap_s = float(args.gap) if args.gap is not None else float(cfg.tone_gap)

    print(f"Calibration profile : {cal_path}")
    print(f"  Method            : {args.method}")
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
