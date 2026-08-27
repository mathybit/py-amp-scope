#!/usr/bin/env python
"""Per-frequency send calibration (v2) for PyAmpScope.

Uses the OutputStream/InputStream callback API for simultaneous playback and
capture on hardware where sd.play()/sd.rec() silently fails during capture.

Inverse correction filter is ALWAYS computed and saved alongside the profile.

Two operating modes:
  sequential  : one OutputStream+InputStream pair per frequency (like v1, but with callbacks)
  single : Single-Capture Mode - one OutputStream + one InputStream for all frequencies;
                   OutputStream switches tones during the recording. Faster, no stream
                   lifecycle overhead between bins.

After calibration, run validate_send_calibration.py to verify whether the correction is
worth applying by measuring frequency deviation before and after correction.

Usage:
    python calibrate_send.py --mode single   # faster, recommended
    python calibrate_send.py --mode sequential        # per-frequency streams
    python calibrate_send.py --dry-run                # show config and frequency table only
"""

import argparse
import math
import numpy as np
import os
from pathlib import Path
from scipy.io import wavfile
import shutil
import sounddevice as sd
import sys
from typing import Optional


# Add repo root to path so we can import utils directly
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import config as cfg  # noqa: E402
from utils.audio.analysis_utils import fft_db, analyze_noise_response, smooth_moving_average
from utils.audio.signal_utils import generate_noise_signal, play_one_freq_single, play_one_freq_seq, print_freq_table
from utils.charting_utils import build_multichart_png
from utils.file_utils import save_cal_profile


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments merged with config defaults."""
    parser = argparse.ArgumentParser(
        description="Per-frequency calibration v2 — OutputStream/InputStream callback API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  single : Single-Capture Mode - one OutputStream switches tones, one InputStream captures all.\n"
            "                   Faster (~30s for 60 bins). Recommended.\n"
            "  sequential     : One OutputStream+InputStream pair per frequency. More robust\n"
            "                   on hardware with unstable PortAudio state between streams.\n"
        ),
    )

    parser.add_argument(
        "--mode", choices=["sequential", "single"], default="single",
        help="Capture mode (default: single)",
    )
    parser.add_argument("--method", choices=["sweep", "pink", "white", "brown"],
                        default=None, help="Calibration signal type (from config by default)")
    parser.add_argument("--freq-min", type=int, default=None, help=f"Lowest analysis frequency Hz (config: {cfg.freq_min})")
    parser.add_argument("--freq-max", type=int, default=None, help=f"Highest analysis frequency Hz (config: {cfg.freq_max})")

    # Device parameters
    parser.add_argument("--send-device", type=int, default=None, help=f"Send device index (config: {cfg.send_device})")
    parser.add_argument("--recv-device", type=int, default=None, help=f"Receive device index (config: {cfg.recv_device})")
    parser.add_argument("--send-ch", choices=["LEFT", "RIGHT", "STEREO"], default=None,
                        help=f"Send channel (config: {cfg.send_ch})")
    parser.add_argument("--recv-ch", choices=["LEFT", "RIGHT", "STEREO"], default=None,
                        help=f"Receive channel (config: {cfg.recv_ch})")

    # Gain parameters (percentage 0-100)
    parser.add_argument("--send-gain", type=float, default=None, help=f"Send gain pct (config: {cfg.send_gain})")
    parser.add_argument("--recv-gain", type=float, default=None, help=f"Receive gain pct (config: {cfg.recv_gain})")

    # Sample rate
    parser.add_argument("--fs", type=int, default=None, help=f"Sample rate Hz (config: {cfg.fs})")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output directory for profile files (config: {cfg.data_dir})")

    # Tone parameters
    parser.add_argument("--num-freqs", type=int, default=None,
                        help=f"Number of frequency bins (config: {cfg.num_freqs_default})")
    parser.add_argument("--tone-duration", type=float, default=None,
                        help=f"Tone duration per frequency in seconds (config: {cfg.tone_duration})")
    parser.add_argument("--gap", type=float, default=None,
                        help=f"Gap between tones in seconds (config: {cfg.tone_gap})")

    # Actions
    parser.add_argument("--dry-run", action="store_true",
                        help="Show config and frequency table; skip hardware")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ── Load config values directly from config/config.py ────────────────────
    method = cfg.cal_method
    fs = int(cfg.fs)
    freq_min = int(cfg.freq_min)
    freq_max = int(cfg.freq_max)
    send_device = cfg.send_device
    recv_device = cfg.recv_device
    tone_amplitude = float(cfg.tone_amplitude)

    # CLI overrides for gain (config values are 90, but user may want to override)
    send_gain = args.send_gain if args.send_gain is not None else cfg.send_gain
    recv_gain = args.recv_gain if args.recv_gain is not None else cfg.recv_gain
    output_dir = Path(cfg.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)

    # CLI overrides for config values
    if args.method is not None:
        method = args.method
    if args.freq_min is not None:
        freq_min = args.freq_min
    if args.freq_max is not None:
        freq_max = args.freq_max
    if args.send_device is not None:
        send_device = args.send_device
    if args.recv_device is not None:
        recv_device = args.recv_device
    if args.fs is not None:
        fs = args.fs
    if args.num_freqs is not None:
        # num_freqs overrides the auto-computed floor; use max(time-constrained, CLI-specified)
        num_freqs_arg = int(args.num_freqs)
        min_time = float(cfg.min_calibration_time)
        config_default_bins = int(cfg.num_freqs_default)
    else:
        num_freqs_arg = None
        min_time = 0.0
        config_default_bins = int(cfg.num_freqs_default)

    # Channel overrides (not-implemented for actual stream routing; stored for metadata)
    send_ch = args.send_ch if args.send_ch is not None else cfg.send_ch
    recv_ch = args.recv_ch if args.recv_ch is not None else cfg.recv_ch

    # Tone duration / gap — from config, overridable via CLI
    tone_duration = float(cfg.tone_duration)
    gap_s = float(cfg.tone_gap)
    if args.tone_duration is not None:
        tone_duration = float(args.tone_duration)
    if args.gap is not None:
        gap_s = float(args.gap)

    # Frequency bin count: max(time-constrained floor, config default)
    auto_bins = math.ceil(min_time / (tone_duration + gap_s))
    if num_freqs_arg is not None:
        num_freqs = max(auto_bins, num_freqs_arg)
    else:
        num_freqs = max(auto_bins, config_default_bins)

    # Total signal duration for metadata (needed before dry-run check)
    total_s = num_freqs * (tone_duration + gap_s)

    # Generate log-spaced frequency array directly from computed count
    freq_array = np.logspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    print("=" * 60)
    print(f"PyAmpScope Send Calibration Profile (v2)")
    print(f"  Method       : {method}")
    print(f"  Mode         : {args.mode} (OutputStream/InputStream callbacks)")
    print(f"  Tone duration: {tone_duration}s")
    print(f"  Gap          : {gap_s}s between tones")
    print(f"  Sample rate  : {fs} Hz")
    print(f"  Freq range   : {freq_min}-{freq_max} Hz")
    print(f"  Send device  : {send_device}")
    print(f"  Recv device  : {recv_device}")
    print(f"  Send gain    : {send_gain}%")
    print(f"  Recv gain    : {recv_gain}%")
    print(f"  Send ch      : {send_ch} (phase-2: not yet applied to streams)")
    print(f"  Recv ch      : {recv_ch} (phase-2: not yet applied to streams)")
    print(f"  Output dir   : {output_dir}")
    print("=" * 60)

    # Print frequency table
    print_freq_table(freq_array, fs, args.mode, tone_duration=tone_duration, gap_s=gap_s)

    if args.dry_run:
        print("\n[Dry run -- skipping hardware play/capture and file writes.]")
        # Don't overwrite existing profile data with zeros.
        return

    # -----------------------------------------------------------------------
    # Run calibration
    # -----------------------------------------------------------------------

    results = []  # list of (freq, amplitude_db, rms) per target frequency
    valid_results = []

    if method in ("pink", "white", "brown"):
        # Broadband noise pipeline: generate 60s continuous signal, stream + capture, analyze via FFT.
        print(f"\n[{method.upper()} noise calibration: generating and streaming {cfg.noise_calibration_time}-second noise signal...]", flush=True)
        n_samples = fs * cfg.noise_calibration_time  # capture duration

        # Generate noise signal
        noise_signal = generate_noise_signal(
            method=method, n_samples=n_samples, fs=int(fs),
            tone_amplitude=float(cfg.tone_amplitude), send_gain=float(send_gain),
        )
        clip_ratio = float(np.sum(np.abs(noise_signal) > 0.99)) / len(noise_signal) * 100

        print(f"  Signal: {len(noise_signal)} samples ({len(noise_signal)/fs:.1f}s), RMS={np.sqrt(np.mean(noise_signal**2)):.6f}, clip={clip_ratio:.3f}%")

        if clip_ratio > 0.5:
            print("  WARNING: Signal exceeds 0.99 threshold -- check gain settings.")

        # Stream + capture via OutputStream/InputStream callbacks
        out_offset = [0]
        in_offset = [0]
        capture_data = np.empty(n_samples, dtype="float32")

        def _noise_in_cb(indata, frame_count, time_flag, status):
            n = min(frame_count, len(capture_data) - in_offset[0])
            if n > 0:
                capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
            in_offset[0] += n

        def _noise_out_cb(outdata, frame_count, time_flag, status):
            start = out_offset[0]
            end = min(start + frame_count, len(noise_signal))
            n = min(frame_count, end - start)
            buf = np.zeros(n, dtype=np.float64)
            buf[:n] = noise_signal[start:end]
            if outdata.ndim == 1:
                outdata[:n] = buf
            else:
                outdata[:n, 0] = buf
            out_offset[0] = end
            return (outdata, "continue")

        in_stream = sd.InputStream(device=int(recv_device), samplerate=int(fs), channels=1, callback=_noise_in_cb, blocksize=512, latency="low")
        out_stream = sd.OutputStream(device=int(send_device), samplerate=int(fs), channels=1, callback=_noise_out_cb, blocksize=512, latency="low")
        in_stream.start()
        out_stream.start()

        elapsed_s = 0
        while in_offset[0] < n_samples and elapsed_s < n_samples / fs * 1.5:
            sd.sleep(100)
            elapsed_s += 0.1
            if int(elapsed_s) % 10 == 0:
                pct = in_offset[0] / n_samples * 100
                bar_len = 20
                filled = int(bar_len * in_offset[0] / n_samples)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  Capturing [{bar}] {pct:5.1f}% ({elapsed_s:.0f}/{n_samples/fs:.0f}s)", end="", flush=True)

        if in_offset[0] < n_samples:
            print(f"\n  WARNING: Capture incomplete ({in_offset[0]}/{n_samples} samples)")

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()

        # Clear progress line
        print(f"\r  Capturing [{'=' * 20}] 100.0% ({in_offset[0]}/{n_samples} samples)\n", flush=True)

        rec_flat = capture_data[:in_offset[0]].astype(float)
        print(f"  Captured {len(rec_flat)} samples ({len(rec_flat)/fs:.1f}s), RMS={np.sqrt(np.mean(rec_flat**2)):.6f}")

        # Analyze via FFT
        freqs_out, amp_db_all, rms_all = analyze_noise_response(captured_signal=rec_flat, freq_array=freq_array, fs=int(fs))
        for f_val, a_val, r_val in zip(freqs_out, amp_db_all, rms_all):
            results.append((f_val, a_val, r_val))

        # Save single combined WAV capture
        wave_path = output_dir / "cal_send_captured.wav"
        wavfile.write(str(wave_path), int(fs), rec_flat.astype(np.float32))
        print(f"\n  WAV saved : {wave_path}")

        # Print progress per-bin (noise method does a single FFT, no per-tone bar needed)
        valid_count = sum(1 for _, a, _ in results if not np.isnan(a))
        print(f"  FFT analysis complete: {valid_count}/{len(results)} bins measured\n")

    elif method in ("sweep"):
        # Tone-based pipeline (sequential or single-capture mode)
        print(f"\n[{num_freqs} frequencies across {args.mode} mode...]", flush=True)
        results = []  # list of (freq, amplitude_db)
        valid_results = []

        if args.mode == "sequential":
            print("\n[Sequential mode: one OutputStream+InputStream per frequency]")

            # Save WAV capture per-frequency files
            wave_dir = output_dir / "cal_send_captured"
            try:
                shutil.rmtree(wave_dir, ignore_errors=True)
            except Exception as e:
                pass
            wave_dir.mkdir(exist_ok=True)

            for i, target_freq in enumerate(freq_array):
                pct = (i + 1) / num_freqs * 100
                bar_len = 50
                filled = int(bar_len * pct / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  [{bar}] {pct:5.1f}% ({i + 1}/{num_freqs}) {target_freq:.1f}Hz", end="", flush=True)
                try:
                    rec_flat = play_one_freq_seq(
                        freq=target_freq, duration_s=tone_duration, fs=fs,
                        send_device=send_device, recv_device=recv_device,
                        send_gain=send_gain, tone_amplitude=tone_amplitude, corr_factor=None
                    )
                except Exception as e:
                    print(f"\n  [ERROR: {e}]", flush=True)
                    results.append((target_freq, float("nan"), 0.0))
                    continue

                if len(rec_flat) == 0:
                    print(f"  [SKIP: no data]", flush=True)
                    results.append((target_freq, float("nan"), 0.0))
                    continue

                rms = float(np.sqrt(np.mean(rec_flat**2)))
                db_target = fft_db(rec_flat, target_freq, fs)
                results.append((target_freq, db_target, rms))
                #print(f"  Captured {len(rec_flat)} samples @ {fs}Hz", flush=True)
                print(f"  RMS={rms:.6f}  {target_freq:.1f}Hz@{db_target:.1f}dBFS", end="", flush=True)

                # Save WAV capture per-frequency file
                wave_path = wave_dir / f"freq_{i:03d}.wav"
                wavfile.write(str(wave_path), fs, rec_flat.astype(np.float32))

            print("\n", flush=True)

        elif args.mode == "single":
            # Single-capture mode
            n_samples = int(total_s * fs)
            capture_data = np.empty(n_samples, dtype="float32")

            print(f"\n[{total_s:.1f}s of simultaneous recording active]", flush=True)
            try:
                rec_flat = play_one_freq_single(
                    freqs=freq_array, duration_s=tone_duration, fs=fs, gap_s=gap_s,
                    send_device=send_device, recv_device=recv_device,
                    send_gain=send_gain, tone_amplitude=tone_amplitude,
                    capture_data=capture_data, verbose=True,
                )
            except Exception as e:
                print(f"\n  [ERROR during single capture: {e}]", file=sys.stderr)
                sys.exit(1)

            # Analyze each tone window in the captured signal
            hop = int((tone_duration + gap_s) * fs)
            for i, target_freq in enumerate(freq_array):
                pct = (i + 1) / num_freqs * 100
                bar_len = 50
                filled = int(bar_len * pct / 100)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  Analyzing [{bar}] {pct:5.1f}% ({i + 1}/{num_freqs})", end="", flush=True)
                seg_start = i * hop
                seg_end = min(seg_start + int(tone_duration * fs), len(rec_flat))
                seg = rec_flat[seg_start:seg_end]

                if len(seg) < 64:
                    results.append((target_freq, float("nan"), 0.0))
                    continue

                rms = float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 0.0
                db_target = fft_db(seg, target_freq, fs)
                results.append((target_freq, db_target, rms))

            # Clear analysis progress line
            print("\r  Analyzing [" + "=" * bar_len + f"] 100.0% ({num_freqs}/{num_freqs})\n", flush=True)
            print(f"\nCaptured {len(rec_flat)} samples ({len(rec_flat)/fs:.1f}s), RMS={np.sqrt(np.mean(rec_flat**2)):.6f}")

            # Save WAV capture file (single combined file in data/ root)
            wave_path = output_dir / "cal_send_captured.wav"
            wavfile.write(str(wave_path), fs, rec_flat.astype(np.float32))
            print(f"\n  WAV saved : {wave_path}")

        else:
            raise NotImplementedError(f"Unsupported mode: {args.mode}")


    # -----------------------------------------------------------------------
    # Compile and save profile
    # -----------------------------------------------------------------------
    amp_array = np.array([r[1] for r in results])
    valid_results = [(f, a) for f, a, _ in results if not np.isnan(a)]

    print(f"\n{'=' * 52}")
    print("Frequency Response Summary")
    print("=" * 52)
    print(f"{'Freq':>10} {'Amplitude':>12} {'RMS':>10} {'Status'}")
    print("-" * 52)

    for freq, amp, rms in results:
        if np.isnan(amp):
            status = "SKIP"
        elif rms >= 1.0:
            status = "CLIP"
        elif amp < -80:
            status = "WEAK"
        else:
            status = "GOOD"
        print(f"{freq:>10.1f} Hz {amp:>12.1f} dBFS {rms:>10.4f} {status}")

    if valid_results:
        valid_freqs = np.array([r[0] for r in valid_results])
        valid_amps = np.array([r[1] for r in valid_results])
        print(f"\n  Valid bins : {len(valid_results)}/{num_freqs}")
        print(f"  Mean amp   : {np.mean(valid_amps):.1f} dBFS")
        print(f"  Std dev    : {np.std(valid_amps):.1f} dB")
        print(f"  Min amp    : {np.min(valid_amps):.1f} dBFS")
        print(f"  Max amp    : {np.max(valid_amps):.1f} dBFS")

        # Build metadata for profile saving
        metadata = {
            "method": method,
            "mode": args.mode,
            "tone_duration": tone_duration,
            "gap": gap_s,
            "duration": total_s,
            "fs": fs,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "num_freqs": num_freqs,
            "send_device": send_device,
            "recv_device": recv_device,
            "send_gain": send_gain,
            "recv_gain": recv_gain,
            "send_ch": send_ch,
            "recv_ch": recv_ch,
        }

        # Smoothing: centered moving average over nearest neighbors for correction factor
        H_db_input = valid_amps if valid_results else amp_array
        smoothed_db = smooth_moving_average(H_db_input, window_size=cfg.smoothing_neighbors)

        # Correction factor in linear space: correction = mean_linear / smoothed_linear
        mean_db_val = float(np.mean(H_db_input))
        mean_linear = 10 ** (mean_db_val / 20.0)
        smoothed_linear = 10 ** (smoothed_db / 20.0)
        corr_factors = mean_linear / smoothed_linear

        # Build multi-chart (3 panels: response + smoothed trend / deviation sigma / correction factor)
        png_bytes, _, _ = build_multichart_png(
            freqs=freq_array,
            H_db=H_db_input,
            num_neighbors=cfg.smoothing_neighbors,
            title="Send Calibration Response (v2)",
        )

        npz_path = save_cal_profile(
            output_dir, "cal_send", metadata,
            response_H=valid_amps if valid_results else amp_array,
            freqs=freq_array,
            correction_filter=None,  # corrected via saved per-bin factors instead
        )
        print(f"  Profile saved: {npz_path}")

        # Save smoothed data and correction factors to profile NPZ
        correction_npz_path = output_dir / "cal_send_corrections.npz"
        np.savez(
            str(correction_npz_path),
            freqs=freq_array,
            response_H_linear=np.maximum(valid_amps if valid_results else amp_array, -200),
            smoothed_H_db=smoothed_db,
            correction_factor=corr_factors,
            mean_H_linear=np.array([mean_linear]),
        )
        print(f"  Correction factors saved : {correction_npz_path}")

        # Save multi-chart PNG alongside profile
        chart_path = output_dir / "cal_send_chart.png"
        chart_path.write_bytes(png_bytes)
        print(f"  Chart saved            : {chart_path}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
