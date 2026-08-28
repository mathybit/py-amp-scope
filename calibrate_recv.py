#!/usr/bin/env python
"""Per-frequency receive DI circuitry calibration for PyAmpScope.

Sends send-corrected tones (pre-scaled using send path correction factors) through
the full loopback path (USB out -> send DI -> receive DI -> USB line-in), measures
each bin's response, and computes per-bin correction factors to cancel the receive
DI circuitry's frequency response during amplifier profiling.

This script isolates what the receive DI circuitry does to a flat signal by sending
a send-corrected tone at each frequency. What we measure is the receive DI's contribution
alone (send effects are canceled by pre-correction).

Two path variants:
  --recv-path dir : Direct coupling capacitor path (default)
  --recv-path iso : Isolation transformer path

Usage:
    python calibrate_recv.py --recv-path dir   # direct path calibration
    python calibrate_recv.py --recv-path iso   # isolated path calibration

After calibration, the saved correction factors are applied during amplifier profiling
to undo the receive DI's coloration and reveal the amp's true spectral response.
"""

import argparse
import math
import numpy as np
import os
from scipy.io import wavfile
import shutil
import sounddevice as sd
import sys
from pathlib import Path


# Add repo root to path so we can import utils directly
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import config as cfg  # noqa: E402
from utils.audio.analysis_utils import fft_db, analyze_noise_response, smooth_moving_average, compare_noise_spectral_shape, print_noise_shape_report
from utils.audio.signal_utils import generate_noise_signal, play_one_freq_single, play_one_freq_seq, print_freq_table
from utils.charting_utils import build_multichart_png
from utils.file_utils import save_cal_profile, load_send_corrections


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments merged with config defaults."""
    parser = argparse.ArgumentParser(
        description="Per-frequency receive DI circuitry calibration v2 -- OutputStream/InputStream callback API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Receive path variants:\n"
            "  dir : Direct coupling capacitor path (default)\n"
            "  iso : Isolation transformer path\n\n"
            "Modes:\n"
            "  single : Single-Capture mode- one OutputStream switches tones, one InputStream captures all.\n"
            "                   Faster (~30s for 60 bins). Recommended.\n"
            "  sequential     : One OutputStream+InputStream pair per frequency. More robust\n"
            "                   on hardware with unstable PortAudio state between streams.\n"
        ),
    )

    parser.add_argument(
        "--mode", choices=["sequential", "single"], default="single",
        help="Capture mode (default: single)",
    )
    parser.add_argument("--recv-path", choices=["dir", "iso"], default=None,
                        help="Receive path variant (default: config value 'dir')")

    parser.add_argument(
        "--correct-send", action="store_true", default=False,
        help="Apply send-correction factors (from cal_send_corrections.npz) to sent tones. "
             "Default: uniform tone (baseline).",
    )

    # Frequency range parameters
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

    # Method override
    parser.add_argument(
        "--method", choices=["sweep", "pink", "white", "brown"], default=None,
        help="Calibration signal type (from config by default)",
    )

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
    method = args.method if args.method is not None else cfg.cal_method
    fs = int(cfg.fs)
    freq_min = int(cfg.freq_min)
    freq_max = int(cfg.freq_max)
    send_device = cfg.send_device
    recv_device = cfg.recv_device
    tone_amplitude = float(cfg.tone_amplitude)

    # CLI overrides for gain
    send_gain = args.send_gain if args.send_gain is not None else cfg.send_gain
    recv_gain = args.recv_gain if args.recv_gain is not None else cfg.recv_gain
    output_dir = Path(cfg.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)

    # CLI overrides for config values
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

    # Receive path resolution
    recv_path = args.recv_path if args.recv_path is not None else cfg.recv_path  # "dir" or "iso"
    if recv_path not in ("dir", "iso"):
        print(f"ERROR: Invalid --recv-path '{recv_path}'. Use 'dir' or 'iso'.", file=sys.stderr)
        sys.exit(1)

    # Tone duration / gap — from config, overridable via CLI
    tone_duration = float(cfg.tone_duration)
    gap_s = float(cfg.tone_gap)
    if args.tone_duration is not None:
        tone_duration = float(args.tone_duration)
    if args.gap is not None:
        gap_s = float(args.gap)

    # Frequency bin count: max(time-constrained floor, config default)
    min_time = float(cfg.min_calibration_time)
    config_default_bins = int(cfg.num_freqs_default)
    auto_bins = math.ceil(min_time / (tone_duration + gap_s))
    num_freqs_arg = int(args.num_freqs) if args.num_freqs is not None else None
    if num_freqs_arg is not None:
        num_freqs = max(auto_bins, num_freqs_arg)
    else:
        num_freqs = max(auto_bins, config_default_bins)

    # Channel overrides (not-implemented for actual stream routing; stored for metadata)
    send_ch = args.send_ch if args.send_ch is not None else cfg.send_ch
    recv_ch = args.recv_ch if args.recv_ch is not None else cfg.recv_ch

    # Total signal duration for metadata
    total_s = num_freqs * (tone_duration + gap_s)

    # Generate log-spaced frequency array directly from computed count
    freq_array = np.logspace(np.log10(freq_min), np.log10(freq_max), num=num_freqs)

    print("=" * 60)
    print(f"PyAmpScope Receive DI Calibration Profile")
    print(f"  Mode              : {args.mode} (OutputStream/InputStream callbacks)")
    print(f"  Tone duration     : {tone_duration}s")
    print(f"  Gap               : {gap_s}s between tones")
    print(f"  Sample rate       : {fs} Hz")
    print(f"  Freq range        : {freq_min}-{freq_max} Hz")
    print(f"  Send device       : {send_device}")
    print(f"  Recv device       : {recv_device}")
    print(f"  Send gain         : {send_gain}%")
    print(f"  Recv gain         : {recv_gain}%")
    print(f"  Send ch           : {send_ch} (phase-2: not yet applied to streams)")
    print(f"  Recv ch           : {recv_ch} (phase-2: not yet applied to streams)")
    print(f"  Recv path         : {recv_path}")
    print(f"  Output dir        : {output_dir}")
    print("=" * 60)
    mode_label = "send-corrected" if args.correct_send else "baseline (uniform tone)"
    print(f"  Mode: {mode_label}")

    variant = "corr" if args.correct_send else "base"

    # Print frequency table
    print_freq_table(freq_array, fs, args.mode, tone_duration=tone_duration, gap_s=gap_s)
    
    print(f"  Base tone amplitude (unscaled) : {tone_amplitude:.4f}")
    if args.correct_send:
        # Apply send-correction factors
        send_corr_factors = load_send_corrections(output_dir)[:num_freqs]
        print("  Using send-corrected amplitudes")
        print(f"  Corrected amp range            : {send_corr_factors.min() * tone_amplitude:.4f} - {send_corr_factors.max() * tone_amplitude:.4f}")
        max_correction = float(np.max(send_corr_factors[:num_freqs]))
        print(f"  Max send correction factor     : {max_correction:.3f} ({max_correction * 100 - 100:+.1f}%)")
    else:
        # Baseline: uniform tone, no send correction applied
        send_corr_factors = np.ones(num_freqs, dtype=np.float64)
        print("  Using uniform tone amplitudes (no send correction)")
        print(f"  Uniform amp values             : {send_corr_factors[0] * tone_amplitude:.4f}")

    if args.dry_run:
        print("\n[Dry run -- skipping hardware play/capture and file writes.]")
        return

    # -----------------------------------------------------------------------
    # Run calibration
    # -----------------------------------------------------------------------
    print(f"\n[{num_freqs} frequencies across {args.mode} mode...]", flush=True)
    results = []  # list of (freq, amplitude_db, rms)
    valid_results = []

    if method in ("pink", "white", "brown"):
        print(f"\n[{method.upper()} noise calibration: generating and streaming {cfg.noise_calibration_time}-second noise signal...]", flush=True)
        n_samples = fs * cfg.noise_calibration_time

        # Generate noise signal
        noise_signal = generate_noise_signal(
            method=method, n_samples=n_samples, fs=fs,
            tone_amplitude=float(cfg.tone_amplitude), send_gain=float(send_gain)
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

        in_stream = sd.InputStream(device=int(recv_device), samplerate=fs, channels=1, callback=_noise_in_cb, blocksize=512, latency="low")
        out_stream = sd.OutputStream(device=int(send_device), samplerate=fs, channels=1, callback=_noise_out_cb, blocksize=512, latency="low")
        in_stream.start()
        out_stream.start()

        elapsed_s = 0
        while in_offset[0] < n_samples and elapsed_s < n_samples / fs * 1.5:
            sd.sleep(100)
            elapsed_s += 0.1
            if int(elapsed_s) % 10 == 0:
                pct = in_offset[0] / n_samples * 100
                filled = int(20 * in_offset[0] / n_samples)
                bar_str = "=" * filled + "-" * (20 - filled)
                print(f"\r  Capturing [{bar_str}] {pct:5.1f}% ({elapsed_s:.0f}/{n_samples/fs:.0f}s)", end="", flush=True)

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
        freqs_out, amp_db_arr, rms_arr = analyze_noise_response(captured_signal=rec_flat, freq_array=freq_array, fs=fs)
        for fv, av, rv in zip(freqs_out, amp_db_arr, rms_arr):
            results.append((fv, av, rv))

        # Noise spectral shape analysis (pink/white/brown — reporting only)
        shape_result = compare_noise_spectral_shape(rec_flat, method, freq_array, fs)
        print_noise_shape_report(shape_result, method)

        # Save single combined WAV capture
        wave_path = output_dir / f"cal_recv_{recv_path}_{variant}_captured.wav"
        wavfile.write(str(wave_path), fs, rec_flat.astype(np.float32))
        print(f"\n  WAV saved : {wave_path}")

        # Print progress per-bin (noise method does a single FFT, no per-tone bar needed)
        valid_count = sum(1 for _, a, _ in results if not np.isnan(a))
        print(f"  FFT analysis complete: {valid_count}/{len(results)} bins measured")


    elif method in ("sweep"):
        # Tone-based pipeline (sequential or single-capture mode)
        print(f"\n[{num_freqs} frequencies across {args.mode} mode...]", flush=True)
        if args.mode == "sequential":
            print("\n[Sequential mode: one OutputStream+InputStream per frequency]")

            # Save WAV capture per-frequency files
            wave_dir = output_dir / f"cal_recv_{recv_path}_{variant}_captured"
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

                # Build per-frequency amplitude for sequential mode
                corr_factor_i = send_corr_factors[i] if i < len(send_corr_factors) else 1.0
                try:
                    rec_flat = play_one_freq_seq(
                        freq=target_freq, duration_s=tone_duration, fs=fs,
                        send_device=send_device, recv_device=recv_device,
                        send_gain=send_gain, tone_amplitude=tone_amplitude, corr_factor=corr_factor_i,
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
                    send_gain=send_gain, tone_amplitude=tone_amplitude, corr_factors=send_corr_factors,
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
            wave_path = output_dir / f"cal_recv_{recv_path}_{variant}_captured.wav"
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
            "recv_path": recv_path,
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

        # Method-gated: pink/brown uses theoretical-reference; sweep/white uses arithmetic mean
        png_bytes = None
        correction_npz_path = None
        corr_factors = None
        if method in ("pink", "brown"):
            # Generate expected profile for this noise method
            f_ref = freq_array[0]
            expected_db = np.zeros_like(freq_array)
            if method == "pink":
                mask = freq_array > 0
                expected_db[mask] = -3.0103 * np.log10(freq_array[mask] / f_ref)
            else:  # brown
                mask = freq_array > 0
                expected_db[mask] = -6.0206 * np.log10(freq_array[mask] / f_ref)

            # Least-squares shift of theoretical reference to match measurement level
            valid_for_shift = ~np.isnan(smoothed_db) & (freq_array > 0)
            diff_all = smoothed_db[valid_for_shift] - expected_db[valid_for_shift]
            shift_db = float(np.mean(diff_all))
            expected_shifted_db = expected_db + shift_db

            # Deviation from shifted theory as percent difference (consistent with sweep/white)
            shifted_lin = 10 ** (expected_shifted_db / 20.0)
            measured_lin = 10 ** (smoothed_db / 20.0)
            deviation_pct = np.abs((measured_lin - shifted_lin) / np.maximum(shifted_lin, 1e-30) * 100.0)

            # Correction: linear inverse of (measured / shifted_theory), preserves expected slope
            measured_to_expected = measured_lin / np.maximum(shifted_lin, 1e-6)
            corr_factors = np.where(measured_to_expected > 1e-6, 1.0 / measured_to_expected, np.ones_like(measured_to_expected))

            # Deviation in dB for backward compat
            deviation_db = smoothed_db - expected_shifted_db

            # Build pink/brown-specific chart (3 panels: response+theory / deviation / correction)
            from utils.charting_utils import build_noise_chart_png as _build_noise_chart
            png_bytes, _, _ = _build_noise_chart(
                freqs=freq_array,
                H_db=H_db_input,
                smoothed_db=smoothed_db,
                expected_db=expected_shifted_db,  # shifted reference for charting
                deviation_db=deviation_pct,        # percent deviation for charting
                corr_factors=corr_factors,
                num_neighbors=cfg.smoothing_neighbors,
                title="Receive DI Response ({}) Calibration - {} Noise".format(recv_path, method.capitalize()),
            )
        else:
            # sweep / white — mean-based correction and standard chart
            mean_db_val = float(np.mean(H_db_input))
            mean_linear = 10 ** (mean_db_val / 20.0)
            smoothed_linear = 10 ** (smoothed_db / 20.0)
            corr_factors = mean_linear / smoothed_linear

            png_bytes, _, _ = build_multichart_png(
                freqs=freq_array,
                H_db=H_db_input,
                num_neighbors=cfg.smoothing_neighbors,
                title=f"Receive DI Response ({recv_path}) Calibration {'(send-corrected)' if args.correct_send else '(uniform tone)'}",
            )

        if png_bytes is not None:
            # Save profile NPZ (directly in output_dir)
            npz_path = save_cal_profile(
                output_dir, f"cal_recv_{recv_path}_{variant}", metadata,
                response_H=valid_amps if valid_results else amp_array,
                freqs=freq_array,
                correction_filter=None,  # corrected via saved per-bin factors instead
            )
            print(f"  Profile saved: {npz_path}")

            # Save smoothed data and correction factors to corrections NPZ
            correction_npz_path = output_dir / f"cal_recv_{recv_path}_{variant}_corrections.npz"
            np.savez(
                str(correction_npz_path),
                freqs=freq_array,
                response_H_linear=np.maximum(valid_amps if valid_results else amp_array, -200),
                smoothed_H_db=smoothed_db,
                correction_factor=corr_factors,
                mean_H_linear=np.array([float(np.mean(corr_factors))]),
            )
            print(f"  Correction factors saved : {correction_npz_path}")

            # Save multi-chart PNG alongside profile
            chart_path = output_dir / f"cal_recv_{recv_path}_{variant}_chart.png"
            chart_path.write_bytes(png_bytes)
            print(f"  Chart saved            : {chart_path}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
