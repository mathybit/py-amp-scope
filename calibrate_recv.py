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
from utils.audio.sweep_utils import save_cal_profile  # noqa: E402
from utils.charting_utils import build_multichart_png, _smooth_moving_average  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: load send correction factors
# ---------------------------------------------------------------------------
def _load_send_corrections(data_dir: Path) -> np.ndarray:
    """Load per-bin send correction factors from pre-computed calibration output.

    Returns correction_factor array or exits with error if not found.
    """
    search_paths = [
        data_dir / "cal_send_corrections.npz",
        _REPO_ROOT / "data" / "cal_send_corrections.npz",
        Path.cwd() / "data" / "cal_send_corrections.npz",
        _REPO_ROOT / "logs" / "cal_send_corrections.npz",
    ]
    for p in search_paths:
        if p.exists():
            d = np.load(str(p))
            factors = d["correction_factor"]
            print(f"  Loaded send corrections from : {p}")
            return factors
    print("ERROR: cal_send_corrections.npz not found. Run calibrate_send.py first.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------
def _print_freq_table(freqs, fs, mode, recv_path="dir", tone_duration=1.0, gap_s=0.3):
    """Print a formatted frequency table."""
    spacing = "log" if np.all(np.diff(np.log10(freqs)) > 0) else "linear"

    border = "=" * 52
    title = f"PyAmpScope -- {spacing.capitalize()} Spaced Frequencies ({mode} mode, recv={recv_path})"
    print(f"\n{title}")
    print(border)
    print(f"  Sample rate   : {fs} Hz")
    if mode == "sequential":
        est_total = len(freqs) * (tone_duration + gap_s)
        print(f"  Mode          : sequential (one OutputStream/InputStream per freq)")
        print(f"  Est. duration : ~{est_total:.0f}s at {tone_duration:.1f}s tone + {gap_s:.1f}s gap")

    col_w = 52
    print(f"\n{'Bin':>4}  {'Frequency (Hz)':>16}  {'Samples per wave period':>20}")
    print("-" * col_w)

    for i, f in enumerate(freqs):
        samples_per_cycle = fs / f
        print(f"{i + 1:>4}  {f:>12.1f} Hz  | {samples_per_cycle:>9.0f} samples/period")

    border2 = "=" * col_w
    print(border2)
    print(f"{'Total':>4} bins : {len(freqs)}")


# ---------------------------------------------------------------------------
# ToneSwitcher — re-used from calibrate_send.py (accepts per-bin amp_factors)
# ---------------------------------------------------------------------------
class _ToneSwitcher:
    """Manages per-frequency tone segments for a single OutputStream.

    Accepts pre-computed amp_factors (per-bin corrected amplitudes) rather than
    a raw tone_amplitude — the caller handles correction factor multiplication.
    """

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
            # amp_factors[i] already includes send correction: base_tone * corr_factor[f]
            self.tone_arrays.append(
                (np.sin(2 * np.pi * freq * t) * amp_factors[i]).astype(np.float64)
            )
            self.tone_starts.append(offset)
            offset += tone_samples + self.gap_samples
        self.total_out_samples = offset


# ---------------------------------------------------------------------------
# Single-capture mode — one OutputStream that switches tones
# ---------------------------------------------------------------------------
def _play_one_freq_single(
    freqs, duration_s, fs, gap_s,
    send_device, recv_device, send_gain, amp_factors,
    capture_data, verbose=False,
):
    """Run a single-capture cycle: one OutputStream switching tones + one InputStream.

    amp_factors are pre-computed corrected amplitudes (tone_amplitude * correction_factor[f]).

    Returns the captured signal (already written into capture_data array).
    """
    switcher = _ToneSwitcher(freqs, duration_s, fs, gap_s, amp_factors)
    total_out_samples = switcher.total_out_samples
    total_s = total_out_samples / fs
    in_offset = [0]
    out_offset = [0]

    def _in_cb(indata, frame_count, time_flag, status):
        if status:
            print(f"    [Input status: {status}]", flush=True)
        n = min(frame_count, len(capture_data) - in_offset[0])
        capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
        in_offset[0] += n

    def _out_cb(outdata, frame_count, time_flag, status):
        if status:
            print(f"    [Output status: {status}]", flush=True)

        start = out_offset[0]
        end = min(start + frame_count, total_out_samples)
        n = min(frame_count, end - start)

        buf = np.zeros(n, dtype=np.float64)
        gain_factor = send_gain / 100.0
        for i in range(len(switcher.tone_arrays)):
            t_start = switcher.tone_starts[i]
            t_end = t_start + len(switcher.tone_arrays[i])
            lo = max(start, t_start)
            hi = min(end, t_end)
            if lo < hi:
                t_lo = lo - t_start
                t_hi = hi - t_start
                buf[lo - start : hi - start] = switcher.tone_arrays[i][t_lo:t_hi] * gain_factor

        if outdata.ndim == 1:
            outdata[:n] = buf
        else:
            outdata[:n, 0] = buf
        out_offset[0] = end

        if out_offset[0] >= switcher.total_out_samples:
            return outdata
        return (outdata, "continue")

    try:
        out_stream = sd.OutputStream(
            device=send_device, samplerate=fs, channels=1,
            callback=_out_cb, blocksize=512, latency="low",
        )
        in_stream = sd.InputStream(
            device=recv_device, samplerate=fs, channels=1,
            callback=_in_cb, blocksize=512, latency="low",
        )
        out_stream.start()
        in_stream.start()

        elapsed = 0
        last_reported_s = -1.0
        while in_offset[0] < total_out_samples and elapsed < total_s * 3:
            if verbose and (elapsed - last_reported_s) >= 0.5:
                pct = in_offset[0] / total_out_samples * 100
                bar_len = 50
                filled = int(bar_len * in_offset[0] / total_out_samples)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  Capturing [{bar}] {pct:5.1f}% ({elapsed:.0f}/{total_s:.0f}s)", end="", flush=True)
                last_reported_s = elapsed
            sd.sleep(100)
            elapsed += 0.1

        if verbose:
            print(f"\r  Capturing [{'' + '=' * bar_len}] 100.0% ({total_s:.0f}/{total_s:.0f}s)\n", flush=True)

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()
    except Exception as e:
        raise RuntimeError(f"Single-capture stream error: {e}") from e

    rec_flat = capture_data[: in_offset[0]].astype(float)
    return rec_flat


# ---------------------------------------------------------------------------
# FFT analysis helper
# ---------------------------------------------------------------------------
def _fft_db(sig, target_hz, fs):
    """Get dB value at target frequency via FFT."""
    N = len(sig)
    fft_vals = np.abs(np.fft.rfft(sig.astype(float)))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    idx = np.argmin(np.abs(freqs - target_hz))
    return 20 * np.log10(max(fft_vals[idx], 1e-10))


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
            "  single-capture : One OutputStream switches tones, one InputStream captures all.\n"
            "                   Faster (~30s for 60 bins). Recommended.\n"
        ),
    )

    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="single-capture",
        help="Capture mode (default: single-capture)",
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

    base_tone = float(cfg.tone_amplitude) * (float(send_gain) / 100.0)  # e.g. 0.2 * 0.70 = 0.14

    if args.correct_send:
        # Apply send-correction factors
        send_corr_factors = _load_send_corrections(output_dir)
        amp_factors = base_tone * send_corr_factors[:num_freqs]  # clip to our bin count
        print("  Using send-corrected amplitudes")
    else:
        # Baseline: uniform tone, no send correction applied
        amp_factors = np.full(num_freqs, base_tone, dtype=np.float64)
        print("  Using uniform tone amplitudes (no send correction)")

    print(f"  Base tone amplitude (unscaled) : {base_tone:.4f}")
    if args.correct_send:
        print(f"  Corrected amp range            : {amp_factors.min():.4f} - {amp_factors.max():.4f}")
        max_correction = float(np.max(send_corr_factors[:num_freqs]))
        print(f"  Max send correction factor     : {max_correction:.3f} ({max_correction * 100 - 100:+.1f}%)")
    else:
        print(f"  Uniform amp value              : {amp_factors[0]:.4f}")

    # Print frequency table
    _print_freq_table(freq_array, fs, args.mode, recv_path=recv_path,
                      tone_duration=tone_duration, gap_s=gap_s)

    if args.dry_run:
        print("\n[Dry run -- skipping hardware play/capture and file writes.]")
        return

    # -----------------------------------------------------------------------
    # Run calibration
    # -----------------------------------------------------------------------
    print(f"\n[{num_freqs} frequencies across {args.mode} mode...]", flush=True)
    results = []  # list of (freq, amplitude_db, rms)
    valid_results = []

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
            amp_i = amp_factors[i] if i < len(amp_factors) else base_tone
            try:
                rec_flat = _play_one_freq_seq(
                    freq=target_freq, duration_s=tone_duration, fs=fs,
                    send_device=send_device, recv_device=recv_device,
                    send_gain=send_gain, amp_factor=amp_i,
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
            db_target = _fft_db(rec_flat, target_freq, fs)
            results.append((target_freq, db_target, rms))
            print(f"  RMS={rms:.6f}  {target_freq:.1f}Hz@{db_target:.1f}dBFS", end="", flush=True)

            # Save WAV capture per-frequency file
            wave_path = wave_dir / f"freq_{i:03d}.wav"
            wavfile.write(str(wave_path), fs, rec_flat.astype(np.float32))

        print("\n", flush=True)

    else:
        # Single-capture mode
        n_samples = int(total_s * fs)
        capture_data = np.empty(n_samples, dtype="float32")

        print(f"\n[{total_s:.1f}s of simultaneous recording active]", flush=True)
        try:
            rec_flat = _play_one_freq_single(
                freqs=freq_array, duration_s=tone_duration, fs=fs, gap_s=gap_s,
                send_device=send_device, recv_device=recv_device,
                send_gain=send_gain, amp_factors=amp_factors,
                capture_data=capture_data, verbose=True,
            )
        except Exception as e:
            print(f"\n  [ERROR during single-capture: {e}]", file=sys.stderr)
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
            db_target = _fft_db(seg, target_freq, fs)
            results.append((target_freq, db_target, rms))

        # Clear analysis progress line
        print("\r  Analyzing [" + "=" * bar_len + f"] 100.0% ({num_freqs}/{num_freqs})\n", flush=True)
        print(f"\nCaptured {len(rec_flat)} samples ({len(rec_flat)/fs:.1f}s), RMS={np.sqrt(np.mean(rec_flat**2)):.6f}")

        # Save WAV capture file (single combined file in data/ root)
        wave_path = output_dir / f"cal_recv_{recv_path}_{variant}_captured.wav"
        wavfile.write(str(wave_path), fs, rec_flat.astype(np.float32))
        print(f"\n  WAV saved : {wave_path}")

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
        smoothed_db = _smooth_moving_average(H_db_input, window_size=cfg.smoothing_neighbors)

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
            title=f"Receive DI Response ({recv_path}) Calibration {'(send-corrected)' if args.correct_send else '(uniform tone)'}",
        )

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
            mean_H_linear=np.array([mean_linear]),
        )
        print(f"  Correction factors saved : {correction_npz_path}")

        # Save multi-chart PNG alongside profile
        chart_path = output_dir / f"cal_recv_{recv_path}_{variant}_chart.png"
        chart_path.write_bytes(png_bytes)
        print(f"  Chart saved            : {chart_path}")

    print("\n[Done]")


# ---------------------------------------------------------------------------
# Sequential mode helper (same pattern as calibrate_send.py, but accepts per-bin amp_factor)
# ---------------------------------------------------------------------------
def _play_one_freq_seq(
    freq: float, duration_s: float, fs: int,
    send_device, recv_device,
    send_gain: float, amp_factor: float,
) -> np.ndarray:
    """Play a single frequency with pre-computed amplitude and capture it.

    Returns the captured signal as a numpy array.
    """
    n_samples = int(duration_s * fs)
    t_total = np.arange(n_samples) / fs
    tone_full = (np.sin(2 * np.pi * freq * t_total) * amp_factor).astype(np.float64)

    out_offset = [0]
    in_offset = [0]
    capture_data = np.empty(n_samples, dtype="float32")

    def _out_cb(outdata, frame_count, time_flag, status):
        if status:
            print(f"    [Output status: {status}]", flush=True)
        start = out_offset[0]
        end = min(start + frame_count, len(tone_full))
        n = min(frame_count, end - start)
        gain_factor = send_gain / 100.0
        scaled_tone = tone_full[start:end] * gain_factor
        if outdata.ndim == 1:
            outdata[:n] = scaled_tone
        else:
            outdata[:n, 0] = scaled_tone
        if end < len(tone_full):
            outdata[n:] = 0.0
            out_offset[0] = end
            return (outdata, "continue")
        else:
            outdata[n:] = 0.0
            return outdata

    def _in_cb(indata, frame_count, time_flag, status):
        if status:
            print(f"    [Input status: {status}]", flush=True)
        n = min(frame_count, len(capture_data) - in_offset[0])
        capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
        in_offset[0] += n

    try:
        out_stream = sd.OutputStream(
            device=send_device, samplerate=fs, channels=1,
            callback=_out_cb, blocksize=512, latency="low",
        )
        in_stream = sd.InputStream(
            device=recv_device, samplerate=fs, channels=1,
            callback=_in_cb, blocksize=512, latency="low",
        )
        out_stream.start()
        in_stream.start()

        elapsed = 0
        while in_offset[0] < n_samples and elapsed < int(duration_s * 3):
            sd.sleep(100)
            elapsed += 0.1

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()
    except Exception as e:
        raise RuntimeError(f"Stream error at {freq:.1f}Hz: {e}") from e

    rec_flat = capture_data[: in_offset[0]].astype(float)
    return rec_flat


if __name__ == "__main__":
    main()
