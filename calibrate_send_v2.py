#!/usr/bin/env python
"""Per-frequency send calibration (v2) for PyAmpScope.

Uses the OutputStream/InputStream callback API for simultaneous playback and
capture on hardware where sd.play()/sd.rec() silently fails during capture.

Inverse correction filter is ALWAYS computed and saved alongside the profile.

Two operating modes:
  sequential  : one OutputStream+InputStream pair per frequency (like v1, but with callbacks)
  single-capture : one OutputStream + one InputStream for all frequencies;
                   OutputStream switches tones during the recording. Faster, no stream
                   lifecycle overhead between bins.

After calibration, run validate_send_calibration.py to verify whether the correction is
worth applying by measuring frequency deviation before and after correction.

Usage:
    python calibrate_send_v2.py --mode single-capture   # faster, recommended
    python calibrate_send_v2.py --mode sequential        # per-frequency streams
    python calibrate_send_v2.py --dry-run                # show config and frequency table only
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Add repo root to path so we can import utils directly
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import config as cfg  # noqa: E402
from config import config as cfg  # noqa: E402
from utils.audio.sweep_utils import compute_inverse_filter, save_cal_profile
from utils.charting_utils import build_multichart_png  # noqa: E402


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------
def _print_freq_table(freqs, fs, mode, tone_duration=1.0, gap_s=0.3):
    """Print a formatted frequency table."""
    spacing = "log" if np.all(np.diff(np.log10(freqs)) > 0) else "linear"

    border = "=" * 52
    title = f"PyAmpScope -- {spacing.capitalize()} Spaced Frequencies ({mode} mode)"
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
# Sequential mode helpers — per-frequency OutputStream/InputStream
# ---------------------------------------------------------------------------
def _play_one_freq_seq(
    freq: float, duration_s: float, fs: int,
    send_device: Optional[int], recv_device: Optional[int],
    send_gain: float, tone_amplitude: float,
) -> np.ndarray:
    """Play a single frequency and capture it via separate OutputStream/InputStream per call.

    Returns the captured signal as a numpy array.
    """
    n_samples = int(duration_s * fs)
    t_total = np.arange(n_samples) / fs
    tone_full = (np.sin(2 * np.pi * freq * t_total) * tone_amplitude).astype(np.float64)

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


# ---------------------------------------------------------------------------
# Single-capture mode helpers — one OutputStream that switches tones
# ---------------------------------------------------------------------------
class _ToneSwitcher:
    """Manages per-frequency tone segments for a single OutputStream."""

    def __init__(self, freqs, duration_s, fs, gap_s, tone_amplitude):
        self.fs = fs
        self.tone_duration_s = duration_s
        self.gap_samples = int(gap_s * fs)
        self.total_out_samples = 0

        offset = 0
        self.tone_starts = []
        self.tone_arrays = []
        for freq in freqs:
            tone_samples = int(duration_s * fs)
            t = np.arange(tone_samples) / fs
            self.tone_arrays.append(
                (np.sin(2 * np.pi * freq * t) * tone_amplitude).astype(np.float64)
            )
            self.tone_starts.append(offset)
            offset += tone_samples + self.gap_samples
        self.total_out_samples = offset

def _play_one_freq_single(
    freqs, duration_s, fs, gap_s,
    send_device, recv_device, send_gain, tone_amplitude,
    capture_data, verbose=False,
):
    """Run a single-capture cycle: one OutputStream switching tones + one InputStream.

    Returns the captured signal (already written into capture_data array).
    """
    switcher = _ToneSwitcher(freqs, duration_s, fs, gap_s, tone_amplitude)
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
        description="Per-frequency calibration v2 — OutputStream/InputStream callback API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  single-capture : One OutputStream switches tones, one InputStream captures all.\n"
            "                   Faster (~30s for 60 bins). Recommended.\n"
            "  sequential     : One OutputStream+InputStream pair per frequency. More robust\n"
            "                   on hardware with unstable PortAudio state between streams.\n"
        ),
    )

    parser.add_argument(
        "--mode", choices=["sequential", "single-capture"], default="single-capture",
        help="Capture mode (default: single-capture)",
    )
    parser.add_argument("--method", choices=["sweep", "multitone", "pink", "white"],
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
    print(f"  Method     : {method}")
    print(f"  Mode       : {args.mode} (OutputStream/InputStream callbacks)")
    print(f"  Tone duration: {tone_duration}s")
    print(f"  Gap        : {gap_s}s between tones")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Freq range : {freq_min}-{freq_max} Hz")
    print(f"  Send device: {send_device}")
    print(f"  Recv device: {recv_device}")
    print(f"  Send gain  : {send_gain}%")
    print(f"  Recv gain  : {recv_gain}%")
    print(f"  Send ch    : {send_ch} (phase-2: not yet applied to streams)")
    print(f"  Recv ch    : {recv_ch} (phase-2: not yet applied to streams)")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    # Print frequency table
    _print_freq_table(freq_array, fs, args.mode, tone_duration=tone_duration, gap_s=gap_s)

    if args.dry_run:
        print("\n[Dry run -- skipping hardware play/capture and file writes.]")
        # Don't overwrite existing profile data with zeros.
        return

    # -----------------------------------------------------------------------
    # Run calibration
    # -----------------------------------------------------------------------
    print(f"\n[{num_freqs} frequencies across {args.mode} mode...]", flush=True)
    results = []  # list of (freq, amplitude_db)
    valid_results = []

    if args.mode == "sequential":
        print("\n[Sequential mode: one OutputStream+InputStream per frequency]")
        for i, target_freq in enumerate(freq_array):
            pct = (i + 1) / num_freqs * 100
            bar_len = 50
            filled = int(bar_len * pct / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}% ({i + 1}/{num_freqs}) {target_freq:.1f}Hz", end="", flush=True)
            try:
                rec_flat = _play_one_freq_seq(
                    freq=target_freq, duration_s=tone_duration, fs=fs,
                    send_device=send_device, recv_device=recv_device,
                    send_gain=send_gain, tone_amplitude=tone_amplitude,
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
            #print(f"  Captured {len(rec_flat)} samples @ {fs}Hz", flush=True)
            print(f"  RMS={rms:.6f}  {target_freq:.1f}Hz@{db_target:.1f}dBFS", end="", flush=True)
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
                send_gain=send_gain, tone_amplitude=tone_amplitude,
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

    # Save per-frequency WAV files
    v2_dir = output_dir / "v2_cal_send_wav"
    v2_dir.mkdir(exist_ok=True)
    wavfile.write(str(v2_dir / "captured_all.wav"), fs, rec_flat.astype(np.float32))
    print(f"\nWAV saved: {v2_dir}/captured_all.wav")

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

        # Always compute and save inverse correction filter from magnitude response
        if valid_results:
            amp_linear = 10 ** (valid_amps / 20.0)
        else:
            amp_linear = 10 ** (amp_array / 20.0)

        H_complex = amp_linear  # real part only, magnitude as complex with zero phase
        W_complex, ir = compute_inverse_filter(
            H=H_complex, freqs=freq_array, fft_len=int(tone_duration * fs),
        )

        metadata["filter_length"] = len(ir)
        metadata["filter_peak"] = float(np.max(np.abs(ir)))
        metadata["response_H"] = valid_amps if valid_results else amp_array
        metadata["freqs"] = valid_freqs if valid_results else freq_array

        # Build multi-chart (3 panels: response, deviation dB, deviation sigma) with correction filter overlay
        H_db_for_chart = valid_amps if valid_results else amp_array
        png_bytes = build_multichart_png(
            freqs=freq_array,
            H_db=H_db_for_chart,
            correction_filter=W_complex,
            title="Send Calibration Response (v2)",
        )

        # Save WAV capture file
        v2_dir = output_dir / "v2_cal_send_wav"
        v2_dir.mkdir(exist_ok=True)
        wav_path = v2_dir / "captured_all.wav"
        wavfile.write(str(wav_path), fs, rec_flat.astype(np.float32))
        print(f"\n  WAV saved : {wav_path}")

        npz_path = save_cal_profile(
            output_dir, "v2_cal_send", metadata,
            response_H=amp_linear[:len(freq_array)],
            freqs=freq_array,
        )
        print(f"  Profile saved: {npz_path}")

        # Save multi-chart PNG alongside profile
        chart_path = output_dir / "v2_cal_send_chart.png"
        chart_path.write_bytes(png_bytes)
        print(f"  Chart saved : {chart_path}")

        save_cal_profile(
            output_dir, "v2_cal_send", metadata,
            correction_filter=W_complex, ir=ir, response_H=H_complex, freqs=freq_array,
        )
        print(f"  Correction filter saved alongside profile.")
        print(f"  FIR length: {len(ir)} samples")

    print("\n[Done]")


if __name__ == "__main__":
    main()
