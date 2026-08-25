#!/usr/bin/env python
"""Send calibration utility for PyAmpScope.

Plays a calibration signal through the audio interface's headphone output,
captures it via the line-in input, computes frequency response by FFT
deconvolution, and optionally generates a regularized inverse correction
filter.

Usage:
    python calibrate_send.py                        # interactive (uses config defaults)
    python calibrate_send.py --dry-run              # generate signal only, no hardware
    python calibrate_send.py --correct              # compute & save inverse filter
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Add repo root to path so we can import utils directly
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from utils.config_loader import load_config, merge_args  # noqa: E402
from utils.audio.sweep_utils import (  # noqa: E402
    compute_frequency_response,
    compute_inverse_filter,
    generate_cal_signal,
    play_and_capture,
    save_cal_profile,
    build_chart_png,
    _DEFAULT_DURATION,
    _FS_DEFAULT,
)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments merged with config defaults."""
    parser = argparse.ArgumentParser(
        description="Calibrate audio path: send signal, capture response, compute profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Methods:\n"
            "  sweep      Exponential chirp (perceptual weighting)\n"
            "  multitone  Equal-energy multi-tone (recommended for wire-through)\n"
            "  pink       Pink noise\n"
            "  white      White noise\n"
        ),
    )

    # Signal parameters
    parser.add_argument("--method", choices=["sweep", "multitone", "pink", "white"],
                        default=None, help="Calibration signal type (default: from config)")
    parser.add_argument("--duration", type=float, default=None,
                        help=f"Signal duration in seconds (default: {_DEFAULT_DURATION})")
    parser.add_argument("--freq-min", type=int, default=None, help="Lowest analysis frequency Hz")
    parser.add_argument("--freq-max", type=int, default=None, help="Highest analysis frequency Hz")

    # Device parameters
    parser.add_argument("--send-device", type=int, default=None, help="Send device index")
    parser.add_argument("--recv-device", type=int, default=None, help="Receive device index")
    parser.add_argument("--send-ch", choices=["LEFT", "RIGHT", "STEREO"], default=None,
                        help="Send channel (default: from config)")
    parser.add_argument("--recv-ch", choices=["LEFT", "RIGHT", "STEREO"], default=None,
                        help="Receive channel (default: from config)")

    # Gain parameters (percentage 0-100)
    parser.add_argument("--send-gain", type=float, default=None, help="Send gain %")
    parser.add_argument("--recv-gain", type=float, default=None, help="Receive gain %")

    # Sample rate
    parser.add_argument("--fs", type=int, default=None, help="Sample rate Hz (default: 48000)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for profile files (default: data/)")

    # Actions
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signal only; skip hardware play/capture")
    parser.add_argument("--correct", action="store_true",
                        help="Compute and save inverse correction filter")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load config defaults and merge with CLI overrides
    config = load_config()
    merged = merge_args(vars(args), config)

    method = merged["cal_method"]
    duration = float(merged.get("duration") or args.duration or _DEFAULT_DURATION)
    fs = int(merged.get("fs") or args.fs or _FS_DEFAULT)
    freq_min = int(merged.get("freq_min") or args.freq_min or 20)
    freq_max = int(merged.get("freq_max") or args.freq_max or 24000)
    send_device = merged.get("send_device")
    recv_device = merged.get("recv_device")
    send_gain = float(merged.get("send_gain") or args.send_gain or 30)
    recv_gain = float(merged.get("recv_gain") or args.recv_gain or 30)
    output_dir = Path(merged.get("data_dir", "data"))

    if args.output_dir:
        output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"PyAmpScope Send Calibration Profile")
    print(f"  Method     : {method}")
    duration_str = f"{duration:.0f}s"
    print(f"  Duration   : {duration_str}")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Freq range : {freq_min}-{freq_max} Hz")
    print(f"  Send device: {send_device}")
    print(f"  Recv device: {recv_device}")
    print(f"  Send gain  : {send_gain}%")
    print(f"  Recv gain  : {recv_gain}%")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    # Step 1: Generate signal
    print("\n[Step 1/6] Generating calibration signal...")
    signal_result = generate_cal_signal(
        method=method, duration=duration, fs=fs,
        freq_min=freq_min, freq_max=freq_max,
    )
    if method == "multitone":
        generated, frame_freqs = signal_result
    else:
        generated = signal_result

    n_samples = len(generated)
    print(f"  Signal length: {n_samples} samples ({n_samples/fs:.1f}s)")
    print(f"  Peak amplitude: {np.max(np.abs(generated)):.3f}")

    if args.dry_run:
        print("\n[Dry run -- skipping hardware play/capture.]")
        # Save profile even in dry-run with just freqs and zero response for inspection
        metadata = {
            "method": method,
            "duration": duration,
            "fs": fs,
            "freq_min": freq_min,
            "freq_max": freq_max,
            "send_device": send_device,
            "recv_device": recv_device,
            "send_gain": send_gain,
            "recv_gain": recv_gain,
            "generated_peak": float(np.max(np.abs(generated))),
            "captured_available": False,
        }
        save_cal_profile(output_dir, "cal_send", metadata)
        print(f"[Dry run profile saved to {output_dir / 'cal_send_profile.npz'}]")
        return

    # Step 2: Play and capture
    print("\n[Step 2/6] Playing calibration signal and capturing response...")
    print("  [This takes ~30 seconds -- do not interrupt.]")
    try:
        captured, generated_trimmed = play_and_capture(
            generated_signal=generated, duration=duration, fs=fs,
            send_device=send_device, recv_device=recv_device,
            send_gain=send_gain, recv_gain=recv_gain,
        )
    except Exception as e:
        print(f"\n  ERROR during play/capture: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Captured {len(captured)} samples ({len(captured)/fs:.1f}s)")

    if len(captured) < 256:
        print("  ERROR: captured too few samples. Check device selection.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Save WAV files for playback comparison (unconditional)
    wav_dir = output_dir / "cal_send_wav"
    wav_dir.mkdir(exist_ok=True)
    wavfile.write(str(wav_dir / "generated.wav"), fs, generated_trimmed)
    wavfile.write(str(wav_dir / "captured.wav"), fs, captured)
    print(f"\n  WAV files saved: {wav_dir}/generated.wav, {wav_dir}/captured.wav")

    # Step 4: Compute frequency response
    print("\n[Step 4/6] Computing frequency response...")
    freqs, H, G = compute_frequency_response(
        generated=generated_trimmed, captured=captured, fs=fs,
    )

    # Filter to valid band for stats
    passband_mask = (freqs >= freq_min) & (freqs <= freq_max)
    if np.any(passband_mask):
        H_pb = abs(H[passband_mask])
        db_values = 20 * np.log10(np.maximum(H_pb, 1e-10))

        # Skip bins where generated signal had near-zero energy
        peak_G = np.max(np.abs(G))
        # Apply energy mask within passband
        G_pb = G[passband_mask]
        energy_mask = np.abs(G_pb) > 0.01 * peak_G
        if np.any(energy_mask):
            db_valid = db_values[energy_mask]
        else:
            db_valid = db_values

        print(f"  Passband ({freq_min}-{freq_max} Hz):")
        print(f"    Min dB : {db_valid.min():.1f}")
        print(f"    Max dB : {db_valid.max():.1f}")
        print(f"    Mean dB: {db_valid.mean():.1f}")
        print(f"    Std dB : {db_valid.std():.1f}")

    # Always save a standalone chart PNG for the measured response
    png_bytes = build_chart_png(freqs, H, title="Send Calibration Response")
    chart_path = output_dir / "cal_send_chart.png"
    chart_path.write_bytes(png_bytes)
    print(f"  Chart saved: {chart_path}")

    # Step 5: Save profile
    print("\n[Step 5/6] Saving calibration profile...")
    metadata = {
        "method": method,
        "duration": duration,
        "fs": fs,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "send_device": send_device,
        "recv_device": recv_device,
        "send_gain": send_gain,
        "recv_gain": recv_gain,
        "generated_peak": float(np.max(np.abs(generated))),
        "captured_peak": float(np.max(np.abs(captured))),
        "captured_samples": len(captured),
    }

    npz_path = save_cal_profile(output_dir, "cal_send", metadata,
                                response_H=H, freqs=freqs)
    print(f"  Profile saved: {npz_path}")

    # Step 6: Optional inverse filter and correction visualization
    if args.correct:
        print("\n[Step 6/6] Computing inverse correction filter...")
        fft_len = len(generated_trimmed)
        W, ir = compute_inverse_filter(H=H, freqs=freqs, fft_len=fft_len)

        # Build visualization chart (3 curves: measured, compensation, corrected)
        png_bytes = build_chart_png(freqs, H, W=W, title="Send Calibration Response")
        metadata["chart_png_bytes"] = png_bytes

        # Save standalone PNG file on disk
        chart_path = output_dir / "cal_send_chart.png"
        chart_path.write_bytes(png_bytes)
        print(f"  Chart saved: {chart_path}")

        metadata["filter_length"] = len(ir)
        metadata["filter_peak"] = float(np.max(np.abs(ir)))

        # Save with correction filter embedded
        save_cal_profile(output_dir, "cal_send", metadata,
                         correction_filter=W, ir=ir, response_H=H, freqs=freqs)
        print(f"  Correction filter saved alongside profile.")
        print(f"  FIR length       : {len(ir)} samples")
        print(f"  FIR peak amplitude: {metadata['filter_peak']:.4f}")

        # Show correction stats in passband
        pb_W = abs(W[passband_mask])
        db_W = 20 * np.log10(np.maximum(pb_W, 1e-10))
        print(f"  Compensation range ({freq_min}-{freq_max} Hz):")
        print(f"    Min dB: {db_W.min():.1f}")
        print(f"    Max dB: {db_W.max():.1f}")

        corrected_db = 20 * np.log10(np.maximum(abs(H) * abs(W), 1e-10))
        if np.any(energy_mask):
            corr_pb = corrected_db[passband_mask]
            print(f"  Corrected passband stats ({freq_min}-{freq_max} Hz):")
            print(f"    Min dB : {corr_pb.min():.2f}")
            print(f"    Max dB : {corr_pb.max():.2f}")
            print(f"    Std dB : {corr_pb.std():.2f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
