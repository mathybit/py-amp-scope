#!/usr/bin/env python
"""Generate a send-path frequency-response correction profile for PyAmpScope.

The default is intentionally *uncorrected*.  Calibration uses the exact same
single-capture sweep/noise engines as the GUI and validation scripts.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from config import config as cfg
from utils.audio.analysis_utils import analyze_noise_measurement, analyze_sweep_measurement
from utils.audio.calibration import derive_inverse_correction
from utils.audio.signal_utils import geometric_frequencies
from utils.audio.streaming import run_noise_measurement, run_sweep_measurement
from utils.charting_utils import build_multichart_png
from utils.storage import save_cal_profile, save_correction_profile

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate the PyAmpScope send path.")
    p.add_argument("--method", choices=["sweep","white"],#"pink","brown"],
                   default=cfg.cal_method, help="Calibration signal type")
    p.add_argument("--freq-min", type=float, default=cfg.freq_min, help=f"Lowest analysis frequency Hz (config: {cfg.freq_min})")
    p.add_argument("--freq-max", type=float, default=cfg.freq_max, help=f"Highest analysis frequency Hz (config: {cfg.freq_max})")
    p.add_argument("--num-freqs", type=int, default=cfg.num_freqs_default, help=f"Number of (sweep) frequency bins (config: {cfg.num_freqs_default})")
    p.add_argument("--tone-duration", type=float, default=cfg.tone_duration, help=f"Sweep tone duration in seconds (config: {cfg.tone_duration})")
    p.add_argument("--gap", type=float, default=cfg.tone_gap, help=f"Gap between (sweep) tones in seconds (config: {cfg.tone_gap})")
    p.add_argument("--tone-amplitude", type=float, default=cfg.tone_amplitude, help='Tone amplitude')
    p.add_argument("--send-gain", type=float, default=cfg.send_gain, help='Send gain')
    p.add_argument("--noise-duration", type=float, default=cfg.noise_calibration_time, help=f'Noise calibration duration in seconds (config: {cfg.noise_calibration_time})')
    p.add_argument("--fs", type=int, default=cfg.fs, help=f'Sampling rate (default: {cfg.fs})')
    p.add_argument("--send-device",type=int,default=cfg.send_device, help=f"Send device index (config: {cfg.send_device})")
    p.add_argument("--recv-device",type=int,default=cfg.recv_device, help=f"Receive device index (config: {cfg.recv_device})")
    p.add_argument("--send-ch",choices=["LEFT","RIGHT","STEREO"],default=cfg.send_ch, help=f"Send channel (config: {cfg.send_ch})")
    p.add_argument("--recv-ch",choices=["LEFT","RIGHT","STEREO"],default=cfg.recv_ch, help=f"Receive channel (config: {cfg.recv_ch})")
    p.add_argument("--output-dir",type=Path,default=ROOT/cfg.data_dir)
    p.add_argument("--dry-run",action="store_true", help="Show config and frequency table; skip hardware")
    return p.parse_args()


def main():
    args = parse_args()
    freqs = geometric_frequencies(args.freq_min, args.freq_max, args.num_freqs)
    if args.freq_max >= args.fs/2:
        raise SystemExit(f"freq-max must be below Nyquist ({args.fs/2:.0f} Hz)")
    print(f"PyAmpScope send calibration: {args.method}, {len(freqs)} bins, corrections OFF")
    print(f"Tone peak={args.tone_amplitude:.4f}, Send Gain={args.send_gain:.1f}%")
    if args.dry_run:
        return

    if args.method == "sweep":
        print('Running sweep measurement...')
        cap = run_sweep_measurement(
            freqs=freqs, fs=args.fs, tone_duration=args.tone_duration, gap_s=args.gap,
            tone_amplitude=args.tone_amplitude, send_gain=args.send_gain,
            send_device=args.send_device,recv_device=args.recv_device,
            send_ch=args.send_ch, recv_ch=args.recv_ch,
            peak_headroom=cfg.sweep_peak_headroom
        )
        metrics = analyze_sweep_measurement(
            cap.capture, cap.stimulus.metadata,args.fs, frequency_bands=cfg.frequency_bands
        )
    else:
        print('Running noise measurement...')
        cap = run_noise_measurement(
            method=args.method, duration_s=args.noise_duration, fs=args.fs,
            tone_amplitude=args.tone_amplitude, send_gain=args.send_gain,
            send_device=args.send_device, recv_device=args.recv_device,
            send_ch=args.send_ch,recv_ch=args.recv_ch,
            peak_headroom=cfg.noise_peak_headroom
        )
        metrics = analyze_noise_measurement(
            cap.capture, cap.stimulus.reference_samples, args.fs, freqs, frequency_bands=cfg.frequency_bands
        )

    response = np.asarray(metrics["relative_response_db"], float)
    factors, smooth,reference = derive_inverse_correction(freqs, response, smoothing_window=cfg.smoothing_neighbors)

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    meta={
        "method": args.method,
        "fs": args.fs,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "num_freqs": len(freqs),
        "tone_amplitude": args.tone_amplitude,
        "send_gain": args.send_gain,
        "send_ch": args.send_ch,
        "recv_ch": args.recv_ch,
        "requested_rms": cap.stimulus.requested_rms,
        "actual_rms": cap.stimulus.actual_rms,
        "actual_peak": cap.stimulus.actual_peak,
        "headroom_scale": cap.stimulus.headroom_scale,
        "correction_applied": False  # no correction is to be applied during Send calibration
    }
    save_cal_profile(out, "cal_send", meta, freqs=freqs, response_H=response)
    corr_path = save_correction_profile(out, "cal_send_corrections.npz", freqs, factors, response, smooth, reference, meta)

    png, _, _ = build_multichart_png(freqs, response, cfg.smoothing_neighbors, "Send Path Calibration")
    (out/"cal_send_chart.png").write_bytes(png)

    wavfile.write(str(out/"cal_send_captured.wav"), args.fs, cap.capture.astype(np.float32))
    print(f"Saved correction: {corr_path}")
    print(f"Response std: {np.nanstd(response):.3f} dB; correction range: {np.nanmin(factors):.4f}..{np.nanmax(factors):.4f}")


if __name__ == "__main__":
    main()
