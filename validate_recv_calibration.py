#!/usr/bin/env python
"""Validate receive-path calibration combinations.

Both ``--correct-send`` and ``--correct-recv`` are optional and OFF by default.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from config import config as cfg
from utils.audio.analysis_utils import analyze_noise_measurement, analyze_sweep_measurement, deviation_report
from utils.audio.calibration import load_receive_correction, load_send_correction
from utils.audio.signal_utils import geometric_frequencies
from utils.audio.streaming import run_noise_measurement, run_sweep_measurement
from utils.charting_utils import build_validate_chart_png


ROOT=Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Validate PyAmpScope receive-path calibration.")
    p.add_argument("--correct-send", action="store_true", help="Apply send correction; off by default.")
    p.add_argument("--correct-recv", action="store_true", help="Apply receive correction; off by default.")
    p.add_argument("--path", choices=["dir","iso"], default=cfg.recv_path)
    p.add_argument("--method", choices=["sweep","white"],#"pink","brown"],
                           default=cfg.cal_method, help="Calibration signal type")
    p.add_argument("--freq-min", type=float, default=cfg.freq_min)
    p.add_argument("--freq-max", type=float, default=cfg.freq_max)
    p.add_argument("--num-freqs", type=int, default=cfg.num_freqs_default)
    p.add_argument("--tone-duration", type=float, default=cfg.tone_duration)
    p.add_argument("--gap", type=float, default=cfg.tone_gap)
    p.add_argument("--tone-amplitude", type=float, default=cfg.tone_amplitude)
    p.add_argument("--send-gain", type=float, default=cfg.send_gain)
    p.add_argument("--noise-duration", type=float, default=cfg.noise_calibration_time)
    p.add_argument("--fs", type=int, default=cfg.fs)
    p.add_argument("--send-device", type=int, default=cfg.send_device)
    p.add_argument("--recv-device", type=int, default=cfg.recv_device)
    p.add_argument("--send-ch", choices=["LEFT","RIGHT","STEREO"], default=cfg.send_ch)
    p.add_argument("--recv-ch", choices=["LEFT","RIGHT","STEREO"], default=cfg.recv_ch)
    p.add_argument("--output-dir", type=Path,default=ROOT/cfg.logs_dir)
    return p.parse_args()


def main():
    args = parse_args()
    freqs = geometric_frequencies(args.freq_min, args.freq_max, args.num_freqs)
    data_dir=ROOT/cfg.data_dir

    send_corr = load_send_correction(data_dir) if args.correct_send else None
    recv_corr = load_receive_correction(data_dir, args.path, prefer_corrected_send=args.correct_send) if args.correct_recv else None
    if args.correct_send and send_corr is None:
        raise SystemExit("--correct-send requested but no send correction profile was found")
    if args.correct_recv and recv_corr is None:
        raise SystemExit(f"--correct-recv requested but no {args.path} receive correction profile was found")

    sf = send_corr.frequencies if send_corr else None
    sx = send_corr.factors if send_corr else None
    rf = recv_corr.frequencies if recv_corr else None
    rx = recv_corr.factors if recv_corr else None

    if args.method == "sweep":
        cap = run_sweep_measurement(
            freqs=freqs, fs=args.fs, tone_duration=args.tone_duration, gap_s=args.gap,
            tone_amplitude=args.tone_amplitude, send_gain=args.send_gain,
            send_device=args.send_device, recv_device=args.recv_device, send_ch=args.send_ch, recv_ch=args.recv_ch,
            send_correction_freqs=sf, send_correction_factors=sx,
            peak_headroom=cfg.sweep_peak_headroom
        )
        metrics = analyze_sweep_measurement(
            cap.capture, cap.stimulus.metadata, args.fs, recv_correction_freqs=rf, recv_correction_factors=rx, frequency_bands=cfg.frequency_bands
        )
    else:
        cap = run_noise_measurement(
            method=args.method, duration_s=args.noise_duration, fs=args.fs,
            tone_amplitude=args.tone_amplitude, send_gain=args.send_gain,
            send_device=args.send_device, recv_device=args.recv_device, send_ch=args.send_ch, recv_ch=args.recv_ch,
            send_correction_freqs=sf, send_correction_factors=sx,
            peak_headroom=cfg.noise_peak_headroom
        )
        metrics = analyze_noise_measurement(
            cap.capture, cap.stimulus.reference_samples, args.fs, freqs, recv_correction_freqs=rf, recv_correction_factors=rx, frequency_bands=cfg.frequency_bands
            )
    response=np.asarray(metrics["relative_response_db"],float); valid=np.isfinite(response)
    label=("sc" if send_corr else "sb")+("rc" if recv_corr else "rb")

    deviation_report(response, freqs, f"Receive Validation ({args.path}/{label})", bool(send_corr), bool(recv_corr))
    pct = (np.power(10.0,response/20.0)-1.0) * 100.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    png = build_validate_chart_png(freqs[valid], response[valid], np.abs(pct[valid]), f"Receive Validation ({args.path}/{label})")
    (out_dir/f"validate_recv_{args.path}_{label}_chart.png").write_bytes(png)

    wavfile.write(str(out_dir/f"validate_recv_{args.path}_{label}_captured.wav"), args.fs, cap.capture.astype(np.float32))

    np.savez(
        str(out_dir/f"validate_recv_{args.path}_{label}_profile.npz"),
        frequencies=freqs,
        relative_response_db=response,
        send_correction=np.asarray([bool(send_corr)]),
        receive_correction=np.asarray([bool(recv_corr)]),
        requested_rms=np.asarray([cap.stimulus.requested_rms]),
        actual_rms=np.asarray([cap.stimulus.actual_rms])
    )
    print(f"Relative response std: {np.nanstd(response):.3f} dB")


if __name__ == "__main__":
    main()
