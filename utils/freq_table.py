#!/usr/bin/env python
"""Print frequency tables for both modes.

Usage:
    python -m utils.freq_table --mode sequential  # one tone at a time (like v1)
    python -m utils.freq_table --mode single-capture  # all in one recording
    python -m utils.freq_table --mode all          # both
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config, merge_args  # noqa: E402


def freq_table(log_spaced=True, mode="sequential"):
    """Print a frequency table for the given spacing and capture mode."""
    config = load_config()
    merged = merge_args({}, config)

    freq_min = int(merged.get("freq_min", 20))
    freq_max = int(merged.get("freq_max", 20000))
    fs = int(merged.get("fs", 44100))

    if log_spaced:
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), num=60)
    else:
        n = int((freq_max - freq_min) // 5) + 1
        freqs = np.linspace(freq_min, freq_max, n)

    if mode == "sequential":
        # One tone per window; gap between tones needed (0.3s default)
        # Approximate number of tones for typical calibration (60s total capture)
        tone_s = 2.0
        gap_s = 0.3
        num_tones = min(len(freqs), int(60 / (tone_s + gap_s)))
    else:
        num_tones = len(freqs)

    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), num=num_tones) if log_spaced else np.linspace(freq_min, freq_max, num=num_tones)

    return freqs, fs


def print_header(mode="sequential", spacing="log"):
    _, fs = _build_frequencies(spacing=spacing, mode=mode)
    title = f"PyAmpScope -- {spacing.capitalize()} Spaced Frequencies ({mode} mode)"
    border = "=" * len(title)

    print(f"\n{title}")
    print(border)
    print(f"  Sample rate   : {fs} Hz")
    if mode == "sequential":
        print(f"  Mode          : sequential (one OutputStream/InputStream pair per frequency)")
        print(f"  Est. tones/signal: ~60 at 2s tone + 0.3s gap (~150s total signal duration)")
    else:
        print(f"  Mode          : single-capture (one OutputStream switches tones, one InputStream captures all)")

    print(f"\n{'Bin':>4}  {'Frequency (Hz)':>16}  {'Samples per wave period':>20}")
    print("-" * 57)


def _build_frequencies(spacing="log", mode="sequential"):
    config = load_config()
    merged = merge_args({}, config)
    freq_min = int(merged.get("freq_min", 20))
    freq_max = int(merged.get("freq_max", 20000))
    fs = int(merged.get("fs", 44100))

    if spacing == "log":
        n = 60 if mode == "sequential" else None
        freq_array = np.logspace(np.log10(freq_min), np.log10(freq_max), num=n)
    else:
        n = 60 if mode == "sequential" else None
        freq_array = np.linspace(freq_min, freq_max, n)

    return freq_array, fs


def print_table(mode="sequential"):
    spacing = "log"
    freqs, fs = _build_frequencies(spacing=spacing, mode=mode)
    print_header(mode=mode, spacing=spacing)

    col_w = 57
    for i, f in enumerate(freqs):
        samples_per_cycle = fs / f
        print(f"{i+1:>4}  {f:>12.1f} Hz  | {samples_per_cycle:>9.0f} samples/period")

    border = "=" * col_w
    print(border)
    print(f"{'Total':>4} bins: {len(freqs)}")
    if spacing == "log":
        avg_spacing = np.mean(np.diff(np.log10(freqs)))
        print(f"Spacing  : logarithmic (avg step={avg_spacing:.2f} decades)")
    else:
        avg_spacing = np.mean(np.diff(freqs))
        print(f"Spacing  : linear (avg step={avg_spacing:.0f} Hz)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Print frequency tables")
    parser.add_argument("--mode", choices=["sequential", "single-capture", "all"], default="all")
    args = parser.parse_args()

    if args.mode == "all":
        for m in ["sequential", "single-capture"]:
            print_table(mode=m)

    elif args.mode == "sequential":
        # Log spaced
        print_table(mode="sequential")
        # Linear spaced
        spacing_table(mode="sequential", spacing="linear")

    elif args.mode == "single-capture":
        print_table(mode="single-capture")


def spacing_table(mode="sequential", spacing="log"):
    freqs, fs = _build_frequencies(spacing=spacing, mode=mode)
    col_w = 46
    title = f"PyAmpScope -- {'Linear' if spacing == 'linear' else 'Log'} Spaced Frequencies ({mode} mode)"
    border = "=" * len(title)
    print(f"\n{title}")
    print(border)
    print(f"  Sample rate   : {fs} Hz")
    print(f"\n{'Bin':>4}  {'Frequency (Hz)':>16}  {'Samples per (wave) period':>20}")
    print("-" * 40)

    for i, f in enumerate(freqs):
        samples_per_cycle = fs / f
        print(f"{i+1:>4}  {f:>12.1f} Hz  | {samples_per_cycle:>10.0f}samples/period")

    border2 = "=" * col_w
    print(border2)
    print(f"{'Total':>4} bins: {len(freqs)}")


if __name__ == "__main__":
    main()
