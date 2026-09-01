#!/usr/bin/env python
"""List and select audio devices for PyAmpScope.

Uses sounddevice (PortAudio) to enumerate available audio hardware,
play test tones, record samples, and save device selections to config/config.py.

Usage:
    python list_audio_devices.py                    # interactive mode
    python list_audio_devices.py --dry-run          # list only, no interaction
    python list_audio_devices.py --save-devices 3 5 # non-interactive save
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
try:
    import sounddevice as sd
except ImportError:
    print("Warning: unable to import sounddevice")
    sd = None

# Resolve repo root (one level up from this script's directory)
REPO_ROOT = Path(__file__).resolve().parent

from config import config as cfg

TONE_FREQ = cfg.device_test_tone_freq
TONE_DURATION = cfg.device_test_tone_duration
RECORD_SECONDS = cfg.device_test_tone_record_secs
TEST_TONE_VOL = cfg.tone_amplitude
SEND_GAIN = cfg.send_gain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_tone(freq=TONE_FREQ, duration=TONE_DURATION, fs=None, send_gain=SEND_GAIN, vol=TEST_TONE_VOL):
    """Generate a sine wave test tone."""
    if fs is None:
        fs = cfg.fs
    t = np.arange(int(fs * duration)) / fs
    amplitude = vol * (send_gain / 100.0)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dev_channel_count(dev, direction):
    """Get max channels for 'in' or 'out', handling legacy key names."""
    out_key = "max_output_channels" if direction == "out" else "max_input_channels"
    in_key = "max_output_streams" if direction == "out" else "max_input_streams"
    return dev.get(out_key, dev.get(in_key, 0))


def get_device_type(dev):
    """Return 'output', 'input', or 'both' for a device dict."""
    out = dev_channel_count(dev, "out")
    inp = dev_channel_count(dev, "in")
    if out > 0 and inp > 0:
        return "both"
    elif out > 0:
        return "output"
    elif inp > 0:
        return "input"
    return "none"


def is_default_device(idx):
    """Check if device idx is the system default output or input."""
    try:
        default_out, default_in = sd.default.device
        out_ok = (default_out is None) or (default_out == idx)
        in_ok = (default_in is None) or (default_in == idx)
        return out_ok or in_ok
    except Exception:
        return False


def format_name(name, max_len=45):
    """Truncate device name with ellipsis."""
    if len(name) > max_len:
        return name[:max_len - 3] + "..."
    return name


# ---------------------------------------------------------------------------
# Device listing
# ---------------------------------------------------------------------------
def list_devices(dry_run=False):
    """Query and display all available audio devices."""
    try:
        all_devs = sd.query_devices()
        # Handle scalar case (single device)
        if isinstance(all_devs, dict):
            all_devs = [all_devs]
    except Exception as e:
        print(f"Error querying devices: {e}", file=sys.stderr)
        sys.exit(1)

    output_devs = [d for d in all_devs if dev_channel_count(d, "out") > 0]
    input_devs = [d for d in all_devs if dev_channel_count(d, "in") > 0]

    # Header format: fixed-width columns
    header_fmt = "{:>4}  {:<42}  {:<6}  {:>5}  {:>5}  {:>8}  {}"
    divider = "-" * 72

    print("=" * 72)
    print("=== OUTPUT DEVICES (Playback / Send) ===")
    print(divider)
    print(header_fmt.format("Idx", "Name", "Type", "ChOut", "ChIn", "Sr(Hz)", "Def"))
    print(divider)

    for dev in output_devs:
        idx = dev.get("index", -1)
        name = format_name(dev.get("name", "Unknown"), max_len=42)
        dtype = get_device_type(dev)
        ch_out = dev_channel_count(dev, "out")
        ch_in = dev_channel_count(dev, "in")
        sr = dev.get("default_samplerate", "?")
        if isinstance(sr, (int, float)):
            sr_str = f"{sr/1000:.0f}k" if sr >= 1000 else str(int(sr))
        else:
            sr_str = str(sr)
        default = "*" if is_default_device(idx) else " "
        print(header_fmt.format(idx, name, dtype, ch_out, ch_in, sr_str, default))

    print()
    print("=" * 72)
    print("=== INPUT DEVICES (Capture / Receive) ===")
    print(divider)
    print(header_fmt.format("Idx", "Name", "Type", "ChOut", "ChIn", "Sr(Hz)", "Def"))
    print(divider)

    for dev in input_devs:
        idx = dev.get("index", -1)
        name = format_name(dev.get("name", "Unknown"), max_len=42)
        dtype = get_device_type(dev)
        ch_out = dev_channel_count(dev, "out")
        ch_in = dev_channel_count(dev, "in")
        sr = dev.get("default_samplerate", "?")
        if isinstance(sr, (int, float)):
            sr_str = f"{sr/1000:.0f}k" if sr >= 1000 else str(int(sr))
        else:
            sr_str = str(sr)
        default = "*" if is_default_device(idx) else " "
        print(header_fmt.format(idx, name, dtype, ch_out, ch_in, sr_str, default))

    print()
    print("Commands: type 'p <idx>' to play test tone, 'r <idx>' to record.")
    print("Type 'q' to quit or when prompted, select send/receive devices.")

    if dry_run:
        print("\n[Dry run -- no device selection performed.]")
        return None


# ---------------------------------------------------------------------------
# Device testing
# ---------------------------------------------------------------------------
def play_tone(device_idx):
    """Play a test tone through the specified device."""
    # Validate device supports output
    all_devs = sd.query_devices()
    if isinstance(all_devs, dict):
        all_devs = [all_devs]
    dev = all_devs[device_idx] if 0 <= device_idx < len(all_devs) else None
    if dev is None or dev_channel_count(dev, "out") == 0:
        print(f"Device {device_idx} does not support playback.", file=sys.stderr)
        return False

    tone = generate_tone()
    sr = cfg.fs
    try:
        sd.play(tone, samplerate=sr, device=device_idx)
        sd.wait()  # block until playback completes
        print(f"  [Played 440Hz tone through device {device_idx}]")
        return True
    except Exception as e:
        print(f"  Error playing tone on device {device_idx}: {e}", file=sys.stderr)
        return False


def record_from_device(device_idx):
    """Record from a device and play back the captured audio."""
    all_devs = sd.query_devices()
    if isinstance(all_devs, dict):
        all_devs = [all_devs]
    dev = all_devs[device_idx] if 0 <= device_idx < len(all_devs) else None
    if dev is None or dev_channel_count(dev, "in") == 0:
        print(f"Device {device_idx} does not support capture.", file=sys.stderr)
        return False

    # Use the device's default sample rate if available, fall back to config fs
    sr = dev.get("default_samplerate", None)
    if sr is None or sr <= 0:
        sr = cfg.fs
    frames = int(sr * RECORD_SECONDS)

    buf = np.empty(frames, dtype='float32')
    offset = 0
    def _cb(indata, frame_count, time_flag, status):
        nonlocal offset
        n = min(frame_count, frames - offset)
        buf[offset:offset + n] = indata.flatten()[:n]
        offset += n

    try:
        print(f"  [Recording from device {device_idx} ({sr/1000:.0f}kHz) for {RECORD_SECONDS}s...]")
        stream = sd.InputStream(
            device=device_idx, samplerate=sr, channels=1,
            callback=_cb, blocksize=512,
        )
        stream.start()
        # Wait for the recording duration
        time.sleep(RECORD_SECONDS)
        stream.stop()
        stream.close()

        mono = buf.flatten()
        if offset == 0:
            print(f"  No audio captured from device {device_idx}.", file=sys.stderr)
            return False

        # Compute stats
        rms = float(np.sqrt(np.mean(mono ** 2)))
        peak = float(np.max(np.abs(mono)))
        print(f"  [Recording complete] RMS={rms:.4f}  Peak={peak:.4f}")

        # Play back through system default output at the same sample rate
        try:
            sd.play(mono, samplerate=sr)
            sd.wait()
            print(f"  [Played back captured audio on default device]")
        except Exception as e:
            print(f"  Warning: could not play back: {e}")

        return True
    except Exception as e:
        print(f"  Error recording from device {device_idx}: {e}", file=sys.stderr)
        return False


def interactive_test(output_indices, input_indices):
    """Enter interactive loop for testing devices."""
    output_set = set(output_indices)
    input_set = set(input_indices)

    while True:
        try:
            cmd = input("\nEnter command (p <idx> / r <idx> / q): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd or cmd.lower() == "q":
            break

        parts = cmd.split()
        if len(parts) != 2:
            print("Usage: p <idx> to play, r <idx> to record, q to quit")
            continue

        action, idx_str = parts[0].lower(), parts[1]
        try:
            idx = int(idx_str)
        except ValueError:
            print(f"Invalid index: {idx_str}")
            continue

        if action == "p":
            if idx not in output_set and idx not in input_set:
                print(f"Device {idx} not found.")
                continue
            play_tone(idx)
        elif action == "r":
            if idx not in input_set:
                print(f"Device {idx} does not support capture.")
                continue
            record_from_device(idx)
        else:
            print(f"Unknown command: {action}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="List and select audio devices for PyAmpScope",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python list_audio_devices.py                 # interactive mode\n"
            "  python list_audio_devices.py --dry-run       # list only\n"
            "  python list_audio_devices.py --save-devices 3 5   # non-interactive save\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Just list devices, no interactive selection")
    group.add_argument(
        "--save-devices", nargs=2, type=int, metavar=("SEND_IDX", "RECV_IDX"),
        help="Non-interactive: save these indices as send/recv devices",
    )
    args = parser.parse_args()

    # Query and display devices
    output_indices = []
    input_indices = []
    all_devs = sd.query_devices()
    if isinstance(all_devs, dict):
        all_devs = [all_devs]

    for dev in all_devs:
        dtype = get_device_type(dev)
        idx = dev.get("index", -1)
        if dtype in ("output", "both"):
            output_indices.append(idx)
        if dtype in ("input", "both"):
            input_indices.append(idx)

    result = list_devices(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)

    # Non-interactive save mode
    if args.save_devices is not None:
        send_idx, recv_idx = args.save_devices

        # Validate indices
        all_devs = sd.query_devices()
        if isinstance(all_devs, dict):
            all_devs = [all_devs]
        if send_idx < 0 or send_idx >= len(all_devs):
            print(f"Error: send device index {send_idx} is out of range (0-{len(all_devs) - 1}).", file=sys.stderr)
            sys.exit(1)
        if recv_idx < 0 or recv_idx >= len(all_devs):
            print(f"Error: recv device index {recv_idx} is out of range (0-{len(all_devs) - 1}).", file=sys.stderr)
            sys.exit(1)
        if dev_channel_count(all_devs[send_idx], "out") == 0:
            print(f"Error: device {send_idx} does not support playback.", file=sys.stderr)
            sys.exit(1)
        if dev_channel_count(all_devs[recv_idx], "in") == 0:
            print(f"Error: device {recv_idx} does not support capture.", file=sys.stderr)
            sys.exit(1)

        sys.exit(0)

    # Interactive mode - continue to device selection

    print("Testing devices (type 'q' when done):")
    interactive_test(output_indices, input_indices)


if __name__ == "__main__":
    main()
