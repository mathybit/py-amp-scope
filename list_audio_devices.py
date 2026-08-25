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
import sounddevice as sd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TONE_FREQ = 440       # Hz - standard A4 tuning note
TONE_DURATION = 2.0   # seconds
RECORD_SECONDS = 3    # seconds
TEST_TONE_VOL = 0.3   # cap at 30% for safety

# Resolve repo root (one level up from this script's directory)
REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_tone(freq=TONE_FREQ, duration=TONE_DURATION, fs=48000, vol=TEST_TONE_VOL):
    """Generate a sine wave test tone."""
    t = np.arange(int(fs * duration)) / fs
    return (np.sin(2 * np.pi * freq * t) * vol).astype(np.float32)


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
    try:
        sd.play(tone, samplerate=48000, device=device_idx)
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

    # Use the device's default sample rate if available
    sr = dev.get("default_samplerate", 48000)
    if not isinstance(sr, (int, float)) or sr <= 0:
        sr = 48000
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
# Config saving
# ---------------------------------------------------------------------------
def load_config():
    """Load existing config values, returning defaults if file missing."""
    config_path = REPO_ROOT / "config" / "config.py"
    defaults = {}
    try:
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Try to parse Python literals safely
                try:
                    parsed = eval(val)
                    defaults[key] = parsed
                except Exception:
                    defaults[key] = val
    except FileNotFoundError:
        pass
    return defaults


def save_config(send_idx, recv_idx):
    """Write device selections to config/config.py, preserving other keys."""
    config_path = REPO_ROOT / "config" / "config.py"
    existing = load_config()

    # Build the full config text
    lines = []

    # Header comment
    lines.append("# PyAmpScope configuration - do not remove this marker")
    lines.append("# Audio interface selection (user fills these in after running list_audio_devices.py)")

    # Only write send_device if we have a valid selection
    if send_idx is not None:
        existing["send_device"] = send_idx
    if recv_idx is not None:
        existing["recv_device"] = recv_idx

    config_keys_order = [
        "send_device", "recv_device",
        "send_ch", "recv_ch",
        "send_gain", "recv_gain",
        "cal_method", "freq_min", "freq_max", "fs",
        "recv_path",
        "data_dir", "logs_dir",
        "cal_send_file", "cal_recv_file",
    ]

    for key in config_keys_order:
        if key in existing:
            val = existing[key]
            lines.append(f"{key} = {val}")

    # Any user-defined keys not in our known list
    for key, val in existing.items():
        if key not in config_keys_order:
            lines.append(f"\n# User-defined\n{key} = {val}")

    config_path.write_text("\n".join(lines) + "\n")

    # Resolve device names for confirmation message
    try:
        all_devs = sd.query_devices()
        if isinstance(all_devs, dict):
            all_devs = [all_devs]
        send_name = format_name(all_devs[send_idx]["name"]) if send_idx is not None and 0 <= send_idx < len(all_devs) else "default"
        recv_name = format_name(all_devs[recv_idx]["name"]) if recv_idx is not None and 0 <= recv_idx < len(all_devs) else "default"
    except Exception:
        send_name = str(send_idx) if send_idx is not None else "default"
        recv_name = str(recv_idx) if recv_idx is not None else "default"

    print(f"\nConfig saved to config/config.py:")
    if send_idx is not None:
        print(f"  send_device = {send_idx}   ({send_name})")
    if recv_idx is not None:
        print(f"  recv_device = {recv_idx}   ({recv_name})")


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

        #save_config(send_idx, recv_idx)
        sys.exit(0)

    # Interactive mode - continue to device selection

    print("Testing devices (type 'q' when done):")
    interactive_test(output_indices, input_indices)

    # Prompt device selection
    print()
    send_idx = None
    recv_idx = None

    while True:
        try:
            send_input = input(f"Select send device (output/both index [{', '.join(map(str, output_indices))}]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not send_input:
            break
        try:
            send_idx = int(send_input)
            if send_idx not in set(output_indices):
                print(f"  Please choose from: {output_indices}")
                continue
            break
        except ValueError:
            print("  Enter a numeric index or press Enter to skip")

    while True:
        try:
            recv_input = input(f"Select receive device (input/both index [{', '.join(map(str, input_indices))}]): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not recv_input:
            break
        try:
            recv_idx = int(recv_input)
            if recv_idx not in set(input_indices):
                print(f"  Please choose from: {input_indices}")
                continue
            break
        except ValueError:
            print("  Enter a numeric index or press Enter to skip")

    # Ensure both were entered before saving
    if send_idx is None or recv_idx is None:
        print("Both send and receive devices must be selected.")
        return

    #save_config(send_idx, recv_idx)


if __name__ == "__main__":
    main()
