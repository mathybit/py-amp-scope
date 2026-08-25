"""Shared config loading for PyAmpScope calibration/analysis scripts."""

import json
from pathlib import Path


_DEFAULTS = {
    "send_device": None,
    "recv_device": None,
    "send_ch": "LEFT",
    "recv_ch": "LEFT",
    "send_gain": 30,
    "recv_gain": 30,
    "cal_method": "multitone",
    "duration": 30,
    "freq_min": 20,
    "freq_max": 24000,
    "fs": 48000,
    "recv_path": "dir",
    "data_dir": "data",
    "logs_dir": "logs",
    "cal_send_file": "di_send_profile.npz",
    "cal_recv_file": "di_receive_profile.npz",
}


def _get_repo_root():
    """Find repo root by walking up from this file's location."""
    p = Path(__file__).resolve().parent.parent
    # Verify it has the config directory
    if (p / "config" / "config.py").exists():
        return p
    # Fallback: assume same directory as the caller script
    for candidate in [Path(".").resolve(), Path(".").resolve().parent]:
        if (candidate / "config" / "config.py").exists():
            return candidate
    return p


def load_config(config_path=None):
    """Load existing config values, returning defaults if file missing.

    Returns a dict with all PyAmpScope configuration keys.
    Existing values in config.py override the hardcoded defaults.
    """
    result = dict(_DEFAULTS)

    if config_path is None:
        repo = _get_repo_root()
        config_path = repo / "config" / "config.py"

    try:
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                try:
                    parsed = eval(val)
                    result[key] = parsed
                except Exception:
                    pass
    except FileNotFoundError:
        pass

    return result


def merge_args(args_dict, config):
    """Merge CLI args with config defaults.

    For each key, if the value in args_dict is not the argparse default (None for
    optional args), use the CLI value; otherwise fall back to config.

    Returns a merged dict suitable for passing to calibration functions.
    """
    merged = dict(config)
    for key, val in args_dict.items():
        # Skip keys whose value is still the argparse default (None or empty)
        if val is not None:
            merged[key] = val
    return merged
