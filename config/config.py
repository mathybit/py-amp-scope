"""PyAmpScope configuration.

The signal-processing code deliberately does not depend on ``log_f``.  It is kept
here only as an experimental helper for exploring alternative logarithm bases.
All dB calculations use log10 and octave calculations use log2 explicitly.
"""
from __future__ import annotations

import numpy as np

# Audio interface selection (sounddevice device indices, or None for default).
send_device = 13
recv_device = 2

# Channels: LEFT, RIGHT, or STEREO.
send_ch = "LEFT"
recv_ch = "LEFT"

# Digital source scaling.  Tone Amplitude is peak amplitude; Send Gain is an
# additional percentage multiplier.
send_gain = 100
device_test_tone_freq = 200  # the frequency of the test tone played by list_audio_devices.py
device_test_tone_duration = 5  # the duration of the test tone played by list_audio_devices.py
device_test_tone_record_secs = 5  # the duration of the test tone played by list_audio_devices.py

# Analysis / calibration defaults.
cal_method = "sweep"  # sweep | white | pink | brown
freq_min = 40
freq_max = 20000
fs = 44100  # sampling frequency (hardware dependent)

# Experimental logarithm helper.  Intentionally unused by production DSP code.
log_base = 2
log_f = lambda x: np.log(x) / np.log(log_base)

noise_calibration_time = 30
num_freqs_default = 30
tone_amplitude = 0.5       # sine PEAK before Send Gain
tone_duration = 0.7
tone_gap = 0.2
noise_peak_headroom = 0.95 # globally scale generated broadband noise below this peak
sweep_peak_headroom = 0.95 # same protection after per-frequency send correction
smoothing_neighbors = 5

# Conventional mixing/mastering frequency-band terminology.  Intervals are
# [low, high), except Brilliance includes the final endpoint.
frequency_bands = {
    "Sub-bass": (20.0, 60.0),
    "Bass": (60.0, 250.0),
    "Low-mid": (250.0, 500.0),
    "Mid": (500.0, 2000.0),
    "Upper-mid": (2000.0, 4000.0),
    "Presence": (4000.0, 6000.0),
    "Brilliance": (6000.0, 20000.0)
}

# Receive path: dir | iso
recv_path = "dir"

# Relative repository paths.
data_dir = "data"
logs_dir = "logs"

# Preferred correction-file names.
cal_send_corrections_file = "cal_send_corrections.npz"
cal_recv_dir_corrections_file = "cal_recv_dir_base_corrections.npz"
cal_recv_iso_corrections_file = "cal_recv_iso_base_corrections.npz"

# NPZ keys retained for compatibility with existing profiles.
_NPZ_KEY_FREQS = "frequencies"
_NPZ_KEY_RESPONSE = "response_H"
_NPZ_KEY_CORRECTION = "correction_W"
_NPZ_KEY_IR = "impulse_response"
_NPZ_KEY_CAL_DATA = "calibration_data"
_NPZ_KEY_PNG_BYTES = "chart_png_bytes"
