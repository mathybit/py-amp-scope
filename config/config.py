# PyAmpScope configuration - do not remove this marker
# Audio interface selection (user fills these in after running list_audio_devices.py)
send_device = 13      # sounddevice device index or None for default
recv_device = 16      # same

# Channels (LEFT, RIGHT, STEREO)
send_ch = "LEFT"
recv_ch = "LEFT"

# Gains as percentage (0-100)
send_gain = 70
recv_gain = 50

# Calibration signal method and frequency range
cal_method = "sweep"   # "sweep" | "pink" | "white" | "brown"
freq_min = 40
freq_max = 20000       # adjusted for USB interface spec
fs = 44100             # default sample rate (USB interface lowest common)

# Calibration settings
min_calibration_time = 30      # minimum total calibration time in seconds
noise_calibration_time = 30    # total time to capture broadband noise for pink/white/brown methods
num_freqs_default = 60        # default frequency bin count (floor)
tone_duration = 0.7            # per-tone duration in seconds
tone_gap = 0.2                 # gap between tones in seconds
tone_amplitude = 0.2           # base tone amplitude (before send_gain scaling)

# Smoothing for correction factor calculation (total window size, centered around each bin)
smoothing_neighbors = 5   # total points in moving average window; edges use whatever's available

# Receive path (direct vs isolated)
recv_path = "dir"      # "dir" | "iso"

# File paths (relative to repo root)
data_dir = "data"
logs_dir = "logs"
cal_send_file = "di_send_profile.npz"
cal_recv_file = "di_receive_profile.npz"
cal_send_correction_file = "cal_send_correction.npz"

# NPZ keys
_NPZ_KEY_FREQS = "frequencies"
_NPZ_KEY_RESPONSE = "response_H"
_NPZ_KEY_CORRECTION = "correction_W"
_NPZ_KEY_IR = "impulse_response"
_NPZ_KEY_CAL_DATA = "calibration_data"
_NPZ_KEY_PNG_BYTES = "chart_png_bytes"
