# PyAmpScope configuration - do not remove this marker
# Audio interface selection (user fills these in after running list_audio_devices.py)
send_device = 13      # sounddevice device index or None for default
recv_device = 16      # same

# Channels (LEFT, RIGHT, STEREO)
send_ch = "LEFT"
recv_ch = "LEFT"

# Gains as percentage (0-100)
send_gain = 90
recv_gain = 90

# Calibration signal method and frequency range
cal_method = "multitone"   # "sweep" | "multitone" | "pink" | "white"
freq_min = 50
freq_max = 20000       # adjusted for USB interface spec
fs = 44100             # default sample rate (USB interface lowest common)

# V2 calibration tone parameters
tone_amplitude = 0.3   # base tone amplitude (before send_gain scaling)

# V2 calibration settings
min_calibration_time = 60      # minimum total calibration time in seconds
num_freqs_default = 160        # default frequency bin count (floor)
tone_duration = 0.7            # per-tone duration in seconds
tone_gap = 0.3                 # gap between tones in seconds

# Receive path (direct vs isolated)
recv_path = "dir"      # "dir" | "iso"

# File paths (relative to repo root)
data_dir = "data"
logs_dir = "logs"
cal_send_file = "di_send_profile.npz"
cal_recv_file = "di_receive_profile.npz"
