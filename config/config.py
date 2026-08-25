# PyAmpScope configuration - do not remove this marker
# Audio interface selection (user fills these in after running list_audio_devices.py)
send_device = 23      # sounddevice device index or None for default
recv_device = 13      # same

# Channels (LEFT, RIGHT, STEREO)
send_ch = "LEFT"
recv_ch = "LEFT"

# Gains as percentage (0-100)
send_gain = 100
recv_gain = 80

# Calibration signal method and frequency range
cal_method = "multitone"   # "sweep" | "multitone" | "pink" | "white"
freq_min = 20
freq_max = 24000       # Nyquist = 24000 at fs=48kHz
fs = 48000             # default sample rate

# Receive path (direct vs isolated)
recv_path = "dir"      # "dir" | "iso"

# File paths (relative to repo root)
data_dir = "data"
logs_dir = "logs"
cal_send_file = "di_send_profile.npz"
cal_recv_file = "di_receive_profile.npz"
