import math
import numpy as np
from pathlib import Path
import sounddevice as sd
import sys


# Add repo root to path so we can import config directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import log_f


# ---------------------------------------------------------------------------
# Single-capture mode helpers — one OutputStream that switches tones
# ---------------------------------------------------------------------------
class _ToneSwitcher:
    """
    Manages per-frequency tone segments for a single OutputStream.
    """
    def __init__(self, freqs, duration_s, fs, gap_s, tone_amplitude=1.0, corr_factors=None):
        """
        Initialize the tone switcher with the given parameters.
        """
        self.fs = fs
        self.tone_duration_s = duration_s
        self.gap_samples = int(gap_s * fs)
        self.total_out_samples = 0

        offset = 0
        self.tone_starts = []
        self.tone_arrays = []

        for i, freq in enumerate(freqs):
            tone_samples = int(duration_s * fs)
            t = np.arange(tone_samples) / fs

            if corr_factors is not None:
                amplitude = tone_amplitude * corr_factors[i]
            else:
                amplitude = tone_amplitude
            
            self.tone_arrays.append(
                (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float64)
            )
            self.tone_starts.append(offset)
            offset += tone_samples + self.gap_samples

        self.total_out_samples = offset


def play_one_freq_single(
    freqs, duration_s, fs, gap_s,
        send_device, recv_device, send_gain, tone_amplitude,
        capture_data, corr_factors=None, verbose=False,
):
    """
    Run a single-capture cycle: one OutputStream switching tones + one InputStream.

    Returns the captured signal (already written into capture_data array).
    """
    switcher = _ToneSwitcher(freqs, duration_s, fs, gap_s, tone_amplitude, corr_factors=corr_factors)
    total_out_samples = switcher.total_out_samples
    total_s = total_out_samples / fs
    in_offset = [0]
    out_offset = [0]

    def _in_cb(indata, frame_count, time_flag, status):
        if status:
            print(f"    [Input status: {status}]", flush=True)
        n = min(frame_count, len(capture_data) - in_offset[0])
        capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
        in_offset[0] += n

    def _out_cb(outdata, frame_count, time_flag, status):
        if status:
            print(f"    [Output status: {status}]", flush=True)

        start = out_offset[0]
        end = min(start + frame_count, total_out_samples)
        n = min(frame_count, end - start)

        buf = np.zeros(n, dtype=np.float64)
        gain_factor = send_gain / 100.0
        for i in range(len(switcher.tone_arrays)):
            t_start = switcher.tone_starts[i]
            t_end = t_start + len(switcher.tone_arrays[i])
            lo = max(start, t_start)
            hi = min(end, t_end)
            if lo < hi:
                t_lo = lo - t_start
                t_hi = hi - t_start
                buf[lo - start : hi - start] = switcher.tone_arrays[i][t_lo:t_hi] * gain_factor

        if outdata.ndim == 1:
            outdata[:n] = buf
        else:
            outdata[:n, 0] = buf
        out_offset[0] = end

        if out_offset[0] >= switcher.total_out_samples:
            return outdata
        return (outdata, "continue")

    try:
        out_stream = sd.OutputStream(
            device=send_device, samplerate=fs, channels=1,
            callback=_out_cb, blocksize=512, latency="low",
        )
        in_stream = sd.InputStream(
            device=recv_device, samplerate=fs, channels=1,
            callback=_in_cb, blocksize=512, latency="low",
        )
        out_stream.start()
        in_stream.start()

        elapsed = 0
        last_reported_s = -1.0
        while in_offset[0] < total_out_samples and elapsed < total_s * 3:
            if verbose and (elapsed - last_reported_s) >= 0.5:
                pct = in_offset[0] / total_out_samples * 100
                bar_len = 50
                filled = int(bar_len * in_offset[0] / total_out_samples)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r  Capturing [{bar}] {pct:5.1f}% ({elapsed:.0f}/{total_s:.0f}s)", end="", flush=True)
                last_reported_s = elapsed
            sd.sleep(100)
            elapsed += 0.1

        if verbose:
            print(f"\r  Capturing [{'' + '=' * bar_len}] 100.0% ({total_s:.0f}/{total_s:.0f}s)\n", flush=True)

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()
    except Exception as e:
        raise RuntimeError(f"Single-capture stream error: {e}") from e

    rec_flat = capture_data[: in_offset[0]].astype(float)
    return rec_flat


# ---------------------------------------------------------------------------
# Sequential mode helpers — per-frequency OutputStream/InputStream
# ---------------------------------------------------------------------------
def play_one_freq_seq(
    freq: float, duration_s: float, fs: int,
    send_device, recv_device,
    send_gain: float, tone_amplitude: float, corr_factor: float = None,
) -> np.ndarray:
    """
    Play a single frequency and capture it via separate OutputStream/InputStream per call.

    Returns the captured signal as a numpy array.
    """
    n_samples = int(duration_s * fs)
    t_total = np.arange(n_samples) / fs
    if corr_factor is not None:
        amplitude = tone_amplitude * corr_factor
    else:
        amplitude = tone_amplitude
    tone_full = (np.sin(2 * np.pi * freq * t_total) * amplitude).astype(np.float64)

    out_offset = [0]
    in_offset = [0]
    capture_data = np.empty(n_samples, dtype="float32")

    def _out_cb(outdata, frame_count, time_flag, status):
        if status:
            print(f"    [Output status: {status}]", flush=True)
        start = out_offset[0]
        end = min(start + frame_count, len(tone_full))
        n = min(frame_count, end - start)
        gain_factor = send_gain / 100.0
        scaled_tone = tone_full[start:end] * gain_factor
        if outdata.ndim == 1:
            outdata[:n] = scaled_tone
        else:
            outdata[:n, 0] = scaled_tone
        if end < len(tone_full):
            outdata[n:] = 0.0
            out_offset[0] = end
            return (outdata, "continue")
        else:
            outdata[n:] = 0.0
            return outdata

    def _in_cb(indata, frame_count, time_flag, status):
        if status:
            print(f"    [Input status: {status}]", flush=True)
        n = min(frame_count, len(capture_data) - in_offset[0])
        capture_data[in_offset[0] : in_offset[0] + n] = indata.flatten()[:n].astype(np.float32)
        in_offset[0] += n

    try:
        out_stream = sd.OutputStream(
            device=send_device, samplerate=fs, channels=1,
            callback=_out_cb, blocksize=512, latency="low",
        )
        in_stream = sd.InputStream(
            device=recv_device, samplerate=fs, channels=1,
            callback=_in_cb, blocksize=512, latency="low",
        )
        out_stream.start()
        in_stream.start()

        elapsed = 0
        while in_offset[0] < n_samples and elapsed < int(duration_s * 3):
            sd.sleep(100)
            elapsed += 0.1

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()
    except Exception as e:
        raise RuntimeError(f"Stream error at {freq:.1f}Hz: {e}") from e

    rec_flat = capture_data[: in_offset[0]].astype(float)
    return rec_flat



# ---------------------------------------------------------------------------
# Noise signal generation (for pink/white/brown methods)
# ---------------------------------------------------------------------------

def generate_noise_signal(method, n_samples, fs, tone_amplitude, send_gain):
    """
    Generate a broadband noise signal with the specified spectral density.

    Uses frequency-domain approach: random phases -> apply spectral shaping -> IFFT.
    RMS is scaled to match the target tone amplitude for consistent hardware output level.

    Args:
        method: "white", "pink", or "brown"
        n_samples: total number of samples in the output signal
        fs: sample rate in Hz
        tone_amplitude: base tone amplitude from config (e.g. 0.2)
        send_gain: gain percentage (e.g. 70)

    Returns:
        np.ndarray of float64 audio samples, RMS-matched to a sine wave at tone_amplitude*send_gain/100
    """
    # Target RMS = same as a sine wave at the calibrated amplitude
    target_amp = tone_amplitude * send_gain / 100.0
    target_rms = target_amp / math.sqrt(2)

    # Frequency-domain approach
    N = n_samples
    freqs_fft = np.fft.rfftfreq(N, d=1.0 / fs)

    # Random phases (uniform [0, 2pi])
    rng = np.random.default_rng(seed=42)  # deterministic for reproducibility
    phases = rng.uniform(0, 2 * math.pi, size=N // 2 + 1)

    # Spectral magnitude shaping (flat before scaling)
    mag = np.ones_like(freqs_fft)

    if method == "white":
        # Flat PSD: equal energy per Hz bin
        pass  # mag stays all ones
    elif method == "pink":
        # Pink noise: equal energy per octave (-3 dB/octave, ~1/sqrt(f))
        # Use |f|^(-0.5) but clamp at very low frequencies to avoid blow-up
        eps = 1.0  # Hz minimum frequency floor for spectral shaping
        mask = freqs_fft > eps
        mag[mask] = freqs_fft[mask] ** (-0.5)
    elif method == "brown":
        # Brown noise: integrated pink noise (-6 dB/octave, ~1/f)
        eps = 1.0
        mask = freqs_fft > eps
        mag[mask] = freqs_fft[mask] ** (-1.0)

    # Zero DC component to prevent offset on audio hardware
    mag[0] = 0.0

    # Apply spectral shaping and random phases in frequency domain
    spectrum = mag * np.exp(1j * phases)

    # Inverse FFT to time domain (irfft correctly reconstructs N samples from rfft output)
    time_signal = np.fft.irfft(spectrum, n=N).real

    # Normalize RMS to target level
    current_rms = math.sqrt(np.mean(time_signal ** 2))
    if current_rms > 1e-30:
        time_signal *= target_rms / current_rms

    return time_signal


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------
def print_freq_table(freqs, fs, mode, tone_duration=1.0, gap_s=0.3):
    """Print a formatted frequency table."""
    spacing = "log" if np.all(np.diff(log_f(freqs)) > 0) else "linear"

    border = "=" * 52
    title = f"PyAmpScope -- {spacing.capitalize()} Spaced Frequencies ({mode} mode)"
    print(f"\n{title}")
    print(border)
    print(f"  Sample rate   : {fs} Hz")
    if mode == "sequential":
        est_total = len(freqs) * (tone_duration + gap_s)
        print(f"  Mode          : sequential (one OutputStream/InputStream per freq)")
        print(f"  Est. duration : ~{est_total:.0f}s at {tone_duration:.1f}s tone + {gap_s:.1f}s gap")

    col_w = 52
    print(f"\n{'Bin':>4}  {'Frequency (Hz)':>16}  {'Samples per wave period':>20}")
    print("-" * col_w)

    for i, f in enumerate(freqs):
        samples_per_cycle = fs / f
        print(f"{i + 1:>4}  {f:>12.1f} Hz  | {samples_per_cycle:>9.0f} samples/period")

    border2 = "=" * col_w
    print(border2)
    print(f"{'Total':>4} bins : {len(freqs)}")
