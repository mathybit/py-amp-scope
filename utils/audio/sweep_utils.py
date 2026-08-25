"""Signal generation, FFT analysis, and inverse filter utilities for PyAmpScope.

Produces calibration signals (multitone sine burst, exponential sweep, pink/white
noise), captures them via sounddevice callback streaming, computes frequency response
by FFT deconvolution, and optionally generates regularized inverse correction filters
(in both complex and time-domain FIR forms).

Signal flow
-----------
  generated (play on send device) --> hardware path --> captured (record on recv device)
  H(f) = captured_fft * conj(generated_fft) / |generated_fft|^2

Multitone equal-energy design
-----------------------------
  The signal is divided into N frames. Each frame contains one pure sine at a
  discrete frequency completing an integer number of cycles, guaranteeing zero
  spectral leakage and equal energy per bin by construction.
"""

import argparse
import io
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DURATION = 30   # seconds
_FS_DEFAULT = 48000
_FREQ_MIN_DEFAULT = 20
_FREQ_MAX_DEFAULT = 24000
_REG_TOL = 1e-3           # regularization floor for inverse filter

# NPZ keys
_NPZ_KEY_FREQS = "frequencies"
_NPZ_KEY_RESPONSE = "response_H"
_NPZ_KEY_CORRECTION = "correction_W"
_NPZ_KEY_IR = "impulse_response"
_NPZ_KEY_CAL_DATA = "calibration_data"
_NPZ_KEY_PNG_BYTES = "chart_png_bytes"


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def _sinc(x: np.ndarray) -> np.ndarray:
    """Safe sinc function: sin(pi*x) / (pi*x), with sinc(0)=1."""
    x = np.asarray(x, dtype=np.float64)
    result = np.ones_like(x)
    nonzero = np.abs(x) > 1e-12
    result[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    return result


def generate_multitone(
    duration: float = _DEFAULT_DURATION,
    fs: int = _FS_DEFAULT,
    freq_min: int = _FREQ_MIN_DEFAULT,
    freq_max: int = _FREQ_MAX_DEFAULT,
) -> np.ndarray:
    """Generate equal-energy multi-tone signal.

    Divides `duration` into N frames. Each frame is a pure sine at one
    discrete frequency completing an integer number of cycles. Frequencies are
    evenly spaced across [freq_min, freq_max] with ~60 Hz spacing (giving
    fine resolution while keeping per-bin energy high).

    Returns:
        numpy.ndarray[float32]: mono signal samples.
    """
    n_frames = max(int(duration // 0.5), 1)      # at least one frame per 0.5 s
    frame_samples = int(fs * 0.5)                  # each frame is 500 ms
    total_samples = n_frames * frame_samples
    signal = np.zeros(total_samples, dtype=np.float64)

    freqs = np.linspace(freq_min, freq_max, n_frames)
    for i, (f_center, frame_start) in enumerate(zip(freqs, range(0, total_samples, frame_samples))):
        f_i = freq_min + (freq_max - freq_min) * i / max(n_frames - 1, 1)
        t_frame = np.arange(frame_samples) / fs
        # Integer cycles: round to nearest to minimize leakage
        n_cycles = int(round(f_i * frame_samples / fs))
        if n_cycles < 1:
            n_cycles = 1
        signal[frame_start:frame_start + frame_samples] += (
            np.sin(2 * np.pi * n_cycles * t_frame)
        )

    # Normalize to float32 range without clipping
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= 1.0 / peak                        # scale to [-1, +1]
    return signal.astype(np.float32), freqs[:n_frames]


def generate_sweep(
    duration: float = _DEFAULT_DURATION,
    fs: int = _FS_DEFAULT,
    freq_min: int = _FREQ_MIN_DEFAULT,
    freq_max: int = _FREQ_MAX_DEFAULT,
) -> np.ndarray:
    """Generate exponential chirp/sweep signal.

    Frequency rises exponentially from freq_min to freq_max over `duration`.
    Has unequal energy per bin (low frequencies get more cycles), kept for
    non-wire-through tests where perceptual weighting matters.
    """
    t = np.arange(int(duration * fs)) / fs
    # Exponential sweep: f(t) = freq_min * (freq_max/freq_min)^(t/T)
    k = np.log(freq_max / freq_min) / duration
    phase = 2 * np.pi * freq_min / k * (np.exp(k * t) - 1)
    signal = np.sin(phase)

    # Hann taper at start/end to reduce edge transients
    taper_len = int(0.01 * fs)                      # 10 ms
    if taper_len > 0:
        taper = np.hanning(2 * taper_len)
        signal[:taper_len] *= taper[:taper_len]
        signal[-taper_len:] *= taper[taper_len:]

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return signal.astype(np.float32)


def generate_pink(
    duration: float = _DEFAULT_DURATION,
    fs: int = _FS_DEFAULT,
) -> np.ndarray:
    """Generate pink (1/f) noise using Voss-McCartney algorithm.

    Each new sample adds one random value from the previous state,
    giving 1/f spectral density.
    """
    n_samples = int(duration * fs)
    levels = 6                              # number of random-walk layers
    states = np.zeros(levels)
    pink = np.empty(n_samples)
    for i in range(n_samples):
        level = np.random.randint(0, levels)
        states[level] = np.random.randn()     # new random walk position
        pink[i] = np.sum(states) / levels     # average all layers
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink.astype(np.float32)


def generate_white(
    duration: float = _DEFAULT_DURATION,
    fs: int = _FS_DEFAULT,
) -> np.ndarray:
    """Generate white (flat spectral density) noise."""
    n_samples = int(duration * fs)
    signal = np.random.randn(n_samples).astype(np.float64)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return signal.astype(np.float32)


def generate_cal_signal(
    method: str = "multitone",
    duration: float = _DEFAULT_DURATION,
    fs: int = _FS_DEFAULT,
    freq_min: int = _FREQ_MIN_DEFAULT,
    freq_max: int = _FREQ_MAX_DEFAULT,
):
    """Dispatch to signal generator by method name.

    Returns:
        For 'multitone': (signal, freq_array) where freq_array holds per-frame center freqs.
        For all others: signal only (np.ndarray[float32]).
    """
    if method == "multitone":
        return generate_multitone(duration=duration, fs=fs,
                                  freq_min=freq_min, freq_max=freq_max)
    elif method == "sweep":
        return generate_sweep(duration=duration, fs=fs,
                              freq_min=freq_min, freq_max=freq_max)
    elif method == "pink":
        return generate_pink(duration=duration, fs=fs)
    elif method == "white":
        return generate_white(duration=duration, fs=fs)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------
def compute_frequency_response(
    generated: np.ndarray,
    captured: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frequency response H(f) = C(f) * conj(G(f)) / |G(f)|^2.

    Regularization floor prevents division-by-zero in deep nulls. Bins where the
    generated signal has <1% peak energy are zeroed (avoids bogus values from
    epsilon floor). Returns (freqs, response_H, G) where:
      freqs  : array of frequencies in Hz
      response_H : complex frequency response H(f)
      G         : complex FFT of the generated signal (same length as H)
    """
    fft_len = max(len(generated), len(captured))

    G = np.fft.rfft(generated, n=fft_len)
    C = np.fft.rfft(captured, n=fft_len)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / fs)

    # Regularized deconvolution: H = C * conj(G) / max(|G|, tol)
    G_mag_sq = np.abs(G) ** 2
    tol = _REG_TOL
    # Energy mask: only trust bins where generated signal has real energy
    peak_energy = np.max(np.abs(G))
    valid_low = np.abs(G) > 0.01 * peak_energy
    valid_tol = np.abs(G) > tol
    valid_mask = valid_low | valid_tol

    G_norm_sq = G_mag_sq.copy()
    G_norm_sq[G_norm_sq < tol ** 2] = tol ** 2   # regularization floor
    H = C * np.conj(G) / G_norm_sq

    # Zero out invalid bins so they don't pollute stats
    H[~valid_mask] = 0.0 + 0.0j

    return freqs, H, G


# ---------------------------------------------------------------------------
# Inverse filter
# ---------------------------------------------------------------------------
def compute_inverse_filter(
    H: np.ndarray,
    freqs: np.ndarray,
    fft_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute regularized inverse correction filter W(f).

    W(f) = conj(H(f)) / max(|H(f)|, tol)

    Returns:
        W: complex frequency-domain filter (same length as H)
        ir: time-domain FIR impulse response (length fft_len)
    """
    H_mag = np.abs(H)
    W = np.conj(H) / np.maximum(H_mag, _REG_TOL)

    # IFFT to get FIR impulse response
    ir = np.fft.ifft(W, n=fft_len).real

    return W, ir


# ---------------------------------------------------------------------------
# Visualization chart
# ---------------------------------------------------------------------------
def build_chart_png(
    freqs: np.ndarray,
    H: np.ndarray,
    W: Optional[np.ndarray] = None,
    title: str = "Calibration Frequency Response",
    alpha: float = 0.85,
) -> bytes:
    """Build a 3-curve PNG chart (matplotlib) showing response, compensation, corrected.

    Returns raw PNG bytes.
    """
    import matplotlib
    matplotlib.use("Agg")                      # headless backend
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    H_mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))

    # Build corrected response if W is provided: H_corrected = H * W
    if W is not None:
        W_mag_db = 20 * np.log10(np.maximum(np.abs(W), 1e-10))
        H_corr_mag_db = 20 * np.log10(
            np.maximum(np.abs(H) * np.abs(W), 1e-10)
        )

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-80, 60)
    ax.set_xscale("log")
    ax.set_xlim(max(freqs[0], 20), freqs[-1])
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.grid(True, which="major", axis="x", alpha=0.3)

    # Plot corrected response first (on top)
    if W is not None:
        ax.plot(freqs, H_corr_mag_db, "g-", linewidth=1.2, alpha=0.65, label="Corrected")
        ax.plot(freqs, W_mag_db, "b--", linewidth=1.0, alpha=0.65, label="Compensation (W)")
    ax.plot(freqs, H_mag_db, "r-", linewidth=1.5, alpha=0.75, label="Measured (H)")

    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_cal_profile(
    output_dir: Path,
    prefix: str = "cal_send",
    metadata_dict: Optional[dict] = None,
    correction_filter: Optional[np.ndarray] = None,
    ir: Optional[np.ndarray] = None,
    freqs: Optional[np.ndarray] = None,
    response_H: Optional[np.ndarray] = None,
) -> Path:
    """Save calibration profile (and optional correction filter/IR) as NPZ.

    If PNG chart data is in `metadata_dict["chart_png_bytes"]`, it is stored under
    ``NPZ_KEY_PNG_BYTES`` for programmatic access.

    Returns the path to the saved NPZ file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{prefix}_profile.npz"

    save_dict: dict = {
        _NPZ_KEY_FREQS: freqs if freqs is not None else np.array([]),
        _NPZ_KEY_RESPONSE: response_H if response_H is not None else np.array([], dtype=np.complex128),
    }

    if correction_filter is not None:
        save_dict[_NPZ_KEY_CORRECTION] = correction_filter
    if ir is not None:
        save_dict[_NPZ_KEY_IR] = ir
    if "chart_png_bytes" in (metadata_dict or {}):
        save_dict[_NPZ_KEY_PNG_BYTES] = metadata_dict["chart_png_bytes"]

    np.savez(str(npz_path), **save_dict)

    # Also write a metadata JSON alongside for human readability
    meta_path = output_dir / f"{prefix}_profile.meta.json"
    meta_for_json = {}
    if metadata_dict:
        clean_meta = {k: v for k, v in metadata_dict.items()
                       if not isinstance(v, (np.ndarray, bytes))}
        meta_for_json.update(clean_meta)
    meta_path.write_text(json.dumps(meta_for_json, indent=2, default=str))

    return npz_path


# ---------------------------------------------------------------------------
# Play and capture via sounddevice callback streaming
# ---------------------------------------------------------------------------
def play_and_capture(
    generated_signal: np.ndarray,
    duration: float,
    fs: int,
    send_device: Optional[int] = None,
    recv_device: Optional[int] = None,
    send_gain: float = 1.0,
    recv_gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play ``generated_signal`` on the send device while capturing from the recv device.

    Uses simultaneous OutputStream + InputStream callback streaming. Callbacks stop
    automatically when signal samples are exhausted. Returns (captured, generated_trimmed)
    where both arrays have been scaled by their respective gain percentages.
    """
    send_scaled = np.clip(generated_signal * (send_gain / 100.0), -1.0, 1.0).astype(np.float32)
    expected_samples = int(duration * fs)

    # Validate devices support required I/O
    all_devs = sd.query_devices()
    if isinstance(all_devs, dict):
        all_devs = [all_devs]

    if send_device is not None and dev_channel_count(all_devs[send_device], "out") == 0:
        raise ValueError(f"Device {send_device} does not support playback.")
    if recv_device is not None and dev_channel_count(all_devs[recv_device], "in") == 0:
        raise ValueError(f"Device {recv_device} does not support capture.")

    # Buffers
    buf = np.zeros(expected_samples, dtype='float32')
    outbuf = np.zeros(512, dtype='float32').reshape(-1, 1)
    offset = 0
    output_done = False

    def _out_cb(outdata, frame_count, time_flag, status):
        nonlocal offset, output_done
        if status:
            print(f"  [Output stream status: {status}]")
        start = offset
        end = min(offset + frame_count, len(send_scaled))
        n_out = min(end - start, len(outbuf))
        outbuf[:n_out] = send_scaled[start:end].reshape(-1, 1)
        if end < len(send_scaled):
            # More signal remaining
            outbuf[n_out:] = 0.0
            offset = end
            return (outbuf, "continue")
        else:
            # Signal exhausted — zero-fill rest and stop stream
            outbuf[n_out:] = 0.0
            output_done = True
            return outbuf[:n_out]

    def _in_cb(indata, frame_count, time_flag, status):
        nonlocal offset
        if status:
            print(f"  [Input stream status: {status}]")
        n = min(frame_count, expected_samples - offset)
        buf[offset:offset + n] = indata.flatten()[:n]
        offset += n

    try:
        print(f"  [Playing on device {send_device} / capturing on device {recv_device}]")
        out_stream = sd.OutputStream(
            device=send_device, samplerate=fs, channels=1,
            callback=_out_cb, blocksize=512,
        )
        in_stream = sd.InputStream(
            device=recv_device, samplerate=fs, channels=1,
            callback=_in_cb, blocksize=512,
        )

        out_stream.start()
        in_stream.start()

        # Wait for both buffers to fill up
        elapsed = 0
        wait_limit = int(duration * 1.5) + 2      # generous margin
        while (offset < expected_samples or not output_done) and elapsed < wait_limit:
            sd.sleep(100)                            # non-blocking sleep
            elapsed += 0.1

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()

    except Exception as e:
        raise RuntimeError(f"Audio I/O failed: {e}") from e

    # Truncate to actual played duration for consistent FFT lengths
    actual_samples = min(offset, expected_samples)
    if actual_samples == 0:
        raise RuntimeError("No audio captured — check device selection and gains.")

    return buf[:actual_samples].copy(), send_scaled[:actual_samples].copy()


# ---------------------------------------------------------------------------
# Helpers (used above and externally)
# ---------------------------------------------------------------------------
def dev_channel_count(dev: dict, direction: str) -> int:
    """Get max channels for 'in' or 'out', handling legacy PortAudio key names."""
    out_key = "max_output_channels" if direction == "out" else "max_input_channels"
    in_key = "max_output_streams" if direction == "out" else "max_input_streams"
    return dev.get(out_key, dev.get(in_key, 0))
