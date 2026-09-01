"""Shared sounddevice streaming engine for GUI, calibration, and validation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import time
from typing import Callable, Optional
try:
    import sounddevice as sd
except ImportError:  # allows offline DSP/unit testing without PortAudio bindings
    print("Warning: unable to import sounddevice")
    sd = None

from .signal_utils import SignalBundle, build_noise_signal, build_sweep_signal

ProgressCallback = Callable[[float, float], None]


@dataclass
class CaptureResult:
    capture: np.ndarray
    stimulus: SignalBundle
    sample_rate: int
    send_device: object
    recv_device: object
    send_channel: str
    recv_channel: str
    stream_status: list[str]


def _require_sounddevice():
    if sd is None:
        raise RuntimeError("sounddevice is required for hardware capture (pip install sounddevice)")


def _device_max_channels(device, direction: str) -> int:
    if sd is None:
        return 2
    try:
        info = sd.query_devices(device)
        key = "max_output_channels" if direction == "out" else "max_input_channels"
        return max(1, int(info.get(key, 1)))
    except Exception:
        return 2


def _stream_channels(device, mode: str, direction: str) -> int:
    max_ch = _device_max_channels(device, direction)
    if mode.upper() in {"RIGHT", "STEREO"} and max_ch >= 2:
        return 2
    return 1


def _write_output(outdata: np.ndarray, mono: np.ndarray, mode: str) -> None:
    outdata[:] = 0.0
    if outdata.ndim == 1:
        outdata[: len(mono)] = mono
        return
    # Single-channel stream produces (n, 1) arrays — assign to column 0.
    if outdata.shape[1] == 1:
        outdata[: len(mono), 0] = mono
        return
    mode = mode.upper()
    if mode == "RIGHT":
        outdata[: len(mono), 1] = mono
    elif mode == "STEREO":
        outdata[: len(mono), 0] = mono
        outdata[: len(mono), 1] = mono
    else:
        outdata[: len(mono), 0] = mono


def _read_input(indata: np.ndarray, mode: str) -> np.ndarray:
    if indata.ndim == 1 or indata.shape[1] == 1:
        return np.asarray(indata).reshape(-1)
    mode = mode.upper()
    if mode == "RIGHT":
        return indata[:, 1]
    if mode == "STEREO":
        # Average stereo channels rather than summing, preserving digital scale.
        return np.mean(indata[:, :2], axis=1)
    return indata[:, 0]


def duplex_capture(
            output_signal: np.ndarray,
            fs: int,
            send_device,
            recv_device,
            *,
            send_ch: str = "LEFT",
            recv_ch: str = "LEFT",
            tail_s: float = 0.25,
            blocksize: int = 512,
            latency: str | float = "low",
            stop_event=None,
            progress_callback: Optional[ProgressCallback] = None,
        ) -> tuple[np.ndarray, list[str]]:
    """Play one mono analysis waveform and capture the selected input channel.

    Input starts before output.  The generated waveform itself contains a short
    pre-roll for sweep measurements, which makes small independent-device latency
    offsets harmless to per-tone analysis.
    """
    _require_sounddevice()
    signal = np.asarray(output_signal, dtype=np.float64)
    tail_n = max(0, int(round(tail_s * fs)))
    capture_n = len(signal) + tail_n
    capture = np.zeros(capture_n, dtype=np.float32)
    in_offset = 0
    out_offset = 0
    statuses: list[str] = []

    out_channels = _stream_channels(send_device, send_ch, "out")
    in_channels = _stream_channels(recv_device, recv_ch, "in")

    def in_cb(indata, frames, time_info, status):
        nonlocal in_offset
        if status:
            statuses.append(f"input: {status}")
        if stop_event is not None and stop_event.is_set():
            return
        mono = _read_input(indata, recv_ch)
        n = min(frames, capture_n - in_offset)
        if n > 0:
            capture[in_offset:in_offset+n] = mono[:n].astype(np.float32, copy=False)
            in_offset += n

    def out_cb(outdata, frames, time_info, status):
        nonlocal out_offset
        if status:
            statuses.append(f"output: {status}")
        if stop_event is not None and stop_event.is_set():
            outdata[:] = 0.0
            return
        n = min(frames, max(0, len(signal) - out_offset))
        buf = np.zeros(frames, dtype=np.float64)
        if n > 0:
            buf[:n] = signal[out_offset:out_offset+n]
            out_offset += n
        _write_output(outdata, buf, send_ch)

    start = time.monotonic()
    input_stream = sd.InputStream(
        device=recv_device, samplerate=fs, channels=in_channels,
        callback=in_cb, blocksize=blocksize, latency=latency, dtype="float32",
    )
    output_stream = sd.OutputStream(
        device=send_device, samplerate=fs, channels=out_channels,
        callback=out_cb, blocksize=blocksize, latency=latency, dtype="float32",
    )

    try:
        input_stream.start()
        output_stream.start()
        expected_s = capture_n / fs
        timeout_s = max(expected_s * 1.75, expected_s + 2.0)
        while in_offset < capture_n and (time.monotonic() - start) < timeout_s:
            if stop_event is not None and stop_event.is_set():
                break
            if progress_callback is not None:
                progress_callback(min(in_offset / max(capture_n, 1), 1.0), time.monotonic() - start)
            sd.sleep(50)
    finally:
        for stream in (output_stream, input_stream):
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    if progress_callback is not None:
        progress_callback(min(in_offset / max(capture_n, 1), 1.0), time.monotonic() - start)
    return capture[:in_offset].astype(np.float64), statuses


def run_sweep_measurement(
            *,
            freqs: np.ndarray,
            fs: int,
            tone_duration: float,
            gap_s: float,
            tone_amplitude: float,
            send_gain: float,
            send_device,
            recv_device,
            send_ch: str = "LEFT",
            recv_ch: str = "LEFT",
            send_correction_freqs: Optional[np.ndarray] = None,
            send_correction_factors: Optional[np.ndarray] = None,
            peak_headroom: float = 0.95,
            stop_event=None,
            progress_callback: Optional[ProgressCallback] = None,
        ) -> CaptureResult:
    stimulus = build_sweep_signal(
        freqs, fs, tone_duration, gap_s, tone_amplitude, send_gain,
        correction_freqs=send_correction_freqs,
        correction_factors=send_correction_factors,
        peak_headroom=peak_headroom,
    )
    capture, statuses = duplex_capture(
        stimulus.samples, fs, send_device, recv_device,
        send_ch=send_ch, recv_ch=recv_ch, stop_event=stop_event,
        progress_callback=progress_callback,
    )
    return CaptureResult(capture, stimulus, fs, send_device, recv_device, send_ch, recv_ch, statuses)


def run_noise_measurement(
            *,
            method: str,
            duration_s: float,
            fs: int,
            tone_amplitude: float,
            send_gain: float,
            send_device,
            recv_device,
            send_ch: str = "LEFT",
            recv_ch: str = "LEFT",
            send_correction_freqs: Optional[np.ndarray] = None,
            send_correction_factors: Optional[np.ndarray] = None,
            peak_headroom: float = 0.95,
            seed: int = 12345,
            stop_event=None,
            progress_callback: Optional[ProgressCallback] = None,
        ) -> CaptureResult:
    stimulus = build_noise_signal(
        method, int(round(duration_s * fs)), fs, tone_amplitude, send_gain,
        correction_freqs=send_correction_freqs,
        correction_factors=send_correction_factors,
        peak_headroom=peak_headroom, seed=seed,
    )
    capture, statuses = duplex_capture(
        stimulus.samples, fs, send_device, recv_device,
        send_ch=send_ch, recv_ch=recv_ch, stop_event=stop_event,
        progress_callback=progress_callback,
    )
    # Trim capture to nominal stimulus duration for noise analysis; the tail is
    # deliberately excluded from RMS/PSD metrics.
    capture = capture[: len(stimulus.samples)]
    return CaptureResult(capture, stimulus, fs, send_device, recv_device, send_ch, recv_ch, statuses)
