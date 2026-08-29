#!/usr/bin/env python
"""PyAmpScope Amplifier Analyzer — GUI tool.

Sends noise or sweep signals to an amplifier via the USB audio interface,
captures the response on line-in, and displays live frequency-response charts
and harmonic/distortion metrics.

Architecture:
  Main thread  -- Tkinter event loop + matplotlib canvas (chart) + metrics panel
  Worker thread -- sounddevice streaming (OutputStream+InputStream callbacks)
                   progressive FFT analysis every N seconds
                   communicates results back to main via queue.Queue
"""
import math
import sys
import time
import json
import queue
import threading
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ── Local imports ─────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from config import log_f, config as _cfg_module
from utils.audio.signal_utils import generate_noise_signal
from utils.audio.analysis_utils import analyze_noise_response, compare_noise_spectral_shape, smooth_moving_average, extract_tone_measurements


class _Config:
    """Thin wrapper exposing config.py module-level vars as attributes."""
    send_device = property(lambda self: getattr(_cfg_module, "send_device", None))
    recv_device = property(lambda self: getattr(_cfg_module, "recv_device", None))
    send_ch = property(lambda self: getattr(_cfg_module, "send_ch", "LEFT"))
    recv_ch = property(lambda self: getattr(_cfg_module, "recv_ch", "LEFT"))
    send_gain = property(lambda self: getattr(_cfg_module, "send_gain", 70))
    recv_gain = property(lambda self: getattr(_cfg_module, "recv_gain", 50))
    cal_method = property(lambda self: getattr(_cfg_module, "cal_method", "sweep"))
    freq_min = property(lambda self: getattr(_cfg_module, "freq_min", 40))
    freq_max = property(lambda self: getattr(_cfg_module, "freq_max", 20000))
    fs = property(lambda self: getattr(_cfg_module, "fs", 44100))
    min_calibration_time = property(lambda self: getattr(_cfg_module, "min_calibration_time", 30))
    noise_calibration_time = property(lambda self: getattr(_cfg_module, "noise_calibration_time", 30))
    num_freqs_default = property(lambda self: getattr(_cfg_module, "num_freqs_default", 60))
    tone_duration = property(lambda self: getattr(_cfg_module, "tone_duration", 0.7))
    tone_gap = property(lambda self: getattr(_cfg_module, "tone_gap", 0.2))
    tone_amplitude = property(lambda self: getattr(_cfg_module, "tone_amplitude", 0.2))
    smoothing_neighbors = property(lambda self: getattr(_cfg_module, "smoothing_neighbors", 5))
    recv_path = property(lambda self: getattr(_cfg_module, "recv_path", "dir"))
    data_dir = property(lambda self: getattr(_cfg_module, "data_dir", "data"))
    logs_dir = property(lambda self: getattr(_cfg_module, "logs_dir", "logs"))

config = _Config()


def generate_sweep_sequence(freq_array, fs, duration_s=30.0, gap_s=0.2, tone_amplitude=1.0):
    """Generate a sweep signal: each freq is played sequentially with inter-tone gaps.

    Args:
        freq_array: np.ndarray of frequencies in Hz (each element is one tone).
        fs: sample rate in Hz.
        duration_s: duration per tone in seconds.
        gap_s: gap between tones in seconds.
        tone_amplitude: amplitude for each sine tone.

    Returns:
        np.ndarray of float64 samples, concatenated sequence of all tones + gaps.
    """
    n_tones = len(freq_array)
    # Calculate total length
    tone_samples = int(duration_s * fs)
    gap_samples = int(gap_s * fs)
    total_len = n_tones * tone_samples + (n_tones - 1) * gap_samples
    signal = np.zeros(total_len, dtype=np.float64)

    offset = 0
    for i, freq in enumerate(freq_array):
        t = np.arange(tone_samples) / fs
        tone = tone_amplitude * np.sin(2.0 * math.pi * freq * t)
        signal[offset:offset + tone_samples] = tone.astype(np.float64)
        offset += tone_samples
        if i < n_tones - 1:
            offset += gap_samples  # gap of zeros

    return signal


# ==========================================================================
#  Signal analysis helpers (new: THD + harmonics)
# ==========================================================================

def _find_persistent_peaks(sig, fs, peak_thresh_db=-60.0, min_prominence_hz=50):
    """Find peaks in a signal's FFT that persist across the full capture."""
    fft_vals = np.fft.rfft(sig.astype(np.float64))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    mag_db = 20 * log_f(np.maximum(np.abs(fft_vals), 1e-30))
    return freqs, mag_db


def compute_thd(signal, fs):
    """Compute Total Harmonic Distortion (THD) of a signal.

    Finds the fundamental (highest non-DC peak above 20 Hz), then sums power
    in integer harmonics 2..10. Returns dict with thd_pct, thdn_pct, and harmonic_levels_dB.
    """
    freqs, mag_db = _find_persistent_peaks(signal, fs)

    # Find fundamental: highest peak above threshold in [20, 500] Hz
    fund_candidates = (freqs > 20) & (freqs < 500) & (mag_db > -90)
    if not np.any(fund_candidates):
        return {"thd_pct": None, "thdn_pct": None, "harmonic_levels_dB": {}}

    fund_freqs_on_mask = freqs[fund_candidates]
    fund_mag_on_mask = mag_db[fund_candidates]
    best_idx = np.argmax(fund_mag_on_mask)
    fund_freq = float(fund_freqs_on_mask[best_idx])
    fund_mag_db = float(fund_mag_on_mask[best_idx])

    thd_numer = 0.0
    thdn_numer = 0.0
    harmonics_dB = {}

    for h in range(2, 11):
        h_freq = fund_freq * h
        if h_freq > fs / 2.0:
            break
        # Find nearest FFT bin
        idx = int(np.argmin(np.abs(freqs - h_freq)))
        h_mag_db = float(mag_db[idx])
        harmonics_dB["%dth" % h] = h_mag_db
        h_lin = 10 ** ((h_mag_db - fund_mag_db) / 20.0)
        thd_numer += h_lin ** 2
        if h <= 5:
            thdn_numer += h_lin ** 2

    thd_pct = float(math.sqrt(thd_numer)) * 100.0
    # THD+N: fundamental + harmonics total relative to noise floor
    noise_floor_idx = (freqs > fund_freq * 20) & (freqs < fs / 3.0)
    if np.any(noise_floor_idx):
        noise_rms = float(np.sqrt(np.mean(10 ** ((mag_db[noise_floor_idx] - fund_mag_db) / 10.0))))
        thdn_pct = math.sqrt(thd_numer + noise_rms ** 2) * 100.0
    else:
        thdn_pct = thd_pct

    return {
        "fundamental_freq_hz": round(fund_freq, 1),
        "fundamental_mag_dB": round(fund_mag_db, 2),
        "thd_pct": round(thd_pct, 3),
        "thdn_pct": round(thdn_pct, 3),
        "harmonic_levels_dB": {k: round(v, 2) for k, v in harmonics_dB.items()},
    }


def compute_harmonics_for_overdrive(signal, fs, fundamental_hz=None):
    """Find prominent harmonics above a threshold (for overdrive analysis).

    If fundamental is not given, finds it automatically as the strongest peak above 20 Hz.
    Returns list of dicts with freq, level_dB, order keys.
    """
    freqs, mag_db = _find_persistent_peaks(signal, fs)

    if fundamental_hz is None:
        candidates = (freqs > 20) & (mag_db > -90)
        if not np.any(candidates):
            return []
        fund_idx = np.argmax(mag_db[candidates])
        fundamental_hz = float(freqs[candidates][fund_idx])

    # Find peaks near integer harmonics of the fundamental
    fundamentals_lin = 10 ** (mag_db[np.argmin(np.abs(freqs - fundamental_hz))] / 20.0)

    result = []
    for h in range(2, min(16, int((fs / 2.0) / fundamental_hz))):
        h_freq = fundamental_hz * h
        if h_freq > fs / 2.0:
            break
        idx = np.argmin(np.abs(freqs - h_freq))
        if mag_db[idx] > -90:
            h_lin = 10 ** (mag_db[idx] / 20.0) / fundamentals_lin
            result.append({
                "order": h,
                "freq_hz": round(h_freq, 1),
                "level_dB_relative_to_fund": round(20 * log_f(max(abs(h_lin), 1e-30)), 2),
                "level_linear": round(abs(h_lin), 6),
            })

    return result


def compute_odd_even_ratio(signal, fs):
    """Compute the ratio of odd-order to even-order harmonic power."""
    freqs, mag_db = _find_persistent_peaks(signal, fs)
    candidates = (freqs > 20) & (mag_db > -90)
    if not np.any(candidates):
        return None

    fund_idx = np.argmax(mag_db[candidates])
    fund_freq = float(freqs[candidates][fund_idx])

    odd_power = 0.0
    even_power = 0.0
    for h in range(2, min(16, int((fs / 2.0) / fund_freq))):
        h_freq = fund_freq * h
        idx = np.argmin(np.abs(freqs - h_freq))
        if mag_db[idx] > -90:
            h_lin = 10 ** (mag_db[idx] / 20.0)
            if h % 2 == 1:
                odd_power += h_lin ** 2
            else:
                even_power += h_lin ** 2

    if even_power < 1e-30:
        return {"odd_even_ratio": None, "odd_power_dB": round(10 * log_f(max(odd_power, 1e-30)), 2),
                "even_power_dB": -99.0}
    return {
        "odd_even_ratio": round(math.sqrt(odd_power / max(even_power, 1e-30)), 4),
        "odd_power_dB": round(10 * log_f(max(odd_power, 1e-30)), 2),
        "even_power_dB": round(10 * log_f(max(even_power, 1e-30)), 2),
    }


def compute_octave_band_stats(signal, freq_array, fs):
    """Compute per-octave-band mean and std of measured dB."""
    fft_vals = np.abs(np.fft.rfft(signal.astype(np.float64))) / (len(signal) // 2)
    fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    scale = 2.0 / len(signal)
    mag_db = 20 * log_f(np.maximum(fft_vals * scale, 1e-30))

    octaves = [(20, 100, "Sub-bass"), (100, 300, "Bass"), (300, 800, "Low-mid"),
               (800, 2000, "Mid"), (2000, 5000, "Upper-mid"), (5000, 10000, "Presence"),
               (10000, 20000, "Brilliance")]

    bands = {}
    for lo, hi, name in octaves:
        mask = (fft_freqs >= lo) & (fft_freqs < hi) & (~np.isnan(mag_db)) & (fft_freqs > 0)
        if np.sum(mask) > 2:
            vals = mag_db[mask]
            bands[name] = {
                "mean_dB": round(float(np.mean(vals)), 2),
                "std_dB": round(float(np.std(vals)), 2),
                "min_dB": round(float(np.min(vals)), 2),
                "max_dB": round(float(np.max(vals)), 2),
                "bins": int(np.sum(mask)),
            }
    return bands


# ==========================================================================
#  Device discovery
# ==========================================================================

def _get_audio_devices():
    """Return list of (index, name, status) tuples via sounddevice."""
    import sounddevice as sd
    devs = sd.query_devices()
    devices = []
    for d in devs:
        i = d["index"] if "index" in d else len(devices)
        name = str(d.get("name", "Unknown"))
        available = d.get("hostapi", -1) >= 0 and "None" not in name
        status = "available" if available else "disconnected"
        devices.append((i, name[:60], status))
    return devices


def _device_display_name(dev):
    """Short display name for a device (index + first 40 chars of name)."""
    idx, name, status = dev
    return "[%d] %s (%s)" % (idx, name[:50], status)


# ==========================================================================
#  Device picker dialog (Tkinter)
# ==========================================================================

class DevicePickerDialog:
    """Modal dialog for selecting send/recv audio devices."""

    def __init__(self, parent, current_send=None, current_recv=None):
        self.result = {"send": None, "recv": None}
        self.top = tk.Toplevel(parent)
        self.top.title("Select Audio Devices")
        self.top.geometry("580x420")
        self.top.resizable(False, False)
        self._build_ui(current_send, current_recv)

    def _build_ui(self, send_idx, recv_idx):
        ttk.Label(self.top, text="Send device (output/headphone):", font=("Segoe UI", 9)).pack(
            padx=10, pady=(8, 2), anchor="w")
        self._make_device_picker("send", send_idx)

        ttk.Label(self.top, text="Recv device (input/line-in):", font=("Segoe UI", 9)).pack(
            padx=10, pady=(14, 2), anchor="w")
        self._make_device_picker("recv", recv_idx)

        btn_frame = ttk.Frame(self.top)
        btn_frame.pack(pady=12)
        ttk.Button(btn_frame, text="OK", width=10, command=self._ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", width=10, command=self._cancel).pack(side=tk.LEFT, padx=5)

        # Update device list on open
        self.top.after(50, self._refresh_devices)

    def _make_device_picker(self, key, default_idx):
        frame = ttk.Frame(self.top)
        frame.pack(padx=10, fill=tk.X)

        cb = ttk.Combobox(frame, width=72, state="readonly")
        cb.pack(fill=tk.X, expand=True)
        cb.bind("<<ComboboxSelected>>", lambda e, k=key: self._on_select(k))
        setattr(self, "_cb_%s" % key, cb)

        if default_idx is not None:
            devices = _get_audio_devices()
            for i, n, s in devices:
                if i == default_idx:
                    cb.current(devices.index((i, n, s)) if (i, n, s) in devices else -1)

    def _refresh_devices(self):
        devices = _get_audio_devices()
        for key in ("send", "recv"):
            cb = getattr(self, "_cb_%s" % key)
            names = ["[%d] %s (%s)" % (i, n[:60], s) for i, n, s in devices]
            cb["values"] = names
            if not names:
                cb["values"] = ["No devices found"]

    def _on_select(self, key):
        cb = getattr(self, "_cb_%s" % key)
        sel = cb.get()
        # Parse index from "[13] ..."
        try:
            self.result[key] = int(sel.strip().lstrip("["))
        except ValueError:
            self.result[key] = None

    def _ok(self):
        self.top.destroy()

    def _cancel(self):
        self.result = {"send": None, "recv": None}
        self.top.destroy()

    @property
    def send_idx(self):
        return self.result["send"]

    @property
    def recv_idx(self):
        return self.result["recv"]


# ==========================================================================
#  Save dialog (Tkinter)
# ==========================================================================

class SaveDialog:
    """Simple filename entry dialog."""

    def __init__(self, parent, default_name="amplifier_profile"):
        self.filename = None
        self.top = tk.Toplevel(parent)
        self.top.title("Save Results")
        self.top.geometry("360x120")
        self.top.resizable(False, False)

        ttk.Label(self.top, text="Enter filename (no extension):").pack(pady=(10, 4))
        entry = ttk.Entry(self.top, width=40)
        entry.pack(pady=4)
        entry.insert(0, default_name)
        entry.select_range(0, tk.END)
        entry.focus_set()

        btn_frame = ttk.Frame(self.top)
        btn_frame.pack(pady=(6, 10))
        ttk.Button(btn_frame, text="OK", width=8, command=lambda: self._save(entry.get())).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", width=8, command=self._cancel).pack(side=tk.LEFT, padx=5)

    def _save(self, name):
        if name.strip():
            self.filename = name.strip().replace(" ", "_")
        self.top.destroy()

    def _cancel(self):
        self.top.destroy()


# ==========================================================================
#  Worker thread: streaming + progressive analysis
# ==========================================================================

class AnalyzerWorker(threading.Thread):
    """Background worker that streams audio and runs progressive FFT analysis.

    Communicates results back to the GUI thread via `result_queue` (queue.Queue).
    Signal types: 'noise' or 'sweep'.
    """

    # Message types sent to GUI queue
    MSG_STATUS = "status"      # payload: string message
    MSG_CHART  = "chart"       # payload: dict {freqs, amp_db, smoothed_db}
    MSG_METRICS = "metrics"    # payload: dict of computed metrics
    MSG_DONE   = "done"        # payload: None

    def __init__(self, signal_type, **params):
        super().__init__(daemon=True)
        self.signal_type = signal_type
        self.params = params  # method, n_samples, fs, freq_array, tone_duration, gap_s, etc.
        self._stop_event = threading.Event()
        self._capture_data = None

    def stop(self):
        self._stop_event.set()

    def _stream_and_analyze(self, capture_queue):
        """Core streaming loop: generate signal -> stream to DAC -> capture via ADC."""
        import sounddevice as sd
        fs = self.params["fs"]
        method = self.params.get("method", "white")
        n_samples = self.params.get("n_samples", int(fs * 30))
        freq_array = self.params.get("freq_array", None)
        tone_duration = self.params.get("tone_duration", 0.7)
        gap_s = self.params.get("gap_s", 0.2)
        tone_amplitude = self.params.get("tone_amplitude", 0.2)
        send_gain = self.params.get("send_gain", 70)
        send_device = self.params["send_device"]
        recv_device = self.params["recv_device"]

        # Generate signal
        capture_queue.put((self.MSG_STATUS, "Generating signal..."))

        if self.signal_type == "noise":
            noise_signal = generate_noise_signal(
                method=method, n_samples=n_samples, fs=int(fs),
                tone_amplitude=float(tone_amplitude), send_gain=float(send_gain))
        else:
            # sweep: generate sequential tones at each frequency in freq_array
            noise_signal = generate_sweep_sequence(
                freq_array=freq_array, fs=int(fs),
                duration_s=self.params.get("tone_duration", 0.7),  # per-tone duration (not total)
                gap_s=float(gap_s), tone_amplitude=float(tone_amplitude))

        max_abs = float(np.max(np.abs(noise_signal))) or 1e-30

        capture_queue.put((self.MSG_STATUS, "Streaming signal..."))

        # Progressive analysis buffers
        captured_total = []
        capture_start = [0.0]
        last_report_s = [0.0]
        report_interval_s = max(self.params.get("report_interval_s", 2), 1)

        # Capture buffer for full signal (for post-capture analysis)
        full_capture = np.zeros(int(fs * self.params.get("duration_s", 60)), dtype="float32")

        def _in_cb(indata, frame_count, time_flag, status):
            idx = int(capture_start[0])
            if idx >= len(full_capture):
                return
            n = min(frame_count, len(full_capture) - idx)
            if n > 0 and not self._stop_event.is_set():
                full_capture[idx : idx + n] = indata.flatten()[:n].astype(np.float32)
            capture_start[0] += float(n)


        def _out_cb(outdata, frame_count, time_flag, status):
            if self._stop_event.is_set():
                outdata[:] = 0
                return (outdata, "done")
            start = out_offset[0]
            end = min(start + frame_count, len(noise_signal))
            n = min(frame_count, end - start)
            buf = np.zeros(n, dtype=np.float64)
            buf[:n] = noise_signal[start:end]
            if outdata.ndim == 1:
                outdata[:n] = buf
            else:
                outdata[:n, 0] = buf
            out_offset[0] = end
            return (outdata, "continue")

        out_offset = [0]

        in_stream = sd.InputStream(device=int(recv_device), samplerate=int(fs), channels=1,
                                   callback=_in_cb, blocksize=512, latency="low")
        out_stream = sd.OutputStream(device=int(send_device), samplerate=int(fs), channels=1,
                                     callback=_out_cb, blocksize=512, latency="low")
        in_stream.start()
        out_stream.start()

        duration_s = self.params.get("duration_s", 30)
        expected_samples = int(fs * duration_s)
        total_time = time.time()

        while (not self._stop_event.is_set() and
               capture_start[0] < expected_samples and
               time.time() - total_time < duration_s * 1.5):
            sd.sleep(50)

            # Periodic progress pulse (all modes) + live analysis (noise only)
            elapsed = time.time() - total_time
            prog = min(capture_start[0] / expected_samples * 100, 100)
            if self.signal_type == "noise" and elapsed - last_report_s[0] >= report_interval_s:
                last_report_s[0] = elapsed
                n_to_analyze = int(capture_start[0])
                if n_to_analyze > fs:  # need at least 1 second of data
                    seg = full_capture[:n_to_analyze].astype(np.float64)
                    # Progressive FFT on log-spaced frequency bins
                    freqs_out, amp_db, rms_arr = analyze_noise_response(
                        captured_signal=seg, freq_array=list(freq_array) if len(freq_array) > 0 else [], fs=int(fs))

                    # Smooth the response for chart display
                    smoothed_db = smooth_moving_average(amp_db, window_size=_cfg_module.smoothing_neighbors)

                    capture_queue.put((self.MSG_CHART, {
                        "freqs": freqs_out,
                        "amp_db": amp_db,
                        "smoothed_db": smoothed_db,
                        "rms": rms_arr,
                        "elapsed_s": elapsed,
                        "progress": prog,
                        "method": method,
                    }))

                    # Compute metrics on the segment
                    freqs_for_metrics = list(freq_array) if len(freq_array) > 0 else list(freqs_out)
                    metrics = self._compute_metrics(seg, np.array(freqs_for_metrics), fs, method)
                    if metrics is not None:
                        capture_queue.put((self.MSG_METRICS, metrics))

            # Always emit a progress pulse so the status label stays updated
            if not self._stop_event.is_set():
                capture_queue.put((self.MSG_STATUS, "Streaming %.1f%% (%.1fs)" % (prog, elapsed)))

        # Signal end
        elapsed = time.time() - total_time

        out_stream.stop()
        in_stream.stop()
        out_stream.close()
        in_stream.close()

        captured_len = int(capture_start[0])
        final_seg = full_capture[:captured_len].astype(np.float64) if captured_len > 0 else np.array([])

        # Final analysis — sweep uses per-tone FFT; noise uses broadband analysis
        if len(final_seg) > fs * 2:  # need enough data
            if self.signal_type == "sweep":
                # Per-tone extraction: splits capture into segments, FFTs each
                measured_db, rms_arr = extract_tone_measurements(
                    final_seg, freq_array,
                    tone_duration_s=self.params.get("tone_duration", 0.7),
                    gap_s=float(gap_s),
                    fs=int(fs)
                )
                valid = ~np.isnan(measured_db)
                if np.any(valid):
                    capture_queue.put((self.MSG_CHART, {
                        "freqs": freq_array[valid],
                        "amp_db": measured_db,
                        "smoothed_db": smooth_moving_average(measured_db[valid], window_size=_cfg_module.smoothing_neighbors),
                        "rms": rms_arr[valid],
                        "elapsed_s": elapsed,
                        "progress": 100.0,
                        "method": None,
                    }))
                    # Sweep metrics (no noise spectral shape)
                    final_metrics = self._compute_final_metrics(
                        final_seg, np.array(list(freq_array)), fs, None)
                    capture_queue.put((self.MSG_METRICS, final_metrics))
            else:
                freqs_out, amp_db, rms_arr = analyze_noise_response(
                    captured_signal=final_seg, freq_array=list(freq_array) if len(freq_array) > 0 else [], fs=int(fs))

                smoothed_db_final = smooth_moving_average(amp_db, window_size=_cfg_module.smoothing_neighbors)

                capture_queue.put((self.MSG_CHART, {
                    "freqs": freqs_out,
                    "amp_db": amp_db,
                    "smoothed_db": smoothed_db_final,
                    "rms": rms_arr,
                    "elapsed_s": elapsed,
                    "progress": 100.0,
                    "method": method,
                }))

                # Final metrics + correction analysis (noise)
                final_freqs = list(freq_array) if len(freq_array) > 0 else list(freqs_out)
                final_metrics = self._compute_final_metrics(
                    final_seg, np.array(final_freqs), fs, method)
                capture_queue.put((self.MSG_METRICS, final_metrics))

        capture_queue.put((self.MSG_DONE, None))

    def _compute_metrics(self, seg, freq_array, fs, noise_method):
        """Compute live metrics for a segment of captured data."""
        if len(seg) < fs * 2:
            return None

        valid_freqs, amp_db, rms = analyze_noise_response(
            captured_signal=seg, freq_array=freq_array, fs=fs)

        n_valid = np.sum(~np.isnan(amp_db))
        if n_valid < 3:
            return None

        db_valid = amp_db[~np.isnan(amp_db)]

        metrics = {
            "valid_bins": int(n_valid),
            "mean_dBFS": round(float(np.mean(db_valid)), 2),
            "std_dB": round(float(np.std(db_valid)), 3),
            "min_dBFS": round(float(np.min(db_valid)), 2),
            "max_dBFS": round(float(np.max(db_valid)), 2),
        }

        # Octave band stats
        octave_stats = compute_octave_band_stats(seg, freq_array if len(freq_array) > 0 else np.array([40]), fs)
        metrics["octave_bands"] = octave_stats

        return metrics

    def _compute_final_metrics(self, final_seg, freq_array, fs, noise_method):
        """Compute full set of final metrics including THD and harmonics."""
        metrics = {}

        # Frequency response stats (on correction-applied data if enabled)
        valid_freqs, amp_db, rms = analyze_noise_response(
            captured_signal=final_seg, freq_array=freq_array, fs=fs)
        db_valid = amp_db[~np.isnan(amp_db)]
        metrics["freq_response"] = {
            "valid_bins": int(np.sum(~np.isnan(amp_db))),
            "mean_dBFS": round(float(np.mean(db_valid)), 2),
            "std_dB": round(float(np.std(db_valid)), 3),
            "min_dBFS": round(float(np.min(db_valid)), 2),
            "max_dBFS": round(float(np.max(db_valid)), 2),
        }

        # THD analysis
        thd = compute_thd(final_seg, fs)
        metrics["thd"] = thd

        # Harmonics
        harmonics = compute_harmonics_for_overdrive(final_seg, fs)
        metrics["harmonics"] = harmonics

        # Odd/even ratio
        oe = compute_odd_even_ratio(final_seg, fs)
        if oe:
            metrics["odd_even_ratio"] = oe

        # Noise spectral shape analysis (for noise methods)
        if noise_method in ("pink", "brown", "white") and len(freq_array) > 0:
            shape_result = compare_noise_spectral_shape(final_seg, noise_method, freq_array=freq_array, fs=fs)
            metrics["noise_shape"] = shape_result

        # Octave band stats (final)
        if self.signal_type == "sweep":
            # For sweep: use per-tone dBFS values for meaningful octave bands
            measured_db, rms_arr = extract_tone_measurements(
                final_seg, freq_array, tone_duration_s=self.params.get("tone_duration", 0.7),
                gap_s=float(self.params.get("gap_s", 0.2)), fs=int(fs))
            valid_tones = ~np.isnan(measured_db)

            # RMS / signal strength for sweep (useful metric not available from dBFS alone)
            if len(final_seg) > 0:
                metrics["rms_signal"] = round(float(np.sqrt(np.mean(final_seg.astype(float) ** 2))), 6)
                metrics["peak_dBFS"] = round(float(20 * log_f(max(np.max(np.abs(final_seg)), 1e-30))), 2)

            octaves = [(20, 100, "Sub-bass"), (100, 300, "Bass"), (300, 800, "Low-mid"),
                       (800, 2000, "Mid"), (2000, 5000, "Upper-mid"), (5000, 10000, "Presence"),
                       (10000, 20000, "Brilliance")]

            octave_bands = {}
            for lo, hi, name in octaves:
                tone_mask = valid_tones & (freq_array >= lo) & (freq_array < hi)
                if np.sum(tone_mask) > 0:
                    vals = measured_db[tone_mask]
                    octave_bands[name] = {
                        "kind": "sweep",
                        "mean_dB": round(float(np.mean(vals)), 2),
                        "std_dB": round(float(np.std(vals)), 2),
                        "tones": int(np.sum(tone_mask)),
                    }
            metrics["octave_bands"] = octave_bands
        else:
            # Noise mode: use full FFT-based analysis
            octave_stats = compute_octave_band_stats(
                final_seg, freq_array if len(freq_array) > 0 else np.array([40]), fs)
            # Tag noise-mode bands for renderer
            for n in octave_stats:
                octave_stats[n]["kind"] = "noise"
            metrics["octave_bands"] = octave_stats

        return metrics

    def run(self):
        self._stream_and_analyze(self.params.get("result_queue"))


# ==========================================================================
#  GUI Application
# ==========================================================================

class AmpAnalyzerApp:
    """Main GUI application window."""

    def __init__(self, root):
        self.root = root
        self.root.title("PyAmpScope Amplifier Analyzer")
        self.root.geometry("1280x700")
        self.root.minsize(960, 540)

        # State
        self.worker = None
        self.is_running = False
        self.current_freqs = np.array([])
        self.current_amp_db = np.array([])
        self.result_queue = queue.Queue()

        # Load config defaults
        self._send_device = config.send_device
        self._recv_device = config.recv_device
        self._send_ch = config.send_ch
        self._recv_ch = config.recv_ch
        self._send_gain = config.send_gain
        self._recv_gain = config.recv_gain
        self._noise_method = "white"  # default
        self._signal_type = "noise"  # matches UI default (White Noise)
        self._num_freqs = int(config.num_freqs_default)
        self._tone_duration = float(config.tone_duration)
        self._tone_gap = float(config.tone_gap)
        self._noise_duration = config.noise_calibration_time
        self._min_freq = int(config.freq_min)
        self._max_freq = int(config.freq_max)
        self._apply_send_correction = True
        self._apply_recv_correction = False
        self._recv_path = config.recv_path

        # Correction data loaded lazily
        self._send_correction_lin = None
        self._recv_correction_lin = None
        self._corr_loaded = {"send": False, "recv": False}

        # Results for save
        self._last_metrics = {}
        self._last_chart_data = None

        self._build_ui()
        self._refresh_device_display()

    def _build_ui(self):
        """Build the entire GUI layout."""
        # Style configuration
        style = ttk.Style()
        style.configure("Start.TButton", font=("Segoe UI", 14, "bold"), foreground="green")
        style.configure("Stop.TButton", font=("Segoe UI", 12, "bold"), foreground="red")
        style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))

        # ── Main container (3 columns: left controls, center chart, right metrics) ──
        main_frame = ttk.Frame(self.root, padding=6)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(main_frame)
        self._build_center_panel(main_frame)
        self._build_right_panel(main_frame)

    def _build_left_panel(self, parent):
        """Left panel: controls."""
        left = ttk.LabelFrame(parent, text="  CONTROLS  ", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.configure(width=280)
        left.pack_propagate(False)

        # Signal type selector
        ttk.Label(left, text="Signal Type:", style="Header.TLabel").pack(anchor="w", pady=(4, 0))
        self._var_signal_type = tk.StringVar(value="White Noise")  # white noise is most neutral for flatness analysis
        sig_cb = ttk.Combobox(left, textvariable=self._var_signal_type,
                              values=["Sweep", "Pink Noise", "White Noise", "Brown Noise"],
                              state="readonly", width=25)
        sig_cb.pack(fill=tk.X, pady=2)
        sig_cb.bind("<<ComboboxSelected>>", self._on_signal_type_change)

        # Frequency bins (sweep only)
        ttk.Label(left, text="Frequency Bins:").pack(anchor="w", pady=(4, 0))
        self._var_bins = tk.IntVar(value=self._num_freqs)
        self._spin_bins = ttk.Spinbox(left, from_=16, to=1024, textvariable=self._var_bins, width=12)
        self._spin_bins.pack(fill=tk.X, pady=2)

        # Min / Max frequency
        freq_frame = ttk.Frame(left)
        freq_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(freq_frame, text="Min F:").pack(side=tk.LEFT)
        self._var_min_freq = tk.IntVar(value=self._min_freq)
        ttk.Spinbox(freq_frame, from_=20, to=10000, textvariable=self._var_min_freq, width=8).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(freq_frame, text="Max F:").pack(side=tk.LEFT)
        self._var_max_freq = tk.IntVar(value=self._max_freq)
        ttk.Spinbox(freq_frame, from_=10000, to=24000, textvariable=self._var_max_freq, width=8).pack(side=tk.LEFT, padx=(2, 6))

        # Noise Duration (s)
        self._noise_dur_frame = ttk.Frame(left)
        ttk.Label(self._noise_dur_frame, text="Noise Duration (s):").pack(anchor="w", pady=(4, 0))
        self._var_duration = tk.DoubleVar(value=self._noise_duration)
        self._spin_duration = ttk.Spinbox(self._noise_dur_frame, from_=1.0, to=120.0, increment=0.5,
                                           textvariable=self._var_duration, width=12)
        self._spin_duration.pack(fill=tk.X, pady=2)

        # Tone duration (sweep only)
        self._tone_dur_frame = ttk.Frame(left)
        ttk.Label(self._tone_dur_frame, text="Tone Duration:").pack(anchor="w", pady=(4, 0))
        self._var_tone_dur = tk.DoubleVar(value=self._tone_duration)
        ttk.Spinbox(self._tone_dur_frame, from_=0.1, to=2.0, increment=0.1,
                    textvariable=self._var_tone_dur, width=12).pack(fill=tk.X, pady=2)

        # Inter-tone gap (sweep only)
        self._gap_frame = ttk.Frame(left)
        ttk.Label(self._gap_frame, text="Gap between tones:").pack(anchor="w", pady=(4, 0))
        self._var_gap = tk.DoubleVar(value=self._tone_gap)
        ttk.Spinbox(self._gap_frame, from_=0.05, to=1.0, increment=0.05,
                    textvariable=self._var_gap, width=12).pack(fill=tk.X, pady=2)

        self._noise_dur_frame.pack(fill=tk.X)
        self._tone_dur_frame.pack(fill=tk.X)
        self._gap_frame.pack(fill=tk.X)

        # Gains
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="GAINS", style="Header.TLabel").pack(anchor="w")

        ttk.Label(left, text="Send Gain (%):").pack(anchor="w", pady=(4, 0))
        self._var_send_gain = tk.IntVar(value=self._send_gain)
        #self._lbl_send_gain = ttk.Label(left, text="%d" % self._send_gain + "%", font=("Consolas", 9))
        #self._lbl_send_gain.pack(fill=tk.X, pady=2)
        def _on_send_gain(*args):
            v = self._var_send_gain.get()
            if v > 100:
                self._var_send_gain.set(100)
            elif v < 0:
                self._var_send_gain.set(0)
            #self._lbl_send_gain.config(text="%d%%" % self._var_send_gain.get())
        self._var_send_gain.trace_add("write", _on_send_gain)
        ttk.Spinbox(left, from_=0, to=100, increment=1,
                    textvariable=self._var_send_gain, width=6).pack(fill=tk.X, pady=(0, 4))

        ttk.Label(left, text="Recv Gain (%):").pack(anchor="w", pady=(4, 0))
        self._var_recv_gain = tk.IntVar(value=self._recv_gain)
        #self._lbl_recv_gain = ttk.Label(left, text="%d" % self._recv_gain + "%", font=("Consolas", 9))
        #self._lbl_recv_gain.pack(fill=tk.X, pady=2)
        def _on_recv_gain(*args):
            v = self._var_recv_gain.get()
            if v > 100:
                self._var_recv_gain.set(100)
            elif v < 0:
                self._var_recv_gain.set(0)
            #self._lbl_recv_gain.config(text="%d%%" % self._var_recv_gain.get())
        self._var_recv_gain.trace_add("write", _on_recv_gain)
        ttk.Spinbox(left, from_=0, to=100, increment=1,
                    textvariable=self._var_recv_gain, width=6).pack(fill=tk.X, pady=(0, 4))

        # Corrections
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="CORRECTIONS", style="Header.TLabel").pack(anchor="w")

        self._chk_send_corr = tk.BooleanVar(value=False)  # no correction applied by default
        ttk.Checkbutton(left, text="Apply send correction", variable=self._chk_send_corr).pack(anchor="w", pady=1)

        self._chk_recv_corr = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Apply receive correction", variable=self._chk_recv_corr).pack(anchor="w", pady=1)

        self._corr_status_lbl = ttk.Label(left, text="", foreground="gray", font=("Segoe UI", 8))
        self._corr_status_lbl.pack(anchor="w", pady=2)

        # Path
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="PATH", style="Header.TLabel").pack(anchor="w")

        ttk.Label(left, text="Recv Path:").pack(anchor="w", pady=(4, 0))
        self._var_recv_path = tk.StringVar(value=self._recv_path)
        ttk.Combobox(left, textvariable=self._var_recv_path, values=["dir", "iso"], state="readonly", width=25).pack(fill=tk.X, pady=2)

        # Device selection — inline Combobox dropdowns (replaces popup dialog)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text="DEVICE", style="Header.TLabel").pack(anchor="w")

        self._cb_send_device = tk.StringVar()
        self._combo_send_dev = ttk.Combobox(
            left, textvariable=self._cb_send_device, state="readonly", width=30)
        self._combo_send_dev.pack(fill=tk.X, pady=(4, 2))
        self._populate_device_combo(self._combo_send_dev)

        self._cb_recv_device = tk.StringVar()
        self._combo_recv_dev = ttk.Combobox(
            left, textvariable=self._cb_recv_device, state="readonly", width=30)
        self._combo_recv_dev.pack(fill=tk.X, pady=(4, 2))
        self._populate_device_combo(self._combo_recv_dev)

        # START / STOP buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        self._btn_start = ttk.Button(btn_frame, text="START", style="Start.TButton", command=self._start_capture)
        self._btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self._btn_stop = ttk.Button(btn_frame, text="STOP", style="Stop.TButton", command=self._stop_capture)
        self._btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Status / progress label
        self._status_lbl = ttk.Label(left, text="Ready.", foreground="gray")
        self._status_lbl.pack(anchor="w", pady=(6, 0))

    def _build_center_panel(self, parent):
        """Center panel: live chart."""
        center = ttk.LabelFrame(parent, text="  LIVE CHART  ", padding=6)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Chart figure and canvas
        self._fig = Figure(figsize=(7, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("Frequency (Hz)")
        self._ax.set_ylabel("Magnitude (dBFS)")
        self._ax.set_xscale("log")   # log frequency axis
        self._freq_min = int(getattr(_cfg_module, "freq_min", 40))
        self._freq_max = int(getattr(_cfg_module, "freq_max", 20000))
        self._ax.grid(True, which="both", alpha=0.3)

        # Three separate lines: raw (red), smoothed (green solid), expected (green dashed)
        self._raw_line, = self._ax.plot([], [], "r-", linewidth=1.0, alpha=0.4, label="Raw")
        self._smoothed_line, = self._ax.plot([], [], "b-", linewidth=1.2, alpha=0.8, label="Smoothed")
        self._expected_line, = self._ax.plot([], [], "g--", linewidth=1.0, alpha=0.8, visible=False, label="Expected")
        self._ax.legend(loc="upper right", fontsize=8)

        # Embed canvas in Tkinter
        self._canvas = FigureCanvasTkAgg(self._fig, master=center)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Chart save button and overlay toggles
        chart_frame = ttk.Frame(center)
        chart_frame.place(relx=0, rely=1, x=6, y=-32, anchor="sw")
        self._btn_save_chart = ttk.Button(chart_frame, text="Save Chart", state=tk.DISABLED, command=self._save_chart)
        self._btn_save_chart.pack(side=tk.LEFT, padx=(0, 0))

    def _build_right_panel(self, parent):
        """Right panel: metrics display."""
        right = ttk.LabelFrame(parent, text="  METRICS  ", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.configure(width=300)
        right.pack_propagate(False)

        # Frequency Response Summary
        freq_grp = ttk.Frame(right)
        freq_grp.pack(fill=tk.X, pady=(2, 4), anchor="w")
        self._metrics = {}  # populated during capture
        self._lbl_metrics_frame = ttk.Label(freq_grp, font=("Consolas", 9))
        self._lbl_metrics_frame.pack(anchor="nw")

        # Octave Band Analysis
        oct_grp = ttk.Frame(right)
        oct_grp.pack(fill=tk.X, pady=(6, 2), anchor="w")
        ttk.Label(oct_grp, text="Octave Bands:", font=("Consolas", 9, "bold")).pack(anchor="nw")

        self._octave_labels = {}
        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            lbl = ttk.Label(oct_grp, text="%-12s  n/a" % (name + ":"), font=("Consolas", 8))
            lbl.pack(anchor="nw")
            self._octave_labels[name] = lbl

        # RMS / Signal strength
        rms_frame = ttk.Frame(right)
        rms_frame.pack(fill=tk.X, pady=(4, 2), anchor="w")
        ttk.Label(rms_frame, text="Signal Level:", font=("Consolas", 9, "bold")).pack(anchor="nw")
        self._rms_label = ttk.Label(rms_frame, text="  RMS: -- / Peak dBFS: --", font=("Consolas", 8))
        self._rms_label.pack(anchor="nw")

        # Harmonics / THD
        harm_grp = ttk.Frame(right)
        harm_grp.pack(fill=tk.X, pady=(6, 2), anchor="w")
        ttk.Label(harm_grp, text="THD / Harmonics:", font=("Consolas", 9, "bold")).pack(anchor="nw")

        self._thd_label = ttk.Label(harm_grp, text="  THD: -- %", font=("Consolas", 8))
        self._thd_label.pack(anchor="nw")
        self._fund_label = ttk.Label(harm_grp, text="  Fundamental: -- Hz", font=("Consolas", 8))
        self._fund_label.pack(anchor="nw")
        self._harmonics_frame = ttk.Frame(harm_grp)
        self._harmonics_frame.pack(fill=tk.X, anchor="w")

        # Noise shape analysis (for noise methods)
        self._noise_shape_frame = ttk.Frame(right)
        self._noise_shape_frame.pack(fill=tk.X, pady=(6, 2), anchor="w")
        self._lbl_noise_shape_quality = ttk.Label(self._noise_shape_frame, text="", font=("Consolas", 8))
        self._lbl_noise_shape_quality.pack(anchor="nw")

    def _get_noise_method(self):
        """Derive noise method from the Signal Type selection."""
        val = self._var_signal_type.get()
        if "Noise" in val:
            return val.replace(" Noise", "").lower()
        return None

    def _on_signal_type_change(self, event=None):
        """Toggle visibility of parameters based on signal type selection."""
        val = self._var_signal_type.get()
        is_noise = "Noise" in val

        if is_noise:
            self._signal_type = "noise"
        else:
            self._signal_type = "sweep"

    def _change_devices(self):
        """Open device picker dialog."""
        picker = DevicePickerDialog(
            self.root,
            current_send=self._send_device,
            current_recv=self._recv_device,
        )
        self.root.wait_window(picker.top)
        if picker.send_idx is not None and picker.recv_idx is not None:
            self._send_device = picker.send_idx
            self._recv_device = picker.recv_idx
            # Update inline comboboxes to reflect the new selection
            devices = _get_audio_devices()
            for combo, idx in [(self._combo_send_dev, picker.send_idx), (self._combo_recv_dev, picker.recv_idx)]:
                for opt in combo["values"]:
                    if str(opt).startswith("[%d]" % idx):
                        combo.set(opt)
                        break

    def _populate_device_combo(self, combobox_widget):
        """Refresh a device Combobox with live sounddevice query."""
        devices = _get_audio_devices()
        values = []
        for i, name, status in devices:
            values.append("[%d] %s (%s)" % (i, name[:50], status))
        combobox_widget["values"] = values
        if not values:
            combobox_widget.set("[none]")

    def _update_device_combo_selection(self, combobox_widget, device_idx):
        """Set Combobox to the option matching a given device index."""
        for opt in combobox_widget["values"]:
            if str(opt).startswith("[%d]" % device_idx):
                combobox_widget.set(opt)
                break

    def _refresh_device_display(self):
        """Update device display labels (now uses comboboxes)."""
        # Refresh both comboboxes with live device list and update to current selection
        self._populate_device_combo(self._combo_send_dev)
        self._update_device_combo_selection(self._combo_send_dev, self._send_device)
        self._populate_device_combo(self._combo_recv_dev)
        self._update_device_combo_selection(self._combo_recv_dev, self._recv_device)

    def _load_corrections(self):
        """Load send/recv corrections from NPZ files if they exist."""
        data_dir = _REPO_ROOT / config.data_dir
        self._send_correction_lin = None
        self._corr_loaded["send"] = False
        self._corr_loaded["recv"] = False

        # Load send correction
        send_corr_files = list(data_dir.glob("cal_send*correction*.npz")) + list(data_dir.glob("di_send_profile.npz"))
        if send_corr_files:
            try:
                data = np.load(str(send_corr_files[0]))
                self._send_correction_lin = 10 ** (data.get("response_H", np.zeros(1)) / 20.0)
                self._corr_loaded["send"] = True
            except Exception:
                pass

        # Load receive correction
        recv_path_dir = data_dir / "cal_recv"
        for path_suffix in ["dir", "iso"]:
            if not path_suffix:
                continue
            corr_files = list(data_dir.glob("di_receive_%s*correction*.npz" % path_suffix))
            if corr_files:
                try:
                    data = np.load(str(corr_files[0]))
                    self._recv_correction_lin = 10 ** (data.get("response_H", np.zeros(1)) / 20.0)
                    self._corr_loaded["recv"] = True
                except Exception:
                    break

        status_parts = []
        if self._corr_loaded["send"]:
            status_parts.append("Send OK")
        if self._corr_loaded["recv"]:
            status_parts.append("Recv OK")
        if not status_parts:
            status_parts.append("No corrections loaded")
        self._corr_status_lbl.config(text="  ".join(status_parts))

    def _start_capture(self):
        """Start streaming + capture in a background worker thread."""
        if self.is_running:
            return

        # Gather parameters from UI controls
        self._apply_send_correction = self._chk_send_corr.get()
        self._apply_recv_correction = self._chk_recv_corr.get()
        self._recv_path = self._var_recv_path.get()

        if not self._corr_loaded["send"] or not self._corr_loaded["recv"]:
            self._load_corrections()

        # Determine signal-specific parameters (read from UI vars, not stale instance attrs)
        freq_min = int(self._var_min_freq.get())
        freq_max = int(self._var_max_freq.get())
        n_freqs = self._var_bins.get()
        duration_s = float(self._var_duration.get())
        tone_dur = float(self._var_tone_dur.get())
        gap_s = float(self._var_gap.get())

        if "Noise" in self._var_signal_type.get():
            noise_method = self._get_noise_method()
        else:
            noise_method = None

        ui_signal = self._var_signal_type.get()
        is_sweep = ui_signal == "Sweep"

        # Generate frequency array for sweep or freq analysis
        if is_sweep:
            # Geometric spacing matches log-scale x-axis so points appear evenly spaced visually
            freq_array = np.geomspace(freq_min, freq_max, num=n_freqs)
            duration_s = float(tone_dur * n_freqs + gap_s * (n_freqs - 1)) + 2.0  # add margin
        else:
            freq_array = np.geomspace(freq_min, freq_max, num=60)  # fixed bin count for noise

        # Build worker params
        self._last_metrics = {}
        self._last_chart_data = None
        duration_s = max(duration_s, float(self._var_duration.get()))

        signal_type_for_worker = "sweep" if is_sweep else "noise"
        worker_params = {
            "fs": int(config.fs),
            "send_device": int(self._send_device),
            "recv_device": int(self._recv_device),
            "duration_s": duration_s,
            "freq_array": freq_array,
            "tone_duration": tone_dur,
            "gap_s": gap_s,
            "n_samples": int(config.fs * duration_s),
            "method": noise_method,
            "send_gain": self._var_send_gain.get(),
            "tone_amplitude": float(config.tone_amplitude),
            "apply_send_correction": self._apply_send_correction,
            "recv_path": self._recv_path,
            "result_queue": self.result_queue,
        }

        # Start worker
        self.worker = AnalyzerWorker(signal_type_for_worker, **worker_params)
        self.is_running = True
        self.worker.start()
        self._btn_start.config(state=tk.DISABLED)
        self._status_lbl.config(text="Running...", foreground="blue")
        self.root.after(100, self._check_worker_status)

    def _stop_capture(self):
        """Stop the running worker."""
        if not self.is_running or not self.worker:
            return
        self.worker.stop()
        self._btn_stop.config(state=tk.DISABLED)
        # Clear chart state so partial payloads after stop are skipped
        self.current_freqs = np.array([])
        self.current_amp_db = np.array([])
        self._status_lbl.config(text="Stopped", foreground="red")

    def _check_worker_status(self):
        """Periodic check of worker thread state and result queue."""
        if not self.is_running:
            return

        # Process any pending results from the queue
        while True:
            try:
                msg_type, payload = self.result_queue.get_nowait()
            except queue.Empty:
                break

            if msg_type == AnalyzerWorker.MSG_STATUS:
                self._status_lbl.config(text=str(payload), foreground="blue")
            elif msg_type == AnalyzerWorker.MSG_CHART:
                # Guard: skip partial chart data that doesn't match the sweep's full freq count
                amp_db = payload.get("amp_db", np.array([]))
                freqs = payload.get("freqs", np.array([]))
                if len(amp_db) < 2 or len(freqs) < 2:
                    continue
                # If this is a sweep, the full chart should have all freq_array bins.
                # Partial data (fewer bins than expected) is ignored.
                if self.worker and hasattr(self.worker, 'params'):
                    fa = self.worker.params.get("freq_array", None)
                    if fa is not None and len(freqs) < len(fa):
                        continue
                self._update_chart(payload)
                # Append progress percentage to status label
                prog = payload.get("progress")
                elapsed_s = payload.get("elapsed_s", 0)
                if prog is not None and prog < 99.5:
                    self._status_lbl.config(text="Streaming %.1f%% (%.1fs)" % (prog, elapsed_s))
            elif msg_type == AnalyzerWorker.MSG_METRICS:
                self._last_metrics = payload
                self._update_metrics_panel()
            elif msg_type == AnalyzerWorker.MSG_DONE:
                self.is_running = False
                self.worker = None
                self._btn_start.config(state=tk.NORMAL)
                self._btn_stop.config(state=tk.NORMAL)
                self._btn_save_chart.config(state=tk.NORMAL)
                # Only show "Complete!" if STOP wasn't pressed (which already set status)
                cur_status = self._status_lbl.cget("text")
                if cur_status not in ("Stopped",):
                    self._status_lbl.config(text="Complete!", foreground="green")

        # Continue polling if still running
        if self.is_running:
            self.root.after(200, self._check_worker_status)
        else:
            # Worker finished; try one more time to get final results
            while True:
                try:
                    msg_type, payload = self.result_queue.get_nowait()
                    if msg_type == AnalyzerWorker.MSG_CHART:
                        self._update_chart(payload)
                    elif msg_type == AnalyzerWorker.MSG_METRICS:
                        self._last_metrics = payload
                        self._update_metrics_panel()
                except queue.Empty:
                    break

    def _update_chart(self, data):
        """Update the matplotlib chart with raw, smoothed, and expected frequency response data."""
        freqs = data.get("freqs", np.array([]))
        amp_db = data.get("amp_db", np.array([]))
        smoothed_db = data.get("smoothed_db", np.array([]))
        method = data.get("method")

        if len(freqs) > 0 and len(amp_db) > 0:
            self._last_chart_data = {"freqs": freqs, "amp_db": amp_db}
            self.current_freqs = freqs
            self.current_amp_db = amp_db
            self._btn_save_chart.config(state=tk.NORMAL)

        if len(freqs) != len(self.current_freqs):
            # Update cached state before using it
            self.current_freqs = freqs
            self.current_amp_db = amp_db

        if len(self.current_freqs) > 0:
            raw_mask = ~np.isnan(self.current_amp_db)
            self._raw_line.set_data(
                self.current_freqs[raw_mask],
                self.current_amp_db[raw_mask],
            )
            if len(smoothed_db) > 0 and not np.all(np.isnan(smoothed_db)):
                smooth_mask = ~np.isnan(smoothed_db)
                self._smoothed_line.set_data(
                    self.current_freqs[smooth_mask],
                    smoothed_db[smooth_mask],
                )
            else:
                self._smoothed_line.set_data(
                    self.current_freqs[raw_mask],
                    self.current_amp_db[raw_mask],
                )

            # Build and update expected profile
            if method in ("white", "pink", "brown"):
                valid_amps = self.current_amp_db[~np.isnan(self.current_amp_db)]
                if len(valid_amps) > 0:
                    mean_level = float(np.mean(valid_amps))
                    expected_db = self._build_expected_profile(
                        self.current_freqs, method, mean_level,
                    )
                    self._expected_line.set_data(
                        self.current_freqs, expected_db,
                    )
                    self._expected_line.set_visible(True)

            # Legend — only include visible lines
            handles = []
            labels = []
            for line_obj, name in (
                (self._raw_line, "Raw"),
                (self._smoothed_line, "Smoothed"),
                (self._expected_line, "Expected"),
            ):
                if line_obj.get_visible():
                    handles.append(line_obj)
                    labels.append(name)
            self._ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)

            # Autoscale Y around signal only (no 0 reference)
            y_min = float(np.nanmin(self.current_amp_db))
            y_max = float(np.nanmax(self.current_amp_db))
            if method in ("white", "pink", "brown"):
                expected_db = self._build_expected_profile(
                    self.current_freqs, method, (y_min + y_max) / 2,
                )
                y_min = min(y_min, float(np.nanmin(expected_db)))
                y_max = max(y_max, float(np.nanmax(expected_db)))
            y_margin = max((y_max - y_min) * 0.15, 3.0)
            self._ax.set_ylim(y_min - y_margin, y_max + y_margin)

            # Force X-axis to configured frequency range
            self._ax.set_xlim(self._freq_min, self._freq_max)
            self._fig.canvas.draw_idle()

    def _build_expected_profile(self, freqs, method, ref_level):
        """Build theoretical expected spectrum shifted to match ``ref_level`` (dBFS).

        White / sweep -> flat horizontal line.
        Pink  -> -3 dB/oct slope centered on the geometric-mean frequency.
        Brown -> -6 dB/oct power slope (~-3 dB/oct amplitude) centered the same way.
        """
        geo_mean = float(np.exp(np.mean(np.log(freqs))))
        if method == "pink":
            return ref_level + 3 * np.log2(geo_mean / freqs)
        elif method == "brown":
            return ref_level + 6 * np.log2(geo_mean / freqs)
        # white / sweep
        return np.full_like(freqs, ref_level, dtype=float)

    def _update_metrics_panel(self):
        """Update the right-side metrics panel with latest computed values."""
        m = self._last_metrics
        if not m:
            return

        # Frequency Response Summary
        fr = m.get("freq_response", {})
        lines = [
            "Freq Response:",
            "  Valid bins : %d" % fr.get("valid_bins", 0),
            "  Mean amp   : %+.1f dBFS" % fr.get("mean_dBFS", 0),
            "  Std dev    : %.2f dB" % fr.get("std_dB", 0),
            "  Max amp    : %+.1f dBFS" % fr.get("max_dBFS", 0),
            "  Min amp    : %+.1f dBFS" % fr.get("min_dBFS", 0),
        ]

        # Octave bands
        octave_data = m.get("octave_bands", {})
        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            d = octave_data.get(name, {})
            if not d:
                val = "n/a"
            elif d.get("kind") == "sweep":
                # Sweep mode: show mean dBFS and tone count (per-tone FFT results)
                mean_val = d.get("mean_dB", "--")
                std_val = d.get("std_dB", "--")
                tones = d.get("tones", 0)
                val = "%+.1f / %+.1f dB (%d tones)" % (mean_val, std_val, tones)
            else:
                # Noise mode: min/max from FFT-based analysis
                mn = d.get("min_dB", "--")
                mx = d.get("max_dB", "--")
                val = "%+.1f / %+.1f dB (%.1f std)" % (mn, mx, d.get("std_dB", 0))
            self._octave_labels[name].config(text="%-12s  %s" % (name + ":", val))

        # RMS / signal strength (available for sweep mode)
        if "rms_signal" in m:
            rms = m["rms_signal"]
            peak = m.get("peak_dBFS", "--")
            self._rms_label.config(text="  RMS: %+.2f (%+.1f dBFS) Peak: %+.1f dBFS" % (rms, rms, peak))
        else:
            self._rms_label.config(text="  RMS: -- / Peak dBFS: --")

        # THD / Harmonics
        thd = m.get("thd", {})
        thd_val = thd.get("thd_pct")
        if thd_val is not None:
            self._thd_label.config(text="  THD: %.3f %% (%.1f Hz)" % (thd_val, thd.get("fundamental_freq_hz", 0)))
            self._fund_label.config(text="  Fundamental: %.1f Hz" % thd.get("fundamental_freq_hz", 0))
        else:
            self._thd_label.config(text="  THD: --")

        # Noise shape analysis
        if "noise_shape" in m:
            ns = m["noise_shape"]
            shape_std = ns.get("shape_std_pct", ns.get("shape_std_db", None))
            if shape_std is not None:
                quality = "excellent" if shape_std < 5.0 else "good" if shape_std < 10.0 else "fair" if shape_std < 15.0 else "poor"
                self._lbl_noise_shape_quality.config(text="Shape quality: %s (%.1f)" % (quality, shape_std))

    def _save_chart(self):
        """Save the current chart as PNG."""
        if len(self.current_freqs) == 0:
            messagebox.showwarning("No data", "No chart to save.")
            return

        path = filedialog.asksaveasfilename(
            initialdir=str(_REPO_ROOT / config.logs_dir),
            defaultextension=".png",
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
            title="Save Chart",
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")

    def _save_results(self):
        """Save complete results (NPZ + JSON)."""
        if not self._last_chart_data or not self._last_metrics:
            messagebox.showwarning("No data", "No results to save yet.")
            return

        path = filedialog.asksaveasfilename(
            initialdir=str(_REPO_ROOT / config.logs_dir),
            defaultextension=".npz",
            filetypes=[("NPZ files", "*.npz"), ("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Results",
        )
        if not path:
            return

        base = path.rsplit(".", 1)[0]

        # Save NPZ
        chart_data = self._last_chart_data
        np.savez(base + ".npz",
                 freqs=chart_data["freqs"],
                 amp_db=chart_data["amp_db"],
                 metrics=np.array([json.dumps(self._last_metrics)]),  # JSON string of metrics dict
                 params=np.array([json.dumps({
                     "signal_type": self._signal_type,
                     "method": self._get_noise_method() or "white",
                     "send_gain": self._var_send_gain.get(),
                     "recv_gain": self._var_recv_gain.get(),
                 })]),
        )

        # Save JSON
        json_path = base + ".json"
        with open(json_path, "w") as f:
            json.dump({
                "freq_response": self._last_metrics.get("freq_response", {}),
                "thd": self._last_metrics.get("thd", {}),
                "harmonics": self._last_metrics.get("harmonics", []),
                "odd_even_ratio": self._last_metrics.get("odd_even_ratio", {}),
                "octave_bands": self._last_metrics.get("octave_bands", {}),
                "noise_shape": {k: str(v) if k != "freqs" and k != "measured_db" else v for k, v in self._last_metrics.get("noise_shape", {}).items()},
                "params": {
                    "signal_type": self._signal_type,
                    "method": self._get_noise_method() or "white",
                    "send_gain": self._var_send_gain.get(),
                    "recv_gain": self._var_recv_gain.get(),
                    "num_freqs": self._var_bins.get(),
                    "duration_s": float(self._var_duration.get()),
                },
            }, f, indent=2)

        # Save chart PNG
        png_path = base + "_chart.png"
        self._fig.savefig(png_path, dpi=150, bbox_inches="tight")

        messagebox.showinfo("Saved", "Results saved to:\n%s\n%s\n%s" % (base + ".npz", json_path, png_path))


def main():
    """Entry point for the amplifier analyzer GUI."""
    root = tk.Tk()
    root.state('zoomed')  # maximize on start (Windows)

    # Apply dark-ish theme
    style = ttk.Style()
    style.theme_use("clam")

    app = AmpAnalyzerApp(root)

    # Keyboard shortcuts
    def _on_key(event):
        if event.state & 0x4 and event.keysym == "s":  # Ctrl+S
            app._save_results()
            return "break"
        if event.keysym == "F5":  # F5 = Start
            if not app.is_running:
                app._start_capture()
            else:
                app._stop_capture()
        if event.keysym == "Escape":  # ESC = Stop
            if app.is_running:
                app._stop_capture()
        return "break"

    root.bind("<Key>", _on_key)
    root.mainloop()


if __name__ == "__main__":
    main()
