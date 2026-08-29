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
from utils.audio.signal_utils import generate_noise_signal, generate_sweep_sequence
from utils.audio.analysis_utils import (analyze_noise_response, compare_noise_spectral_shape,
    smooth_moving_average, extract_tone_measurements, compute_thd, compute_thd_per_tone_with_freqs,
    compute_harmonics_for_overdrive, compute_odd_even_ratio, compute_octave_band_stats)


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

        # Overall RMS/Peak (all modes — live only for noise; sweep uses _compute_final_metrics at end)
        if len(seg) > 0:
            metrics["rms_signal"] = round(float(np.sqrt(np.mean(seg.astype(float) ** 2))), 6)
            metrics["peak_dBFS"] = round(float(20 * log_f(max(np.max(np.abs(seg)), 1e-30))), 2)

        # Per-band RMS (noise): spectral density (RMS/Hz) per band, independent of bin count.
        # FFT zero-out + IFFT extracts the band's signal component; divide bins by sqrt(N_B)
        # to normalize out the number-of-bins bias so white noise shows equal values per band.
        fft_full = np.fft.rfft(seg.astype(np.float64))
        fft_freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        n = len(seg)
        rms_per_band = {}
        for lo, hi, name in [(20, 100, "Sub-bass"), (100, 300, "Bass"), (300, 800, "Low-mid"),
                              (800, 2000, "Mid"), (2000, 5000, "Upper-mid"), (5000, 10000, "Presence"),
                              (10000, 20000, "Brilliance")]:
            band_mask = (fft_freqs >= lo) & (fft_freqs < hi)
            if np.sum(band_mask) > 2:
                fft_band = fft_full.copy()
                fft_band[~band_mask] = 0
                N_B = np.sum(band_mask)
                band_signal = np.real(np.fft.irfft(fft_band / np.sqrt(N_B), n=n))
                band_rms = float(np.sqrt(np.mean(band_signal ** 2)))
                rms_per_band[name] = {
                    "mean": round(band_rms, 6),
                    "max": round(float(np.max(np.abs(band_signal))), 6),
                    "tones": int(N_B),
                }
            metrics["rms_per_band"] = rms_per_band

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

        # RMS / peak signal level (all modes)
        if len(final_seg) > 0:
            metrics["rms_signal"] = round(float(np.sqrt(np.mean(final_seg.astype(float) ** 2))), 6)
            metrics["peak_dBFS"] = round(float(20 * log_f(max(np.max(np.abs(final_seg)), 1e-30))), 2)

        # THD / Harmonics — sweep only (noise has no periodic fundamental)
        if self.signal_type == "sweep":
            metrics["thd"] = compute_thd(final_seg, fs)
            metrics["harmonics"] = compute_harmonics_for_overdrive(final_seg, fs)
            oe = compute_odd_even_ratio(final_seg, fs)
            if oe:
                metrics["odd_even_ratio"] = oe

        # Noise spectral shape analysis (for noise methods)
        if noise_method in ("pink", "brown", "white") and len(freq_array) > 0:
            shape_result = compare_noise_spectral_shape(final_seg, noise_method, freq_array=freq_array, fs=fs)
            metrics["noise_shape"] = shape_result

        # Octave band stats (final)
        oct_groups = [(20, 100, "Sub-bass"), (100, 300, "Bass"), (300, 800, "Low-mid"),
                      (800, 2000, "Mid"), (2000, 5000, "Upper-mid"), (5000, 10000, "Presence"),
                      (10000, 20000, "Brilliance")]

        if self.signal_type == "sweep":
            # For sweep: use per-tone dBFS values for meaningful octave bands
            measured_db, rms_arr = extract_tone_measurements(
                final_seg, freq_array, tone_duration_s=self.params.get("tone_duration", 0.7),
                gap_s=float(self.params.get("gap_s", 0.2)), fs=int(fs))
            valid_tones = ~np.isnan(measured_db)

            octave_bands = {}
            thd_per_band = {}
            all_thd = []  # collect every valid tone's THD for global average

            for lo, hi, name in oct_groups:
                tone_mask = valid_tones & (freq_array >= lo) & (freq_array < hi)
                if np.sum(tone_mask) > 0:
                    vals = measured_db[tone_mask]
                    octave_bands[name] = {
                        "kind": "sweep",
                        "mean_dB": round(float(np.mean(vals)), 2),
                        "std_dB": round(float(np.std(vals)), 2),
                        "tones": int(np.sum(tone_mask)),
                    }

            # Per-tone THD for sweep mode
            tone_dur_s = self.params.get("tone_duration", 0.7)
            gap_s_val = self.params.get("gap_s", 0.2)
            params = {"tone_duration": tone_dur_s, "gap_s": gap_s_val}
            freqs_thd, thd_vals, _ = compute_thd_per_tone_with_freqs(
                final_seg, int(fs), freq_array, params=params)
            valid_thd = ~np.isnan(thd_vals) if hasattr(thd_vals, '__iter__') else thd_vals > 0

            for lo, hi, name in oct_groups:
                band_mask = (freqs_thd >= lo) & (freqs_thd < hi) & valid_thd
                if np.sum(band_mask) > 0:
                    band_thd = thd_vals[band_mask]
                    thd_per_band[name] = {
                        "mean_pct": round(float(np.mean(band_thd)), 3),
                        "std_pct": round(float(np.std(band_thd)), 3),
                        "tones": int(np.sum(band_mask)),
                    }
                    all_thd.extend(band_thd.tolist())

            if len(all_thd) > 0:
                metrics["thd_global"] = round(float(np.mean(all_thd)), 3)
            else:
                metrics["thd_global"] = None

            # Per-band RMS (sweep): rms_arr from extract_tone_measurements matches freq_array order
            rms_per_band = {}
            for lo, hi, name in oct_groups:
                band_mask = (freq_array >= lo) & (freq_array < hi) & valid_tones
                if np.sum(band_mask) > 0:
                    band_rms = rms_arr[band_mask]
                    rms_per_band[name] = {
                        "mean": round(float(np.mean(band_rms)), 6),
                        "max": round(float(np.max(np.abs(band_rms))), 6),
                        "tones": int(np.sum(band_mask)),
                    }

            metrics["thd_per_band"] = thd_per_band
            metrics["octave_bands"] = octave_bands
            metrics["rms_per_band"] = rms_per_band
        else:
            # Noise mode: FFT-based octave band stats
            octave_stats = compute_octave_band_stats(
                final_seg, freq_array if len(freq_array) > 0 else np.array([40]), fs)
            # Tag noise-mode bands for renderer
            for n in octave_stats:
                octave_stats[n]["kind"] = "noise"
            metrics["octave_bands"] = octave_stats

            # Per-band RMS (noise): compute true time-domain signal energy per band,
            # normalized by number of bins so the result is spectral density (RMS/Hz).
            # This ensures that for white noise (flat PSD), all bands show approximately
            # equal values — regardless of how many FFT bins fall in each band.
            fft_full = np.fft.rfft(final_seg.astype(np.float64))
            fft_freqs = np.fft.rfftfreq(len(final_seg), d=1.0 / fs)
            n = len(final_seg)
            rms_per_band = {}
            for lo, hi, name in oct_groups:
                band_mask = (fft_freqs >= lo) & (fft_freqs < hi)
                if np.sum(band_mask) > 2:
                    fft_band = fft_full.copy()
                    fft_band[~band_mask] = 0
                    # Normalize by sqrt(N_B * n): FFT zero-out gives sqrt(sum(E_k / n)),
                    # dividing bins by sqrt(N_B) converts from "total per-band amplitude" to
                    # "spectral density (RMS/Hz)" — independent of bin count.
                    N_B = np.sum(band_mask)
                    band_signal = np.real(np.fft.irfft(fft_band / np.sqrt(N_B), n=n))
                    band_rms = float(np.sqrt(np.mean(band_signal ** 2)))
                    rms_per_band[name] = {
                        "mean": round(band_rms, 6),
                        "max": round(float(np.max(np.abs(band_signal))), 6),
                        "tones": int(N_B),
                    }

            metrics["rms_per_band"] = rms_per_band

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

        # Signal Level (per-band RMS + overall stats)
        self._sl_frame = ttk.Frame(right)
        self._sl_frame.pack(fill=tk.X, pady=(6, 2), anchor="w")
        ttk.Label(self._sl_frame, text="Signal Level:", font=("Consolas", 9, "bold")).pack(anchor="nw")

        # Per-band RMS labels
        self._sl_rms_labels = {}
        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            lbl = ttk.Label(self._sl_frame, text="%-12s  --" % (name + ":"), font=("Consolas", 8))
            lbl.pack(anchor="nw")
            self._sl_rms_labels[name] = lbl

        # Overall stats
        self._sl_rms_val = ttk.Label(self._sl_frame, text="  RMS: --", font=("Consolas", 8))
        self._sl_rms_val.pack(anchor="nw")
        self._sl_peak_rms_val = ttk.Label(self._sl_frame, text="  Peak RMS: --", font=("Consolas", 8))
        self._sl_peak_rms_val.pack(anchor="nw")
        self._sl_peak_dbfs_val = ttk.Label(self._sl_frame, text="  Peak dBFS: --", font=("Consolas", 8))
        self._sl_peak_dbfs_val.pack(anchor="nw")

        # THD / Harmonics (per-band + global average)
        harm_grp = ttk.Frame(right)
        harm_grp.pack(fill=tk.X, pady=(6, 2), anchor="w")
        ttk.Label(harm_grp, text="THD / Harmonics:", font=("Consolas", 9, "bold")).pack(anchor="nw")

        self._thd_labels = {}
        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            lbl = ttk.Label(harm_grp, text="%-12s  --" % (name + ":"), font=("Consolas", 8))
            lbl.pack(anchor="nw")
            self._thd_labels[name] = lbl

        self._thd_global_lbl = ttk.Label(harm_grp, text="  Global avg: --", font=("Consolas", 8))
        self._thd_global_lbl.pack(anchor="nw")

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

        # Frequency Response Summary (kept as-is for now)
        fr = m.get("freq_response", {})
        fr_text = "Freq Response:\n"
        fr_text += "  Valid bins : %d\n" % fr.get("valid_bins", 0)
        fr_text += "  Mean amp   : %+.1f dBFS\n" % fr.get("mean_dBFS", 0)
        fr_text += "  Std dev    : %.2f dB\n" % fr.get("std_dB", 0)
        fr_text += "  Max amp    : %+.1f dBFS\n" % fr.get("max_dBFS", 0)
        fr_text += "  Min amp    : %+.1f dBFS" % fr.get("min_dBFS", 0)
        self._lbl_metrics_frame.config(text=fr_text)

        # Octave bands (dB stats only — separate from RMS / THD)
        octave_data = m.get("octave_bands", {})

        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            d = octave_data.get(name, {})
            if not d:
                self._octave_labels[name].config(text="%-12s  n/a" % (name + ":"))
                continue
            if d.get("kind") == "sweep":
                mean_val = d.get("mean_dB", "--")
                std_val = d.get("std_dB", "--")
                tones = d.get("tones", 0)
                val = "%+.1f / %+.1f dB (%d tones)" % (mean_val, std_val, tones)
            elif d.get("kind") == "noise":
                mn = d.get("min_dB", "--")
                mx = d.get("max_dB", "--")
                val = "%+.1f / %+.1f dB (%.1f std)" % (mn, mx, d.get("std_dB", 0))
            else:
                val = "n/a"
            self._octave_labels[name].config(text="%-12s  %s" % (name + ":", val))

        # ── Signal Level section: per-band RMS + overall stats ──────────────
        rms_per_band = m.get("rms_per_band", {})

        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            rb = rms_per_band.get(name, {})
            if rb:
                mean_v = rb["mean"]
                # Fixed-point with adaptive precision (no scientific notation)
                if abs(mean_v) >= 0.01:
                    mean_str = "%.4f" % mean_v
                elif abs(mean_v) >= 0.001:
                    mean_str = "%.5f" % mean_v
                else:
                    mean_str = "%.6f" % mean_v
                # Strip trailing zeros
                if "." in mean_str:
                    mean_str = mean_str.rstrip("0").rstrip(".")
                dbfs_str = "%+.1f" % (20 * log_f(max(mean_v, 1e-30)))
                val = "%s (%s dBFS)" % (mean_str, dbfs_str)
            else:
                val = "--"
            self._sl_rms_labels[name].config(text="%-12s  %s" % (name + ":", val))

        # Overall stats
        if m.get("rms_signal") is not None:
            rms_val = m["rms_signal"]
            peak_dbfs = m.get("peak_dBFS", "--")
            self._sl_rms_val.config(text="  RMS: %+.4f (%+.1f dBFS)" % (rms_val, 20 * log_f(max(rms_val, 1e-30))))
            self._sl_peak_rms_val.config(text="  Peak RMS: N/A")
            self._sl_peak_dbfs_val.config(text="  Peak dBFS: %s" % str(peak_dbfs))
        else:
            self._sl_rms_val.config(text="  RMS: --")
            self._sl_peak_rms_val.config(text="  Peak RMS: --")
            self._sl_peak_dbfs_val.config(text="  Peak dBFS: --")

        # ── THD / Harmonics section: per-band + global average (sweep only) ──
        thd_per_band = m.get("thd_per_band", {})
        thd_global = m.get("thd_global")

        for name in ["Sub-bass", "Bass", "Low-mid", "Mid", "Upper-mid", "Presence", "Brilliance"]:
            tb = thd_per_band.get(name)
            if tb and tb.get("tones", 0) > 0:
                val = "%.3f %%" % tb["mean_pct"]
            else:
                val = "--"
            self._thd_labels[name].config(text="%-12s  %s" % (name + ":", val))

        if thd_global is not None and math.isfinite(thd_global):
            self._thd_global_lbl.config(text="  Global avg: %.3f %% (%.0f tones)" % (thd_global, sum(tb.get("tones", 0) for tb in thd_per_band.values())))
        else:
            self._thd_global_lbl.config(text="  Global avg: --")

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
