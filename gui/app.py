"""Tkinter application for PyAmpScope."""
from __future__ import annotations

import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import sounddevice as sd
except ImportError:
    print("Warning: unable to import sounddevice")
    sd = None

from config import config as cfg
from utils.audio.calibration import load_receive_correction, load_send_correction
from utils.audio.levels import db20
from utils.audio.signal_utils import geometric_frequencies
from utils.charting_utils import create_gui_response_figure, update_gui_response_figure
from utils.storage import save_result_bundle
from .worker import AnalyzerWorker


_REPO_ROOT = Path(__file__).resolve().parent.parent
_BAND_NAMES = list(cfg.frequency_bands.keys())


def _device_options(direction: str) -> list[tuple[int, str]]:
    """Return sounddevice choices that actually support the requested direction."""
    options = []
    if sd is None:
        return options
    try:
        devices = sd.query_devices()
    except Exception:
        return options
    channel_key = "max_output_channels" if direction == "out" else "max_input_channels"
    for i, dev in enumerate(devices):
        try:
            if int(dev.get(channel_key, 0)) > 0:
                options.append((i, str(dev.get("name", "Unknown"))))
        except Exception:
            continue
    return options


def _parse_device_index(text: str, fallback=None):
    try:
        if text.startswith("["):
            return int(text.split("]", 1)[0][1:])
    except Exception:
        pass
    return fallback


class AmpAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PyAmpScope Amplifier Analyzer")
        self.root.geometry("1320x760")
        self.root.minsize(1050, 620)

        self.result_queue = queue.Queue()
        self.worker: AnalyzerWorker | None = None
        self.is_running = False
        self._last_metrics: dict = {}
        self._last_chart_data: dict | None = None
        self._build_ui()
        self._refresh_devices()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Start.TButton", font=("Segoe UI", 12, "bold"))
        style.configure("Stop.TButton", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill=tk.BOTH, expand=True)
        self._build_controls(main)
        self._build_chart(main)
        self._build_metrics(main)

    def _build_controls(self, parent):
        left = ttk.LabelFrame(parent, text="  CONTROLS  ", padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.configure(width=285)
        left.pack_propagate(False)

        ttk.Label(left, text="Signal Type:", style="Header.TLabel").pack(anchor="w")
        self.var_signal = tk.StringVar(value="Sweep")
        cb = ttk.Combobox(left, textvariable=self.var_signal,
                          values=["Sweep"],#,"White Noise", "Pink Noise", "Brown Noise"],
                          state="readonly")
        cb.pack(fill=tk.X, pady=(1, 3))

        ttk.Label(left, text="Frequency Bins:").pack(anchor="w")
        self.var_bins = tk.IntVar(value=int(cfg.num_freqs_default))
        ttk.Spinbox(left, from_=8, to=2048, textvariable=self.var_bins).pack(fill=tk.X, pady=(1,3))

        ff = ttk.Frame(left)
        ff.pack(fill=tk.X, pady=(1,3))
        ttk.Label(ff, text="Min F:").pack(side=tk.LEFT)
        self.var_min_f = tk.IntVar(value=int(cfg.freq_min))
        ttk.Spinbox(ff, from_=10, to=10000, textvariable=self.var_min_f, width=8).pack(side=tk.LEFT, padx=(2,8))
        ttk.Label(ff, text="Max F:").pack(side=tk.LEFT)
        self.var_max_f = tk.IntVar(value=int(cfg.freq_max))
        ttk.Spinbox(ff, from_=100, to=96000, textvariable=self.var_max_f, width=8).pack(side=tk.LEFT, padx=(2,0))

        # Required order: Noise Duration -> Tone Amplitude -> Tone Duration -> Tone Gap.
        ttk.Label(left, text="Noise Duration (s):").pack(anchor="w")
        self.var_noise_duration = tk.DoubleVar(value=int(cfg.noise_calibration_time))
        ttk.Spinbox(left, from_=1, to=300, increment=1,
                    textvariable=self.var_noise_duration).pack(fill=tk.X, pady=(1,3))

        ttk.Label(left, text="Tone Amplitude (peak FS):").pack(anchor="w")
        self.var_tone_amplitude = tk.DoubleVar(value=float(cfg.tone_amplitude))
        ttk.Spinbox(left, from_=0.001, to=1.0, increment=0.01,
                    textvariable=self.var_tone_amplitude).pack(fill=tk.X, pady=(1,3))

        ttk.Label(left, text="Tone Duration (s):").pack(anchor="w")
        self.var_tone_duration = tk.DoubleVar(value=float(cfg.tone_duration))
        ttk.Spinbox(left, from_=0.05, to=5.0, increment=0.05,
                    textvariable=self.var_tone_duration).pack(fill=tk.X, pady=(1,3))

        ttk.Label(left, text="Tone Gap (s):").pack(anchor="w")
        self.var_tone_gap = tk.DoubleVar(value=float(cfg.tone_gap))
        ttk.Spinbox(left, from_=0.0, to=2.0, increment=0.05,
                    textvariable=self.var_tone_gap).pack(fill=tk.X, pady=(1,3))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left, text="SEND LEVEL", style="Header.TLabel").pack(anchor="w")
        ttk.Label(left, text="Send Gain (%):").pack(anchor="w")
        self.var_send_gain = tk.DoubleVar(value=int(cfg.send_gain))
        ttk.Spinbox(left, from_=0, to=100, increment=1,
                    textvariable=self.var_send_gain).pack(fill=tk.X, pady=(1,3))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left, text="CORRECTIONS", style="Header.TLabel").pack(anchor="w")
        self.var_send_corr = tk.BooleanVar(value=False)
        self.var_recv_corr = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Apply send correction", variable=self.var_send_corr).pack(anchor="w")
        ttk.Checkbutton(left, text="Apply receive correction", variable=self.var_recv_corr).pack(anchor="w")
        self.lbl_corr = ttk.Label(left, text="Corrections off", foreground="gray", font=("Segoe UI",8))
        self.lbl_corr.pack(anchor="w", pady=(1,2))

        ttk.Label(left, text="Path:", style="Header.TLabel").pack(anchor="w")
        self.var_path = tk.StringVar(value=str(cfg.recv_path))
        ttk.Combobox(left, textvariable=self.var_path, values=["dir", "iso"], state="readonly").pack(fill=tk.X, pady=(1,3))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(left, text="AUDIO DEVICES", style="Header.TLabel").pack(anchor="w")
        self.var_send_dev = tk.StringVar()
        self.var_recv_dev = tk.StringVar()
        self.cb_send = ttk.Combobox(left, textvariable=self.var_send_dev, state="readonly")
        self.cb_recv = ttk.Combobox(left, textvariable=self.var_recv_dev, state="readonly")
        self.cb_send.pack(fill=tk.X, pady=(1,2))
        self.cb_recv.pack(fill=tk.X, pady=(1,3))

        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, pady=(6,0))
        self.btn_start = ttk.Button(bf, text="START", style="Start.TButton", command=self._start)
        self.btn_stop = ttk.Button(bf, text="STOP", style="Stop.TButton", command=self._stop, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,3))
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3,0))
        self.lbl_status = ttk.Label(left, text="Ready.", foreground="gray")
        self.lbl_status.pack(anchor="w", pady=(6,0))

    def _build_chart(self, parent):
        center = ttk.LabelFrame(parent, text="  RESPONSE  ", padding=6)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig, self.ax_level, self.ax_response, self.line_level, self.line_level_smooth, self.line_response = \
            create_gui_response_figure(float(cfg.freq_min), float(cfg.freq_max))
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.btn_save = ttk.Button(center, text="Save", command=self._save, state=tk.DISABLED)
        self.btn_save.pack(anchor="w", pady=(4,0))

    def _build_metrics(self, parent):
        right = ttk.LabelFrame(parent, text="  METRICS  ", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6,0))
        right.configure(width=315)
        right.pack_propagate(False)

        self.lbl_summary = ttk.Label(right, text="No measurement yet.", font=("Consolas",9), justify=tk.LEFT)
        self.lbl_summary.pack(anchor="nw", pady=(0,6))

        ttk.Label(right, text="Band received level:", font=("Consolas",9,"bold")).pack(anchor="nw")
        self.band_level_labels = {}
        for name in _BAND_NAMES:
            lbl = ttk.Label(right,text=f"{name+':':12s} --",font=("Consolas",8))
            lbl.pack(anchor="nw")
            self.band_level_labels[name] = lbl

        ttk.Label(right, text="\nBand THD:", font=("Consolas",9,"bold")).pack(anchor="nw")
        self.band_thd_labels={}
        for name in _BAND_NAMES:
            lbl = ttk.Label(right,text=f"{name+':':12s} --",font=("Consolas",8))
            lbl.pack(anchor="nw")
            self.band_thd_labels[name] = lbl

        self.lbl_harmonics = ttk.Label(right,text="\nTHD: --\nEven/Odd: --",font=("Consolas",9),justify=tk.LEFT)
        self.lbl_harmonics.pack(anchor="nw")

    # -------------------------------------------------------------- devices
    def _refresh_devices(self):
        outs = _device_options("out")
        ins = _device_options("in")
        outvals = [f"[{i}] {name[:55]}" for i,name in outs]
        invals = [f"[{i}] {name[:55]}" for i,name in ins]
        self.cb_send["values"] = outvals
        self.cb_recv["values"] = invals
        self._set_device(self.cb_send, cfg.send_device)
        self._set_device(self.cb_recv, cfg.recv_device)
        if not self.cb_send.get() and outvals:
            self.cb_send.current(0)
        if not self.cb_recv.get() and invals:
            self.cb_recv.current(0)

    @staticmethod
    def _set_device(combo, idx):
        if idx is None:
            return
        for value in combo["values"]:
            if str(value).startswith(f"[{idx}]"):
                combo.set(value)
                return

    # --------------------------------------------------------------- capture
    def _validated_params(self):
        if sd is None:
            raise ValueError("The sounddevice package is not installed. Install requirements.txt before using hardware capture.")
        fs = int(cfg.fs)
        fmin = float(self.var_min_f.get())
        fmax = float(self.var_max_f.get())
        if not (0 < fmin < fmax < fs/2):
            raise ValueError(f"Frequency range must satisfy 0 < min < max < Nyquist ({fs/2:.0f} Hz).")

        bins = int(self.var_bins.get())
        tone_amp = float(self.var_tone_amplitude.get())
        send_gain = float(self.var_send_gain.get())
        tone_duration = float(self.var_tone_duration.get())
        gap_s = float(self.var_tone_gap.get())
        noise_duration = float(self.var_noise_duration.get())

        if bins < 2:
            raise ValueError("Frequency Bins must be at least 2.")
        if not (0 < tone_amp <= 1.0):
            raise ValueError("Tone Amplitude must be in (0, 1].")
        if not (0 < send_gain <= 100):
            raise ValueError("Send Gain must be greater than 0 and at most 100%.")
        if tone_duration <= 0:
            raise ValueError("Tone Duration must be greater than 0.")
        if gap_s < 0:
            raise ValueError("Tone Gap cannot be negative.")
        if noise_duration <= 0:
            raise ValueError("Noise Duration must be greater than 0.")

        send_dev = _parse_device_index(self.var_send_dev.get(), cfg.send_device)
        recv_dev = _parse_device_index(self.var_recv_dev.get(), cfg.recv_device)
        if send_dev is None or recv_dev is None:
            raise ValueError("Select both send and receive audio devices.")

        return {
            "fs": fs,
            "freq_array": geometric_frequencies(fmin,fmax,bins),
            "tone_duration": tone_duration,
            "gap_s": gap_s,
            "tone_amplitude": tone_amp,
            "send_gain": send_gain,
            "duration_s": noise_duration,
            "send_device": send_dev,
            "recv_device": recv_dev,
            "send_ch": cfg.send_ch,
            "recv_ch": cfg.recv_ch,
            "noise_peak_headroom": float(cfg.noise_peak_headroom),
            "sweep_peak_headroom": float(cfg.sweep_peak_headroom),
        }

    def _start(self):
        if self.is_running:
            return
        try:
            p=self._validated_params()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        method = self.var_signal.get().replace(" Noise","").lower()
        signal_type = "sweep" if self.var_signal.get()=="Sweep" else "noise"
        p["method"] = "sweep" if signal_type=="sweep" else method
        data_dir = _REPO_ROOT / cfg.data_dir

        send_profile = None
        recv_profile=None
        if self.var_send_corr.get():
            send_profile = load_send_correction(data_dir)
            if send_profile is None:
                messagebox.showerror("Send correction", f"No send correction profile found in {data_dir}.")
                return
        if self.var_recv_corr.get():
            recv_profile = load_receive_correction(data_dir,self.var_path.get(),prefer_corrected_send=self.var_send_corr.get())
            if recv_profile is None:
                messagebox.showerror("Receive correction", f"No receive correction profile found for path '{self.var_path.get()}'.")
                return
        p["send_profile"] = send_profile
        p["recv_profile"] = recv_profile

        status = []
        if send_profile:
            status.append(f"send={send_profile.path.name}")
        if recv_profile:
            status.append(f"recv={recv_profile.path.name}")
        self.lbl_corr.config(text=", ".join(status) if status else "Corrections off")

        self._last_metrics = {}
        self._last_chart_data = None
        self.worker = AnalyzerWorker(signal_type,self.result_queue,**p)
        self.is_running=True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self.lbl_status.config(text="Starting...", foreground="blue")
        self.worker.start()
        self.root.after(100,self._poll)

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.lbl_status.config(text="Stopping...", foreground="red")

    def _poll(self):
        while True:
            try:
                kind,payload=self.result_queue.get_nowait()
            except queue.Empty:
                break
            if kind == AnalyzerWorker.MSG_STATUS:
                self.lbl_status.config(text=str(payload),foreground="blue")
            elif kind == AnalyzerWorker.MSG_CHART:
                self._last_chart_data=payload
                self._update_chart(payload)
            elif kind == AnalyzerWorker.MSG_METRICS:
                self._last_metrics=payload
                self._update_metrics(payload)
            elif kind == AnalyzerWorker.MSG_ERROR:
                messagebox.showerror("Measurement error",str(payload))
                self.lbl_status.config(text="Error",foreground="red")
            elif kind == AnalyzerWorker.MSG_DONE:
                self.is_running=False
                self.worker=None
                self.btn_start.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
                if self._last_metrics:
                    self.lbl_status.config(text="Complete.", foreground="green")
                    self.btn_save.config(state=tk.NORMAL)
                elif self.lbl_status.cget("text")!="Error":
                    self.lbl_status.config(text="Stopped.", foreground="red")
        if self.is_running:
            self.root.after(150,self._poll)

    # --------------------------------------------------------------- display
    def _update_chart(self,data):
        update_gui_response_figure(
            self.ax_level, self.ax_response, self.line_level, self.line_level_smooth,
            self.line_response, data["freqs"], data["received_dbfs"],
            data["smoothed_dbfs"], data["relative_response_db"],
            float(self.var_min_f.get()), float(self.var_max_f.get()),
        )
        self.canvas.draw_idle()

    def _update_metrics(self,m):
        fr=m.get("freq_response",{})
        stim=m.get("stimulus",{})
        recv_rms = m.get("rms_signal")
        recv_peak = m.get("peak")
        peak_dbfs = m.get("peak_dBFS")
        req_rms = stim.get("requested_rms")
        actual_send_rms = stim.get("actual_tone_rms_mean", stim.get("actual_rms"))
        response_std = fr.get("relative_std_db")

        def f6(v):
            return "--" if v is None or not np.isfinite(v) else f"{float(v):.6f}"
        def f2(v):
            return "--" if v is None or not np.isfinite(v) else f"{float(v):+.2f}"
        def f3(v):
            return "--" if v is None or not np.isfinite(v) else f"{float(v):.3f}"

        summary=(
            f"Received RMS : {f6(recv_rms)} ({f2(db20(recv_rms) if recv_rms is not None else None)} dBFS)\n"
            f"Received peak: {f6(recv_peak)} ({f2(peak_dbfs)} dBFS)\n"
            f"Requested RMS: {f6(req_rms)}\n"
            f"Actual send RMS: {f6(actual_send_rms)}\n"
            f"Response std: {f3(response_std)} dB\n"
            f"ADC clipping: {'YES' if m.get('clipped') else 'no'}"
        )
        self.lbl_summary.config(text=summary)

        bands = m.get("frequency_bands",{})
        for name in _BAND_NAMES:
            d = bands.get(name)
            if d and d.get("received_dbfs") is not None:
                self.band_level_labels[name].config(text=f"{name+':':12s} {d['received_dbfs']:+7.2f} dBFS")
            else: self.band_level_labels[name].config(text=f"{name+':':12s} --")
            if d and d.get("mean_thd_pct") is not None:
                self.band_thd_labels[name].config(text=f"{name+':':12s} {d['mean_thd_pct']:7.3f} %")
            else: self.band_thd_labels[name].config(text=f"{name+':':12s} --")

        thd=m.get("thd_global")
        ratio=m.get("even_odd_ratio")
        ratio_db=m.get("even_odd_ratio_db")
        thd_text="--" if thd is None else f"{thd:.4f} %"
        ratio_text="--" if ratio is None else f"{ratio:.4f} ({ratio_db:+.2f} dB)"
        self.lbl_harmonics.config(text=f"\nTHD: {thd_text}\nEven/Odd: {ratio_text}")

    # ---------------------------------------------------------------- save
    def _save(self):
        if not self._last_metrics or self._last_chart_data is None:
            messagebox.showwarning("No data","No measurement is available to save.")
            return
        
        log_dir = _REPO_ROOT/cfg.logs_dir
        log_dir.mkdir(parents=True,exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=str(log_dir), defaultextension=".png",
            filetypes=[("PNG image","*.png"), ("JSON metrics","*.json"), ("All files","*.*")], title="Save chart and metrics")
        if not path:
            return

        params={
            "signal_type": self.var_signal.get(),
            "path": self.var_path.get(),
            "send_gain_pct": float(self.var_send_gain.get()),
            "tone_amplitude_peak": float(self.var_tone_amplitude.get()),
            "tone_duration_s": float(self.var_tone_duration.get()),
            "tone_gap_s": float(self.var_tone_gap.get()),
            "noise_duration_s": float(self.var_noise_duration.get()),
            "freq_min_hz": float(self.var_min_f.get()),
            "freq_max_hz": float(self.var_max_f.get()),
            "frequency_bins": int(self.var_bins.get()),
            "send_correction": bool(self.var_send_corr.get()),
            "receive_correction": bool(self.var_recv_corr.get())
        }
        png,json_path = save_result_bundle(path,self.fig,self._last_metrics, self._last_chart_data,params)
        messagebox.showinfo("Saved", f"Saved:\n{png}\n{json_path}")


def main():
    root = tk.Tk()
    try:
        root.state("zoomed")
    except Exception:
        pass

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = AmpAnalyzerApp(root)

    def keys(event):
        if event.state & 0x4 and event.keysym.lower()=="s":
            app._save()
            return "break"
        if event.keysym=="F5":
            app._stop() if app.is_running else app._start()
            return "break"
        if event.keysym=="Escape" and app.is_running:
            app._stop()
            return "break"
    
    root.bind("<Key>",keys)
    root.mainloop()
