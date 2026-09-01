"""Background measurement worker used by the Tkinter GUI."""
from __future__ import annotations

import numpy as np
import queue
import threading
import traceback

from config import config as cfg
from utils.audio.analysis_utils import analyze_noise_measurement, analyze_sweep_measurement, smooth_moving_average
from utils.audio.streaming import run_noise_measurement, run_sweep_measurement


class AnalyzerWorker(threading.Thread):
    MSG_STATUS = "status"
    MSG_CHART = "chart"
    MSG_METRICS = "metrics"
    MSG_ERROR = "error"
    MSG_DONE = "done"

    def __init__(self, signal_type: str, result_queue: queue.Queue, **params):
        super().__init__(daemon=True)
        self.signal_type = signal_type
        self.result_queue = result_queue
        self.params = params
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _progress(self, fraction: float, elapsed: float):
        pct = min(max(fraction, 0.0), 1.0) * 100.0
        self.result_queue.put((self.MSG_STATUS, f"Streaming {pct:5.1f}% ({elapsed:.1f}s)"))

    def run(self):
        try:
            p = self.params
            send_profile = p.get("send_profile")
            recv_profile = p.get("recv_profile")
            send_cf = send_profile.frequencies if send_profile is not None else None
            send_factors = send_profile.factors if send_profile is not None else None
            recv_cf = recv_profile.frequencies if recv_profile is not None else None
            recv_factors = recv_profile.factors if recv_profile is not None else None

            self.result_queue.put((self.MSG_STATUS, "Generating stimulus..."))
            if self.signal_type == "sweep":
                capture_result = run_sweep_measurement(
                    freqs=p["freq_array"], fs=p["fs"], tone_duration=p["tone_duration"],
                    gap_s=p["gap_s"], tone_amplitude=p["tone_amplitude"], send_gain=p["send_gain"],
                    send_device=p["send_device"], recv_device=p["recv_device"],
                    send_ch=p["send_ch"], recv_ch=p["recv_ch"],
                    send_correction_freqs=send_cf, send_correction_factors=send_factors,
                    peak_headroom=p.get("sweep_peak_headroom", cfg.sweep_peak_headroom),
                    stop_event=self._stop_event, progress_callback=self._progress,
                )
                if self._stop_event.is_set():
                    return
                metrics = analyze_sweep_measurement(
                    capture_result.capture, capture_result.stimulus.metadata, p["fs"],
                    recv_correction_freqs=recv_cf, recv_correction_factors=recv_factors,
                    frequency_bands=cfg.frequency_bands,
                )
            else:
                capture_result = run_noise_measurement(
                    method=p["method"], duration_s=p["duration_s"], fs=p["fs"],
                    tone_amplitude=p["tone_amplitude"], send_gain=p["send_gain"],
                    send_device=p["send_device"], recv_device=p["recv_device"],
                    send_ch=p["send_ch"], recv_ch=p["recv_ch"],
                    send_correction_freqs=send_cf, send_correction_factors=send_factors,
                    peak_headroom=p.get("noise_peak_headroom", cfg.noise_peak_headroom),
                    stop_event=self._stop_event, progress_callback=self._progress,
                )
                if self._stop_event.is_set():
                    return
                metrics = analyze_noise_measurement(
                    capture_result.capture, capture_result.stimulus.reference_samples,
                    p["fs"], p["freq_array"], recv_correction_freqs=recv_cf,
                    recv_correction_factors=recv_factors, frequency_bands=cfg.frequency_bands,
                )

            stimulus_meta = dict(capture_result.stimulus.metadata)
            # Large ndarray fields are already represented in sweep per-tone metrics;
            # keep level metadata compact in the top-level summary.
            for k in ("frequencies", "tone_starts", "sent_peak_per_tone", "sent_rms_per_tone", "reference_rms_per_tone"):
                stimulus_meta.pop(k, None)
            metrics["stimulus"] = stimulus_meta
            metrics["send_correction_applied"] = send_profile is not None
            metrics["receive_correction_applied"] = recv_profile is not None
            metrics["send_correction_file"] = str(send_profile.path) if send_profile is not None else None
            metrics["receive_correction_file"] = str(recv_profile.path) if recv_profile is not None else None
            metrics["stream_status"] = list(capture_result.stream_status)

            received_dbfs = np.asarray(metrics["received_dbfs"], dtype=float)
            response_db = np.asarray(metrics["relative_response_db"], dtype=float)
            chart = {
                "freqs": np.asarray(metrics["frequency_hz"], dtype=float),
                "received_dbfs": received_dbfs,
                "received_dbfs_raw": np.asarray(metrics.get("received_dbfs_raw", received_dbfs), dtype=float),
                "smoothed_dbfs": smooth_moving_average(received_dbfs, cfg.smoothing_neighbors),
                "relative_response_db": response_db,
                "relative_response_db_raw": np.asarray(metrics.get("relative_response_db_raw", response_db), dtype=float),
                "method": p.get("method", "sweep"),
            }
            self.result_queue.put((self.MSG_CHART, chart))
            self.result_queue.put((self.MSG_METRICS, metrics))
        except Exception as exc:
            self.result_queue.put((self.MSG_ERROR, f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"))
        finally:
            self.result_queue.put((self.MSG_DONE, None))
