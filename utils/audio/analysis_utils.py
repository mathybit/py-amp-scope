"""DSP and measurement analysis for PyAmpScope.

The public measurement paths are:

* :func:`analyze_sweep_measurement` -- one known sine at a time; harmonics are
  measured by exact-frequency sinusoidal projection (lock-in style), not by
  reading a single FFT bin.
* :func:`analyze_noise_measurement` -- Welch power spectral-density estimation;
  true RMS is obtained by integrating PSD over frequency intervals.

All dB amplitude/RMS calculations use base-10 logarithms.  Power/PSD ratios use
10*log10.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal as scipy_signal
from typing import Iterable, Optional

from config import config as cfg
from .levels import db10, db20, peak, rms
from .signal_utils import interpolate_correction


_EPS = 1e-30


def smooth_moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Centered NaN-aware moving average with edge windows shortened naturally."""
    x = np.asarray(arr, dtype=np.float64)
    if x.size == 0 or window_size <= 1:
        return x.copy()
    w = max(1, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo, hi = max(0, i-half), min(len(x), i+half+1)
        vals = x[lo:hi]
        valid = np.isfinite(vals)
        if np.any(valid):
            out[i] = float(np.mean(vals[valid]))
    return out


def measure_sine_component_rms(samples: np.ndarray, freq_hz: float, fs: int) -> float:
    """RMS amplitude of a sinusoidal component at an exact known frequency.

    Least-squares projection onto sine/cosine is robust to non-integer FFT-bin
    alignment and unknown phase.  The signal mean is removed first.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 8 or freq_hz <= 0 or freq_hz >= fs/2:
        return float("nan")
    x = x - np.mean(x)
    t = np.arange(x.size, dtype=np.float64) / fs
    omega = 2.0 * np.pi * float(freq_hz)
    s = np.sin(omega * t)
    c = np.cos(omega * t)
    # Solve the two-column least-squares system.  This remains accurate even when
    # the segment does not contain an integer number of cycles.
    design = np.column_stack((s, c))
    coeff, *_ = np.linalg.lstsq(design, x, rcond=None)
    peak_amp = float(np.hypot(coeff[0], coeff[1]))
    return peak_amp / math.sqrt(2.0)




def measure_harmonic_components_rms(samples: np.ndarray, fundamental_hz: float, fs: int, max_harmonic: int = 10) -> dict[int, float]:
    """Jointly fit the fundamental and observable harmonics.

    A joint least-squares fit avoids leakage from a strong fundamental into the
    harmonic estimates when the analysis segment does not contain an exact
    integer number of cycles.  Returned values are RMS amplitudes keyed by
    harmonic order (1=fundamental).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 16 or fundamental_hz <= 0 or fundamental_hz >= fs / 2:
        return {}
    orders = [h for h in range(1, int(max_harmonic) + 1) if h * fundamental_hz < fs / 2.0]
    if not orders:
        return {}
    t = np.arange(x.size, dtype=np.float64) / fs
    cols = [np.ones_like(t)]
    for h in orders:
        w = 2.0 * np.pi * fundamental_hz * h
        cols.extend((np.sin(w * t), np.cos(w * t)))
    design = np.column_stack(cols)
    coeff, *_ = np.linalg.lstsq(design, x, rcond=None)
    result: dict[int, float] = {}
    j = 1
    for h in orders:
        peak_amp = float(np.hypot(coeff[j], coeff[j + 1]))
        result[h] = peak_amp / math.sqrt(2.0)
        j += 2
    return result

def fft_db(sig, target_hz, fs):
    """Compatibility helper: exact-frequency sinusoidal RMS level in dBFS."""
    return db20(measure_sine_component_rms(np.asarray(sig), float(target_hz), int(fs)))


def _estimate_sweep_latency(capture: np.ndarray, first_tone_start: int, tone_samples: int, fs: int) -> int:
    """Estimate capture-vs-stimulus offset from the first signal-energy onset.

    This is intentionally conservative: if no reliable onset is found, return 0
    and rely on center-of-tone trimming.  Separate USB devices can have small
    independent latencies, so avoiding a hard failure is preferable.
    """
    x = np.asarray(capture, dtype=np.float64)
    if x.size < max(256, first_tone_start + 128):
        return 0
    frame = max(128, int(0.025 * fs))
    hop = frame
    search_end = min(len(x), first_tone_start + tone_samples + int(0.4*fs))
    starts = np.arange(0, max(frame, search_end-frame+1), hop, dtype=int)
    if starts.size < 4:
        return 0
    energies = np.array([rms(x[s:s+frame]) for s in starts], dtype=float)
    baseline_end = max(1, int((first_tone_start * 0.7) / hop))
    baseline = float(np.median(energies[:baseline_end])) if baseline_end > 0 else float(np.median(energies[:3]))
    noise_sigma = float(np.median(np.abs(energies[:max(3, baseline_end)] - baseline))) * 1.4826
    threshold = max(baseline + 6.0 * noise_sigma, baseline * 3.0, 1e-6)
    above = energies > threshold
    # Require two consecutive frames to reject impulses.
    for i in range(max(0, baseline_end-1), len(above)-1):
        if above[i] and above[i+1]:
            onset = int(starts[i])
            offset = onset - int(first_tone_start)
            return int(np.clip(offset, -int(0.2*fs), int(0.4*fs)))
    return 0


def _band_for_frequency(freq: float, bands: dict[str, tuple[float, float]]) -> Optional[str]:
    names = list(bands)
    for i, name in enumerate(names):
        lo, hi = bands[name]
        if lo <= freq < hi or (i == len(names)-1 and lo <= freq <= hi):
            return name
    return None


def _aggregate_sweep_bands(per_tone: list[dict], bands: dict[str, tuple[float, float]]) -> dict:
    result: dict[str, dict] = {}
    for name, (lo, hi) in bands.items():
        rows = [r for r in per_tone if r.get("valid") and _band_for_frequency(r["frequency_hz"], bands) == name]
        if not rows:
            continue
        recv = np.array([r["received_rms"] for r in rows if r.get("received_rms") is not None], dtype=float)
        resp = np.array([r["relative_response_db"] for r in rows if r.get("relative_response_db") is not None], dtype=float)
        thd = np.array([r["thd_pct"] for r in rows if r.get("thd_pct") is not None], dtype=float)
        even = np.array([r["even_harmonic_pct"] for r in rows if r.get("even_harmonic_pct") is not None], dtype=float)
        odd = np.array([r["odd_harmonic_pct"] for r in rows if r.get("odd_harmonic_pct") is not None], dtype=float)
        band: dict = {"low_hz": lo, "high_hz": hi, "tones": len(rows), "kind": "sweep"}
        if recv.size:
            # Mean power across individual-tone measurements, then back to RMS.
            band_rms = float(np.sqrt(np.mean(recv**2)))
            band.update(received_rms=band_rms, received_dbfs=db20(band_rms), mean_dBFS=db20(band_rms))
        if resp.size:
            band.update(mean_response_db=float(np.mean(resp)), std_response_db=float(np.std(resp)))
        if thd.size:
            band.update(mean_thd_pct=float(np.mean(thd)), std_thd_pct=float(np.std(thd)))
        if even.size:
            band["mean_even_harmonic_pct"] = float(np.mean(even))
        if odd.size:
            band["mean_odd_harmonic_pct"] = float(np.mean(odd))
        result[name] = band
    return result


def analyze_sweep_measurement(
            capture: np.ndarray,
            stimulus_metadata: dict,
            fs: int,
            *,
            recv_correction_freqs: Optional[np.ndarray] = None,
            recv_correction_factors: Optional[np.ndarray] = None,
            frequency_bands: Optional[dict[str, tuple[float, float]]] = None,
            max_harmonic: int = 10,
            trim_fraction: float = 0.15,
        ) -> dict:
    """Analyze one captured sweep and return per-tone + aggregate metrics."""
    bands = frequency_bands or cfg.frequency_bands
    freqs = np.asarray(stimulus_metadata["frequencies"], dtype=np.float64)
    starts = np.asarray(stimulus_metadata["tone_starts"], dtype=np.int64)
    tone_samples = int(stimulus_metadata["tone_samples"])
    reference_rms = np.asarray(stimulus_metadata["reference_rms_per_tone"], dtype=np.float64)
    sent_rms = np.asarray(stimulus_metadata["sent_rms_per_tone"], dtype=np.float64)
    cap = np.asarray(capture, dtype=np.float64)

    latency = _estimate_sweep_latency(cap, int(starts[0]), tone_samples, fs)
    trim = min(int(tone_samples * trim_fraction), max(0, tone_samples//3))
    corr_fund = interpolate_correction(freqs, recv_correction_freqs, recv_correction_factors)
    correction_applied = recv_correction_freqs is not None and recv_correction_factors is not None

    per_tone: list[dict] = []
    raw_transfer_db: list[float] = []
    corrected_transfer_db: list[float] = []

    for i, f in enumerate(freqs):
        start = int(starts[i] + latency + trim)
        end = int(starts[i] + latency + tone_samples - trim)
        row = {
            "frequency_hz": float(f),
            "band": _band_for_frequency(float(f), bands),
            "valid": False,
            "sent_rms": float(sent_rms[i]),
            "reference_rms": float(reference_rms[i]),
        }
        if start < 0 or end > len(cap) or end-start < 128:
            row["reason"] = "capture segment unavailable"
            per_tone.append(row)
            raw_transfer_db.append(float("nan"))
            corrected_transfer_db.append(float("nan"))
            continue
        seg = cap[start:end]
        components = measure_harmonic_components_rms(seg, f, fs, max_harmonic=max_harmonic)
        fundamental_raw = components.get(1, float("nan"))
        if not np.isfinite(fundamental_raw) or fundamental_raw <= 0:
            row["reason"] = "fundamental unavailable"
            per_tone.append(row)
            raw_transfer_db.append(float("nan"))
            corrected_transfer_db.append(float("nan"))
            continue

        fundamental_corr = fundamental_raw * corr_fund[i]
        raw_tf = db20(fundamental_raw / max(reference_rms[i], _EPS))
        corr_tf = db20(fundamental_corr / max(reference_rms[i], _EPS))
        raw_transfer_db.append(raw_tf)
        corrected_transfer_db.append(corr_tf)

        harmonic_rows: list[dict] = []
        raw_sq = 0.0
        corr_sq = 0.0
        even_corr_sq = 0.0
        odd_corr_sq = 0.0
        for h in range(2, int(max_harmonic)+1):
            hf = f * h
            if hf >= fs / 2.0:
                break
            h_raw = components.get(h, float("nan"))
            if not np.isfinite(h_raw):
                continue
            h_corr_factor = interpolate_correction(
                np.asarray([hf]), recv_correction_freqs, recv_correction_factors
            )[0]
            h_corr = h_raw * h_corr_factor
            raw_ratio = h_raw / max(fundamental_raw, _EPS)
            corr_ratio = h_corr / max(fundamental_corr, _EPS)
            raw_sq += raw_ratio**2
            corr_sq += corr_ratio**2
            if h % 2 == 0:
                even_corr_sq += corr_ratio**2
            else:
                odd_corr_sq += corr_ratio**2
            harmonic_rows.append({
                "order": h,
                "frequency_hz": float(hf),
                "rms_raw": float(h_raw),
                "rms_corrected": float(h_corr),
                "relative_pct_raw": float(raw_ratio * 100.0),
                "relative_pct": float(corr_ratio * 100.0),
                "relative_db": db20(corr_ratio),
            })

        has_harmonic = bool(harmonic_rows)
        thd_raw = math.sqrt(raw_sq)*100.0 if has_harmonic else None
        thd_corr = math.sqrt(corr_sq)*100.0 if has_harmonic else None
        even_pct = math.sqrt(even_corr_sq)*100.0 if has_harmonic else None
        odd_pct = math.sqrt(odd_corr_sq)*100.0 if has_harmonic else None
        if even_pct is not None and odd_pct is not None and odd_pct > 1e-15:
            eo_ratio = even_pct / odd_pct
            eo_db = db20(eo_ratio)
        else:
            eo_ratio = None
            eo_db = None

        row.update({
            "valid": True,
            "segment_rms": rms(seg),
            "received_rms_raw": float(fundamental_raw),
            "received_dbfs_raw": db20(fundamental_raw),
            "received_rms": float(fundamental_corr),
            "received_dbfs": db20(fundamental_corr),
            "transfer_db_raw": raw_tf,
            "transfer_db": corr_tf,
            "thd_pct_raw": thd_raw,
            "thd_pct": thd_corr,
            "even_harmonic_pct": even_pct,
            "odd_harmonic_pct": odd_pct,
            "even_odd_ratio": eo_ratio,
            "even_odd_ratio_db": eo_db,
            "harmonics": harmonic_rows,
        })
        per_tone.append(row)

    raw_tf_arr = np.asarray(raw_transfer_db, dtype=float)
    corr_tf_arr = np.asarray(corrected_transfer_db, dtype=float)
    valid = np.isfinite(corr_tf_arr)
    if np.any(valid):
        ref_db = float(np.median(corr_tf_arr[valid]))
        rel = corr_tf_arr - ref_db
        raw_ref_db = float(np.median(raw_tf_arr[np.isfinite(raw_tf_arr)]))
        rel_raw = raw_tf_arr - raw_ref_db
    else:
        ref_db = raw_ref_db = float("nan")
        rel = np.full_like(corr_tf_arr, np.nan)
        rel_raw = np.full_like(raw_tf_arr, np.nan)

    for i, row in enumerate(per_tone):
        if row.get("valid"):
            row["relative_response_db"] = float(rel[i])
            row["relative_response_db_raw"] = float(rel_raw[i])

    valid_rows = [r for r in per_tone if r.get("valid")]
    recv_vals = np.asarray([r["received_rms"] for r in valid_rows], dtype=float) if valid_rows else np.array([])
    thd_vals = np.asarray([r["thd_pct"] for r in valid_rows if r.get("thd_pct") is not None], dtype=float)

    # Global even/odd ratio: combine harmonic ratios by power across all tones.
    even_power = 0.0
    odd_power = 0.0
    for r in valid_rows:
        for h in r.get("harmonics", []):
            ratio = float(h["relative_pct"]) / 100.0
            if h["order"] % 2 == 0:
                even_power += ratio**2
            else:
                odd_power += ratio**2
    if odd_power > 1e-30:
        even_odd_ratio = math.sqrt(even_power / odd_power)
        even_odd_ratio_db = db20(even_odd_ratio)
    else:
        even_odd_ratio = even_odd_ratio_db = None

    band_metrics = _aggregate_sweep_bands(per_tone, bands)
    thd_per_band = {
        name: {"mean_pct": d["mean_thd_pct"], "std_pct": d["std_thd_pct"], "tones": d["tones"]}
        for name, d in band_metrics.items() if "mean_thd_pct" in d
    }

    return {
        "mode": "sweep",
        "frequency_hz": freqs,
        "received_dbfs": np.asarray([r.get("received_dbfs", np.nan) for r in per_tone], dtype=float),
        "received_dbfs_raw": np.asarray([r.get("received_dbfs_raw", np.nan) for r in per_tone], dtype=float),
        "relative_response_db": rel,
        "relative_response_db_raw": rel_raw,
        "transfer_reference_db": ref_db,
        "transfer_reference_db_raw": raw_ref_db,
        "latency_samples": int(latency),
        "latency_ms": float(latency / fs * 1000.0),
        "receive_correction_applied": correction_applied,
        "per_tone": per_tone,
        "freq_response": {
            "valid_bins": int(np.sum(valid)),
            "mean_dBFS": db20(float(np.sqrt(np.mean(recv_vals**2)))) if recv_vals.size else None,
            "std_dB": float(np.std(db20(recv_vals))) if recv_vals.size else None,
            "min_dBFS": float(np.min(db20(recv_vals))) if recv_vals.size else None,
            "max_dBFS": float(np.max(db20(recv_vals))) if recv_vals.size else None,
            "relative_std_db": float(np.std(rel[valid])) if np.any(valid) else None,
            "relative_min_db": float(np.min(rel[valid])) if np.any(valid) else None,
            "relative_max_db": float(np.max(rel[valid])) if np.any(valid) else None,
        },
        "rms_signal": float(np.sqrt(np.mean(recv_vals**2))) if recv_vals.size else None,
        "capture_rms": rms(cap),
        "peak": peak(cap),
        "peak_dBFS": db20(peak(cap)),
        "clipped": bool(peak(cap) >= 0.999),
        "frequency_bands": band_metrics,
        "octave_bands": band_metrics,  # compatibility with older GUI naming
        "rms_per_band": {
            name: {"mean": d.get("received_rms"), "dbfs": d.get("received_dbfs"), "tones": d.get("tones", 0)}
            for name, d in band_metrics.items()
        },
        "thd_per_band": thd_per_band,
        "thd_global": float(np.mean(thd_vals)) if thd_vals.size else None,
        "even_odd_ratio": even_odd_ratio,
        "even_odd_ratio_db": even_odd_ratio_db,
    }


def welch_psd(samples: np.ndarray, fs: int, nperseg: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    if x.size < 32:
        return np.array([], dtype=float), np.array([], dtype=float)
    if nperseg is None:
        # 32768 gives ~1.35 Hz resolution at 44.1 kHz while still averaging many
        # segments in a 30-60 s capture.
        nperseg = min(32768, len(x))
    nperseg = max(32, min(int(nperseg), len(x)))
    freqs, psd = scipy_signal.welch(
        x, fs=fs, window="hann", nperseg=nperseg,
        noverlap=nperseg//2, detrend="constant", scaling="density",
        return_onesided=True,
    )
    return freqs.astype(float), psd.astype(float)


def _interp_log_frequency(source_f: np.ndarray, values: np.ndarray, target_f: np.ndarray) -> np.ndarray:
    source_f = np.asarray(source_f, dtype=float)
    values = np.asarray(values, dtype=float)
    target_f = np.asarray(target_f, dtype=float)
    valid = np.isfinite(source_f) & np.isfinite(values) & (source_f > 0)
    out = np.full_like(target_f, np.nan, dtype=float)
    target_valid = np.isfinite(target_f) & (target_f > 0)
    if np.sum(valid) < 2 or not np.any(target_valid):
        return out
    f = source_f[valid]
    v = values[valid]
    order = np.argsort(f)
    out[target_valid] = np.interp(np.log10(target_f[target_valid]), np.log10(f[order]), v[order], left=np.nan, right=np.nan)
    return out


def _log_bin_edges(target_freqs: np.ndarray, min_f: float, max_f: float) -> np.ndarray:
    f = np.asarray(target_freqs, dtype=float)
    edges = np.empty(len(f)+1, dtype=float)
    if len(f) == 1:
        edges[:] = [max(min_f, f[0]/math.sqrt(2)), min(max_f, f[0]*math.sqrt(2))]
        return edges
    mids = np.sqrt(f[:-1] * f[1:])
    edges[1:-1] = mids
    edges[0] = max(min_f, f[0]**2 / mids[0])
    edges[-1] = min(max_f, f[-1]**2 / mids[-1])
    return edges


def integrate_psd_band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate one-sided PSD over [lo, hi] and return RMS."""
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(psd)
    if np.sum(mask) < 2:
        return float("nan")
    trapezoid_fn = getattr(np, "trapezoid", np.trapz)
    power = float(np.trapz(psd[mask], freqs[mask]))
    return math.sqrt(max(power, 0.0))


def analyze_noise_measurement(
            capture: np.ndarray,
            reference_signal: np.ndarray,
            fs: int,
            target_freqs: np.ndarray,
            *,
            recv_correction_freqs: Optional[np.ndarray] = None,
            recv_correction_factors: Optional[np.ndarray] = None,
            frequency_bands: Optional[dict[str, tuple[float, float]]] = None,
        ) -> dict:
    """Analyze broadband noise with Welch PSD and normalized transfer response."""
    bands = frequency_bands or cfg.frequency_bands
    cap = np.asarray(capture, dtype=np.float64)
    ref = np.asarray(reference_signal, dtype=np.float64)
    n = min(len(cap), len(ref))
    cap = cap[:n]
    ref = ref[:n]
    f_recv, p_recv_raw = welch_psd(cap, fs)
    f_ref, p_ref = welch_psd(ref, fs)
    if f_recv.size == 0 or f_ref.size == 0:
        raise ValueError("not enough samples for Welch analysis")

    corr = interpolate_correction(f_recv, recv_correction_freqs, recv_correction_factors)
    correction_applied = recv_correction_freqs is not None and recv_correction_factors is not None
    p_recv = p_recv_raw * np.square(corr)

    target = np.asarray(target_freqs, dtype=np.float64)
    target = target[(target > 0) & (target < fs/2)]
    p_recv_i = _interp_log_frequency(f_recv, p_recv, target)
    p_recv_raw_i = _interp_log_frequency(f_recv, p_recv_raw, target)
    p_ref_i = _interp_log_frequency(f_ref, p_ref, target)
    transfer_db = db10(p_recv_i / np.maximum(p_ref_i, _EPS))
    transfer_db_raw = db10(p_recv_raw_i / np.maximum(p_ref_i, _EPS))
    valid = np.isfinite(transfer_db)
    median_transfer = float(np.median(transfer_db[valid])) if np.any(valid) else float("nan")
    median_transfer_raw = float(np.median(transfer_db_raw[np.isfinite(transfer_db_raw)])) if np.any(np.isfinite(transfer_db_raw)) else float("nan")
    relative = transfer_db - median_transfer
    relative_raw = transfer_db_raw - median_transfer_raw

    # True RMS per logarithmic analysis bin.  This keeps the legacy concept of a
    # received dBFS curve while avoiding mislabeled single-FFT-bin magnitude.
    min_f = max(float(np.min(target)), float(min(f_recv[f_recv > 0])))
    max_f = min(float(np.max(target)), fs/2)
    edges = _log_bin_edges(target, min_f, max_f)
    bin_rms = np.array([integrate_psd_band(f_recv, p_recv, edges[i], edges[i+1]) for i in range(len(target))])
    bin_rms_raw = np.array([integrate_psd_band(f_recv, p_recv_raw, edges[i], edges[i+1]) for i in range(len(target))])
    bin_dbfs = db20(bin_rms)
    bin_dbfs_raw = db20(bin_rms_raw)

    band_metrics: dict[str, dict] = {}
    for name, (lo, hi) in bands.items():
        lo_use = max(lo, 0.0)
        hi_use = min(hi, fs/2)
        if hi_use <= lo_use:
            continue
        r = integrate_psd_band(f_recv, p_recv, lo_use, hi_use)
        rr = integrate_psd_band(f_recv, p_recv_raw, lo_use, hi_use)
        if not np.isfinite(r):
            continue
        band_metrics[name] = {
            "kind": "noise",
            "low_hz": lo,
            "high_hz": hi,
            "received_rms": float(r),
            "received_dbfs": db20(r),
            "received_rms_raw": float(rr) if np.isfinite(rr) else None,
            "received_dbfs_raw": db20(rr) if np.isfinite(rr) else None,
            "mean_dBFS": db20(r),
        }

    audible_mask = (f_recv >= min(b[0] for b in bands.values())) & (f_recv <= min(max(b[1] for b in bands.values()), fs/2))
    corrected_total_rms = math.sqrt(max(float(np.trapz(p_recv[audible_mask], f_recv[audible_mask])), 0.0)) if np.sum(audible_mask) >= 2 else rms(cap)

    return {
        "mode": "noise",
        "frequency_hz": target,
        "received_dbfs": bin_dbfs,
        "received_dbfs_raw": bin_dbfs_raw,
        "relative_response_db": relative,
        "relative_response_db_raw": relative_raw,
        "transfer_reference_db": median_transfer,
        "transfer_reference_db_raw": median_transfer_raw,
        "receive_correction_applied": correction_applied,
        "welch": {
            "frequency_hz": f_recv,
            "received_psd": p_recv,
            "received_psd_raw": p_recv_raw,
            "reference_psd": _interp_log_frequency(f_ref, p_ref, f_recv),
        },
        "freq_response": {
            "valid_bins": int(np.sum(np.isfinite(bin_dbfs))),
            "mean_dBFS": float(np.nanmean(bin_dbfs)),
            "std_dB": float(np.nanstd(bin_dbfs)),
            "min_dBFS": float(np.nanmin(bin_dbfs)),
            "max_dBFS": float(np.nanmax(bin_dbfs)),
            "relative_std_db": float(np.std(relative[valid])) if np.any(valid) else None,
            "relative_min_db": float(np.min(relative[valid])) if np.any(valid) else None,
            "relative_max_db": float(np.max(relative[valid])) if np.any(valid) else None,
        },
        "rms_signal": float(corrected_total_rms),
        "capture_rms": rms(cap),
        "peak": peak(cap),
        "peak_dBFS": db20(peak(cap)),
        "clipped": bool(peak(cap) >= 0.999),
        "frequency_bands": band_metrics,
        "octave_bands": band_metrics,
        "rms_per_band": {
            name: {"mean": d["received_rms"], "dbfs": d["received_dbfs"], "tones": None}
            for name, d in band_metrics.items()
        },
        "thd_per_band": {},
        "thd_global": None,
        "even_odd_ratio": None,
        "even_odd_ratio_db": None,
    }


# ---------------------------------------------------------------------------
# Compatibility / calibration helpers
# ---------------------------------------------------------------------------
def analyze_noise_response(captured_signal, freq_array, fs):
    target = np.asarray(freq_array, dtype=float)
    f, p = welch_psd(np.asarray(captured_signal), int(fs))
    if f.size == 0:
        return target, np.full_like(target, np.nan), np.full_like(target, np.nan)
    edges = _log_bin_edges(target, max(1e-6, target[0]), min(fs/2, target[-1]))
    rms_arr = np.array([integrate_psd_band(f, p, edges[i], edges[i+1]) for i in range(len(target))])
    return target, db20(rms_arr), rms_arr


def compute_correction(H_lin):
    values = np.asarray(H_lin, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return np.ones_like(values)
    ref = float(np.median(values[valid]))
    return np.where(valid, ref / np.maximum(values, _EPS), 1.0)


def compute_thd(signal, fs):
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 128:
        return {"fundamental_hz": None, "thd_pct": None, "harmonic_levels_dB": {}}
    f, p = scipy_signal.periodogram(x - np.mean(x), fs=fs, window="hann", scaling="spectrum")
    valid = (f >= 20) & (f < fs/2)
    if not np.any(valid):
        return {"fundamental_hz": None, "thd_pct": None, "harmonic_levels_dB": {}}
    fi = np.where(valid)[0][int(np.argmax(p[valid]))]
    fund = float(f[fi])
    fund_rms = measure_sine_component_rms(x, fund, fs)
    sq = 0.0
    hlevels = {}
    for h in range(2, 11):
        hf = fund*h
        if hf >= fs/2:
            break
        hr = measure_sine_component_rms(x, hf, fs)
        ratio = hr / max(fund_rms, _EPS)
        sq += ratio**2
        hlevels[f"H{h}"] = db20(ratio)
    return {"fundamental_hz": fund, "fundamental_mag_dB": db20(fund_rms),
            "thd_pct": math.sqrt(sq)*100.0 if hlevels else None,
            "harmonic_levels_dB": hlevels}


def extract_tone_measurements(rec, freqs, tone_duration_s, gap_s, fs):
    freqs = np.asarray(freqs, dtype=float)
    hop = int(round((tone_duration_s + gap_s)*fs))
    tone_n = int(round(tone_duration_s*fs))
    measured = np.full(len(freqs), np.nan)
    rms_list = np.full(len(freqs), np.nan)
    trim = int(tone_n*0.15)
    for i, f in enumerate(freqs):
        start = i*hop + trim
        end = i*hop + tone_n - trim
        if end > len(rec) or end-start < 64:
            continue
        r = measure_sine_component_rms(np.asarray(rec)[start:end], f, fs)
        rms_list[i] = r
        measured[i] = db20(r)
    return measured, rms_list


def compute_thd_per_tone_with_freqs(signal, fs, freq_array, params=None):
    params = params or {}
    tone_dur = float(params.get("tone_duration", 0.7))
    gap_s = float(params.get("gap_s", 0.2))
    freqs = np.asarray(freq_array, dtype=float)
    hop = int(round((tone_dur+gap_s)*fs))
    tone_n = int(round(tone_dur*fs))
    trim = int(tone_n*0.15)
    vals, mags, out_f = [], [], []
    x = np.asarray(signal)
    for i, f in enumerate(freqs):
        seg = x[i*hop+trim:i*hop+tone_n-trim]
        if len(seg) < 64:
            continue
        fund = measure_sine_component_rms(seg, f, fs)
        sq = 0.0
        count = 0
        for h in range(2, 11):
            if h*f >= fs/2:
                break
            hr = measure_sine_component_rms(seg, h*f, fs)
            sq += (hr/max(fund,_EPS))**2
            count += 1
        out_f.append(f); mags.append(db20(fund)); vals.append(math.sqrt(sq)*100.0 if count else np.nan)
    return np.asarray(out_f), np.asarray(vals), np.asarray(mags)


def compute_harmonics_for_overdrive(signal, fs, fundamental_hz=None):
    base = compute_thd(signal, fs)
    if base.get("fundamental_hz") is None:
        return []
    fund = float(fundamental_hz or base["fundamental_hz"])
    fr = measure_sine_component_rms(np.asarray(signal), fund, fs)
    rows=[]
    for h in range(2, 16):
        hf=fund*h
        if hf>=fs/2: break
        hr=measure_sine_component_rms(np.asarray(signal), hf, fs)
        ratio=hr/max(fr,_EPS)
        rows.append({"order":h,"freq_hz":hf,"level_dB_relative_to_fund":db20(ratio),"level_linear":ratio})
    return rows


def compute_odd_even_ratio(signal, fs):
    hs = compute_harmonics_for_overdrive(signal, fs)
    if not hs:
        return None
    odd=sum(h["level_linear"]**2 for h in hs if h["order"]%2==1)
    even=sum(h["level_linear"]**2 for h in hs if h["order"]%2==0)
    if odd <= _EPS:
        return {"even_odd_ratio": None, "even_odd_ratio_db": None}
    ratio=math.sqrt(even/odd)
    return {"even_odd_ratio":ratio,"even_odd_ratio_db":db20(ratio)}


def compute_octave_band_stats(signal, freq_array, fs):
    f,p=welch_psd(np.asarray(signal),fs)
    out={}
    for name,(lo,hi) in cfg.frequency_bands.items():
        r=integrate_psd_band(f,p,lo,min(hi,fs/2))
        if np.isfinite(r):
            out[name]={"mean_dB":db20(r),"rms":r}
    return out


def compare_noise_spectral_shape(captured_signal, noise_method, freq_array, fs, smooth_window=5):
    """Compatibility report comparing measured Welch PSD with ideal noise slope."""
    target=np.asarray(freq_array,dtype=float)
    f,p=welch_psd(np.asarray(captured_signal),fs)
    p_i=_interp_log_frequency(f,p,target)
    measured_db=db10(p_i)
    method=noise_method.lower()
    if method == "white": exponent=0.0
    elif method == "pink": exponent=-1.0
    elif method == "brown": exponent=-2.0
    else: raise ValueError(f"unsupported noise method: {noise_method}")
    ref_freq=float(np.exp(np.mean(np.log(target))))
    expected_db=10.0*exponent*np.log10(target/ref_freq)
    smooth=smooth_moving_average(measured_db,smooth_window)
    valid=np.isfinite(smooth)
    shift=float(np.mean(smooth[valid]-expected_db[valid])) if np.any(valid) else 0.0
    shifted=expected_db+shift
    dev=smooth-shifted
    return {"freqs":target,"measured_db":measured_db,"smoothed_db":smooth,
            "expected_db":expected_db,"expected_shifted_db":shifted,"deviation_db":dev,
            "deviation_pct":(np.power(10.0,dev/10.0)-1.0)*100.0,
            "shape_std_db":float(np.nanstd(dev)),"shape_std_pct":float(np.nanstd((np.power(10.0,dev/10.0)-1.0)*100.0)),
            "shift_db":shift,"global_offset_db":shift}


def print_noise_shape_report(result, method):
    if "error" in result:
        print(result["error"]); return
    print(f"\n{method.capitalize()} noise spectral-shape report")
    print(f"  Shape std deviation: {result.get('shape_std_db', float('nan')):.3f} dB")
    print(f"  Global offset       : {result.get('global_offset_db', float('nan')):.3f} dB")


def deviation_report(measured, freqs, label, send_correction_applied=False, receive_correction_applied=None):
    values=np.asarray(measured,dtype=float); f=np.asarray(freqs,dtype=float)
    valid=np.isfinite(values)
    print(f"\n{'='*56}\n  {label} -- Frequency Deviation Report\n{'='*56}")
    print(f"  Send-correction applied : {'Yes' if send_correction_applied else 'No'}")
    if receive_correction_applied is not None:
        print(f"  Recv-correction applied : {'Yes' if receive_correction_applied else 'No'}")
    if np.sum(valid)<2:
        print("  Insufficient valid data"); return
    vals=values[valid]
    print(f"  Valid bins              : {len(vals)}/{len(values)}")
    print(f"  Mean                     : {np.mean(vals):.3f} dB")
    print(f"  Std deviation            : {np.std(vals):.3f} dB")
    print(f"  Span                     : {np.ptp(vals):.3f} dB")
