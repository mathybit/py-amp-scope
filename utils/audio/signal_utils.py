"""Signal generation for PyAmpScope.

All paths share the same level semantics:

* ``tone_amplitude`` is sine *peak* amplitude in normalized digital full scale.
* ``send_gain`` is one additional percentage multiplier.
* noise is normalized to the RMS of that requested sine.
* frequency-dependent send correction is applied before playback.
* if correction/noise crest factor would exceed the configured peak headroom,
  the whole waveform is uniformly scaled, preserving spectral shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from typing import Iterable, Optional

from .levels import db20, global_peak_scale, peak, requested_sine_levels, rms


@dataclass
class SignalBundle:
    samples: np.ndarray
    reference_samples: np.ndarray
    fs: int
    requested_rms: float
    actual_rms: float
    actual_peak: float
    headroom_scale: float
    send_correction_applied: bool
    metadata: dict


def geometric_frequencies(freq_min: float, freq_max: float, count: int) -> np.ndarray:
    if freq_min <= 0 or freq_max <= freq_min:
        raise ValueError("frequency range must satisfy 0 < freq_min < freq_max")
    if count < 2:
        raise ValueError("frequency count must be at least 2")
    return np.geomspace(float(freq_min), float(freq_max), int(count), dtype=np.float64)


def interpolate_correction(
            target_freqs: np.ndarray,
            correction_freqs: Optional[np.ndarray],
            correction_factors: Optional[np.ndarray],
        ) -> np.ndarray:
    """Interpolate linear correction factors on a logarithmic frequency axis."""
    target = np.asarray(target_freqs, dtype=np.float64)
    if correction_freqs is None or correction_factors is None:
        return np.ones_like(target)
    cf = np.asarray(correction_freqs, dtype=np.float64)
    factors = np.asarray(correction_factors, dtype=np.float64)
    valid = np.isfinite(cf) & np.isfinite(factors) & (cf > 0) & (factors > 0)
    count = int(np.sum(valid))
    if count == 0:
        return np.ones_like(target)
    if count == 1:
        return np.full_like(target, float(factors[valid][0]))
    order = np.argsort(cf[valid])
    cf = cf[valid][order]
    factors = factors[valid][order]
    # Edge values are held constant outside the calibrated interval.
    return np.interp(np.log10(np.maximum(target, cf[0])), np.log10(cf), factors,
                     left=factors[0], right=factors[-1])


def _colored_noise(method: str, n_samples: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    method = method.lower()
    if method not in {"white", "pink", "brown"}:
        raise ValueError(f"unsupported noise method: {method}")

    white = rng.standard_normal(int(n_samples)).astype(np.float64)
    if method == "white":
        return white - np.mean(white)

    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white), 1.0 / fs)
    shape = np.zeros_like(freqs, dtype=np.float64)
    positive = freqs > 0
    exponent = -0.5 if method == "pink" else -1.0
    shape[positive] = np.power(freqs[positive], exponent)
    spectrum *= shape
    spectrum[0] = 0.0
    noise = np.fft.irfft(spectrum, n=len(white))
    return noise - np.mean(noise)


def apply_frequency_domain_correction(
            signal: np.ndarray,
            fs: int,
            correction_freqs: np.ndarray,
            correction_factors: np.ndarray,
        ) -> np.ndarray:
    """Apply a linear magnitude correction curve to a broadband signal.

    Correction is performed in the frequency domain because broadband noise
    contains all frequencies at the same time.  A time-domain amplitude envelope
    would not be a frequency correction.
    """
    x = np.asarray(signal, dtype=np.float64)
    spec = np.fft.rfft(x)
    fft_freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    factors = np.ones_like(fft_freqs)
    positive = fft_freqs > 0
    factors[positive] = interpolate_correction(
        fft_freqs[positive], correction_freqs, correction_factors
    )
    spec *= factors
    spec[0] = 0.0
    return np.fft.irfft(spec, n=len(x))


def build_noise_signal(
            method: str,
            n_samples: int,
            fs: int,
            tone_amplitude: float,
            send_gain: float,
            *,
            correction_freqs: Optional[np.ndarray] = None,
            correction_factors: Optional[np.ndarray] = None,
            peak_headroom: float = 0.95,
            seed: Optional[int] = 12345,
        ) -> SignalBundle:
    levels = requested_sine_levels(tone_amplitude, send_gain)
    target_rms = levels["requested_rms"]
    rng = np.random.default_rng(seed)
    base = _colored_noise(method, int(n_samples), int(fs), rng)
    base_rms = rms(base)
    if base_rms <= 0:
        raise RuntimeError("generated noise has zero RMS")
    base = base * (target_rms / base_rms)

    corrected = base.copy()
    correction_applied = correction_freqs is not None and correction_factors is not None
    if correction_applied:
        corrected = apply_frequency_domain_correction(
            corrected, fs, np.asarray(correction_freqs), np.asarray(correction_factors)
        )

    corrected, scale = global_peak_scale(corrected, peak_headroom)
    # Reference is the intended source spectrum prior to frequency pre-emphasis,
    # but receives the same uniform headroom scaling.  This makes normalized
    # response represent chain coloration, not the deliberate pre-emphasis.
    reference = base * scale

    return SignalBundle(
        samples=corrected.astype(np.float64),
        reference_samples=reference.astype(np.float64),
        fs=int(fs),
        requested_rms=float(target_rms),
        actual_rms=rms(corrected),
        actual_peak=peak(corrected),
        headroom_scale=float(scale),
        send_correction_applied=bool(correction_applied),
        metadata={
            **levels,
            "method": method.lower(),
            "actual_rms": rms(corrected),
            "actual_rms_dbfs": db20(rms(corrected)),
            "actual_peak": peak(corrected),
            "actual_peak_dbfs": db20(peak(corrected)),
            "headroom_scale": float(scale),
        },
    )


def build_sweep_signal(
            freq_array: Iterable[float],
            fs: int,
            tone_duration: float,
            gap_s: float,
            tone_amplitude: float,
            send_gain: float,
            *,
            correction_freqs: Optional[np.ndarray] = None,
            correction_factors: Optional[np.ndarray] = None,
            peak_headroom: float = 0.95,
            pre_roll_s: float = 0.25,
            post_roll_s: float = 0.25,
            fade_s: float = 0.005,
        ) -> SignalBundle:
    freqs = np.asarray(list(freq_array), dtype=np.float64)
    if freqs.size == 0:
        raise ValueError("sweep requires at least one frequency")
    if np.any(freqs <= 0) or np.any(freqs >= fs / 2):
        raise ValueError("all sweep frequencies must be between 0 and Nyquist")

    levels = requested_sine_levels(tone_amplitude, send_gain)
    base_peak = levels["requested_peak"]
    corr = interpolate_correction(freqs, correction_freqs, correction_factors)
    correction_applied = correction_freqs is not None and correction_factors is not None
    corrected_peaks = base_peak * corr
    max_peak = float(np.max(np.abs(corrected_peaks))) if corrected_peaks.size else 0.0
    global_scale = 1.0 if max_peak <= peak_headroom or max_peak == 0 else peak_headroom / max_peak
    corrected_peaks *= global_scale
    reference_peak = base_peak * global_scale

    n_tone = max(1, int(round(float(tone_duration) * fs)))
    n_gap = max(0, int(round(float(gap_s) * fs)))
    n_pre = max(0, int(round(float(pre_roll_s) * fs)))
    n_post = max(0, int(round(float(post_roll_s) * fs)))
    total = n_pre + len(freqs) * n_tone + max(0, len(freqs) - 1) * n_gap + n_post
    out = np.zeros(total, dtype=np.float64)
    ref = np.zeros(total, dtype=np.float64)
    starts: list[int] = []

    fade_n = min(int(round(fade_s * fs)), n_tone // 4)
    envelope = np.ones(n_tone, dtype=np.float64)
    if fade_n > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_n))
        envelope[:fade_n] = ramp
        envelope[-fade_n:] = ramp[::-1]

    offset = n_pre
    t = np.arange(n_tone, dtype=np.float64) / fs
    for i, f in enumerate(freqs):
        starts.append(offset)
        wave = np.sin(2.0 * np.pi * f * t) * envelope
        out[offset:offset+n_tone] = wave * corrected_peaks[i]
        ref[offset:offset+n_tone] = wave * reference_peak
        offset += n_tone
        if i < len(freqs) - 1:
            offset += n_gap

    sent_rms_per_tone = np.abs(corrected_peaks) / math.sqrt(2.0)
    reference_rms_per_tone = np.full_like(freqs, abs(reference_peak) / math.sqrt(2.0))

    return SignalBundle(
        samples=out,
        reference_samples=ref,
        fs=int(fs),
        requested_rms=float(levels["requested_rms"]),
        actual_rms=rms(out),
        actual_peak=peak(out),
        headroom_scale=float(global_scale),
        send_correction_applied=bool(correction_applied),
        metadata={
            **levels,
            "frequencies": freqs,
            "tone_starts": np.asarray(starts, dtype=np.int64),
            "tone_samples": n_tone,
            "gap_samples": n_gap,
            "pre_roll_samples": n_pre,
            "post_roll_samples": n_post,
            "sent_peak_per_tone": corrected_peaks,
            "sent_rms_per_tone": sent_rms_per_tone,
            "reference_rms_per_tone": reference_rms_per_tone,
            "actual_tone_rms_mean": float(np.sqrt(np.mean(sent_rms_per_tone ** 2))),
            "actual_tone_rms_min": float(np.min(sent_rms_per_tone)),
            "actual_tone_rms_max": float(np.max(sent_rms_per_tone)),
            "actual_rms": rms(out),
            "actual_rms_dbfs": db20(rms(out)),
            "actual_peak": peak(out),
            "actual_peak_dbfs": db20(peak(out)),
            "headroom_scale": float(global_scale),
        },
    )


# ---------------------------------------------------------------------------
# Compatibility wrappers used by older scripts.
# ---------------------------------------------------------------------------
def generate_noise_signal(method, n_samples, fs, tone_amplitude, send_gain):
    return build_noise_signal(method, n_samples, fs, tone_amplitude, send_gain).samples


def generate_sweep_sequence(freq_array, fs, duration_s=0.7, gap_s=0.2,
                            tone_amplitude=1.0, send_gain=100.0):
    return build_sweep_signal(
        freq_array, fs, duration_s, gap_s, tone_amplitude, send_gain,
        pre_roll_s=0.0, post_roll_s=0.0,
    ).samples


def print_freq_table(freqs, fs, mode, tone_duration=1.0, gap_s=0.3):
    print(f"\nFrequency table ({mode}, fs={fs} Hz):")
    for i, f in enumerate(freqs, 1):
        print(f"  {i:3d}: {float(f):9.2f} Hz")
    print(f"Tone duration={tone_duration:.3f}s, gap={gap_s:.3f}s")


def play_one_freq_single(freqs, duration_s, fs, gap_s, send_device, recv_device,
                         send_gain, tone_amplitude, capture_data, corr_factors=None,
                         verbose=False):
    """Compatibility wrapper around the shared sweep measurement engine."""
    from .streaming import run_sweep_measurement
    corr_freqs = np.asarray(freqs, dtype=float) if corr_factors is not None else None
    result = run_sweep_measurement(
        freqs=np.asarray(freqs, dtype=float), fs=int(fs), tone_duration=float(duration_s),
        gap_s=float(gap_s), tone_amplitude=float(tone_amplitude), send_gain=float(send_gain),
        send_device=send_device, recv_device=recv_device,
        send_correction_freqs=corr_freqs,
        send_correction_factors=np.asarray(corr_factors, dtype=float) if corr_factors is not None else None,
    )
    n = min(len(capture_data), len(result.capture))
    capture_data[:n] = result.capture[:n]
    return result.capture


def play_one_freq_seq(freq, duration_s, fs, send_device, recv_device, send_gain,
                      tone_amplitude, corr_factor=None):
    """Compatibility wrapper; all sweep measurements now use one shared capture path."""
    factors = None if corr_factor is None else np.asarray([corr_factor], dtype=float)
    return play_one_freq_single(
        [freq], duration_s, fs, 0.0, send_device, recv_device, send_gain,
        tone_amplitude, np.empty(int((duration_s + 0.5) * fs), dtype=np.float32),
        factors, False,
    )
