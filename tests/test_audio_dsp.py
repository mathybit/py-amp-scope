import math
from pathlib import Path

import numpy as np

from config import config as cfg
from utils.audio.levels import db10, db20, requested_sine_levels, rms
from utils.audio.signal_utils import (
    build_noise_signal, build_sweep_signal, geometric_frequencies,
    interpolate_correction, apply_frequency_domain_correction,
)
from utils.audio.analysis_utils import (
    analyze_noise_measurement, analyze_sweep_measurement,
    measure_harmonic_components_rms,
)
from utils.audio.calibration import derive_inverse_correction


def _synthetic_sweep(freqs, fs=44100, amp=0.2, gain=100, tone_duration=0.35, gap=0.08):
    return build_sweep_signal(freqs, fs, tone_duration, gap, amp, gain, pre_roll_s=0.25, post_roll_s=0.25)


def test_db_math_is_base10():
    assert abs(db20(0.5) + 6.020599913) < 1e-8
    assert abs(db10(0.5) + 3.010299956) < 1e-8
    assert abs(db20(1 / math.sqrt(2)) + 3.010299956) < 1e-8


def test_requested_sine_level_semantics():
    x = requested_sine_levels(0.5, 80)
    assert abs(x["requested_peak"] - 0.4) < 1e-12
    assert abs(x["requested_rms"] - 0.4 / math.sqrt(2)) < 1e-12


def test_sweep_send_gain_applied_once():
    b = _synthetic_sweep([1000], amp=0.5, gain=80)
    assert abs(b.metadata["sent_peak_per_tone"][0] - 0.4) < 1e-10
    assert abs(b.metadata["sent_rms_per_tone"][0] - 0.4 / math.sqrt(2)) < 1e-10


def test_sweep_correction_uses_global_headroom_without_limiting_curve():
    freqs = np.array([100.0, 1000.0])
    b = build_sweep_signal(freqs, 44100, 0.2, 0.05, 0.5, 80,
                           correction_freqs=freqs, correction_factors=np.array([1.0, 3.0]),
                           peak_headroom=0.95)
    assert np.max(np.abs(b.samples)) <= 0.9500001
    assert b.headroom_scale < 1.0
    ratio = b.metadata["sent_peak_per_tone"][1] / b.metadata["sent_peak_per_tone"][0]
    assert abs(ratio - 3.0) < 1e-10


def test_noise_colors_match_requested_rms_when_headroom_not_active():
    target = requested_sine_levels(0.1, 50)["requested_rms"]
    for method in ("white", "pink", "brown"):
        b = build_noise_signal(method, 44100 * 2, 44100, 0.1, 50, peak_headroom=0.95, seed=7)
        assert abs(b.actual_rms - target) / target < 1e-10
        assert np.max(np.abs(b.samples)) < 0.95


def test_noise_reports_actual_lower_rms_when_headroom_needed():
    b = build_noise_signal("white", 44100 * 2, 44100, 0.9, 100, peak_headroom=0.95, seed=1)
    assert b.requested_rms > b.actual_rms
    assert np.max(np.abs(b.samples)) <= 0.9500001
    assert b.headroom_scale < 1.0


def test_correction_interpolation_is_log_frequency():
    cf = np.array([100.0, 1000.0, 10000.0])
    factors = np.array([1.0, 2.0, 3.0])
    target = np.array([math.sqrt(100 * 1000), math.sqrt(1000 * 10000)])
    got = interpolate_correction(target, cf, factors)
    assert np.allclose(got, [1.5, 2.5], atol=1e-12)


def test_joint_harmonic_fit_recovers_known_components():
    fs = 44100; f = 997.0; n = int(0.37 * fs)
    t = np.arange(n) / fs
    x = 0.4*np.sin(2*np.pi*f*t + 0.2) + 0.008*np.sin(2*np.pi*2*f*t + 0.7) + 0.004*np.sin(2*np.pi*3*f*t - 0.4)
    comp = measure_harmonic_components_rms(x, f, fs, 5)
    assert abs(comp[2] / comp[1] - 0.02) < 2e-5
    assert abs(comp[3] / comp[1] - 0.01) < 2e-5


def test_sweep_thd_and_even_odd_are_per_tone():
    fs = 44100; freqs = np.array([200.0, 500.0, 1000.0, 3000.0])
    b = _synthetic_sweep(freqs, fs=fs, amp=0.2, gain=100)
    capture = b.samples * 0.5
    # Add H2=2% and H3=1% relative to each captured fundamental.
    n_tone = b.metadata["tone_samples"]
    for f, start in zip(freqs, b.metadata["tone_starts"]):
        t = np.arange(n_tone) / fs
        capture[start:start+n_tone] += 0.5*0.2*0.02*np.sin(2*np.pi*2*f*t)
        capture[start:start+n_tone] += 0.5*0.2*0.01*np.sin(2*np.pi*3*f*t)
    m = analyze_sweep_measurement(capture, b.metadata, fs)
    assert abs(m["thd_global"] - math.sqrt(0.02**2 + 0.01**2)*100) < 0.03
    assert abs(m["even_odd_ratio"] - 2.0) < 0.03
    assert all("harmonics" in r for r in m["per_tone"] if r["valid"])


def test_constant_gain_sweep_normalizes_to_zero_db():
    freqs = geometric_frequencies(80, 10000, 24)
    b = _synthetic_sweep(freqs)
    m = analyze_sweep_measurement(b.samples * 0.2, b.metadata, b.fs)
    valid = np.isfinite(m["relative_response_db"])
    assert np.max(np.abs(m["relative_response_db"][valid])) < 0.03


def test_receive_correction_flattens_frequency_dependent_sweep():
    fs=44100; freqs=np.array([100.0, 1000.0, 8000.0])
    b=_synthetic_sweep(freqs, fs=fs)
    cap=np.zeros_like(b.samples)
    gains=np.array([0.5,1.0,2.0])
    n=b.metadata["tone_samples"]
    # retain pre/post/gaps as zero; scale each tone independently
    for g,start in zip(gains,b.metadata["tone_starts"]):
        cap[start:start+n]=b.samples[start:start+n]*g
    raw=analyze_sweep_measurement(cap,b.metadata,fs)
    corr=analyze_sweep_measurement(cap,b.metadata,fs,recv_correction_freqs=freqs,recv_correction_factors=1/gains)
    assert np.nanstd(raw["relative_response_db"]) > 4.0
    assert np.nanstd(corr["relative_response_db"]) < 0.03


def test_constant_gain_noise_normalizes_to_zero_db_with_welch():
    fs=44100
    b=build_noise_signal("pink",fs*5,fs,0.08,60,peak_headroom=0.95,seed=9)
    target=geometric_frequencies(40,18000,64)
    m=analyze_noise_measurement(b.reference_samples*0.2,b.reference_samples,fs,target)
    valid=np.isfinite(m["relative_response_db"])
    assert np.nanstd(m["relative_response_db"][valid]) < 0.05
    assert abs(np.nanmedian(m["relative_response_db"][valid])) < 1e-8
    assert m["thd_global"] is None


def test_noise_band_levels_are_true_integrated_rms():
    fs=44100
    b=build_noise_signal("white",fs*6,fs,0.05,50,peak_headroom=0.95,seed=3)
    m=analyze_noise_measurement(b.samples,b.reference_samples,fs,geometric_frequencies(40,18000,48))
    assert "Bass" in m["frequency_bands"] and "Mid" in m["frequency_bands"]
    assert m["frequency_bands"]["Bass"]["received_rms"] > 0
    assert m["frequency_bands"]["Mid"]["received_dbfs"] < 0


def test_frequency_band_boundaries_are_standardized():
    assert cfg.frequency_bands == {
        "Sub-bass": (20.0, 60.0), "Bass": (60.0, 250.0), "Low-mid": (250.0, 500.0),
        "Mid": (500.0, 2000.0), "Upper-mid": (2000.0, 4000.0),
        "Presence": (4000.0, 6000.0), "Brilliance": (6000.0, 20000.0),
    }


def test_inverse_correction_has_no_overall_gain_component():
    f=np.array([100,1000,10000],float); response=np.array([-6.0,0.0,6.0])
    factors,smooth,ref=derive_inverse_correction(f,response,smoothing_window=1)
    assert abs(ref) < 1e-12
    assert np.allclose(factors,10**(-response/20.0))


def test_sweep_latency_compensation_handles_small_capture_delay():
    fs = 44100
    freqs = np.array([120.0, 500.0, 1800.0, 7000.0])
    b = _synthetic_sweep(freqs, fs=fs, amp=0.15, gain=100)
    delay = int(0.037 * fs)
    cap = np.concatenate([np.zeros(delay), b.samples * 0.4, np.zeros(delay)])
    m = analyze_sweep_measurement(cap, b.metadata, fs)
    valid = np.isfinite(m["relative_response_db"])
    assert np.sum(valid) == len(freqs)
    assert np.nanstd(m["relative_response_db"]) < 0.05
    assert abs(m["latency_ms"] - 37.0) < 30.0


def test_high_frequency_tone_reports_thd_unavailable_when_h2_above_nyquist():
    fs = 44100
    f = np.array([15000.0])
    b = _synthetic_sweep(f, fs=fs, amp=0.1, gain=100)
    m = analyze_sweep_measurement(b.samples * 0.5, b.metadata, fs)
    assert m["per_tone"][0]["thd_pct"] is None
    assert m["thd_global"] is None


def test_noise_receive_correction_flattens_known_chain_coloration():
    fs = 44100
    target = geometric_frequencies(80, 16000, 48)
    base = build_noise_signal("white", fs * 5, fs, 0.05, 50, peak_headroom=0.95, seed=12)
    cf = np.array([80.0, 1000.0, 16000.0])
    chain = np.array([0.55, 1.0, 1.7])
    received = apply_frequency_domain_correction(base.reference_samples, fs, cf, chain)
    raw = analyze_noise_measurement(received, base.reference_samples, fs, target)
    corrected = analyze_noise_measurement(received, base.reference_samples, fs, target,
                                          recv_correction_freqs=cf, recv_correction_factors=1.0/chain)
    assert np.nanstd(raw["relative_response_db"]) > 2.0
    assert np.nanstd(corrected["relative_response_db"]) < 0.25


def test_noise_send_precorrection_reference_excludes_deliberate_preemphasis():
    fs = 44100
    target = geometric_frequencies(80, 16000, 48)
    cf = np.array([80.0, 1000.0, 16000.0])
    chain = np.array([0.55, 1.0, 1.7])
    send_inverse = 1.0 / chain
    stimulus = build_noise_signal("white", fs * 5, fs, 0.04, 50,
                                  correction_freqs=cf, correction_factors=send_inverse,
                                  peak_headroom=0.95, seed=13)
    received = apply_frequency_domain_correction(stimulus.samples, fs, cf, chain)
    m = analyze_noise_measurement(received, stimulus.reference_samples, fs, target)
    assert np.nanstd(m["relative_response_db"]) < 0.25


def test_sweep_send_precorrection_reference_excludes_preemphasis():
    fs = 44100
    freqs = np.array([100.0, 1000.0, 8000.0])
    chain = np.array([0.5, 1.0, 2.0])
    b = build_sweep_signal(
        freqs, fs, 0.3, 0.05, 0.12, 100,
        correction_freqs=freqs, correction_factors=1.0 / chain,
        pre_roll_s=0.25, post_roll_s=0.25,
    )
    cap = np.zeros_like(b.samples)
    n = b.metadata["tone_samples"]
    for g, start in zip(chain, b.metadata["tone_starts"]):
        cap[start:start+n] = b.samples[start:start+n] * g
    m = analyze_sweep_measurement(cap, b.metadata, fs)
    assert np.nanstd(m["relative_response_db"]) < 0.03


def test_receive_correction_is_applied_at_harmonic_frequency_for_thd():
    fs = 44100
    f0 = 1000.0
    b = _synthetic_sweep([f0], fs=fs, amp=0.2, gain=100)
    n = b.metadata["tone_samples"]
    start = int(b.metadata["tone_starts"][0])
    t = np.arange(n) / fs
    cap = np.zeros_like(b.samples)
    # Receive path: fundamental gain=1, H2 gain=0.5. True amplifier H2 is 2%.
    cap[start:start+n] = 0.2*np.sin(2*np.pi*f0*t) + 0.2*0.02*0.5*np.sin(2*np.pi*2*f0*t)
    raw = analyze_sweep_measurement(cap, b.metadata, fs)
    corrected = analyze_sweep_measurement(
        cap, b.metadata, fs,
        recv_correction_freqs=np.array([f0, 2*f0, 10000.0]),
        recv_correction_factors=np.array([1.0, 2.0, 2.0]),
    )
    assert abs(raw["per_tone"][0]["thd_pct"] - 1.0) < 0.05
    assert abs(corrected["per_tone"][0]["thd_pct"] - 2.0) < 0.05
