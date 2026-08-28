import math
import numpy as np
from pathlib import Path
import sys


# Add repo root to path so we can import config directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import log_f


def smooth_moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving average over *window_size* total points (including center).

    Edge bins use whatever neighbors are available (no padding assumption).

    Example: window_size=5 means +/-2 from the current bin. If fewer than 2 exist on
    one side, only the available points are averaged.
    """
    half = window_size // 2
    result = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(i - half, 0)
        hi = min(i + half + 1, len(arr))
        result[i] = np.mean(arr[lo:hi])
    return result


def fft_db(sig, target_hz, fs):
    """
    FFT Analysis helper

    Get dBFS value at target frequency via FFT.

    Normalizes FFT magnitude by 2/N so a full-scale sine wave (A=1.0) reads +0 dBFS,
    matching the reference used by ``analyze_noise_response``.
    """
    N = len(sig)
    scale = 2.0 / N
    fft_vals = np.abs(np.fft.rfft(sig.astype(float))) * scale
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    idx = np.argmin(np.abs(freqs - target_hz))
    return 20 * log_f(max(fft_vals[idx], 1e-30))


def analyze_noise_response(captured_signal, freq_array, fs):
    """
    Analyze captured broadband noise to extract per-bin frequency response.

    Computes a single FFT of the full capture and extracts magnitude at each
    target frequency in freq_array. Only uses the valid (non-DC) portion of
    the FFT to avoid aliasing artifacts at Nyquist.

    Args:
        captured_signal: np.ndarray of float samples from hardware capture
        freq_array: np.ndarray of target frequencies (Hz) to extract
        fs: sample rate in Hz

    Returns:
        tuple (freqs_out, amp_db, rms) where freqs_out are the target frequencies
              that had valid measurements, amp_db is magnitude in dBFS at each target,
              and rms is RMS of a windowed segment around each target frequency.
    """
    N = len(captured_signal)
    fft_vals = np.abs(np.fft.rfft(captured_signal.astype(float)))
    freq_bins = np.fft.rfftfreq(N, d=1.0 / fs)

    freqs_out = []
    amp_db_list = []
    rms_list = []

    # Use only FFT bins below Nyquist for analysis (last bin is at Nyquist, ambiguous)
    valid_fft_mask = freq_bins < (fs / 2.0 * 0.999)
    fft_vals_valid = fft_vals[valid_fft_mask]
    freq_bins_valid = freq_bins[valid_fft_mask]

    # Scale: FFT magnitude of real signal -> peak amplitude at each bin
    scale = 2.0 / N

    for target_freq in freq_array:
        if target_freq >= fs / 2.0:
            continue
        idx = np.argmin(np.abs(freq_bins_valid - target_freq))
        mag = fft_vals_valid[idx] * scale
        db_val = 20.0 * log_f(max(mag, 1e-30))

        # Compute RMS over a window centered on target frequency
        window_len = int(fs * 0.5)  # 0.5s window for RMS measurement
        seg_center = N // 2         # use middle of signal (stationary broadband noise)
        seg_start_sample = max(seg_center - window_len // 2, 0)
        seg_end_sample = min(seg_center + window_len // 2, N)
        seg = captured_signal[seg_start_sample:seg_end_sample]
        seg_rms = float(np.sqrt(np.mean(seg.astype(float) ** 2))) if len(seg) > 0 else 0.0

        freqs_out.append(target_freq)
        amp_db_list.append(db_val)
        rms_list.append(seg_rms)

    return (np.array(freqs_out), np.array(amp_db_list), np.array(rms_list))


def compute_correction(H_lin):
    """
    Compute regularized inverse correction filter from linear magnitude response.

    Returns W (complex frequency-domain, unit-magnitude phase-flipped) — not used for
    direct waveform scaling here but available if we want to inspect it later.
    """
    tol = 1e-3
    H_mag = np.maximum(np.abs(H_lin), tol)
    return np.conj(H_lin) / H_mag


def extract_tone_measurements(rec, freqs, tone_duration_s, gap_s, fs):
    """Extract per-tone dBFS measurements and RMS from raw capture data.

    Splits ``rec`` into per-frequency segments using hop = (tone_duration + gap) * fs,
    runs FFT on each segment, returns measured dBFS and RMS arrays.

    Returns:
        measured: np.ndarray of dBFS at each target frequency (NaN for short/missing segments)
        rms_list: np.ndarray of RMS values per bin
    """
    hop = int((tone_duration_s + gap_s) * fs)
    n_bins = len(freqs)
    measured = np.full(n_bins, float("nan"))
    rms_list = np.full(n_bins, 0.0)

    for i in range(n_bins):
        seg_start = i * hop
        seg_end = min(seg_start + int(tone_duration_s * fs), len(rec))
        seg = rec[seg_start:seg_end]
        if len(seg) < 64:
            continue
        rms_list[i] = float(np.sqrt(np.mean(seg ** 2)))
        measured[i] = fft_db(seg, freqs[i], fs)

    return measured, rms_list


def deviation_report(measured, freqs, label, send_correction_applied=False, receive_correction_applied=None):
    """Print frequency deviation report for a single measurement set."""
    valid = measured[~np.isnan(measured)]
    if len(valid) < 2:
        print(f"\n{label} -- insufficient data")
        return

    mean_db = float(np.mean(valid))
    std_db = float(np.std(valid))
    min_db = float(np.min(valid))
    max_db = float(np.max(valid))

    # Convert dBFS to linear magnitude for percent-deviation calculation
    lin = 10 ** (valid / 20.0)
    arith_mean = float(np.mean(lin))

    abs_pct_dev = np.abs((lin - arith_mean) / max(arith_mean, 1e-30) * 100.0)

    print(f"\n{'=' * 56}")
    print(f"  {label} -- Frequency Deviation Report")
    print(f"{'=' * 56}")
    print(f"  Send-correction applied  : {'Yes' if send_correction_applied else 'No'}")
    if receive_correction_applied is not None:
        print(f"  Recv-correction applied  : {'Yes' if receive_correction_applied else 'No'}")
    print(f"  Valid bins               : {len(valid)}/{len(measured)}")
    print()
    print(f"  Amplitude stats:")
    print(f"    Mean (dBFS)       : {mean_db:.2f} dBFS")
    print(f"    Std deviation     : {std_db:.3f} dB")
    print(f"    Range             : {min_db:.2f} - {max_db:.2f} dB (span={max_db-min_db:.2f} dB)")
    print()
    print(f"  Deviation from arithmetic mean (linear magnitude):")
    print(f"    Mean abs % dev    : {float(np.mean(abs_pct_dev)):.3f}%")
    print(f"    Median abs % dev  : {float(np.median(abs_pct_dev)):.3f}%")
    print(f"    Max abs % dev     : {float(np.max(abs_pct_dev)):.3f}%")
    print(f"    Std of abs % dev  : {float(np.std(abs_pct_dev)):.3f}%")

    # Octave-band breakdown
    print(f"\n  Octave band std deviation (linear %):")
    octaves = [(20, 100, "sub-bass"), (100, 300, "bass"), (300, 800, "low-mid"),
               (800, 2000, "mid"), (2000, 5000, "upper-mid"), (5000, 10000, "presence"),
               (10000, 20000, "brilliance")]
    for lo, hi, name in octaves:
        mask = (freqs >= lo) & (freqs < hi)
        if np.sum(mask) > 0:
            seg_lin = lin[mask]
            s_std_pct = float(np.std(seg_lin / arith_mean * 100.0))
            print(f"    {name:>14} {lo:>5}-{hi:>6} Hz: std_dev%={s_std_pct:.3f}% bins={np.sum(mask)}")

    # Worst offenders
    sorted_idx = np.argsort(-abs_pct_dev)
    print(f"\n  Top 5 worst bins:")
    for rank in range(min(5, len(sorted_idx))):
        j = sorted_idx[rank]
        if np.isnan(measured[j]):
            continue
        print(f"    #{rank + 1}  {freqs[j]:>8.0f} Hz  =>  "
              f"{measured[j]:>7.2f} dBFS  abs_dev={abs_pct_dev[j]:.3f}%")


# ---------------------------------------------------------------------------
# Noise spectral shape analysis
# ---------------------------------------------------------------------------

def compare_noise_spectral_shape(captured_signal, noise_method, freq_array, fs, smooth_window=5):
    """Compare measured per-bin response against the theoretical spectral density of input noise.

    Answers: does the receive chain preserve the expected pink/brown/white shape within
    an acceptable tolerance, and what is the net global offset the chain introduces?

    Smoothing removes per-bin FFT quantization noise so we measure overall trend tracking,
    not bin-to-bin artifacts.

    Args:
        captured_signal: raw hardware capture (float np.ndarray)
        noise_method: 'white', 'pink', or 'brown' -- matches generate_noise_signal method
        freq_array: target frequencies in Hz (same as used during calibration)
        fs: sample rate in Hz
        smooth_window: moving average window for smoothing measured dB before comparison

    Returns:
        dict with keys: freqs, measured_db, smoothed_db, expected_db (original),
                       expected_shifted_db (least-squares shifted), deviation_db (percent diff
                       between shifted theory and smoothed measurement), shape_std_pct,
                       octave_band_deviation
    """
    # Step 1: get raw per-bin dBFS
    _, measured_db, _ = analyze_noise_response(captured_signal, freq_array, fs)

    # Step 2: smooth with the same window as correction computation
    valid_mask = ~np.isnan(measured_db)
    if np.sum(valid_mask) < 3:
        return {"error": "insufficient valid bins for shape analysis"}

    smoothed_db = smooth_moving_average(measured_db, window_size=smooth_window)

    # Step 3: generate theoretical input profile in dB relative to f_ref
    f_ref = freq_array[0]
    expected_db = np.zeros_like(freq_array)
    if noise_method == "pink":
        # -3.0103 dB per octave ~= amplitude |f|^(-0.5) power |f|^(-1)
        mask = freq_array > 0
        expected_db[mask] = -3.0103 * log_f(freq_array[mask] / f_ref)
    elif noise_method == "brown":
        # -6.0206 dB per octave ~= amplitude |f|^(-1.0) power |f|^(-2)
        mask = freq_array > 0
        expected_db[mask] = -6.0206 * log_f(freq_array[mask] / f_ref)
    # white: all zeros (flat)

    # Step 4: least-squares shift of theoretical reference to match measured level
    # Find scalar `shift_db` that minimizes ||smoothed_db - (expected_db + shift_db)||^2
    # Closed-form: shift_db = mean(smoothed_db - expected_db) over valid bins
    valid_for_shift = valid_mask & (freq_array > 0)
    if np.sum(valid_for_shift) < 3:
        return {"error": "insufficient valid bins for shape analysis"}

    diff_all = smoothed_db[valid_for_shift] - expected_db[valid_for_shift]
    shift_db = float(np.mean(diff_all))           # least-squares optimal match
    expected_shifted_db = expected_db + shift_db   # shifted reference

    # Step 5: deviations as percent differences (consistent with sweep/white)
    shifted_lin = 10 ** (expected_shifted_db / 20.0)
    measured_lin = 10 ** (smoothed_db / 20.0)
    deviation_pct = np.abs((measured_lin - shifted_lin) / np.maximum(shifted_lin, 1e-30) * 100.0)

    # Step 6: shape quality as std of percent deviation
    shape_std_pct = float(np.std(deviation_pct[valid_for_shift]))

    # Step 7: octave-band breakdown of percent deviation (not dB)
    octaves = [(20, 100, "sub-bass"), (100, 300, "bass"), (300, 800, "low-mid"),
               (800, 2000, "mid"), (2000, 5000, "upper-mid"), (5000, 10000, "presence"),
               (10000, 20000, "brilliance")]
    octave_band_deviation = {}
    for lo, hi, name in octaves:
        mask = valid_mask & (freq_array >= lo) & (freq_array < hi)
        if np.sum(mask) > 0:
            octave_band_deviation[name] = float(np.mean(deviation_pct[mask]))

    # Keep old-style dB deviation for backward compat with downstream code that expects it
    deviation_db = smoothed_db - expected_shifted_db  # dB from shifted reference
    global_offset_db = shift_db                        # net chain gain/loss

    return {
        "freqs": freq_array[valid_mask],
        "measured_db": measured_db,
        "smoothed_db": smoothed_db,
        "expected_db": expected_db,         # original (un-shifted) theoretical profile
        "expected_shifted_db": expected_shifted_db,  # least-squares shifted reference
        "deviation_db": deviation_db,       # dB from shifted reference
        "deviation_pct": deviation_pct,     # percent diff between shifted theory and measurement
        "shift_db": shift_db,               # LSB that minimizes ||smoothed - (expected + shift)||^2
        "shape_std_pct": shape_std_pct,     # std of percent deviation (lower = better)
        "octave_band_deviation": octave_band_deviation,
    }


def print_noise_shape_report(result, method):
    """Print a human-readable noise spectral shape analysis report."""
    if "error" in result:
        print(f"\n  [Noise shape analysis skipped: {result['error']}]")
        return

    method_labels = {"white": "White", "pink": "Pink", "brown": "Brown"}
    method_slope = {"white": "0 dB/oct (flat)", "pink": "-3.01 dB/oct", "brown": "-6.02 dB/oct"}

    # Use the new percent-based fields; fall back to old-style keys if available
    shape_std = result.get("shape_std_pct", result.get("shape_std_db", 0))
    pct_unit = "%" if "shape_std_pct" in result else "dB"

    print(f"\n{'=' * 56}")
    print(f"  Noise Spectral Shape Analysis ({method_labels.get(method, method).upper()})")
    print(f"{'=' * 56}")
    print(f"  Input method        : {method} ({method_slope[method]})")
    print(f"  Theory shift (LSB)  : +{result.get('shift_db', result.get('global_offset_db', 0)):.1f} dB (best-fit match to measurement)")
    print(f"  Valid bins          : {len(result['freqs'])}/{len(result['expected_db'])}")
    print(f"  Global offset       : {result.get('global_offset_db', result.get('shift_db', 0)):+.1f} dB (net chain gain/loss)")
    print(f"  Shape std dev       : {shape_std:.2f} {pct_unit}")

    shape_quality = "excellent" if shape_std < 5.0 else \
                    "good" if shape_std < 10.0 else \
                    "fair" if shape_std < 15.0 else "poor"
    print(f"  Quality             : {shape_quality}")

    # Octave-band deviation from expected slope (percent)
    octaves = [("sub-bass", 20, 100), ("bass", 100, 300), ("low-mid", 300, 800),
               ("mid", 800, 2000), ("upper-mid", 2000, 5000), ("presence", 5000, 10000),
               ("brilliance", 10000, 20000)]

    warnings = []
    pct_unit_label = "%" if "shape_std_pct" in result else "dB"
    for name, lo, hi in octaves:
        val = result["octave_band_deviation"].get(name)
        if val is not None:
            warn = " **" if abs(val) > 15.0 else ""  # tighter threshold for percent
            print(f"    {name:>12} {lo:>5}-{hi:>6} Hz: {val:+7.1f}{pct_unit_label}{warn}")
            if abs(val) > 15.0:
                warnings.append((name, val))

    # Interpretation
    print(f"\n  Interpretation:")
    slope_desc = {"pink": "~3 dB per octave", "brown": "~6 dB per octave", "white": "flat"}
    print(f"    The chain preserves the {slope_desc[method]} input shape within "
          f"+/- {shape_std:.1f} {pct_unit_label} of the shifted theoretical reference "
          f"across most bands.")
    if result.get("global_offset_db", 0) < -30:
        print(f"    ** Global offset ({result['global_offset_db']:+.1f} dB) suggests significant")
        print(f"       chain attenuation -- check input/output gain staging.")
    elif result.get("global_offset_db", 0) > 30:
        print(f"    ** Global offset ({result['global_offset_db']:+.1f} dB) suggests significant")
        print(f"       chain amplification -- verify level settings.")

    if warnings:
        print(f"\n  Warnings:")
        for name, val in warnings:
            if val > 0:
                direction = "boosted relative to expected"
            else:
                direction = "attenuated relative to expected"
            print(f"    {name:>12}: {val:+7.1f}{pct_unit_label} -- chain {direction}")

    # Suggest using white noise for correction computation if pink/brown was used
    if method in ("pink", "brown"):
        print(f"\n  Note: '{method}' captures chain + input coloration. For measuring the")
        print(f"  chain alone (correction computation), use --method white.")
        print(f"  chain alone (correction computation), use --method white.")

