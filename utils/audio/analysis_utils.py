import math
import numpy as np


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

    Get dB value at target frequency via FFT.
    """
    N = len(sig)
    fft_vals = np.abs(np.fft.rfft(sig.astype(float)))
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    idx = np.argmin(np.abs(freqs - target_hz))
    return 20 * np.log10(max(fft_vals[idx], 1e-10))


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
        db_val = 20.0 * math.log10(max(mag, 1e-30))

        # Compute RMS over a window centered on target frequency
        lo_bin = max(0, idx - 2)
        hi_bin = min(len(freq_bins_valid), idx + 3)
        seg_start_sample = int(target_freq * N / fs) - 64
        seg_end_sample = seg_start_sample + 128
        seg_start_sample = max(0, seg_start_sample)
        seg_end_sample = min(N, seg_end_sample)
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

