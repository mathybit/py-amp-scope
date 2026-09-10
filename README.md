# PyAmpScope

PyAmpScope is a Python tool for measuring and comparing audio amplifier / preamplifier behavior with an audio interface and an (optional but recommended) passive measurement/DI box.

For an in-depth overview of the DI box design, calibration math, and harmonic analysis theory, see the companion article: [PyAmpScope: Profiling Preamps #1](https://mathybit.github.io/pyampscope-1/).




## Prerequisites

#### Create and activate the environment

```bash
conda env create -f conda/environment.yml
conda activate foundry
```

This creates the `audio` environment with Python 3.12. Then install the pip dependencies:

#### Install dependencies

```bash
pip install -r conda/requirements.txt
```

This installs the core utility packages necessary to run the project.




## Quick Start

1. Edit `config/config.py` to set your audio interface device indices, gain levels, and sample rate. You may use the `list_audio_devices.py` script to verify your send / receive devices (you may need to input some sort of signal to the receive device).

2. Run a send calibration first (using a send calibration is optional, but the file must exist before receive calibration can use it):

```bash
python calibrate_send.py
```

3. Run the receive calibration:

```bash
python calibrate_recv.py
```

4. Validate both paths to check flatness after correction:

```bash
python validate_send_calibration.py --correct-send
python validate_recv_calibration.py --correct-recv
```

5. Run PyAmpScope

```bash
python run.py
```



## Calibration

The tool can be calibrated to account for the inherent frequency response of the measurement equipment (audio interface, DI box, etc.). Calibration generates correction files specific to your equipment.

Calibration/validation entry points:

```text
calibrate_send.py
calibrate_recv.py
validate_send_calibration.py
validate_recv_calibration.py
```

Examples:

```bash
python calibrate_send.py
python calibrate_recv.py --correct-send  # Applies send correction during receive calibration
```

There are additional validation scripts provided to check the measurement equipment response curve after calibration. These allow you to check the response curves after correction is applied. By default, the scripts do not apply any correction.

```bash
python validate_send_calibration.py --correct-send
python validate_recv_calibration.py --correct-recv
python validate_recv_calibration.py --correct-send --correct-recv
```



### Recommended calibration order

Calibration files are produced in a specific dependency chain. Follow this order:

1. `calibrate_send.py` — generates `data/cal_send_corrections.npz`
2. `calibrate_recv.py — generates `data/cal_recv_dir_base_corrections.npz`
4. Validation scripts (see below)

The validation scripts load correction profiles from the files produced above; if a profile does not exist, the script will error.


### Signal type

Currently only sweep mode is supported (the default). Sweep generates a log-spaced tone sequence (one tone per frequency bin, 300 by default from config.py). A `--method` flag exists for future noise-mode support but is not yet functional.

All four scripts also support `--dry-run`, which shows the configuration and frequency table without sending any audio — useful for verifying settings before running hardware tests.


### Output files

Each script writes three outputs:

| File type | Example name | Description |
|---|---|---|
| `*_corrections.npz` | `cal_send_corrections.npz` | Correction factors (loaded by validation scripts during verification) |
| `*_chart.png` | `cal_send_chart.png` | Frequency-response plot |
| `*_captured.wav` | `cal_send_captured.wav` | Raw captured waveform (float32 PCM) |

Calibration files go to `data/`. Validation files go to `logs/`.


### Validation label suffixes

Validation output filenames use compound labels based on which corrections are active:

| Script | Label | Meaning | Example filename |
|---|---|---|---|
| send validation | `sb` | Send baseline (no correction) | `validate_send_sb_chart.png` |
| send validation | `corr` | Send with correction applied | `validate_send_corr_chart.png` |
| receive validation | `sbrb`, `sbrc`, `scrb`, `scrc` | Baseline/corrected prefix for both send (s) and receive (r) | `validate_recv_dir_sbrc_chart.png` |



## GUI

You can run PyAmpScope:

```bash
python run.py
```

The GUI saves one `.png` chart and one `.json` metrics file with the same basename. Either a `.png` or `.json` name can be entered in the Save dialog. Sweep JSON includes the individual per-frequency measurements and harmonic data.



## Measurement

PyAmpScope uses one set of level definitions in the GUI, calibration scripts, and validation scripts:

- **Tone Amplitude** is sine **peak amplitude** in normalized digital full scale. `0.5` means samples range from `-0.5` to `+0.5` before Send Gain and optional correction.
- **Send Gain** is an additional amplitude scalar (percentage).
- The requested sine RMS is `ToneAmplitude * SendGain / 100 / sqrt(2)`.
- `dBFS` uses `20*log10(amplitude_or_RMS)`. Thus a full-scale sine with peak `1.0` has an RMS level of about `-3.01 dBFS`.
- Frequency-response plots are **relative**: overall constant gain or attenuation is removed by subtracting the median transfer gain. A chain that changes every frequency by the same factor therefore plots as a flat `0 dB` uniform response.



## Analysis

One generated waveform contains the complete log-spaced sweep, with silence between tones. The same capture engine is used by the GUI, calibration, and validation programs.

Each tone is analyzed separately. The code jointly fits the fundamental plus H2-H10 with sine/cosine least squares, which avoids single-FFT-bin leakage. It reports:

- received RMS and dBFS for every tone;
- normalized frequency response;
- H2-H10 where those harmonics are below Nyquist;
- THD per tone;
- even-harmonic and odd-harmonic distortion;
- even/odd harmonic ratio;
- per-band averages;
- overall THD as the arithmetic mean of valid per-tone THD values.



## Frequency bands

The canonical band definitions live in `config/config.py`:

- Sub-bass: 20-60 Hz
- Bass: 60-250 Hz
- Low-mid: 250-500 Hz
- Mid: 500 Hz-2 kHz
- Upper-mid: 2-4 kHz
- Presence: 4-6 kHz
- Brilliance: 6-20 kHz



## Using Corrections

Corrections are always **optional and off by default**. To use this feature, you must run the calibration scripts first, which generates the necessary correction files.

### Receive correction

- Receive correction is applied after capture. Sweep harmonic components are corrected at their own harmonic frequencies, not only at the fundamental.
- While optional, my experiments show that using **receive correction is beneficial**.

### Send correction

- Send correction is applied before playback. For broadband noise it is applied in the frequency domain, not as a time-domain volume envelope.
- Send correction is not arbitrarily capped. If pre-emphasis would exceed digital headroom, the entire waveform is uniformly scaled so the correction shape is preserved without DAC clipping.
- Using send correction is **not recommended** as it can unintentionally lead an amplifier into clipping.




## Verification

The repository includes deterministic DSP tests that do not require audio hardware:

```bash
pytest -q
```

Hardware behavior (actual USB levels, analog volume knob, DI attenuation, transformer behavior, ADC clipping point, and device latency) still needs to be validated on the physical setup.
