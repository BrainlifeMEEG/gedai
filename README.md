# GEDAI -- unsupervised EEG denoising (continuous data)

> ⚠️ **Licence / patent notice.** This app wraps
> [GEDAI](https://github.com/neurotuning/gedai), which is licensed under the
> **PolyForm Noncommercial License 1.0.0** and whose core algorithm is
> **patent pending**. The app as a whole may be used for **non-commercial
> purposes only** (research, education, evaluation). Commercial or proprietary
> use requires a separate licence (including a patent grant) from the GEDAI
> authors (tomas.ros@unige.ch). See `license.txt`.

## Description

GEDAI ("Generalized Eigenvalue De-Artifacting Instrument") is an **unsupervised**
EEG denoising method based on **leadfield filtering**. It contrasts the data
covariance of short segments with a physics-based reference covariance derived
from a template head model / leadfield (bundled with GEDAI, a ~343-electrode
10-05 montage). A generalized eigendecomposition keeps the components aligned
with the brain subspace and projects out the rest. The rejection threshold is
selected automatically by GEDAI's **SENSAI** criterion -- no manual component
review, no training data, no per-subject MRI or forward model.

This app runs GEDAI on the **EEG channels** of a continuous MNE `.fif` recording.
Non-EEG channels (MEG, EOG, ECG, stim, ...) are passed through untouched.

Reference: Ros T., Ferat V., Huang Y., Colangelo C., Kia S.M., Wolfers T.,
Vulliemoz S., Michela A. *Return of the GEDAI: Unsupervised EEG Denoising based
on Leadfield Filtering.* bioRxiv 2025. doi:10.1101/2025.10.04.680449

## Inputs

- **mne** (`neuro/mne`): a continuous raw `.fif` file. Must contain EEG channels.

For best results the EEG channels should either carry **standard 10-05 names**
(`Fp1`, `Fz`, `Cz`, ...) or have **digitised positions**, and the data should be
**high-pass filtered** (the app can do this -- see `l_freq`).

## Outputs

- **out_dir/raw.fif** (`neuro/mne`): the denoised data.
- **out_dir/noise_raw.fif** (`neuro/mne`): the *removed* signal (EEG channels
  only) -- `input EEG = denoised EEG + noise`.
- **out_report/report.html**: QC report (SENSAI fit, subspace similarity, PSD
  before/after, ENOVA topomap, time-domain segment).
- **out_figs/*.png**, **product.json**.

> GEDAI applies an **average reference** internally, so the denoised EEG output
> is average-referenced. Chain a `rereference` app if you need another scheme.

## Configuration parameters

| key | type | default | description |
|---|---|---|---|
| `mode` | `broadband` \| `multiband` \| `adaptive` | `adaptive` | GEDAI variant. `adaptive` is the authors' recommended offline pipeline (slowest); `broadband` is the fastest. |
| `l_freq` | float \| null | `0.5` | High-pass (Hz) applied before fitting. Improves covariance estimation. `null` disables it. |
| `noise_multiplier` | float \| str | `3.0` | SENSAI noise weighting. Higher = keep more brain signal. A number > 0, or `standard` (3.0) / `aggressive` (1.5) / `conservative` (6.0). |
| `sensai_method` | `optimize` \| `gridsearch` | `optimize` | Threshold search strategy. |
| `duration` | float | `2.0` | Segment length (s) for covariance estimation (`broadband` / `multiband`). |
| `overlap` | float | `0.5` | Segment overlap fraction, in `[0, 1)`. |
| `reject_by_annotation` | bool | `false` | Exclude `bad`-annotated spans from the fit (GEDAI recommends keeping them). |
| `broadband_pass` | bool | `true` | `multiband` / `adaptive`: initial broadband pre-pass. |
| `wavelet_low_cutoff` | `auto` \| float \| null | `auto` | `multiband` / `adaptive`: wavelet bands below this (Hz) are zeroed. `auto` uses the data high-pass. |
| `cycles_per_wavelet` | int | `10` | `adaptive` only: wavelet cycles per band. |
| `add_reference_channel` | str \| null | `null` | Add this flat reference channel before fitting (preserves rank for non-average references). Kept in the output. |
| `montage` | str \| null | `null` | MNE built-in montage to attach before fitting (e.g. `standard_1005`). Use when channel names are non-standard and no positions are stored. |
| `montage_handling` | `auto` \| `leadfield` \| `channel_positions` \| `drop_unmatched` | `auto` | How to get the reference covariance when some channel names are not in the 10-05 leadfield. `auto`: leadfield → channel-position covariance → drop unmatched. |
| `interpolate_bads` | bool | `false` | Interpolate bad channels before fitting. |
| `output` | `recombine` \| `eeg_only` | `recombine` | `recombine`: full input copy with EEG replaced by the denoised signal. `eeg_only`: only the denoised EEG channels. |
| `n_jobs` | int | `8` | Parallel jobs. |

## Limitations

- **EEG only.** MEG denoising is not supported (the bundled leadfield is an EEG
  leadfield). MEG channels are passed through unchanged.
- Needs standard 10-05 channel names **or** channel positions **or** a `montage`.
- `mode: adaptive` is unavailable for epoched data -- use the `gedai-epo` app
  with `mode: multiband`.
- The denoised EEG output is average-referenced.
- `mode: broadband` does not produce the SENSAI subspace-similarity figure.

## Local testing

```bash
# edit config.json to point "mne" at a real EEG .fif, then:
pip install 'gedai==0.4.2'
python main.py
```

## Authors

- Maximilien Chaumon (https://github.com/dnacombo)
- GEDAI algorithm: T. Ros, V. Ferat et al. (https://github.com/neurotuning/gedai)

## Citations

If you use this app, please cite **GEDAI** and **brainlife.io**:

Ros T., Ferat V., Huang Y., Colangelo C., Kia S.M., Wolfers T., Vulliemoz S.,
Michela A. *Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield
Filtering.* bioRxiv 2025. https://doi.org/10.1101/2025.10.04.680449

Hayashi, S., Caron, B.A., Heinsfeld, A.S. et al. brainlife.io: a decentralized
and open-source cloud platform to support neuroscience research. Nat Methods 21,
809-813 (2024). https://doi.org/10.1038/s41592-024-02237-2

Gramfort A. et al. MEG and EEG data analysis with MNE-Python. Front Neurosci 7,
267 (2013). https://doi.org/10.3389/fnins.2013.00267

## Funding Acknowledgement

brainlife.io is publicly funded. We kindly ask that you acknowledge the funding
below in your code and publications.

[![NSF-BCS-1734853](https://img.shields.io/badge/NSF_BCS-1734853-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1734853)
[![NSF-BCS-1636893](https://img.shields.io/badge/NSF_BCS-1636893-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1636893)
[![NSF-ACI-1916518](https://img.shields.io/badge/NSF_ACI-1916518-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1916518)
[![NSF-IIS-1912270](https://img.shields.io/badge/NSF_IIS-1912270-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1912270)
[![NIH-NIBIB-R01EB029272](https://img.shields.io/badge/NIH_NIBIB-R01EB029272-green.svg)](https://grantome.com/grant/NIH/R01-EB029272-01)
