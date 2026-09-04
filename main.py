"""
Denoise continuous EEG with GEDAI (unsupervised leadfield filtering).

GEDAI ("Generalized Eigenvalue De-Artifacting Instrument", Ros et al., bioRxiv
2025, doi:10.1101/2025.10.04.680449) contrasts the data covariance of short
segments against a physics-based reference covariance derived from a template
head model / leadfield (bundled with the package, a ~343-electrode 10-05
montage). A generalized eigendecomposition keeps the components aligned with the
brain subspace and projects out the rest; the rejection threshold is chosen
automatically by GEDAI's SENSAI criterion.

This app runs GEDAI on the EEG channels of a continuous MNE .fif recording. Any
non-EEG channels (MEG, EOG, ECG, stim, ...) are passed through untouched.

Inputs
------
mne : str
    Path to the input MNE raw .fif file (must contain EEG channels).

Configuration
-------------
mode : 'broadband' | 'multiband' | 'adaptive'
    GEDAI variant. 'broadband' = Gedai (single pass). 'multiband' =
    MultibandGedai (per wavelet-band). 'adaptive' = AdaptiveMultibandGedai
    (the authors' recommended offline pipeline; slowest).
l_freq : float | null
    High-pass frequency (Hz) applied before fitting. Improves covariance
    estimation by removing slow drifts. null disables it.
noise_multiplier : float | str
    SENSAI noise weighting. Higher keeps more brain signal. A float > 0, or a
    preset ('standard'/'auto' = 3.0, 'aggressive' = 1.5, 'conservative' = 6.0).
sensai_method : 'optimize' | 'gridsearch'
    Threshold search strategy.
duration : float
    Segment length (s) for covariance estimation (broadband / multiband).
overlap : float
    Fractional overlap between consecutive segments, in [0, 1).
reject_by_annotation : bool
    Exclude 'bad'-annotated spans from the fit. GEDAI recommends keeping them.
broadband_pass : bool
    multiband / adaptive: run an initial broadband GEDAI pass before the
    per-band decomposition.
wavelet_low_cutoff : 'auto' | float | null
    multiband / adaptive: wavelet bands below this frequency are zeroed.
    'auto' uses the data high-pass (min 0.5 Hz).
cycles_per_wavelet : int
    adaptive only: wavelet cycles per band.
add_reference_channel : str | null
    Name of a flat reference channel to add before fitting (preserves rank when
    the acquisition reference is not average). Kept in the output.
montage : str | null
    Name of an MNE built-in montage to attach before fitting (e.g.
    'standard_1005'). Needed when the file has no channel positions and the
    channel names are not standard 10-05 labels.
montage_handling : 'auto' | 'leadfield' | 'channel_positions' | 'drop_unmatched'
    How to obtain the reference covariance when some channel names are not in
    the bundled 10-05 leadfield. 'auto' tries leadfield, then a
    channel-position covariance, then dropping the unmatched channels.
interpolate_bads : bool
    Interpolate channels marked bad before fitting (default: keep them in).
output : 'recombine' | 'eeg_only'
    'recombine' writes a full copy of the input with the EEG replaced by the
    denoised signal. 'eeg_only' writes just the denoised EEG channels.
n_jobs : int
    Parallel jobs.

Outputs
-------
out_dir/raw.fif : denoised data (datatype neuro/mne).
out_noise/raw.fif : the removed signal, EEG only (datatype neuro/mne).
out_report/report.html : QC report.
out_figs/*.png : QC figures.
product.json : brainlife metadata.
"""

# Copyright (c) 2026 brainlife.io
#
# Wraps GEDAI (https://github.com/neurotuning/gedai). GEDAI is licensed under the
# PolyForm Noncommercial License 1.0.0 and its core algorithm is patent pending --
# see license.txt and README.md. Non-commercial research / evaluation use only.
#
# Authors:
# - Maximilien Chaumon (https://github.com/dnacombo)
# GEDAI algorithm: T. Ros, V. Ferat et al. (https://github.com/neurotuning/gedai)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

import numpy as np
import mne
import matplotlib.pyplot as plt

from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
    add_raw_info_to_product,
    require_config_keys,
)

mne.set_log_level('WARNING')
setup_matplotlib_backend()

ensure_output_dirs('out_dir', 'out_noise', 'out_figs', 'out_report')
config = load_config()
require_config_keys(config, ['mne'])
product_items = []

# ======================================================================
# VALIDATE CONFIGURATION
# ======================================================================
MODES = ('broadband', 'multiband', 'adaptive')
SENSAI_METHODS = ('optimize', 'gridsearch')
MONTAGE_HANDLING = ('auto', 'leadfield', 'forward', 'channel_positions',
                    'drop_unmatched')
OUTPUT_MODES = ('recombine', 'eeg_only')
NOISE_PRESETS = (
    'auto', 'default', 'standard', 'auto+', 'aggressive', 'more',
    'auto-', 'conservative', 'less',
)

errors = []

mode = str(config.get('mode') or 'broadband').lower()
if mode not in MODES:
    errors.append(f"mode must be one of {MODES}, got '{mode}'")

sensai_method = str(config.get('sensai_method') or 'optimize').lower()
if sensai_method not in SENSAI_METHODS:
    errors.append(f"sensai_method must be one of {SENSAI_METHODS}, got '{sensai_method}'")

montage_handling = str(config.get('montage_handling') or 'auto').lower()
if montage_handling not in MONTAGE_HANDLING:
    errors.append(f"montage_handling must be one of {MONTAGE_HANDLING}, got '{montage_handling}'")

output_mode = str(config.get('output') or 'recombine').lower()
if output_mode not in OUTPUT_MODES:
    errors.append(f"output must be one of {OUTPUT_MODES}, got '{output_mode}'")

# noise_multiplier: float > 0 or a known preset string
noise_multiplier = config.get('noise_multiplier', 3.0)
if noise_multiplier is None:
    noise_multiplier = 3.0
try:
    noise_multiplier = float(noise_multiplier)
    if noise_multiplier <= 0:
        errors.append(f"noise_multiplier must be > 0, got {noise_multiplier}")
except (TypeError, ValueError):
    if str(noise_multiplier).lower() not in NOISE_PRESETS:
        errors.append(
            f"noise_multiplier must be a number > 0 or one of {NOISE_PRESETS}, "
            f"got '{noise_multiplier}'"
        )
    else:
        noise_multiplier = str(noise_multiplier).lower()

# numeric parameters: (key, default, python type, predicate, message)
NUMERIC_SPECS = [
    ('duration', 2.0, float, lambda v: v > 0, 'must be > 0'),
    ('overlap', 0.5, float, lambda v: 0.0 <= v < 1.0, 'must be in [0, 1)'),
    ('cycles_per_wavelet', 10, int, lambda v: v >= 1, 'must be >= 1'),
    ('n_jobs', 8, int, lambda v: v >= 1, 'must be >= 1'),
]
numeric = {}
for _key, _default, _cast, _pred, _msg in NUMERIC_SPECS:
    _val = config.get(_key)
    if _val is None or _val == '':
        numeric[_key] = _default
        continue
    try:
        _val = _cast(_val)
    except (TypeError, ValueError):
        errors.append(f"{_key} must be {_cast.__name__}, got '{config.get(_key)}'")
        numeric[_key] = _default
        continue
    if not _pred(_val):
        errors.append(f"{_key} {_msg}, got {_val}")
    numeric[_key] = _val
duration = numeric['duration']
overlap = numeric['overlap']
cycles_per_wavelet = numeric['cycles_per_wavelet']
n_jobs = numeric['n_jobs']

l_freq = config.get('l_freq')
if l_freq not in (None, ''):
    try:
        l_freq = float(l_freq)
        if l_freq <= 0:
            errors.append(f"l_freq must be > 0 or null, got {l_freq}")
    except (TypeError, ValueError):
        errors.append(f"l_freq must be a number or null, got '{config.get('l_freq')}'")
else:
    l_freq = None

wavelet_low_cutoff = config.get('wavelet_low_cutoff', 'auto')
if wavelet_low_cutoff in (None, ''):
    wavelet_low_cutoff = None
elif str(wavelet_low_cutoff).lower() == 'auto':
    wavelet_low_cutoff = 'auto'
else:
    try:
        wavelet_low_cutoff = float(wavelet_low_cutoff)
        if wavelet_low_cutoff < 0:
            errors.append(f"wavelet_low_cutoff must be >= 0, 'auto' or null, got {wavelet_low_cutoff}")
    except (TypeError, ValueError):
        errors.append(f"wavelet_low_cutoff must be a number, 'auto' or null, got '{config.get('wavelet_low_cutoff')}'")

# booleans: accept real JSON bools and common string spellings
BOOL_MAP = {True: True, False: False, None: None,
            'true': True, '1': True, 'yes': True, 'on': True,
            'false': False, '0': False, 'no': False, 'off': False, 'none': False}
bools = {}
for _key, _default in (('reject_by_annotation', False),
                       ('broadband_pass', True),
                       ('interpolate_bads', False)):
    _raw = config.get(_key)
    _norm = BOOL_MAP.get(_raw if isinstance(_raw, bool) or _raw is None
                         else str(_raw).strip().lower(), 'INVALID')
    if _norm == 'INVALID':
        errors.append(f"{_key} must be a boolean, got '{_raw}'")
        bools[_key] = _default
    else:
        bools[_key] = _default if _norm is None else _norm
reject_by_annotation = bools['reject_by_annotation']
broadband_pass = bools['broadband_pass']
interpolate_bads = bools['interpolate_bads']

add_reference_channel = config.get('add_reference_channel') or None
if add_reference_channel is not None:
    add_reference_channel = str(add_reference_channel)

montage_name = config.get('montage') or None
if montage_name is not None:
    montage_name = str(montage_name)
    if montage_name not in mne.channels.get_builtin_montages():
        errors.append(f"montage '{montage_name}' is not an MNE built-in montage")

if errors:
    for e in errors:
        add_info_to_product(product_items, f"Invalid configuration: {e}", 'error')
    create_product_json(product_items)
    sys.exit(1)

add_info_to_product(
    product_items,
    f"GEDAI mode='{mode}', noise_multiplier={noise_multiplier}, "
    f"sensai_method='{sensai_method}'",
)

# ======================================================================
# IMPORT GEDAI (installed at run time by ./main)
# ======================================================================
try:
    import gedai
    from gedai import Gedai, MultibandGedai, AdaptiveMultibandGedai
    from gedai.metrics import compute_enova_per_channel
    from gedai.covariance.covariance import (
        compute_covariance_from_channel_positions,
        compute_covariance_from_forward,
        get_leadfield_cov_path,
    )
except Exception as e:  # pragma: no cover - environment problem
    add_info_to_product(product_items, f"Could not import gedai: {e}", 'error')
    create_product_json(product_items)
    sys.exit(1)

add_info_to_product(product_items, f"gedai version {gedai.__version__}")

# ======================================================================
# LOAD DATA, SPLIT CHANNELS
# ======================================================================
try:
    raw = mne.io.read_raw_fif(config['mne'], preload=True)
except Exception as e:
    add_info_to_product(
        product_items, f"Could not read input file '{config['mne']}': {e}", 'error')
    create_product_json(product_items)
    sys.exit(1)

eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False,
                           stim=False, misc=False, exclude=[])
if len(eeg_picks) == 0:
    add_info_to_product(
        product_items,
        "No EEG channels found in the input. GEDAI denoises EEG data only.",
        'error')
    create_product_json(product_items)
    sys.exit(1)

n_meg = len(mne.pick_types(raw.info, meg=True))
if n_meg:
    add_info_to_product(
        product_items,
        f"{n_meg} MEG channel(s) present -- GEDAI runs on the "
        f"{len(eeg_picks)} EEG channel(s) only; other channels pass through.",
        'warning')

raw_eeg = raw.copy().pick([raw.ch_names[i] for i in eeg_picks])

# ======================================================================
# REFERENCE CHANNEL (rank preservation)
# ======================================================================
if add_reference_channel and add_reference_channel not in raw_eeg.ch_names:
    raw_eeg = mne.add_reference_channels(raw_eeg, add_reference_channel)
    add_info_to_product(
        product_items,
        f"Added flat reference channel '{add_reference_channel}' before fitting.")
elif raw_eeg.info.get('custom_ref_applied') and not add_reference_channel:
    add_info_to_product(
        product_items,
        "Input EEG uses a custom (non-average) reference and no "
        "'add_reference_channel' was given; the data rank may be reduced by one. "
        "GEDAI applies an average reference internally.",
        'warning')

# ======================================================================
# MONTAGE / LEADFIELD-NAME RECONCILIATION -> reference_cov
# ======================================================================
if montage_name:
    raw_eeg.set_montage(montage_name, match_case=False, on_missing='ignore')
    add_info_to_product(product_items, f"Attached montage '{montage_name}'.")

lead_names = {n.lower() for n in mne.read_cov(str(get_leadfield_cov_path())).ch_names}

matched = [c for c in raw_eeg.ch_names if c.lower() in lead_names]
pos_ok = [ch['ch_name'] for ch in raw_eeg.info['chs']
          if np.isfinite(ch['loc'][:3]).all()
          and not np.allclose(ch['loc'][:3], 0.0)]
unusable = [c for c in raw_eeg.ch_names
            if c.lower() not in lead_names and c not in pos_ok]

# Drop channels that can be neither name-matched to the leadfield nor placed by
# position (e.g. a reference channel with no location).
if unusable and montage_handling in (
        'auto', 'leadfield', 'forward', 'channel_positions', 'drop_unmatched'):
    raw_eeg.pick([c for c in raw_eeg.ch_names if c not in unusable])
    add_info_to_product(
        product_items,
        f"Dropped {len(unusable)} EEG channel(s) with neither a standard 10-05 "
        f"name nor a position: {unusable[:8]}{'...' if len(unusable) > 8 else ''}. "
        f"They are NOT denoised in the output.",
        'warning')
    matched = [c for c in raw_eeg.ch_names if c.lower() in lead_names]
    pos_ok = [ch['ch_name'] for ch in raw_eeg.info['chs']
              if np.isfinite(ch['loc'][:3]).all()
              and not np.allclose(ch['loc'][:3], 0.0)]

n_ch = len(raw_eeg.ch_names)
unmatched = [c for c in raw_eeg.ch_names if c.lower() not in lead_names]

# Decide the reference: ref_kind in {'leadfield','forward','positions'} and the
# set of channels it applies to. 'auto': all names match -> bundled leadfield;
# else all channels positioned -> fsaverage template forward solution; else fall
# back to the matched subset (leadfield) or the positioned subset (forward).
ref_kind, keep = None, list(raw_eeg.ch_names)
if not unmatched:
    ref_kind = 'leadfield'
elif montage_handling == 'channel_positions' and pos_ok:
    ref_kind, keep = 'positions', pos_ok
elif montage_handling in ('leadfield', 'drop_unmatched') and matched:
    ref_kind, keep = 'leadfield', matched
elif montage_handling == 'forward' and pos_ok:
    ref_kind, keep = 'forward', pos_ok
elif montage_handling == 'auto':
    if len(pos_ok) == n_ch:
        ref_kind, keep = 'forward', pos_ok
    elif matched:
        ref_kind, keep = 'leadfield', matched
    elif pos_ok:
        ref_kind, keep = 'forward', pos_ok

if ref_kind is None:
    add_info_to_product(
        product_items,
        "Cannot build a GEDAI reference covariance: EEG channel names are not "
        "standard 10-05 labels and the file carries no usable channel positions. "
        "Set 'montage' to an MNE built-in montage that fits this cap.",
        'error')
    create_product_json(product_items)
    sys.exit(1)

if keep != list(raw_eeg.ch_names):
    dropped = [c for c in raw_eeg.ch_names if c not in keep]
    raw_eeg.pick(keep)
    add_info_to_product(
        product_items,
        f"Using {len(keep)} of {n_ch} EEG channels for GEDAI; dropped "
        f"{len(dropped)} ({dropped[:6]}{'...' if len(dropped) > 6 else ''}). "
        f"Dropped channels are NOT denoised in the output.",
        'warning')

reference_cov = 'leadfield'
if ref_kind == 'leadfield':
    add_info_to_product(
        product_items,
        f"Reference: bundled 10-05 leadfield ({len(raw_eeg.ch_names)} channels).")
elif ref_kind == 'positions':
    reference_cov = compute_covariance_from_channel_positions(raw_eeg.info)
    add_info_to_product(
        product_items,
        f"Reference: channel-position covariance over {len(raw_eeg.ch_names)} "
        f"channels (fast, lower fidelity than a forward model).", 'warning')
else:  # forward
    try:
        # prefer an fsaverage already on disk (baked into the
        # brainlifemeeg/mne-fsaverage image) -- fetch_fsaverage() would try to
        # write to it and fail on a read-only container filesystem.
        _sub = os.environ.get('SUBJECTS_DIR') or ''
        _baked = os.path.join(_sub, 'fsaverage') if _sub else ''
        if _baked and os.path.exists(os.path.join(
                _baked, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')):
            fs_dir = _baked
        else:
            fs_dir = str(mne.datasets.fetch_fsaverage(verbose=False))
        fwd = mne.make_forward_solution(
            raw_eeg.info, trans='fsaverage',
            src=os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif'),
            bem=os.path.join(fs_dir, 'bem',
                             'fsaverage-5120-5120-5120-bem-sol.fif'),
            eeg=True, meg=False, mindist=5.0, n_jobs=n_jobs, verbose=False)
        reference_cov = compute_covariance_from_forward(fwd)
        add_info_to_product(
            product_items,
            f"Reference: fsaverage template forward solution "
            f"({len(raw_eeg.ch_names)} channels); "
            f"{len(unmatched)} channel name(s) not in the 10-05 leadfield.")
    except Exception as e:
        add_info_to_product(
            product_items,
            f"Could not build the fsaverage forward solution: {e}. Try "
            f"montage_handling='leadfield' or 'channel_positions'.", 'error')
        create_product_json(product_items)
        sys.exit(1)

# ======================================================================
# OPTIONAL PRE-FILTER / BAD-CHANNEL HANDLING
# ======================================================================
if l_freq is not None:
    raw_eeg.filter(l_freq=l_freq, h_freq=None, n_jobs=n_jobs)
    add_info_to_product(product_items, f"High-pass pre-filtered at {l_freq} Hz.")

if raw_eeg.info['bads']:
    if interpolate_bads:
        raw_eeg.interpolate_bads(reset_bads=True)
        add_info_to_product(
            product_items, "Interpolated bad channels before fitting.")
    else:
        add_info_to_product(
            product_items,
            f"{len(raw_eeg.info['bads'])} bad channel(s) kept in the fit "
            f"(GEDAI recommends keeping them): {raw_eeg.info['bads']}.")

# ======================================================================
# FIT + TRANSFORM
# ======================================================================
# GEDAI average-references internally; do it here too so raw_eeg_in (used for the
# noise output and every QC figure) is on the same reference as the denoised
# data -- otherwise the re-referencing shows up as a huge common-mode "artifact".
raw_eeg.set_eeg_reference('average', projection=False, ch_type='eeg',
                          verbose=False)
raw_eeg_in = raw_eeg.copy()  # exact GEDAI input, for QC and the noise output

common = dict(
    reference_cov=reference_cov,
    sensai_method=sensai_method,
    noise_multiplier=noise_multiplier,
    n_jobs=n_jobs,
    verbose=False,
)

try:
    if mode == 'broadband':
        model = Gedai()
        model.fit_raw(raw_eeg, duration=duration, overlap=overlap,
                      reject_by_annotation=reject_by_annotation, **common)
    elif mode == 'multiband':
        model = MultibandGedai(broadband_pass=broadband_pass)
        model.fit_raw(raw_eeg, duration=duration, overlap=overlap,
                      reject_by_annotation=reject_by_annotation,
                      wavelet_low_cutoff=wavelet_low_cutoff, **common)
    else:  # adaptive
        model = AdaptiveMultibandGedai(
            cycles_per_wavelet=cycles_per_wavelet, broadband_pass=broadband_pass)
        model.fit_raw(raw_eeg, overlap=overlap,
                      reject_by_annotation=reject_by_annotation,
                      wavelet_low_cutoff=wavelet_low_cutoff, **common)
    denoised_eeg = model.transform_raw(raw_eeg, overlap=overlap, n_jobs=n_jobs,
                                       verbose=False)
except Exception as e:
    add_info_to_product(product_items, f"GEDAI fit/transform failed: {e}", 'error')
    create_product_json(product_items)
    sys.exit(1)

print(model.summary())

fm = getattr(model, 'fit_metrics_', {}) or {}
sensai_score = fm.get('sensai_score', float('nan'))

# ENOVA: fraction of per-channel variance attributed to the removed signal
epoch_samples = max(1, int(round(raw_eeg_in.info['sfreq'])))
noise_arr = raw_eeg_in.get_data() - denoised_eeg.get_data()
try:
    enova_ch = compute_enova_per_channel(
        denoised_eeg.get_data(), noise_arr, epoch_samples)
    mean_enova = float(np.mean(enova_ch))
except Exception as e:
    enova_ch = None
    mean_enova = float('nan')
    add_info_to_product(product_items, f"Could not compute ENOVA: {e}", 'warning')

# ======================================================================
# RECOMBINE / SAVE DENOISED OUTPUT
# ======================================================================
if output_mode == 'eeg_only':
    raw_out = denoised_eeg.copy()
else:
    raw_out = raw.copy()
    if add_reference_channel and add_reference_channel not in raw_out.ch_names:
        raw_out = mne.add_reference_channels(raw_out, add_reference_channel)
    idx = {n: i for i, n in enumerate(raw_out.ch_names)}
    dd = denoised_eeg.get_data()
    for j, ch in enumerate(denoised_eeg.ch_names):
        if ch in idx:
            raw_out._data[idx[ch]] = dd[j]
    raw_out.set_eeg_reference('average', projection=False, ch_type='eeg',
                              verbose=False)
    if montage_name:
        raw_out.set_montage(montage_name, match_case=False, on_missing='ignore')

out_raw = os.path.join('out_dir', 'raw.fif')
raw_out.save(out_raw, overwrite=True)
_ = mne.io.read_raw_fif(out_raw, preload=False)
print(f"Wrote and verified {out_raw}")

# ======================================================================
# NOISE OUTPUT (always produced -- registered 2nd neuro/mne output)
# ======================================================================
noise_raw = raw_eeg_in.copy()
noise_raw._data = noise_arr
out_noise = os.path.join('out_noise', 'raw.fif')
noise_raw.save(out_noise, overwrite=True)
_ = mne.io.read_raw_fif(out_noise, preload=False)
print(f"Wrote and verified {out_noise}")

# ======================================================================
# QC FIGURES + REPORT
# ======================================================================
report = mne.Report(title='GEDAI denoising report')
report.add_html(f"<pre>{model.summary()}</pre>", title='GEDAI fit summary')

fig_paths = []

try:
    for i, fig in enumerate(model.plot_fit()):
        p = os.path.join('out_figs', f'plot_fit_{i:02d}.png')
        fig.savefig(p, dpi=120, bbox_inches='tight')
        plt.close(fig)
        fig_paths.append(('GEDAI fit (band %d)' % i if i else 'GEDAI fit', p))
    report.add_image(fig_paths[0][1], title='GEDAI eigenvalue / SENSAI fit')
except Exception as e:
    add_info_to_product(product_items, f"plot_fit failed: {e}", 'warning')

try:
    fig, sensai_metrics = model.plot_sensai(
        raw_before=raw_eeg_in, raw_after=denoised_eeg, show=False)
    p = os.path.join('out_figs', 'sensai.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(('SENSAI subspace similarity', p))
    report.add_image(p, title='SENSAI subspace similarity (before / after)')
    for k, v in sensai_metrics.items():
        add_info_to_product(product_items, f"SENSAI {k}: {v:.3f}")
except Exception as e:
    add_info_to_product(
        product_items,
        f"SENSAI visualisation not available for mode '{mode}': {e}", 'warning')

# PSD overlay: input / denoised / removed noise
try:
    fmax = min(120.0, raw_eeg_in.info['sfreq'] / 2.0 - 1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    for inst, label, color in (
        (raw_eeg_in, 'input', '0.5'),
        (denoised_eeg, 'denoised', 'tab:blue'),
        (noise_raw, 'removed noise', 'tab:red'),
    ):
        psd = inst.compute_psd(fmax=fmax, verbose=False)
        p, f = psd.get_data(return_freqs=True)
        ax.plot(f, 10 * np.log10(np.maximum(p.mean(axis=0), 1e-30)),
                label=label, color=color)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB)',
           title='Mean power spectral density')
    ax.legend()
    p = os.path.join('out_figs', 'psd_overlay.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(('PSD overlay', p))
    report.add_image(p, title='PSD: input vs denoised vs removed noise')
except Exception as e:
    add_info_to_product(product_items, f"PSD overlay failed: {e}", 'warning')

# ENOVA topomap
if enova_ch is not None:
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        mne.viz.plot_topomap(enova_ch, denoised_eeg.info, axes=ax, show=False,
                             cmap='Reds')
        ax.set_title('ENOVA: fraction of variance removed per channel')
        p = os.path.join('out_figs', 'enova_topo.png')
        fig.savefig(p, dpi=120, bbox_inches='tight')
        plt.close(fig)
        fig_paths.append(('ENOVA topomap', p))
        report.add_image(p, title='ENOVA topomap (variance removed per channel)')
    except Exception as e:
        add_info_to_product(product_items, f"ENOVA topomap failed: {e}", 'warning')

# Raw-segment before/after overlay
try:
    seg = min(10.0, raw_eeg_in.times[-1])
    t0 = max(0.0, raw_eeg_in.times[-1] / 2.0 - seg / 2.0)
    picks_plot = list(range(min(6, len(denoised_eeg.ch_names))))
    sfreq = raw_eeg_in.info['sfreq']
    s0, s1 = int(t0 * sfreq), int((t0 + seg) * sfreq)
    tvec = raw_eeg_in.times[s0:s1]
    din = raw_eeg_in.get_data()[picks_plot, s0:s1]
    dout = denoised_eeg.get_data()[picks_plot, s0:s1]
    off = 8 * np.std(din) if np.std(din) > 0 else 1e-5
    fig, ax = plt.subplots(figsize=(9, 7))
    for k, pk in enumerate(picks_plot):
        ax.plot(tvec, din[k] - k * off, color='0.6', lw=0.6)
        ax.plot(tvec, dout[k] - k * off, color='tab:blue', lw=0.6)
    ax.set(xlabel='Time (s)', yticks=[-k * off for k in range(len(picks_plot))],
           yticklabels=[denoised_eeg.ch_names[p] for p in picks_plot],
           title=f'Input (grey) vs denoised (blue) -- {seg:.0f}s segment')
    p = os.path.join('out_figs', 'segment_overlay.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(('Segment overlay', p))
    report.add_image(p, title='Time-domain segment: input vs denoised')
except Exception as e:
    add_info_to_product(product_items, f"Segment overlay failed: {e}", 'warning')

try:
    report.add_raw(raw_out.copy().crop(tmax=min(60.0, raw_out.times[-1])),
                   title='Denoised output (first 60 s)', psd=True)
except Exception as e:
    add_info_to_product(product_items, f"report.add_raw failed: {e}", 'warning')

report.save(os.path.join('out_report', 'report.html'), overwrite=True,
            verbose=False)

# ======================================================================
# PRODUCT.JSON
# ======================================================================
add_info_to_product(
    product_items,
    f"GEDAI '{mode}' denoising complete on {len(denoised_eeg.ch_names)} EEG "
    f"channel(s).", 'success')
if np.isfinite(sensai_score):
    add_info_to_product(product_items, f"Fitted SENSAI score: {sensai_score:.2f} %")
if np.isfinite(mean_enova):
    add_info_to_product(
        product_items, f"Mean ENOVA (variance removed): {mean_enova * 100:.2f} %")
add_raw_info_to_product(product_items, raw_out)
for name, path in fig_paths:
    if os.path.exists(path):
        add_image_to_product(product_items, name, filepath=path)
create_product_json(product_items)
