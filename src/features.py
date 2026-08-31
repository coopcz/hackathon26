"""
CSI -> feature-vector pipeline.

WHY THESE FEATURES (and not raw amplitude)
------------------------------------------
A trained-on-raw-amplitude model learns the *room*, not the *person*.  Absolute
CSI amplitude is dominated by things that have nothing to do with occupancy:
transmitter power, AGC state, the distance between the two radios, and the
static multipath signature of the furniture and walls.  Move the radios 30 cm
and every absolute amplitude changes; a human walking through the room may
barely move the mean at all.

What a human body actually does to the channel is *modulate* it over time.  A
person is a large, moving, water-filled reflector/absorber.  Their motion makes
the multipath sum at the receiver rise and fall from packet to packet.  So the
occupancy signal lives in the **temporal statistics** of the channel, not its
level:

  1. TEMPORAL VARIANCE (std_*) - how much each subcarrier swings over the window.
     An empty room is a static channel: near-flat, only thermal noise.  An
     occupied room breathes.  This is the single most physically direct measure
     of "something in here is moving".

  2. RATE OF CHANGE (roc_*) - mean/max |A[t+1] - A[t]| between consecutive
     packets.  Variance says "the channel is spread out"; rate-of-change says
     "it is moving *fast*".  This separates real motion from slow thermal drift
     or an AGC step, both of which inflate variance without being a person.

  3. PHASE VARIANCE (phase_*) - phase is far more sensitive to sub-wavelength
     displacement than amplitude (at 5 GHz one wavelength is ~6 cm), so small
     motion - a seated person breathing or shifting - shows up in phase before
     it shows up in amplitude.  Raw phase is unusable (see _sanitize_phase), so
     this is measured on linearly-detrended phase.

  4. NORMALISED VARIANCE (cv_*) - std/mean, i.e. coefficient of variation.
     This is the transfer-friendly feature: dividing out the mean cancels the
     absolute power scale, so it means roughly the same thing on an ESP32 as on
     an Intel 5300.  Deliberately included with Phase 3 portability in mind.

  5. SPECTRAL (dop_ratio, acf_lag) - where in the frequency band the fluctuation energy
     sits, and how temporally correlated it is.  Measured on this dataset the
     effect runs *opposite* to the naive expectation, and the reason is
     instructive: an empty room's residual fluctuation is thermal noise, which is
     broadband and decorrelates within milliseconds (high dop_ratio, acf_lag ~0).
     An occupied room's fluctuation is a physical body moving continuously, which
     is low-frequency and stays correlated across 100 ms.  So these
     features do not detect "motion energy" - they detect whether the channel is
     *structured* or merely noisy.  That is a genuinely different axis of
     evidence from raw variance, which is why both are kept.

  6. MEAN AMPLITUDE (mean_*) - included deliberately as a *control*.  It is the
     naive feature, and if the model leans on it we have learned a room
     fingerprint rather than a presence detector.  Its importance is a
     diagnostic, not a goal.
"""

import numpy as np

from .csi_reader import read_bf_file, get_scaled_csi

# ~2.5 s at the dataset's ~50 Hz packet rate.  Long enough for the variance and
# spectral estimates to be stable, short enough that an HVAC decision is timely.
WINDOW = 128
STRIDE = 128  # non-overlapping: overlapping windows would leak between train and test


def _sanitize_phase(phase):
    """Remove the linear-in-subcarrier phase ramp from raw CSI phase.

    Raw CSI phase is unusable as-is: carrier frequency offset, sampling time
    offset and the receiver's random packet detection delay add a large, random,
    per-packet slope across subcarriers that swamps any motion effect.  The
    standard fix is to unwrap across subcarriers and subtract the best-fit line,
    which removes the offsets that are linear in subcarrier index and leaves the
    residual that actually responds to the environment.

    phase: (n_pkt, n_sub, n_rx) -> detrended, same shape.
    """
    n_sub = phase.shape[1]
    unwrapped = np.unwrap(phase, axis=1)
    k = np.arange(n_sub, dtype=np.float64)
    kc = k - k.mean()
    denom = np.sum(kc ** 2)
    # least-squares slope/intercept per (packet, rx), vectorised over subcarriers
    slope = np.tensordot(unwrapped - unwrapped.mean(axis=1, keepdims=True), kc, axes=([1], [0])) / denom
    return unwrapped - slope[:, None, :] * kc[None, :, None] - unwrapped.mean(axis=1, keepdims=True)


def file_to_arrays(path):
    """Parse one .dat capture into (amplitude, sanitized_phase) arrays.

    Only TX stream 0 is used.  Ntx is not constant *within* a file in this
    dataset (packets alternate between 1 and 2 spatial streams), so stream 0 is
    the only one guaranteed present in every packet.  Nrx is always 3.
    """
    entries = read_bf_file(path)
    amps, phases = [], []
    for e in entries:
        csi = get_scaled_csi(e)[:, :, 0]  # (30 subcarriers, 3 rx)
        amps.append(np.abs(csi))
        phases.append(np.angle(csi))
    amp = np.asarray(amps, dtype=np.float32)
    ph = _sanitize_phase(np.asarray(phases, dtype=np.float64)).astype(np.float32)
    return amp, ph


# Both spectral features are pinned to ABSOLUTE frequencies rather than to
# sample indices, so a 10 Hz capture and a 4 kHz one can be compared.
# BAND_HI caps the analysis band at 5 Hz: the lowest packet rate we accept is
# ~10 Hz, whose Nyquist is 5 Hz, and a ratio computed over "everything up to
# Nyquist" would silently mean a different thing at every rate.
BAND_LO, BAND_HI = 2.0, 5.0
ACF_LAG_SECONDS = 0.1


def _spectral_high_freq_ratio(x, fs=50.0, lo=BAND_LO, hi=BAND_HI):
    """Energy fraction in [lo, hi] Hz relative to all energy below `hi` Hz.

    x: (n_pkt, n_channels), mean-removed internally.
    """
    xd = x - x.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(xd, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    total = spec[freqs <= hi].sum(axis=0)
    high = spec[(freqs > lo) & (freqs <= hi)].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, high / total, 0.0)
    return float(np.mean(ratio))


def _autocorr_at_lag(x, fs, lag_seconds=ACF_LAG_SECONDS):
    """Autocorrelation at a fixed TIME lag, not a fixed sample lag.

    Lag-1-sample autocorrelation is not a property of the room, it is a property
    of the packet rate: at 4 kHz consecutive samples are 0.25 ms apart and
    correlate near 1 no matter what is happening, while at 10 Hz they are 100 ms
    apart.  Anchoring the lag in seconds makes the number comparable across the
    Intel 5300, the Broadcom captures and whatever rate our ESP32 ends up at.
    """
    lag = max(1, int(round(fs * lag_seconds)))
    if x.shape[0] <= lag + 1:
        return 0.0
    xd = x - x.mean(axis=0, keepdims=True)
    num = np.sum(xd[lag:] * xd[:-lag], axis=0)
    den = np.sum(xd * xd, axis=0)
    return float(np.mean(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)))


FEATURE_NAMES = [
    "mean_amp", "mean_amp_std_sc",
    "std_amp_mean", "std_amp_max", "std_amp_std_sc", "std_within_frame",
    "roc_mean", "roc_max", "roc_std",
    "cv_mean", "cv_max",
    "phase_std_mean", "phase_std_max", "phase_roc_mean",
    "dop_ratio", "acf_lag",
]


def window_features(amp_w, ph_w, fs=50.0):
    """Compute one feature vector from a window of CSI.

    amp_w: (WINDOW, n_sub, n_rx) amplitude.  ph_w: same shape, sanitized phase.
    fs:    packet rate in Hz.  Must be passed explicitly for non-Intel captures --
           the spectral feature is defined in Hz, so a 100 Hz ESP32 capture scored
           with fs=50 would put the cutoff in the wrong physical place.
    """
    n_t = amp_w.shape[0]
    flat = amp_w.reshape(n_t, -1)          # (t, sub*rx) - one time series per channel
    ph_flat = ph_w.reshape(n_t, -1)

    # --- family 1: static level (control features) ---
    mean_amp = float(flat.mean())
    mean_amp_std_sc = float(amp_w.mean(axis=0).std())  # frequency selectivity of the mean

    # --- family 2: temporal variance = the core "something is moving" signal ---
    per_channel_std = flat.std(axis=0)
    std_amp_mean = float(per_channel_std.mean())
    std_amp_max = float(per_channel_std.max())
    std_amp_std_sc = float(per_channel_std.std())
    # spread *across* subcarriers within a single frame, averaged over time
    std_within_frame = float(amp_w.std(axis=1).mean())

    # --- family 3: rate of change = how fast it is moving ---
    d = np.abs(np.diff(flat, axis=0))
    roc_mean = float(d.mean())
    roc_max = float(d.max())
    roc_std = float(d.std())

    # --- family 4: normalised variance, scale-free and hardware-portable ---
    ch_mean = flat.mean(axis=0)
    cv = np.divide(per_channel_std, ch_mean, out=np.zeros_like(per_channel_std), where=ch_mean > 1e-9)
    cv_mean = float(cv.mean())
    cv_max = float(cv.max())

    # --- family 5: phase dynamics ---
    ph_std = ph_flat.std(axis=0)
    phase_std_mean = float(ph_std.mean())
    phase_std_max = float(ph_std.max())
    phase_roc_mean = float(np.abs(np.diff(ph_flat, axis=0)).mean())

    # --- family 6: spectral / temporal structure ---
    dop_ratio = _spectral_high_freq_ratio(flat, fs=fs)
    # correlation across a fixed 100 ms gap: thermal noise decorrelates instantly,
    # a moving body does not
    acf_lag = _autocorr_at_lag(flat, fs)

    return np.array([
        mean_amp, mean_amp_std_sc,
        std_amp_mean, std_amp_max, std_amp_std_sc, std_within_frame,
        roc_mean, roc_max, roc_std,
        cv_mean, cv_max,
        phase_std_mean, phase_std_max, phase_roc_mean,
        dop_ratio, acf_lag,
    ], dtype=np.float64)


def windows_from_arrays(amp, ph, window=WINDOW, stride=STRIDE, fs=50.0):
    """Slice a full capture into per-window feature vectors."""
    out = []
    for s in range(0, len(amp) - window + 1, stride):
        out.append(window_features(amp[s:s + window], ph[s:s + window], fs=fs))
    return np.asarray(out) if out else np.empty((0, len(FEATURE_NAMES)))


FEATURE_FAMILY = {
    "mean_amp": "mean amplitude (control)", "mean_amp_std_sc": "mean amplitude (control)",
    "std_amp_mean": "temporal variance", "std_amp_max": "temporal variance",
    "std_amp_std_sc": "temporal variance", "std_within_frame": "mean amplitude (control)",
    "roc_mean": "rate of change", "roc_max": "rate of change", "roc_std": "rate of change",
    "cv_mean": "normalised variance", "cv_max": "normalised variance",
    "phase_std_mean": "phase variance", "phase_std_max": "phase variance",
    "phase_roc_mean": "phase variance",
    "dop_ratio": "spectral/Doppler", "acf_lag": "spectral/Doppler",
}


# ---------------------------------------------------------------------------
# Per-site baseline calibration
# ---------------------------------------------------------------------------
# The single most important lesson from the cross-room evaluation (see train.py):
# a model trained in one room does NOT transfer to another on absolute feature
# values, because every environment has its own noise floor.  RoomA's *empty*
# channel is about twice as agitated as RoomC's empty channel, so a threshold
# learned on RoomC labels RoomA's empty room as occupied.
#
# The fix is to express every feature relative to the quietest thing that site
# has ever seen, rather than in absolute units.  Crucially this needs NO LABELS:
# it only requires that the space is unoccupied for at least a few percent of the
# calibration recording, which for a home is close to guaranteed.
#
# This is also what makes the ESP32 hand-off plausible.  An ESP32 produces
# amplitudes on a completely different scale from an Intel 5300, so absolute
# features are meaningless across the two -- but "5x your own noise floor" means
# the same thing on both radios.

# Features that survive the change of hardware: all are ratios, correlations or
# phase quantities, none carry an absolute amplitude scale.
SCALE_FREE_FEATURES = [
    "cv_mean", "cv_max", "acf_lag", "dop_ratio",
    "phase_std_mean", "phase_std_max", "phase_roc_mean",
]

# Not every scale-free feature may be calibrated the same way, and getting this
# wrong is silent.  Two kinds:
#
#   RATIO  - unbounded positive magnitudes (a coefficient of variation, a phase
#            standard deviation).  "5x my own noise floor" is meaningful, so
#            divide.
#   OFFSET - quantities already bounded on a fixed interval: acf1 is a
#            correlation in [-1, 1] and dop_ratio is an energy fraction in
#            [0, 1].  Dividing these by a baseline that can be near zero or
#            negative is nonsense -- it flips signs and explodes magnitudes.
#            Subtract instead, which asks the same question ("how far above this
#            site's quiet state?") while staying well defined.
RATIO_CALIBRATED = ["cv_mean", "cv_max", "phase_std_mean", "phase_std_max", "phase_roc_mean"]
OFFSET_CALIBRATED = ["acf_lag", "dop_ratio"]

BASELINE_PERCENTILE = 5.0


def fit_site_baseline(X_site, feature_names, percentile=BASELINE_PERCENTILE):
    """Learn one site's quiet baseline from unlabelled windows recorded there.

    Returns the per-feature reference level.  Assumes the site is empty for at
    least `percentile`% of the calibration capture.
    """
    idx = [feature_names.index(f) for f in SCALE_FREE_FEATURES]
    base = np.percentile(X_site[:, idx], percentile, axis=0)
    # guard only the divisors; an offset baseline of zero is perfectly fine
    ratio_mask = np.array([f in RATIO_CALIBRATED for f in SCALE_FREE_FEATURES])
    base = np.where(ratio_mask & (np.abs(base) < 1e-6), 1e-6, base)
    return base


def apply_site_baseline(X_site, feature_names, baseline):
    """Express a site's scale-free features relative to its own quiet baseline."""
    idx = [feature_names.index(f) for f in SCALE_FREE_FEATURES]
    sub = X_site[:, idx]
    out = np.empty_like(sub, dtype=float)
    for j, f in enumerate(SCALE_FREE_FEATURES):
        if f in RATIO_CALIBRATED:
            out[:, j] = sub[:, j] / baseline[j]
        else:
            out[:, j] = sub[:, j] - baseline[j]
    return out
