r"""
glrt_radar.py
=============

Python implementation of

    C. H. Lim and D. A. Guimaraes, "GLRT-Based Spectrum Sensing Techniques for
    Pulse Radar Signals", IEEE Communications Letters, vol. 24, no. 2,
    pp. 447-450, Feb. 2020.

Everything follows the equations of the letter:

    (1),(2)   two-level received-signal model
    (3)-(10)  GLRT test statistic for a *known* change-point vector n
    (11)      GLRT, maximised over the unknown {n_0,...,n_{2P-1}} and P
    (12)      OGLRT  (ordering-based GLRT)
    Alg. 1    SOGLRT (simplified ordering-based GLRT)
    Sec.III-B SWGLRT / SWOGLRT with soft (+SC) and hard (+HC) combining

plus the reference detectors used for comparison: matched filter (MF),
energy detection (ED) and max-min eigenvalue ratio detection (EIG).

Implementation notes
--------------------
* Eq. (10) shows L(r|n) = (sigma0^2 / sigma1^2)^N.  Because x -> x^N is strictly
  increasing, every ROC computed here uses the monotonically equivalent ratio
  rho = sigma0^2 / sigma1^2 (numerically far better behaved for N ~ 500).
  The N-th power only matters for the *soft* combining rule, where it is
  exposed through the `power` argument of `sw_stats`.

* Eqs. (5)-(6) write \hat A_0 (resp. \hat A_1) as a *sum of per-segment means*.
  Taken literally that is not the ML estimate of a level that is common to all
  segments; the ML estimate consistent with (7) and (10) is the pooled mean,
  which is what is used here (and what makes L >= 1 always).

* With the pooled means, maximising (10) is equivalent to *minimising*
  N*sigma1^2 = sum|r|^2 - |sum_{S0} r|^2/|S0| - |sum_{S1} r|^2/|S1|,
  i.e. to *maximising* the "gain"

      G = |sum_{S0} r|^2/|S0| + |sum_{S1} r|^2/|S1| ,      rho = S / (S - G)

  with S = sum_n |r(n)|^2.  All detectors below are written in this form, which
  makes them O(N^2) (GLRT) or O(N log N) (OGLRT/SOGLRT) and fully vectorised
  over Monte-Carlo realisations.

* SNR convention (see README): SNR is the *average* received signal power over
  the observation window divided by the (nominal) noise power, therefore the
  in-pulse level obeys  E|A_1|^2 = sigma_nom^2 * SNR / duty_cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_TINY = np.finfo(np.float64).tiny


# --------------------------------------------------------------------------- #
#  System model  (Section II)
# --------------------------------------------------------------------------- #
@dataclass
class RadarConfig:
    """Sampling / waveform parameters of Section IV of the letter."""

    fs: float = 5e6           # sampling rate [Hz]            (5 MHz, [6, Annex D])
    pulse_width: float = 1e-6  # radar pulse width [s]        (1 us)
    prf: float = 100e3        # pulse repetition frequency [Hz] (10 kHz or 100 kHz)
    window: float = 9e-6      # observation window length [s]
    segment: float = 5e-6     # sliding-window (segment) length [s]
    doppler: float = 10.0     # Rayleigh fading Doppler frequency [Hz]

    # ---- derived quantities, in samples --------------------------------- #
    @property
    def N(self) -> int:
        """Observation window length N, in samples."""
        return int(round(self.window * self.fs))

    @property
    def Lp(self) -> int:
        """Pulse width, in samples."""
        return int(round(self.pulse_width * self.fs))

    @property
    def pri(self) -> int:
        """Pulse repetition interval, in samples."""
        return int(round(self.fs / self.prf))

    @property
    def Lseg(self) -> int:
        """Segment length of the sliding-window schemes, in samples."""
        return int(round(self.segment * self.fs))

    @property
    def duty(self) -> float:
        """Radar duty cycle = pulse width * PRF."""
        return self.pulse_width * self.prf


def jakes_fading(M: int, N: int, fd: float, fs: float, rng, n_osc: int = 16):
    """Clarke/Jakes sum-of-sinusoids flat fading, E|h|^2 = 1.

    Provided for completeness: with fd = 10 Hz and windows of at most 100 us,
    fd*T <= 1e-3, so the channel is static over the window to within ~0.4
    degrees of phase -- `generate` therefore uses block fading by default.
    """
    theta = rng.uniform(0, 2 * np.pi, size=(M, n_osc, 1))
    phi = rng.uniform(0, 2 * np.pi, size=(M, n_osc, 1))
    n = np.arange(N)[None, None, :]
    ph = 2 * np.pi * fd * np.cos(theta) * n / fs + phi
    return np.exp(1j * ph).sum(axis=1) / np.sqrt(n_osc)


def generate(M, cfg: RadarConfig, hypothesis: str, snr_db: float,
             npu_db: float = 0.0, rng=None, sigma_nom: float = 1.0,
             fading: str = "block", snr_ref: str = "window"):
    """Draw `M` independent observation windows, eqs. (1) and (2).

    Parameters
    ----------
    hypothesis : 'H0' (noise only) or 'H1' (radar present).
    snr_db     : signal-to-noise ratio [dB], referred either to
                 'window' -- the average signal power over the observation
                             window, i.e. E|A_1|^2 = sigma_nom^2*SNR/duty.  This
                             is the convention that reproduces Fig. 2; or
                 'pulse'  -- the in-pulse level, E|A_1|^2 = sigma_nom^2*SNR.
    npu_db     : noise power uncertainty x [dB]; the true noise variance is
                 sigma^2 = sigma_nom^2 * 10^(u/10) with u ~ U[-x, +x], drawn
                 independently per realisation.  Detector thresholds are set on
                 the *nominal* noise power, which is the standard NPU model.

    Returns
    -------
    r : (M, N) complex ndarray -- received samples.
    s : (M, N) complex ndarray -- the noiseless radar component (the matched
        filter template).  Under H0 it is a "would-be" signal, statistically
        identical to the H1 one; it is never added to r.
    """
    rng = np.random.default_rng() if rng is None else rng
    N, Lp, pri = cfg.N, cfg.Lp, cfg.pri

    # --- pulse train position: start instant uniform over the PRI ---------- #
    # The modulo construction also produces windows that contain only a
    # *fragment* of a pulse at either edge (the three cases of Fig. 1).
    phase = rng.integers(0, pri, size=(M, 1))
    n = np.arange(N)[None, :]
    mask = ((n + phase) % pri) < Lp                       # (M, N) boolean

    # --- static (block) Rayleigh fading, E|h|^2 = 1 ------------------------ #
    if fading == "block":
        h = (rng.standard_normal((M, 1)) + 1j * rng.standard_normal((M, 1)))
        h = h / np.sqrt(2.0)
    elif fading == "jakes":
        h = jakes_fading(M, N, cfg.doppler, cfg.fs, rng)
    else:
        raise ValueError("fading must be 'block' or 'jakes'")

    snr = 10.0 ** (snr_db / 10.0)
    if snr_ref == "window":
        # average power over the window = duty*|A1|^2  ==>  |A1|^2 = snr/duty
        a1 = np.sqrt(snr / cfg.duty) * sigma_nom
    elif snr_ref == "pulse":
        a1 = np.sqrt(snr) * sigma_nom
    else:
        raise ValueError("snr_ref must be 'window' or 'pulse'")
    s = (a1 * h * mask).astype(np.complex128)             # A_0 = 0, A_1 = a1*h

    # --- noise with (possibly uncertain) power ----------------------------- #
    u = rng.uniform(-npu_db, npu_db, size=(M, 1)) if npu_db > 0 else 0.0
    sigma2 = (sigma_nom ** 2) * (10.0 ** (np.asarray(u) / 10.0))
    w = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N)))
    w *= np.sqrt(sigma2 / 2.0)

    r = w + s if hypothesis.upper() == "H1" else w
    return r, s


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _cumsum0(R):
    """Prefix sums with a leading zero: C[:, k] = sum of the first k samples."""
    C = np.empty((R.shape[0], R.shape[1] + 1), dtype=R.dtype)
    C[:, 0] = 0
    np.cumsum(R, axis=1, out=C[:, 1:])
    return C


def _abs2(z):
    return z.real ** 2 + z.imag ** 2


def _ratio(energy, gain):
    """sigma0^2/sigma1^2 = S / (S - G), clipped to be >= 1."""
    den = np.maximum(energy - gain, _TINY)
    return np.maximum(energy / den, 1.0)


# --------------------------------------------------------------------------- #
#  Reference detectors
# --------------------------------------------------------------------------- #
def stat_energy(R):
    """Energy detection: T = sum_n |r(n)|^2."""
    return _abs2(R).sum(axis=1)


def stat_mf(R, S):
    """Coherent matched filter with perfect knowledge of the received signal.

    T = Re{ sum_n r(n) s*(n) } / ||s||, whose H0 law (N(0, sigma^2/2)) does not
    depend on the fading realisation, so a single threshold is CFAR.
    """
    nrm = np.sqrt(_abs2(S).sum(axis=1))
    return np.real(np.einsum("mn,mn->m", R, np.conj(S))) / np.maximum(nrm, _TINY)


def stat_eig(R, smoothing: int = 10, chunk: int = 2048):
    """Max-min eigenvalue ratio detection with the given smoothing factor."""
    M, N = R.shape
    L = smoothing
    K = N - L + 1
    out = np.empty(M)
    for lo in range(0, M, chunk):
        X = R[lo:lo + chunk]
        Y = sliding_window_view(X, L, axis=1)              # (m, K, L)
        Rc = np.einsum("mkl,mkp->mlp", np.conj(Y), Y) / K  # (m, L, L)
        ev = np.linalg.eigvalsh(Rc)                        # ascending
        out[lo:lo + chunk] = ev[:, -1] / np.maximum(ev[:, 0], _TINY)
    return out


# --------------------------------------------------------------------------- #
#  GLRT, eq. (11)
# --------------------------------------------------------------------------- #
def _max_interval_gain(R, chunk_elems: int = 4_000_000):
    """max over every contiguous interval I of |sum_I r|^2/|I| + |sum_Ic r|^2/|Ic|.

    This is the exact maximisation of (11) for P = 1, i.e. for windows short
    enough to contain a single radar pulse, entirely or partially (Fig. 1).
    The sliding-window schemes of Section III-B put every segment in exactly
    this regime, which is precisely why the segmentation is done.
    """
    M, N = R.shape
    G = np.full(M, -np.inf)
    rows = max(1, int(chunk_elems // max(N, 1)))
    for lo in range(0, M, rows):
        X = R[lo:lo + rows]
        C = _cumsum0(X)
        total = C[:, -1]
        best = np.full(X.shape[0], -np.inf)
        for m in range(1, N):                    # interval length |I| = m
            D = C[:, m:] - C[:, :N + 1 - m]      # (rows, N-m+1) interval sums
            E = total[:, None] - D               # complement sums
            val = _abs2(D) / m + _abs2(E) / (N - m)
            np.maximum(best, val.max(axis=1), out=best)
        G[lo:lo + rows] = best
    return G


def stat_glrt(R):
    """GLRT test statistic of eq. (11) (returned as sigma0^2/sigma1^2)."""
    return _ratio(stat_energy(R), _max_interval_gain(R))


# --------------------------------------------------------------------------- #
#  OGLRT, eq. (12), and SOGLRT, Algorithm 1
# --------------------------------------------------------------------------- #
def _ordered_split_gain(R):
    r"""Gain of every prefix/suffix split of the magnitude-ordered samples.

    Returns
    -------
    val : (M, N-1) array; column k holds the gain for n_0 = k + 1, i.e. for
          S_0 = { \hat r(1), ..., \hat r(n_0) } (low magnitudes, level A_0) and
          S_1 = the remaining samples (high magnitudes, level A_1).
    """
    M, N = R.shape
    idx = np.argsort(np.abs(R), axis=1, kind="stable")     # ascending magnitude
    Rs = np.take_along_axis(R, idx, axis=1)                # \hat r
    C = _cumsum0(Rs)
    total = C[:, -1]
    n0 = np.arange(1, N)[None, :]
    D = C[:, 1:N]                                          # sums of first n_0
    E = total[:, None] - D
    return _abs2(D) / n0 + _abs2(E) / (N - n0), Rs


def stat_oglrt(R):
    """Ordering-based GLRT, eq. (12): maximisation over the single n_0."""
    val, _ = _ordered_split_gain(R)
    return _ratio(stat_energy(R), val.max(axis=1))


def stat_soglrt(R):
    """Simplified ordering-based GLRT: n_0 from Algorithm 1 (largest jump)."""
    val, Rs = _ordered_split_gain(R)
    mag = np.abs(Rs)
    j = np.argmax(np.diff(mag, axis=1), axis=1) + 1        # n_0 = j (1..N-1)
    gain = np.take_along_axis(val, (j - 1)[:, None], axis=1)[:, 0]
    return _ratio(stat_energy(R), gain)


# --------------------------------------------------------------------------- #
#  Sliding-window schemes, Section III-B
# --------------------------------------------------------------------------- #
_BASE = {"glrt": stat_glrt, "oglrt": stat_oglrt, "soglrt": stat_soglrt}


def sw_stats(R, seg_len: int, base: str = "glrt", power: int | None = None):
    """Per-segment statistics of the SW(O)GLRT schemes.

    The window is split into non-overlapping segments of `seg_len` samples,
    each short enough to hold at most one pulse (partially or entirely), and
    the corresponding GLRT/OGLRT statistic is evaluated on each segment.

    `power` raises each intermediate statistic to that power before it is
    returned; `power=seg_len` reproduces L = (sigma0^2/sigma1^2)^N literally,
    `power=None` (default) keeps the monotonically equivalent ratio, which is
    numerically stable and is the form used for the soft-combining sum.
    """
    M, N = R.shape
    ns = N // seg_len
    X = np.ascontiguousarray(R[:, :ns * seg_len]).reshape(M * ns, seg_len)
    t = _BASE[base](X).reshape(M, ns)
    return t if power is None else t ** power


def combine_sc(seg):
    """Soft-decision combining: sum of the intermediate test statistics."""
    return seg.sum(axis=1)


def combine_hc(seg):
    """Hard-decision combining with the OR rule.

    Deciding H1 as soon as any segment exceeds a common threshold tau is
    identical to thresholding max_i T_i, so the maximum is the equivalent test
    statistic and sweeping it traces the OR-rule ROC exactly.
    """
    return seg.max(axis=1)


# --------------------------------------------------------------------------- #
#  ROC utilities
# --------------------------------------------------------------------------- #
def roc(t0, t1, pfa_grid):
    """Empirical ROC: thresholds are the (1-Pfa) quantiles of the H0 statistic."""
    pfa_grid = np.asarray(pfa_grid, dtype=float)
    s0 = np.sort(t0)
    s1 = np.sort(t1)
    m0, m1 = s0.size, s1.size
    k = np.clip(np.ceil((1.0 - pfa_grid) * m0).astype(int), 1, m0)
    thr = s0[k - 1]
    pd = 1.0 - np.searchsorted(s1, thr, side="right") / m1
    return pd
