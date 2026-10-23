# GLRT-Based Spectrum Sensing for Pulse Radar Signals

Python implementation of

> C. H. Lim and D. A. Guimarães, "GLRT-Based Spectrum Sensing Techniques for Pulse
> Radar Signals," *IEEE Communications Letters*, vol. 24, no. 2, pp. 447–450, Feb. 2020.

```
glrt_radar.py    signal model + all detectors
run_figures.py   reproduces Figs. 2, 3 and 4
selftest.py      structural checks (incl. brute-force check of the GLRT)
```

```bash
python selftest.py
python run_figures.py --trials 30000        # ~2.5 min, writes fig2/fig3/fig4 .png
python run_figures.py --quick --figures 2   # smoke run
```
Requires only `numpy` and `matplotlib`.

## What is implemented

| Paper | Function |
|---|---|
| (1), (2) two-level model, random pulse phase over the PRI, Rayleigh fading | `generate` |
| (3)–(10) GLRT statistic for known **n** | folded into the closed form below |
| (11) GLRT, maximised over **n** and *P* | `stat_glrt` |
| (12) OGLRT (ordering-based) | `stat_oglrt` |
| Algorithm 1 SOGLRT | `stat_soglrt` |
| Sec. III-B SWGLRT / SWOGLRT, +SC and +HC | `sw_stats`, `combine_sc`, `combine_hc` |
| Reference detectors MF / ED / EIG | `stat_mf`, `stat_energy`, `stat_eig` |

### The closed form used everywhere

With the pooled ML means, `N·σ̂₁² = Σ|r|² − |Σ_{S₀}r|²/|S₀| − |Σ_{S₁}r|²/|S₁|`, so
maximising (10) is equivalent to maximising the *gain*

```
G = |Σ_{S₀} r|²/|S₀| + |Σ_{S₁} r|²/|S₁| ,     σ̂₀²/σ̂₁² = S/(S − G),   S = Σ|r(n)|²
```

Evaluated with prefix sums this makes the exact GLRT **O(N²)** instead of
combinatorial, and the OGLRT/SOGLRT **O(N log N)**, fully vectorised over
Monte-Carlo trials. `selftest.py` checks the closed form against a literal,
brute-force evaluation of (5)–(7) and (10) over every (n₀, n₁) pair.

Because `x ↦ xᴺ` is increasing, the ROCs use `σ̂₀²/σ̂₁²` rather than
`L = (σ̂₀²/σ̂₁²)ᴺ`; the exponent is exposed via `sw_stats(..., power=...)`, since it
is the one place (soft combining) where it changes the result.

## Choices the letter leaves open

1. **`Â₀`, `Â₁`.** Eqs. (5)–(6) write the level estimates as a *sum of per-segment
   means*. Read literally that is not the ML estimate of a level shared by all
   segments and it breaks the identity (10) (it can give `L < 1`). The pooled mean —
   the estimate that is consistent with (7) and (10) — is used instead.

2. **SNR convention.** The letter does not define its SNR reference. Taking it as the
   **average signal power over the observation window** (so `E|A₁|² = σ²·SNR/duty`,
   i.e. a 10 dB in-pulse boost at PRF = 100 kHz) reproduces Fig. 2 essentially
   point-for-point — see the table below. `--snr-ref pulse` switches to the in-pulse
   reference.

3. **Noise power uncertainty.** Standard model: the true noise variance is
   `σ² = σ²_nom·10^(u/10)` with `u ~ U[−x, x]` drawn per realisation, while the signal
   level stays fixed relative to `σ²_nom`. The GLRT/OGLRT/SOGLRT/EIG statistics are
   *exactly* scale invariant (verified in `selftest.py`), so their H₀ law is untouched;
   the small residual degradation visible in Fig. 2 comes only from the H₁ side, where a
   random σ² randomises the effective SNR. That is precisely the letter's "relatively
   small performance degradation".

4. **Observation window for Figs. 3 and 4.** Not stated. It must exceed one PRI for the
   low-PRF case, so 100 µs (500 samples = 20 segments of 5 µs) is used, giving 10 pulses
   at PRF = 100 kHz and 1 pulse at 10 kHz — exactly the situation described in the text
   ("a lower PRF causes the number of segments with no radar pulses to increase").
   Change it with `--window34`.

5. **Doppler.** With f_d = 10 Hz and windows ≤ 100 µs, `f_d·T ≤ 10⁻³`, so the fading is
   static within a window; block fading is used. A Jakes sum-of-sinusoids generator
   (`fading="jakes"`) is included for completeness.

6. **OR-rule ROC.** Deciding H₁ when any segment exceeds τ is identical to thresholding
   `maxᵢ Tᵢ`, so hard combining is implemented as the maximum, which traces the OR-rule
   ROC exactly as τ is swept.

## Reproduction quality (Fig. 2, 30 000 trials/hypothesis)

P_d at P_fa = 0.01 / 0.05 / 0.10, PRF = 100 kHz, window 9 µs:

| SNR | detector | NPU = 0 dB | NPU = 2 dB |
|---|---|---|---|
| −10 dB | MF | 0.53 / 0.68 / 0.76 | – |
| −10 dB | GLRT | 0.15 / 0.26 / 0.35 | 0.15 / 0.27 / 0.35 |
| −10 dB | OGLRT | 0.06 / 0.14 / 0.22 | 0.06 / 0.15 / 0.22 |
| −10 dB | ED | 0.09 / 0.19 / 0.28 | 0.03 / 0.10 / 0.16 |
| −10 dB | EIG | 0.02 / 0.09 / 0.15 | 0.03 / 0.08 / 0.15 |
| 0 dB | MF | 0.91 / 0.94 / 0.95 | – |
| 0 dB | GLRT | 0.75 / 0.81 / 0.84 | 0.74 / 0.80 / 0.83 |
| 0 dB | OGLRT | 0.63 / 0.71 / 0.75 | 0.62 / 0.70 / 0.74 |
| 0 dB | ED | 0.65 / 0.74 / 0.79 | 0.45 / 0.57 / 0.63 |
| 0 dB | EIG | 0.38 / 0.52 / 0.59 | 0.38 / 0.52 / 0.59 |

These sit on top of the published curves, and every claim in Section IV comes out:

* GLRT > OGLRT (cost of the ordering approximation), and the gap survives segmentation;
* GLRT/OGLRT/EIG are essentially unaffected by 2 dB of NPU while the ED loses ~0.2 in P_d;
* both GLRT and OGLRT beat the ED once NPU = 2 dB;
* SOGLRT → OGLRT as the SNR grows (0.23 vs 0.37 at −10 dB, 0.83 vs 0.90 at 0 dB, P_fa = 0.01);
* SC beats HC at high duty cycle (PRF = 100 kHz: 0.42 vs 0.23) and the advantage
  disappears — here it reverses — at low duty cycle (PRF = 10 kHz: 0.54 vs 0.68), because
  noise-only segments then dominate the soft sum.

## Cost

The exact GLRT of (11) is combinatorial in general; the implementation searches the
P = 1 family (one pulse per window, entire or partial — the three cases of Fig. 1),
which is exactly the regime the letter uses it in and exactly what the segmentation of
Section III-B creates. For a window that genuinely holds P > 1 pulses, use the
sliding-window variants — that is the point the letter is making.
