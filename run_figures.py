r"""
run_figures.py -- reproduce Figs. 2, 3 and 4 of

    C. H. Lim and D. A. Guimaraes, "GLRT-Based Spectrum Sensing Techniques for
    Pulse Radar Signals", IEEE Commun. Lett., 24(2):447-450, Feb. 2020.

Usage
-----
    python run_figures.py                    # all three figures, default trials
    python run_figures.py --trials 50000     # smoother curves (slower)
    python run_figures.py --figures 2        # only Fig. 2
    python run_figures.py --quick            # fast smoke run
"""

from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glrt_radar import (RadarConfig, generate, roc, stat_energy, stat_eig,
                        stat_glrt, stat_mf, stat_oglrt, stat_soglrt, sw_stats,
                        combine_hc, combine_sc)

PFA = np.arange(0.01, 0.115, 0.01)          # the low-Pfa region plotted in the paper
TARGETS = (0.01, 0.05, 0.1)                 # target false alarm rates of Section IV


# --------------------------------------------------------------------------- #
#  Fig. 2 -- MF, ED, GLRT, OGLRT, EIG;  single pulse per window
# --------------------------------------------------------------------------- #
def figure2(trials, outfile, seed=2020, snr_ref="window", zoom=True):
    cfg = RadarConfig(prf=100e3, window=9e-6)      # N = 45 samples, P <= 1
    curves = {}
    for snr_db in (-10.0, 0.0):
        for npu in (0.0, 2.0):
            rng = np.random.default_rng(seed + int(snr_db) + int(10 * npu))
            r0, s0 = generate(trials, cfg, "H0", snr_db, npu, rng,
                              snr_ref=snr_ref)
            r1, s1 = generate(trials, cfg, "H1", snr_db, npu, rng,
                              snr_ref=snr_ref)

            def add(name, f0, f1):
                curves[(snr_db, npu, name)] = roc(f0, f1, PFA)

            if npu == 0.0:                          # the letter plots MF only at NPU = 0
                add("MF", stat_mf(r0, s0), stat_mf(r1, s1))
            add("GLRT", stat_glrt(r0), stat_glrt(r1))
            add("OGLRT", stat_oglrt(r0), stat_oglrt(r1))
            add("ED", stat_energy(r0), stat_energy(r1))
            add("EIG", stat_eig(r0, 10), stat_eig(r1, 10))
            del r0, r1, s0, s1

    style = {"MF": ("k", "*"), "GLRT": ("m", "s"), "OGLRT": ("b", "o"),
             "ED": ("r", "^"), "EIG": ("g", "v")}
    order = ["MF", "GLRT", "OGLRT", "ED", "EIG"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, snr_db in zip(axes, (-10.0, 0.0)):
        for name in order:
            c, mk = style[name]
            for npu, ls, fill in ((0.0, "-", True), (2.0, ":", False)):
                key = (snr_db, npu, name)
                if key not in curves:
                    continue
                ax.plot(PFA, curves[key], color=c, ls=ls, marker=mk, ms=5,
                        lw=1.4, mfc=(c if fill else "none"),
                        label=f"{name}(NPU = {npu:.0f}dB)")
        _decorate(ax, f"SNR = {snr_db:.0f} dB",
                  [v for k, v in curves.items() if k[0] == snr_db], zoom)
    fig.suptitle("Fig. 2 - MF, ED, GLRT, OGLRT and EIG "
                 "(PRF = 100 kHz, observation window = 9 $\\mu$s)", y=0.99)
    _save(fig, outfile)
    return curves


# --------------------------------------------------------------------------- #
#  Figs. 3 and 4 -- ED, OGLRT, SOGLRT and the sliding-window variants
# --------------------------------------------------------------------------- #
def figure34(trials, prf, outfile, window=100e-6, npu=0.0, seed=3030,
             snr_ref="window", zoom=True):
    cfg = RadarConfig(prf=prf, window=window)
    n_seg = cfg.N // cfg.Lseg
    curves = {}
    for snr_db in (-10.0, 0.0):
        rng = np.random.default_rng(seed + int(snr_db))
        r0, _ = generate(trials, cfg, "H0", snr_db, npu, rng, snr_ref=snr_ref)
        r1, _ = generate(trials, cfg, "H1", snr_db, npu, rng, snr_ref=snr_ref)

        curves[(snr_db, "ED")] = roc(stat_energy(r0), stat_energy(r1), PFA)
        curves[(snr_db, "OGLRT")] = roc(stat_oglrt(r0), stat_oglrt(r1), PFA)
        curves[(snr_db, "SOGLRT")] = roc(stat_soglrt(r0), stat_soglrt(r1), PFA)
        for base, tag in (("glrt", "SWGLRT"), ("oglrt", "SWOGLRT")):
            a = sw_stats(r0, cfg.Lseg, base)
            b = sw_stats(r1, cfg.Lseg, base)
            curves[(snr_db, tag + "+SC")] = roc(combine_sc(a), combine_sc(b), PFA)
            curves[(snr_db, tag + "+HC")] = roc(combine_hc(a), combine_hc(b), PFA)
        del r0, r1

    style = {"ED": ("r", "^", "-"), "OGLRT": ("b", "o", "-"),
             "SOGLRT": ("c", "d", "--"), "SWGLRT+SC": ("m", "s", "-"),
             "SWGLRT+HC": ("m", "s", ":"), "SWOGLRT+SC": ("g", "v", "-"),
             "SWOGLRT+HC": ("g", "v", ":")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, snr_db in zip(axes, (-10.0, 0.0)):
        for name, (c, mk, ls) in style.items():
            ax.plot(PFA, curves[(snr_db, name)], color=c, ls=ls, marker=mk,
                    ms=5, lw=1.4, mfc=(c if ls != ":" else "none"), label=name)
        _decorate(ax, f"SNR = {snr_db:.0f} dB",
                  [v for k, v in curves.items() if k[0] == snr_db], zoom)
    num = 3 if prf > 50e3 else 4
    fig.suptitle(f"Fig. {num} - ED, OGLRT, SOGLRT and SWGLRT variants "
                 f"(PRF = {prf/1e3:.0f} kHz, window = {window*1e6:.0f} $\\mu$s, "
                 f"{n_seg} segments of {cfg.segment*1e6:.0f} $\\mu$s)", y=0.99)
    _save(fig, outfile)
    return curves


# --------------------------------------------------------------------------- #
def _decorate(ax, title, data=None, zoom=True):
    ax.set_xlabel("False alarm probability")
    ax.set_ylabel("Detection probability")
    ax.set_xlim(0, 0.12)
    ax.set_xticks(np.arange(0, 0.13, 0.02))

    # The letter uses a full 0..1 ordinate.  When every curve is crowded into
    # the top of that range (long observation windows at SNR = 0 dB) the panel
    # becomes unreadable, so the axis is then zoomed onto the occupied range.
    lo, hi = 0.0, 1.0
    if zoom and data:
        vmin = float(np.min([np.min(v) for v in data]))
        if vmin > 0.55:
            lo = np.floor((vmin - 0.03) * 20) / 20
    ax.set_ylim(lo, hi)
    step = 0.1 if (hi - lo) > 0.35 else 0.02
    ax.set_yticks(np.arange(lo, hi + 1e-9, step))
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=0.9)


def _save(fig, outfile):
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"  -> wrote {outfile}")


def _table(curves, keys):
    """Print Pd at the target false alarm rates of Section IV."""
    idx = [int(np.argmin(np.abs(PFA - t))) for t in TARGETS]
    hdr = "  " + "detector".ljust(24) + "".join(f"Pd@Pfa={t:<8.2f}" for t in TARGETS)
    print(hdr)
    for k in keys:
        v = curves[k]
        label = " | ".join(str(x) for x in (k if isinstance(k, tuple) else (k,)))
        print("  " + label.ljust(24) + "".join(f"{v[i]:<14.3f}" for i in idx))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20000,
                   help="Monte Carlo realisations per hypothesis (default 20000)")
    p.add_argument("--window34", type=float, default=100e-6,
                   help="observation window for Figs. 3/4 [s] (default 100 us)")
    p.add_argument("--npu34", type=float, default=0.0,
                   help="noise power uncertainty used in Figs. 3/4 [dB]")
    p.add_argument("--figures", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--snr-ref", choices=["window", "pulse"], default="window",
                   help="SNR reference: average power over the window (default, "
                        "reproduces Fig. 2) or the in-pulse level")
    p.add_argument("--no-zoom", action="store_true",
                   help="always use the paper's full 0..1 ordinate")
    p.add_argument("--quick", action="store_true", help="2000 trials smoke run")
    args = p.parse_args()

    trials = 2000 if args.quick else args.trials
    print(f"Monte Carlo trials per hypothesis: {trials}\n")

    if 2 in args.figures:
        t = time.time()
        print("Fig. 2 ...")
        c = figure2(trials, "fig2_mf_ed_glrt_oglrt_eig.png",
                    snr_ref=args.snr_ref, zoom=not args.no_zoom)
        for snr in (-10.0, 0.0):
            print(f"  SNR = {snr:.0f} dB")
            _table(c, [k for k in c if k[0] == snr])
        print(f"  ({time.time()-t:.1f} s)\n")

    for fignum, prf in ((3, 100e3), (4, 10e3)):
        if fignum not in args.figures:
            continue
        t = time.time()
        print(f"Fig. {fignum} (PRF = {prf/1e3:.0f} kHz) ...")
        c = figure34(trials, prf, f"fig{fignum}_prf{int(prf/1e3)}kHz.png",
                     window=args.window34, npu=args.npu34,
                     snr_ref=args.snr_ref, zoom=not args.no_zoom)
        for snr in (-10.0, 0.0):
            print(f"  SNR = {snr:.0f} dB")
            _table(c, [k for k in c if k[0] == snr])
        print(f"  ({time.time()-t:.1f} s)\n")


if __name__ == "__main__":
    main()
