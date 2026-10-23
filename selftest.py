r"""selftest.py -- structural checks on the implemented statistics.

Run with:  python selftest.py
"""
import numpy as np

from glrt_radar import (RadarConfig, generate, roc, stat_energy, stat_eig,
                        stat_glrt, stat_oglrt, stat_soglrt, sw_stats,
                        combine_hc, combine_sc)

rng = np.random.default_rng(7)
cfg = RadarConfig(prf=100e3, window=9e-6)
R, S = generate(4000, cfg, "H1", -3.0, 0.0, rng)
N = cfg.N
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("1. brute-force cross-check of the GLRT, eq. (11) with P = 1")
# direct evaluation of (5)-(7) and (10) with pooled ML means, over all (n0, n1)
X = R[:20]
brute = np.empty(X.shape[0])
for i, r in enumerate(X):
    s0hat = np.mean(np.abs(r) ** 2)
    best = np.inf
    for a in range(N):
        for b in range(a + 1, N + 1):
            if b - a == N:
                continue
            I = np.zeros(N, bool)
            I[a:b] = True                       # samples at level A_1
            A1 = r[I].mean()
            A0 = r[~I].mean()
            s1hat = (np.abs(r[I] - A1) ** 2).sum() + (np.abs(r[~I] - A0) ** 2).sum()
            best = min(best, s1hat / N)
    brute[i] = s0hat / best
check("closed-form GLRT == brute-force max over (n_0, n_1)",
      np.allclose(brute, stat_glrt(X), rtol=1e-9))

print("2. eq. (10) sanity")
check("L = (sigma0^2/sigma1^2)^N >= 1 for GLRT", np.all(stat_glrt(R) >= 1.0))
check("L >= 1 for OGLRT", np.all(stat_oglrt(R) >= 1.0))
# NB: the OGLRT statistic is *not* upper-bounded by the GLRT one.  Ordering by
# magnitude makes the partition non-contiguous in time, so it lies outside the
# family searched by (11); the OGLRT is sub-optimal in detection performance,
# not in the attained value of L.  What does hold is that the prefix/suffix
# split of (12) is one particular contiguous split of the *ordered* sequence.
_idx = np.argsort(np.abs(R), axis=1, kind="stable")
_Rs = np.take_along_axis(R, _idx, axis=1)
check("OGLRT(r) <= GLRT applied to the magnitude-ordered sequence",
      np.all(stat_oglrt(R) <= stat_glrt(_Rs) + 1e-9))
check("OGLRT(r) == OGLRT of any permutation of r (order-independent)",
      np.allclose(stat_oglrt(R), stat_oglrt(_Rs)))
check("OGLRT >= SOGLRT (Alg. 1 picks one n_0 of the OGLRT search)",
      np.all(stat_oglrt(R) >= stat_soglrt(R) - 1e-9))

print("3. invariance to the noise power (why NPU barely hurts these detectors)")
c = 3.7
check("GLRT scale invariant", np.allclose(stat_glrt(R), stat_glrt(c * R)))
check("OGLRT scale invariant", np.allclose(stat_oglrt(R), stat_oglrt(c * R)))
check("SOGLRT scale invariant", np.allclose(stat_soglrt(R), stat_soglrt(c * R)))
check("EIG scale invariant", np.allclose(stat_eig(R), stat_eig(c * R)))
check("ED is NOT scale invariant", not np.allclose(stat_energy(R),
                                                   stat_energy(c * R)))

print("4. hard-decision combining == OR rule")
seg = sw_stats(R, cfg.Lseg, "glrt")
tau = np.median(seg)
check("max_i T_i > tau  <=>  any(T_i > tau)",
      np.array_equal(combine_hc(seg) > tau, (seg > tau).any(axis=1)))
check("soft combining sums the intermediate statistics",
      np.allclose(combine_sc(seg), seg.sum(axis=1)))

print("5. the empirical ROC is CFAR")
n0, _ = generate(20000, cfg, "H0", -3.0, 0.0, rng)
n1, _ = generate(20000, cfg, "H0", -3.0, 0.0, rng)   # H0 vs H0 -> ROC on the diagonal
pfa = np.array([0.01, 0.05, 0.1])
check("Pd(H0 vs H0) == Pfa within Monte-Carlo error",
      np.max(np.abs(roc(stat_glrt(n0), stat_glrt(n1), pfa) - pfa)) < 0.01)

print("6. pulse-train generation")
r1, s1 = generate(5000, cfg, "H1", 0.0, 0.0, rng)
on = (np.abs(s1) > 0).sum(axis=1)
check("mean number of in-pulse samples == duty * N",
      abs(on.mean() - cfg.duty * N) < 0.15)
check("in-pulse samples are contiguous within the window",
      all(np.count_nonzero(np.diff((np.abs(s) > 0).astype(int)) != 0) <= 2
          for s in s1[:500]))
check("E|A_1|^2 = SNR/duty (window-referred SNR convention)",
      abs(np.mean(np.abs(s1[np.abs(s1) > 0]) ** 2) / (1.0 / cfg.duty) - 1) < 0.06)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
