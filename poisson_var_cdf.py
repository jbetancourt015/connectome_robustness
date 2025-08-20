#!/usr/bin/env python3
"""poisson_var_cdf.py
Compute the CDF of the empirical (population‑denominator) variance of N
independent Poisson(λ) observations.

Two methods are implemented:
  * Monte‑Carlo simulation (exact in the limit, works for any N and λ)
  * Chi‑square/Gamma approximation (accurate once N·λ ≳ 25)

USAGE EXAMPLE
-------------
    # CDF on a 0–10 grid every 0.5 for N=5, λ=3
    python poisson_var_cdf.py 5 3 --grid 0 10 0.5

This prints a table with three columns:
    v   F_MC   F_ChiSq
where F_MC   = Monte‑Carlo estimate  (nsim can be changed)
      F_ChiSq = χ² approximation.
"""

import argparse, math, sys
from collections import namedtuple

import numpy as np
from scipy.stats import chi2
from scipy.special import loggamma


def sample_variances(N: int, lam: float, nsim: int = 200_000, rng=None):
    """Draw *nsim* vectors of length N from Poisson(λ) and return their variances."""
    rng = np.random.default_rng(rng)
    X = rng.poisson(lam, size=(nsim, N))
    return X.var(axis=1, ddof=0)


def cdf_mc(N: int, lam: float, v_vals, nsim: int = 200_000, rng=None):
    """Monte‑Carlo estimate of the CDF at the points in *v_vals*."""
    var_sample = sample_variances(N, lam, nsim, rng)
    var_sorted = np.sort(var_sample)
    idx = np.searchsorted(var_sorted, v_vals, side="right")
    return idx / nsim


def cdf_chisq(N: int, lam: float, v_vals):
    """Chi‑square (Gamma) approximation to the CDF."""
    df = N - 1
    return chi2.cdf(df * v_vals / lam, df)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="CDF of the empirical variance of Poisson counts."
    )
    p.add_argument("N", type=int, help="sample size (number of Poisson draws)")
    p.add_argument("lam", type=float, help="Poisson mean λ")
    p.add_argument(
        "--grid",
        nargs=3,
        type=float,
        metavar=("v_min", "v_max", "step"),
        default=(0.0, 5.0, 0.25),
        help="Grid: start end step (default 0 5 0.25)",
    )
    p.add_argument(
        "--nsim",
        type=int,
        default=200_000,
        help="Monte‑Carlo draws (default 200 000)",
    )
    args = p.parse_args(argv)

    v_min, v_max, step = args.grid
    v_vals = np.arange(v_min, v_max + 1e-12, step)

    F_mc = cdf_mc(args.N, args.lam, v_vals, args.nsim)
    F_chi2 = cdf_chisq(args.N, args.lam, v_vals)

    header = (
        f"# CDF of V_N with N={args.N}, λ={args.lam}, grid={v_min}:{step}:{v_max}, "
        f"nsim={args.nsim}"
    )
    print(header)
    print("# v\tF_MC\tF_ChiSq")
    for v, f1, f2 in zip(v_vals, F_mc, F_chi2):
        print(f"{v:.6g}\t{f1:.6g}\t{f2:.6g}")


if __name__ == "__main__":  # pragma: no cover
    main()


# ----------------------------------------------------------------------
# Exact CDF by Poisson–multinomial enumeration
# ----------------------------------------------------------------------
from functools import lru_cache
from math import comb, exp

def _poisson_weight(k, mean):
    """Poisson PMF."""
    return exp(-mean) * mean**k / math.factorial(k)

@lru_cache(maxsize=None)
def _cond_prob_q_le(ncells, s, q_limit, current_q=0):
    """Recursive multinomial enumeration.

    Returns P(sum Xi^2 + current_q <= q_limit | sum Xi = s, equal cell probs),
    where Xi counts across *ncells* remaining cells.
    """
    if current_q > q_limit:
        return 0.0
    if ncells == 1:
        # Last cell takes all remaining counts
        return 1.0 if current_q + s * s <= q_limit else 0.0

    p = 1.0 / ncells
    prob = 0.0
    for x in range(s + 1):
        q_new = current_q + x * x
        if q_new > q_limit:
            break  # increasing x only increases q_new
        # Binomial probability for the first cell
        binom_p = comb(s, x) * (p ** x) * ((1 - p) ** (s - x))
        prob += binom_p * _cond_prob_q_le(ncells - 1, s - x, q_limit, q_new)
    return prob


def cdf_exact(N: int, lam: float, v_val: float, eps: float = 1e-6, s_max: int = None):
    """Exact CDF of V_N at a single point v_val.

    The outer Poisson sum is truncated at the first s where the remaining tail
    mass < *eps* (unless *s_max* is given explicitly).

    WARNING: The enumeration grows quickly with N and v; practical for
    N\le8–10 or moderate λ.  For larger problems, use Monte‑Carlo or χ² approx.
    """

    mean_total = N * lam
    tail_mass = 1.0
    s = 0
    cdf_val = 0.0

    # Precompute Poisson PMFs until tail small enough
    while tail_mass > eps:
        if s_max is not None and s > s_max:
            break
        log_pmf = -mean_total + s*np.log(mean_total) - loggamma(s+1)
        pois_pmf = exp(log_pmf)
        q_limit = int(math.floor(N * v_val + (s * s) / N))
        cond_prob = _cond_prob_q_le(N, s, q_limit)
        cdf_val += pois_pmf * cond_prob
        tail_mass -= pois_pmf
        s += 1

    return min(max(cdf_val, 0.0), 1.0)
