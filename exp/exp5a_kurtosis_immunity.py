"""
Experiment 5a: Kurtosis Immunity and Variance Inflation for Scale-Invariant Functionals.

Verifies Proposition 8: Var_t(AR_K)/Var_gaussian(AR_K) = 1+kappa for elliptical distributions.
Tests coverage of delta-method confidence intervals with and without kappa correction.
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, qis
from src.simulation import generate_spiked_cov, generate_elliptical_returns, generate_gaussian_returns
from src.functionals import (
    compute_AR_K, compute_f1, compute_spectral_entropy, compute_trace,
    delta_method_var_AR_K, delta_method_var_f1, delta_method_var_H,
    delta_method_var_trace, estimate_kappa_radial_mle,
)


SPIKES = [6.0, 3.5, 1.2]
SIGMA2 = 0.5
K = 2
M = 252
N_REPS = 25
SEED = 42

DISTRIBUTIONS = {
    "Gaussian": {"nu": None, "kappa_true": 0.0},
    "t_8":     {"nu": 8.0,   "kappa_true": 0.5},
    "t_5":     {"nu": 5.0,   "kappa_true": 2.0},
}

Z95 = 1.959964  # z_{0.975}


def _true_values(N: int) -> dict:
    """Compute true population functional values."""
    Sigma = generate_spiked_cov(N, SPIKES, SIGMA2)
    pop_evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    return {
        "AR_K": compute_AR_K(pop_evals, K),
        "f1": compute_f1(pop_evals),
        "H": compute_spectral_entropy(pop_evals),
        "tr": compute_trace(pop_evals),
        "pop_evals": pop_evals,
        "Sigma": Sigma,
    }


def run_for_N_dist(N: int, dist_name: str) -> dict:
    """
    Run variance inflation and CI coverage check for one (N, distribution).
    """
    rng = np.random.default_rng(SEED)
    truth = _true_values(N)
    Sigma = truth["Sigma"]
    pop_evals = truth["pop_evals"]
    dist_cfg = DISTRIBUTIONS[dist_name]
    kappa_true = dist_cfg["kappa_true"]
    nu = dist_cfg["nu"]

    # Storage
    ar_vals = []
    f1_vals = []
    h_vals = []
    tr_vals = []

    cov_ar_gauss = []
    cov_ar_kappa = []
    kappa_hats = []

    for _ in range(N_REPS):
        if nu is None:
            R = generate_gaussian_returns(Sigma, M, rng)
        else:
            R = generate_elliptical_returns(Sigma, M, nu, rng)
        S = sample_cov(R)
        C = qis(S, M)
        evals_C = np.sort(np.linalg.eigvalsh(C))[::-1]

        ar = compute_AR_K(evals_C, K)
        f1 = compute_f1(evals_C)
        h = compute_spectral_entropy(evals_C)
        tr = compute_trace(evals_C)

        ar_vals.append(ar)
        f1_vals.append(f1)
        h_vals.append(h)
        tr_vals.append(tr)

        # Estimate kappa
        kappa_hat = estimate_kappa_radial_mle(R, S)
        kappa_hats.append(kappa_hat)

        # CI for AR_K
        var_gauss = delta_method_var_AR_K(evals_C, K, kappa=0.0) / M
        var_kappa = delta_method_var_AR_K(evals_C, K, kappa=kappa_hat) / M
        ci_gauss = (ar - Z95 * np.sqrt(max(0, var_gauss)), ar + Z95 * np.sqrt(max(0, var_gauss)))
        ci_kappa = (ar - Z95 * np.sqrt(max(0, var_kappa)), ar + Z95 * np.sqrt(max(0, var_kappa)))
        cov_ar_gauss.append(1 if ci_gauss[0] <= truth["AR_K"] <= ci_gauss[1] else 0)
        cov_ar_kappa.append(1 if ci_kappa[0] <= truth["AR_K"] <= ci_kappa[1] else 0)

    # Empirical variances
    emp_var_ar = np.var(ar_vals, ddof=1)
    emp_var_f1 = np.var(f1_vals, ddof=1)
    emp_var_h = np.var(h_vals, ddof=1)
    emp_var_tr = np.var(tr_vals, ddof=1)

    # Gaussian theoretical variances (using pop eigenvalues)
    gauss_var_ar = delta_method_var_AR_K(pop_evals, K, kappa=0.0) / M
    gauss_var_f1 = delta_method_var_f1(pop_evals, kappa=0.0) / M
    gauss_var_h = delta_method_var_H(pop_evals, kappa=0.0) / M
    gauss_var_tr = delta_method_var_trace(pop_evals, kappa=0.0) / M

    # Variance ratios
    ratio_ar = emp_var_ar / gauss_var_ar if gauss_var_ar > 0 else np.nan
    ratio_f1 = emp_var_f1 / gauss_var_f1 if gauss_var_f1 > 0 else np.nan
    ratio_h = emp_var_h / gauss_var_h if gauss_var_h > 0 else np.nan
    ratio_tr = emp_var_tr / gauss_var_tr if gauss_var_tr > 0 else np.nan

    return {
        "N": N,
        "dist": dist_name,
        "kappa_true": kappa_true,
        "mean_kappa_hat": np.mean(kappa_hats),
        "var_ratio_AR_K": ratio_ar,
        "var_ratio_f1": ratio_f1,
        "var_ratio_H": ratio_h,
        "var_ratio_tr": ratio_tr,
        "expected_ratio": 1 + kappa_true,
        "cov_AR_gauss_%": 100 * np.mean(cov_ar_gauss),
        "cov_AR_kappa_%": 100 * np.mean(cov_ar_kappa),
    }


def run_experiment_5a() -> pd.DataFrame:
    records = []
    for N in [30, 115]:
        for dist_name in DISTRIBUTIONS:
            print(f"  (N={N}, {dist_name}) ...", flush=True)
            r = run_for_N_dist(N, dist_name)
            records.append(r)
            print(
                f"    Var ratio AR_K={r['var_ratio_AR_K']:.3f} (expected {r['expected_ratio']:.1f})  "
                f"Cov Gauss={r['cov_AR_gauss_%']:.1f}%  Cov kappa={r['cov_AR_kappa_%']:.1f}%"
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5a: Kurtosis Immunity and Variance Inflation")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table6 = run_experiment_5a()
    print("\nTable 6:")
    print(table6.to_string(index=False))

    table6.to_csv("/workspace/project/results/table6_kurtosis_immunity.csv", index=False)
    print("\nSaved to results/table6_kurtosis_immunity.csv")
