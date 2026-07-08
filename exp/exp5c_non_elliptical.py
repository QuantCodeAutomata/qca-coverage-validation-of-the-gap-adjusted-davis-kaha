"""
Experiment 5c: Necessity of Elliptical Assumption.

Tests whether kappa-correction remains valid under non-elliptical distributions.
Confirms failure for log-normal and asymmetric mixture.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, qis
from src.simulation import generate_spiked_cov, generate_elliptical_returns, generate_gaussian_returns
from src.functionals import (
    compute_AR_K, delta_method_var_AR_K,
    estimate_kappa_radial_mle,
)


SPIKES = [6.0, 3.5, 1.2]
SIGMA2 = 0.5
K = 2
M = 252
N_REPS = 25
SEED = 2024
Z95 = 1.959964


def _true_cov_lognormal(Sigma_N: np.ndarray) -> np.ndarray:
    """Compute true covariance of log-normal-transformed distribution."""
    N = Sigma_N.shape[0]
    Cov_LN = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Cov_LN[i, j] = (
                np.exp(Sigma_N[i, i] / 2.0)
                * np.exp(Sigma_N[j, j] / 2.0)
                * (np.exp(Sigma_N[i, j]) - 1.0)
            )
    return Cov_LN


def _gen_lognormal(Sigma_N: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """Generate log-normal non-elliptical returns."""
    N = Sigma_N.shape[0]
    L = np.linalg.cholesky(Sigma_N)
    z = rng.standard_normal((M, N))
    r = (L @ z.T).T
    # Elementwise transform: exp(r_i) - exp(Sigma_N[i,i]/2)
    shift = np.exp(0.5 * np.diag(Sigma_N))
    return np.exp(r) - shift


def _gen_mixture(Sigma_N: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """Generate asymmetric mixture returns."""
    N = Sigma_N.shape[0]
    mu = np.zeros(N)
    mu[0] = 1.0
    L = np.linalg.cholesky(Sigma_N)
    probs = rng.uniform(0, 1, size=M)
    z = rng.standard_normal((M, N))
    base = (L @ z.T).T
    shift = np.where(probs[:, None] < 0.3, mu[None, :], 0.0)
    raw = base + shift
    # Center to zero mean
    raw -= 0.3 * mu
    return raw


def _true_cov_mixture(Sigma_N: np.ndarray) -> np.ndarray:
    """True covariance for 0.7*N(0,Sigma)+0.3*N(mu,Sigma), mu=[1,0,...,0]."""
    N = Sigma_N.shape[0]
    mu = np.zeros(N)
    mu[0] = 1.0
    # Cov = Sigma + 0.7*0.3*(mu-0)^T * (mu-0) = Sigma + 0.21*mu*mu^T
    return Sigma_N + 0.21 * np.outer(mu, mu)


def run_for_N(N: int) -> list:
    rng = np.random.default_rng(SEED)
    Sigma_N = generate_spiked_cov(N, SPIKES, SIGMA2)

    # True AR_2 values per distribution
    pop_evals_base = np.sort(np.linalg.eigvalsh(Sigma_N))[::-1]
    ar_true_base = compute_AR_K(pop_evals_base, K)

    # Log-normal true covariance
    Cov_LN = _true_cov_lognormal(Sigma_N)
    pop_evals_LN = np.sort(np.linalg.eigvalsh(Cov_LN))[::-1]
    ar_true_LN = compute_AR_K(pop_evals_LN, K)

    # Mixture true covariance
    Cov_mix = _true_cov_mixture(Sigma_N)
    pop_evals_mix = np.sort(np.linalg.eigvalsh(Cov_mix))[::-1]
    ar_true_mix = compute_AR_K(pop_evals_mix, K)

    distributions = {
        "Gaussian":  {"gen": lambda: generate_gaussian_returns(Sigma_N, M, rng), "ar_true": ar_true_base},
        "t_5":       {"gen": lambda: generate_elliptical_returns(Sigma_N, M, 5.0, rng), "ar_true": ar_true_base},
        "LogNormal": {"gen": lambda: _gen_lognormal(Sigma_N, M, rng), "ar_true": ar_true_LN},
        "Mixture":   {"gen": lambda: _gen_mixture(Sigma_N, M, rng), "ar_true": ar_true_mix},
    }

    records = []
    for dist_name, dist_cfg in distributions.items():
        ar_true = dist_cfg["ar_true"]
        cov_gauss = 0
        cov_kappa = 0
        kappa_hats = []

        for _ in range(N_REPS):
            R = dist_cfg["gen"]()
            S = sample_cov(R)
            C = qis(S, M)
            evals_C = np.sort(np.linalg.eigvalsh(C))[::-1]
            ar_hat = compute_AR_K(evals_C, K)

            kappa_hat = estimate_kappa_radial_mle(R, S)
            kappa_hats.append(kappa_hat)

            # Gaussian CI
            v_gauss = delta_method_var_AR_K(evals_C, K, 0.0) / M
            h_g = Z95 * np.sqrt(max(0, v_gauss))
            cov_gauss += 1 if abs(ar_hat - ar_true) <= h_g else 0

            # Kappa-corrected CI
            v_kappa = delta_method_var_AR_K(evals_C, K, kappa_hat) / M
            h_k = Z95 * np.sqrt(max(0, v_kappa))
            cov_kappa += 1 if abs(ar_hat - ar_true) <= h_k else 0

        records.append({
            "N": N,
            "dist": dist_name,
            "ar_true": round(ar_true, 4),
            "mean_kappa_hat": round(np.mean(kappa_hats), 3),
            "cov_Gauss_%": round(100 * cov_gauss / N_REPS, 1),
            "cov_kappa_%": round(100 * cov_kappa / N_REPS, 1),
        })
        print(
            f"  (N={N}, {dist_name}): Gauss cov={100*cov_gauss/N_REPS:.1f}%  "
            f"kappa cov={100*cov_kappa/N_REPS:.1f}%  kappa_hat={np.mean(kappa_hats):.3f}",
            flush=True,
        )
    return records


def run_experiment_5c() -> pd.DataFrame:
    all_records = []
    for N in [30, 115]:
        print(f"\nN = {N}", flush=True)
        all_records.extend(run_for_N(N))
    return pd.DataFrame(all_records)


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5c: Non-Elliptical Coverage Failure")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table8 = run_experiment_5c()
    print("\nTable 8:")
    print(table8.to_string(index=False))

    table8.to_csv("/workspace/project/results/table8_non_elliptical.csv", index=False)
    print("\nSaved to results/table8_non_elliptical.csv")
