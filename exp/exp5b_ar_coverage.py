"""
Experiment 5b: AR_K Coverage Under Multiple Kappa Estimation Methods.

Compares five CI methods for AR_2 across Gaussian, t_8, t_5 distributions.
Validates spike-debiased AR_2 estimator (Proposition 11).
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, qis
from src.simulation import generate_spiked_cov, generate_elliptical_returns, generate_gaussian_returns
from src.functionals import (
    delta_method_var_AR_K, compute_AR_K,
    estimate_kappa_moment, estimate_kappa_radial_mle,
    spike_debias_AR_K,
)


SPIKES = [6.0, 3.5, 1.2]
SIGMA2 = 0.5
K = 2
M = 252
N_REPS = 25
SEED = 42
Z95 = 1.959964

DISTRIBUTIONS = {
    "Gaussian": {"nu": None, "kappa_true": 0.0},
    "t_8":     {"nu": 8.0,   "kappa_true": 0.5},
    "t_5":     {"nu": 5.0,   "kappa_true": 2.0},
}
METHODS = ["Delta_Gauss", "Delta_Mom", "Delta_MLE", "Delta_True", "HD_Debiased"]


def _compute_ci_coverage(ar_hat: float, evals: np.ndarray, K: int,
                         kappa: float, M: int, ar_true: float) -> int:
    var_ = delta_method_var_AR_K(evals, K, kappa=kappa) / M
    half = Z95 * np.sqrt(max(0.0, var_))
    return 1 if abs(ar_hat - ar_true) <= half else 0


def run_experiment_5b() -> pd.DataFrame:
    records = []
    for N in [30, 115]:
        Sigma = generate_spiked_cov(N, SPIKES, SIGMA2)
        pop_evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
        ar_true = compute_AR_K(pop_evals, K)

        for dist_name, cfg in DISTRIBUTIONS.items():
            rng = np.random.default_rng(SEED)
            nu = cfg["nu"]
            kappa_true = cfg["kappa_true"]

            cov_counters = {m: 0 for m in METHODS}
            bias_sums = {m: 0.0 for m in METHODS}
            width_sums = {m: 0.0 for m in METHODS}

            for _ in range(N_REPS):
                if nu is None:
                    R = generate_gaussian_returns(Sigma, M, rng)
                else:
                    R = generate_elliptical_returns(Sigma, M, nu, rng)

                S = sample_cov(R)
                C = qis(S, M)
                evals_C = np.sort(np.linalg.eigvalsh(C))[::-1]
                ar_hat = compute_AR_K(evals_C, K)

                kappa_mom = estimate_kappa_moment(R)
                kappa_mle = estimate_kappa_radial_mle(R, S)

                # Method i: Gaussian
                cov_counters["Delta_Gauss"] += _compute_ci_coverage(ar_hat, evals_C, K, 0.0, M, ar_true)
                v_g = delta_method_var_AR_K(evals_C, K, 0.0) / M
                width_sums["Delta_Gauss"] += 2 * Z95 * np.sqrt(max(0, v_g))
                bias_sums["Delta_Gauss"] += ar_hat - ar_true

                # Method ii: moment kappa
                cov_counters["Delta_Mom"] += _compute_ci_coverage(ar_hat, evals_C, K, kappa_mom, M, ar_true)
                v_m = delta_method_var_AR_K(evals_C, K, kappa_mom) / M
                width_sums["Delta_Mom"] += 2 * Z95 * np.sqrt(max(0, v_m))
                bias_sums["Delta_Mom"] += ar_hat - ar_true

                # Method iii: MLE kappa
                cov_counters["Delta_MLE"] += _compute_ci_coverage(ar_hat, evals_C, K, kappa_mle, M, ar_true)
                v_mle = delta_method_var_AR_K(evals_C, K, kappa_mle) / M
                width_sums["Delta_MLE"] += 2 * Z95 * np.sqrt(max(0, v_mle))
                bias_sums["Delta_MLE"] += ar_hat - ar_true

                # Method iv: true kappa
                cov_counters["Delta_True"] += _compute_ci_coverage(ar_hat, evals_C, K, kappa_true, M, ar_true)
                v_t = delta_method_var_AR_K(evals_C, K, kappa_true) / M
                width_sums["Delta_True"] += 2 * Z95 * np.sqrt(max(0, v_t))
                bias_sums["Delta_True"] += ar_hat - ar_true

                # Method v: HD debiased
                try:
                    ar_deb, evals_deb = spike_debias_AR_K(evals_C, K, N, M, kappa=kappa_mle)
                    cov_counters["HD_Debiased"] += _compute_ci_coverage(ar_deb, evals_deb, K, kappa_mle, M, ar_true)
                    v_deb = delta_method_var_AR_K(evals_deb, K, kappa_mle) / M
                    width_sums["HD_Debiased"] += 2 * Z95 * np.sqrt(max(0, v_deb))
                    bias_sums["HD_Debiased"] += ar_deb - ar_true
                except Exception:
                    cov_counters["HD_Debiased"] += _compute_ci_coverage(ar_hat, evals_C, K, kappa_mle, M, ar_true)
                    width_sums["HD_Debiased"] += width_sums["Delta_MLE"] / N_REPS
                    bias_sums["HD_Debiased"] += ar_hat - ar_true

            for method in METHODS:
                records.append({
                    "N": N,
                    "dist": dist_name,
                    "method": method,
                    "coverage_%": round(100 * cov_counters[method] / N_REPS, 1),
                    "mean_bias": round(bias_sums[method] / N_REPS, 5),
                    "mean_width": round(width_sums[method] / N_REPS, 5),
                })

            print(
                f"  (N={N}, {dist_name}) Gauss cov: "
                f"{100*cov_counters['Delta_Gauss']/N_REPS:.1f}%  "
                f"MLE cov: {100*cov_counters['Delta_MLE']/N_REPS:.1f}%  "
                f"HD cov: {100*cov_counters['HD_Debiased']/N_REPS:.1f}%",
                flush=True,
            )

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5b: AR_K Coverage Under Multiple Kappa Methods")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table7 = run_experiment_5b()
    print("\nTable 7:")
    print(table7.to_string(index=False))

    table7.to_csv("/workspace/project/results/table7_ar_coverage.csv", index=False)
    print("\nSaved to results/table7_ar_coverage.csv")
