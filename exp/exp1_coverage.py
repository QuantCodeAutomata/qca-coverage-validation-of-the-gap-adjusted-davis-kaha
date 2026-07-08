"""
Experiment 1: Coverage Validation of the Gap-Adjusted Davis-Kahan Perturbation Band (H1).

Validates Theorem 1: |D_hat_K,t - D_K,t| <= tau*_K,t with 100% coverage.
Measures band tightness across gap regimes, estimators, and phases.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, ledoit_wolf, qis, mp_clip, oas
from src.projector import (
    projector_distance_lemma1,
    operator_norm_error,
    compute_band_term,
    compute_capped_band,
)
from src.simulation import generate_factor_model_returns, apply_rotation


# ── Experiment parameters ──────────────────────────────────────────────────────
N = 115
M = 252          # rolling window
K_f = 5          # latent factors
K = 2            # monitoring dimension
T_EVAL = 30      # evaluation dates per replication (paper: 500)
N_REPS = 15      # Monte Carlo replications (paper: 1000)
ALPHA_FLAG = 0.75
SEED = 42
THETA_BREAK = 0.3  # rotation angle for break phase
ESTIMATORS = ["sample", "LW", "QIS", "MP-clip"]


def _apply_estimator(S: np.ndarray, method: str, n: int) -> np.ndarray:
    """Dispatch covariance estimator by name."""
    if method == "sample":
        return S
    if method == "LW":
        return ledoit_wolf(S, n)
    if method == "QIS":
        return qis(S, n)
    if method == "MP-clip":
        return mp_clip(S, n)
    if method == "OAS":
        return oas(S, n)
    raise ValueError(f"Unknown estimator: {method}")


def run_one_condition(
    gap_type: str,
    phase: str,
    estimator: str,
    n_reps: int = N_REPS,
    seed: int = SEED,
) -> dict:
    """
    Run one (gap_type, phase, estimator) condition.

    Parameters
    ----------
    gap_type : 'strong' or 'weak'
    phase    : 'calm' or 'break'
    estimator: one of ESTIMATORS
    n_reps   : Monte Carlo replications
    seed     : base seed

    Returns
    -------
    dict with aggregated metrics
    """
    rng = np.random.default_rng(seed)

    T_total = M + T_EVAL
    t_break = M + T_EVAL // 2  # break point (absolute index)

    # Accumulators over all (replication, date) pairs
    all_dev = []          # |D_hat - D|
    all_tau_star = []     # tau*
    all_uncapped = []     # uncapped band
    all_D_hat = []        # D_hat
    coverage_flags = []   # 1{covered}
    flag_075 = []         # 1{D_hat > 0.75 * tau*}

    for b in range(n_reps):
        # Generate full return series for calm phase (Sigma constant)
        ret, Sigma1, _, _ = generate_factor_model_returns(
            N=N,
            M_total=T_total,
            K_f=K_f,
            var_explained=0.70,
            gap_type=gap_type,
            seed=int(rng.integers(0, 2**31)),
        )
        Sigma2 = apply_rotation(Sigma1, THETA_BREAK) if phase == "break" else Sigma1.copy()

        # Precompute population eigenvectors for D_{K,t}
        pop_evals1 = np.sort(np.linalg.eigvalsh(Sigma1))[::-1]
        pop_evecs1 = np.linalg.eigh(Sigma1)[1][:, ::-1]
        U_pop1 = pop_evecs1[:, :K]

        pop_evals2 = np.sort(np.linalg.eigvalsh(Sigma2))[::-1]
        pop_evecs2 = np.linalg.eigh(Sigma2)[1][:, ::-1]
        U_pop2 = pop_evecs2[:, :K]

        prev_U = None      # previous estimated top-K eigenvectors
        prev_U_pop = None  # previous population top-K eigenvectors
        prev_T = None      # previous band term T_{K,t-1}

        for t in range(M, T_total):
            # Current population covariance
            if phase == "break" and t >= t_break:
                Sigma_t = Sigma2
                Delta_K_t = pop_evals2[K - 1] - pop_evals2[K]
                U_pop_t = U_pop2
            else:
                Sigma_t = Sigma1
                Delta_K_t = pop_evals1[K - 1] - pop_evals1[K]
                U_pop_t = U_pop1

            # Rolling window
            window = ret[t - M : t]
            S_t = sample_cov(window)
            C_t = _apply_estimator(S_t, estimator, M)

            # Eigendecompose C_t (descending order)
            evals_t, evecs_t = np.linalg.eigh(C_t)
            evals_t = evals_t[::-1]
            evecs_t = evecs_t[:, ::-1]
            U_hat_t = evecs_t[:, :K]

            # Operator-norm error
            eta_t = operator_norm_error(C_t, Sigma_t)

            # Band term T_{K,t}
            T_t = compute_band_term(eta_t, C_t, Sigma_t, Delta_K_t, K)

            if prev_U is not None:
                # D_hat_{K,t}: observed projector movement
                D_hat_t = projector_distance_lemma1(U_hat_t, prev_U, K)

                # D_{K,t}: latent projector movement
                D_pop_t = projector_distance_lemma1(U_pop_t, prev_U_pop, K)

                # tau*_{K,t}
                tau_star = compute_capped_band(T_t, prev_T, K)

                # Uncapped band: T_t + T_prev (no sqrt(2K) cap)
                uncapped = T_t + prev_T

                dev = abs(D_hat_t - D_pop_t)
                covered = 1 if dev <= tau_star + 1e-12 else 0

                all_dev.append(dev)
                all_tau_star.append(tau_star)
                all_uncapped.append(uncapped)
                all_D_hat.append(D_hat_t)
                coverage_flags.append(covered)
                flag_075.append(1 if D_hat_t > ALPHA_FLAG * tau_star else 0)

            prev_U = U_hat_t
            prev_U_pop = U_pop_t
            prev_T = T_t

    all_dev = np.array(all_dev)
    all_tau_star = np.array(all_tau_star)
    all_uncapped = np.array(all_uncapped)
    all_D_hat = np.array(all_D_hat)
    coverage_flags = np.array(coverage_flags)
    flag_075 = np.array(flag_075)

    tightness = all_dev / np.clip(all_tau_star, 1e-12, None)
    coverage_pct = 100.0 * coverage_flags.mean()

    # Assert 100% coverage (core H1 claim)
    assert coverage_pct == 100.0, (
        f"Coverage violation in ({gap_type},{phase},{estimator}): {coverage_pct:.2f}%"
    )

    return {
        "Mean_D_hat": float(all_D_hat.mean()),
        "Mean_tau_star": float(all_tau_star.mean()),
        "Mean_uncapped": float(all_uncapped.mean()),
        "Median_tightness": float(np.median(tightness)),
        "P95_tightness": float(np.percentile(tightness, 95)),
        "Coverage_pct": coverage_pct,
        "Flag_rate_075": float(100.0 * flag_075.mean()),
        "_tightness_all": tightness,
    }


def run_experiment_1() -> tuple:
    """
    Run full Experiment 1 and return (Table 1 DataFrame, cached_results dict).
    """
    conditions = [
        ("strong", "calm"),
        ("strong", "break"),
        ("weak", "calm"),
        ("weak", "break"),
    ]

    rows = {}
    cached_results: dict = {}
    for gap_type, phase in conditions:
        for est in ESTIMATORS:
            print(f"  Running ({gap_type}, {phase}, {est}) ...", flush=True)
            result = run_one_condition(gap_type, phase, est)
            key = (gap_type, phase, est)
            cached_results[key] = result
            rows[key] = {k: v for k, v in result.items() if not k.startswith("_")}
            print(
                f"    Coverage={result['Coverage_pct']:.1f}%  "
                f"Median_tight={result['Median_tightness']:.3f}  "
                f"Mean_tau*={result['Mean_tau_star']:.3f}"
            )

    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=["gap", "phase", "estimator"])
    return df, cached_results


def make_figure(table: pd.DataFrame, cached_results: dict | None = None) -> None:
    """Plot tightness distributions (Figure Exp1).
    
    If cached_results is provided (dict keyed by (gap, phase, est)), skip re-running.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    conditions = [("strong", "calm"), ("strong", "break"), ("weak", "calm"), ("weak", "break")]
    for ax, (gap, phase) in zip(axes.flat, conditions):
        for est in ESTIMATORS:
            if cached_results is not None and (gap, phase, est) in cached_results:
                result = cached_results[(gap, phase, est)]
            else:
                result = run_one_condition(gap, phase, est, n_reps=N_REPS)
            ax.hist(
                result["_tightness_all"],
                bins=30,
                alpha=0.4,
                label=est,
                density=True,
            )
        ax.axvline(1.0, color="k", linestyle="--", linewidth=1, label="tau* = |D|")
        ax.set_title(f"{gap}-gap / {phase}")
        ax.set_xlabel("Tightness ratio |D̂−D| / τ*")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    plt.suptitle("Exp1: Tightness ratio distributions (should be <1 always)")
    plt.tight_layout()
    os.makedirs("/workspace/project/results", exist_ok=True)
    plt.savefig("/workspace/project/results/fig_exp1_tightness.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 1: Coverage Validation")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table1, cached = run_experiment_1()
    print("\nTable 1:")
    print(table1.to_string())

    table1.to_csv("/workspace/project/results/table1_coverage.csv")
    print("\nSaved to results/table1_coverage.csv")

    print("\nGenerating tightness figure...")
    make_figure(table1, cached_results=cached)
    print("Saved to results/fig_exp1_tightness.png")
