"""
Experiment 2: Calibration of the Alert Scale Alpha for Null False-Flag Rate Control.

Find alpha* such that the calm-phase false-flag rate is <= 5%.
Validate alpha*=0.75 across gap regimes and estimators.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, ledoit_wolf, qis, mp_clip
from src.projector import (
    projector_distance_lemma1,
    operator_norm_error,
    compute_band_term,
    compute_capped_band,
)
from src.simulation import generate_factor_model_returns


N = 115
M = 252
K_f = 5
K = 2
T_EVAL = 30
N_REPS = 15
SEED = 42
ALPHA_GRID = [0.25, 0.50, 0.75, 0.90, 1.00]
ESTIMATORS = ["sample", "LW", "QIS", "MP-clip"]


def _apply_estimator(S: np.ndarray, method: str, n: int) -> np.ndarray:
    if method == "sample":
        return S
    if method == "LW":
        return ledoit_wolf(S, n)
    if method == "QIS":
        return qis(S, n)
    if method == "MP-clip":
        return mp_clip(S, n)
    raise ValueError(f"Unknown estimator: {method}")


def run_calm_phase(
    gap_type: str,
    estimator: str,
    n_reps: int = N_REPS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Run calm-phase simulation and collect per-date (D_hat, tau*) pairs.

    Returns
    -------
    np.ndarray of shape (n_pairs, 2): columns are [D_hat, tau_star]
    """
    rng = np.random.default_rng(seed)
    T_total = M + T_EVAL

    D_hats = []
    tau_stars = []

    for _ in range(n_reps):
        ret, Sigma, _, _ = generate_factor_model_returns(
            N=N, M_total=T_total, K_f=K_f, gap_type=gap_type,
            seed=int(rng.integers(0, 2**31)),
        )
        pop_evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
        Delta_K = pop_evals[K - 1] - pop_evals[K]

        prev_U = None
        prev_T = None

        for t in range(M, T_total):
            window = ret[t - M : t]
            S_t = sample_cov(window)
            C_t = _apply_estimator(S_t, estimator, M)

            evals_t, evecs_t = np.linalg.eigh(C_t)
            U_hat_t = evecs_t[:, ::-1][:, :K]

            eta_t = operator_norm_error(C_t, Sigma)
            T_t = compute_band_term(eta_t, C_t, Sigma, Delta_K, K)

            if prev_U is not None:
                D_hat_t = projector_distance_lemma1(U_hat_t, prev_U, K)
                tau_star = compute_capped_band(T_t, prev_T, K)
                D_hats.append(D_hat_t)
                tau_stars.append(tau_star)

            prev_U = U_hat_t
            prev_T = T_t

    return np.column_stack([np.array(D_hats), np.array(tau_stars)])


def run_experiment_2() -> pd.DataFrame:
    """
    Compute Table 2: flag rates per (gap, estimator) for each alpha.
    """
    records = []
    for gap_type in ["strong", "weak"]:
        for est in ESTIMATORS:
            print(f"  ({gap_type}, {est}) ...", flush=True)
            data = run_calm_phase(gap_type, est)
            D_hat = data[:, 0]
            tau_star = data[:, 1]

            row = {"gap": gap_type, "estimator": est}
            for alpha in ALPHA_GRID:
                flags = (D_hat > alpha * tau_star).astype(float)
                flag_rate = flags.mean() * 100.0
                row[f"alpha={alpha:.2f}"] = round(flag_rate, 2)

            # Find alpha*
            alpha_star = None
            for alpha in ALPHA_GRID:
                if row[f"alpha={alpha:.2f}"] <= 5.0:
                    alpha_star = alpha
                    break
            row["alpha_star"] = alpha_star
            records.append(row)
            print(
                f"    alpha=0.25: {row['alpha=0.25']:.1f}%  "
                f"alpha=0.75: {row['alpha=0.75']:.1f}%  "
                f"alpha*={alpha_star}"
            )

    df = pd.DataFrame(records).set_index(["gap", "estimator"])
    return df


def make_figure2(table2: pd.DataFrame) -> None:
    """Figure 2: flag rate vs alpha curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"sample": "tab:blue", "LW": "tab:orange", "QIS": "tab:green", "MP-clip": "tab:red"}

    for ax, gap_type in zip(axes, ["strong", "weak"]):
        for est in ESTIMATORS:
            row = table2.loc[(gap_type, est)]
            rates = [row[f"alpha={a:.2f}"] for a in ALPHA_GRID]
            ax.plot(ALPHA_GRID, rates, marker="o", label=est, color=colors[est])

        ax.axhline(5.0, color="k", linestyle="--", linewidth=1.5, label="5% threshold")
        ax.axvline(0.75, color="gray", linestyle=":", linewidth=1.5, label="α*=0.75")
        ax.set_title(f"{gap_type}-gap design")
        ax.set_xlabel("α")
        ax.set_ylabel("Flag rate (%)")
        ax.legend(fontsize=8)
        ax.set_xticks(ALPHA_GRID)

    plt.suptitle("Exp2: Flag Rate vs Alert Scale α (calm phase)")
    plt.tight_layout()
    os.makedirs("/workspace/project/results", exist_ok=True)
    plt.savefig("/workspace/project/results/fig2_alpha_calibration.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2: Alpha Calibration")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table2 = run_experiment_2()
    print("\nTable 2:")
    print(table2.to_string())

    table2.to_csv("/workspace/project/results/table2_alpha.csv")
    print("\nSaved to results/table2_alpha.csv")

    print("\nGenerating Figure 2...")
    make_figure2(table2)
    print("Saved to results/fig2_alpha_calibration.png")
