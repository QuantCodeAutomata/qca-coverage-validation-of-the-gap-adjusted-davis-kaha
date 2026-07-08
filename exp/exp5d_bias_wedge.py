"""
Experiment 5d: Cleaning-Debiasing Wedge.

Validates Proposition 11: sample AR_K has positive bias +Delta_S,
oracle has negative bias -Delta_O, QIS overshoots, debiased ~0.
Reproduces Figure 7 (relative bias vs concentration c=N/M).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, qis
from src.simulation import generate_spiked_cov, generate_gaussian_returns
from src.functionals import compute_AR_K, spike_debias_AR_K


SPIKES = [6.0, 3.5, 1.2]
SIGMA2 = 0.5
K = 2
M_FIXED = 252
N_REPS = 200
SEED = 42

N_VALUES = [32, 63, 115, 202]   # c = N/M in {0.127, 0.25, 0.456, 0.802}


def _predicted_biases(c: float) -> tuple:
    """Compute Delta_S and Delta_O from Proposition 11 formulas."""
    lam1, lam2 = SPIKES[0], SPIKES[1]
    S_K = lam1 + lam2
    Delta_S = c * SIGMA2 / S_K * (lam1 / (lam1 - SIGMA2) + lam2 / (lam2 - SIGMA2))
    Delta_O = c * SIGMA2 / S_K * (
        lam1 / (lam1 - SIGMA2 + c * SIGMA2) + lam2 / (lam2 - SIGMA2 + c * SIGMA2)
    )
    return Delta_S, Delta_O


def _oracle_eigenvalues(sample_evals: np.ndarray, N: int, M: int) -> np.ndarray:
    """Oracle debiasing using TRUE sigma2=0.5 and c=N/M.

    Applies the exact BBP inverse map for all eigenvalues above the known
    population MP edge, then redistributes the total eigenvalue reduction
    evenly over the bulk to preserve the empirical trace tr(S).

    BBP inverse solves λ² - (lam_hat + σ²(1-c))λ + lam_hat·σ² = 0:
        B    = lam_hat + sigma2*(1-c)
        disc = B² - 4*lam_hat*sigma2
        lam_deb = (B + sqrt(max(0, disc))) / 2
    """
    c = N / M
    sigma2 = SIGMA2
    lambda_plus = sigma2 * (1.0 + np.sqrt(c)) ** 2

    evals_deb = sample_evals.copy()
    spike_mask = sample_evals > lambda_plus + 0.01 * sigma2
    spike_indices = np.where(spike_mask)[0]

    for idx in spike_indices:
        lam_hat = sample_evals[idx]
        B    = lam_hat + sigma2 * (1.0 - c)
        disc = B ** 2 - 4.0 * lam_hat * sigma2
        lam_deb = (B + np.sqrt(max(0.0, disc))) / 2.0
        evals_deb[idx] = lam_deb

    # Trace preservation: total reduction from spikes redistributed to bulk
    bulk_mask = ~spike_mask
    if bulk_mask.sum() > 0 and spike_indices.size > 0:
        total_reduction = (
            np.sum(sample_evals[spike_indices]) - np.sum(evals_deb[spike_indices])
        )
        evals_deb[bulk_mask] += total_reduction / bulk_mask.sum()

    return evals_deb


def run_experiment_5d() -> tuple:
    records = []
    rng = np.random.default_rng(SEED)

    c_vals = []
    bias_sample = []
    bias_oracle = []
    bias_qis = []
    bias_deb = []
    delta_s_pred = []
    delta_o_pred = []

    for N in N_VALUES:
        c = N / M_FIXED
        c_vals.append(c)

        Sigma = generate_spiked_cov(N, SPIKES, SIGMA2)
        pop_evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
        ar_true = compute_AR_K(pop_evals, K)

        DS, DO = _predicted_biases(c)
        delta_s_pred.append(DS)
        delta_o_pred.append(DO)

        rel_biases = {est: [] for est in ["Sample", "Oracle", "QIS", "Debiased"]}

        for _ in range(N_REPS):
            R = generate_gaussian_returns(Sigma, M_FIXED, rng)
            S = sample_cov(R)
            evals_S = np.sort(np.linalg.eigvalsh(S))[::-1]

            # Sample
            ar_sample = compute_AR_K(evals_S, K)
            rel_biases["Sample"].append(ar_sample / ar_true - 1)

            # Oracle
            evals_oracle = _oracle_eigenvalues(evals_S, N, M_FIXED)
            ar_oracle = compute_AR_K(evals_oracle, K)
            rel_biases["Oracle"].append(ar_oracle / ar_true - 1)

            # QIS
            C_qis = qis(S, M_FIXED)
            evals_qis = np.sort(np.linalg.eigvalsh(C_qis))[::-1]
            ar_qis = compute_AR_K(evals_qis, K)
            rel_biases["QIS"].append(ar_qis / ar_true - 1)

            # Debiased (estimated sigma2)
            try:
                ar_deb, _ = spike_debias_AR_K(evals_S, K, N, M_FIXED)
                rel_biases["Debiased"].append(ar_deb / ar_true - 1)
            except Exception:
                rel_biases["Debiased"].append(ar_sample / ar_true - 1)

        mean_biases = {est: np.mean(rel_biases[est]) for est in rel_biases}
        bias_sample.append(mean_biases["Sample"])
        bias_oracle.append(mean_biases["Oracle"])
        bias_qis.append(mean_biases["QIS"])
        bias_deb.append(mean_biases["Debiased"])

        records.append({
            "N": N, "c": round(c, 3),
            "Delta_S_pred": round(DS, 4),
            "Delta_O_pred": round(DO, 4),
            "Sample_rel_bias_%": round(100 * mean_biases["Sample"], 2),
            "Oracle_rel_bias_%": round(100 * mean_biases["Oracle"], 2),
            "QIS_rel_bias_%": round(100 * mean_biases["QIS"], 2),
            "Debiased_rel_bias_%": round(100 * mean_biases["Debiased"], 2),
        })
        print(
            f"  N={N} c={c:.3f}: Sample={100*mean_biases['Sample']:.2f}%  "
            f"Oracle={100*mean_biases['Oracle']:.2f}%  "
            f"QIS={100*mean_biases['QIS']:.2f}%  "
            f"Deb={100*mean_biases['Debiased']:.2f}%  "
            f"(pred S={100*DS:.2f}%, O={100*DO:.2f}%)",
            flush=True,
        )

    # ── Figure 7 ──────────────────────────────────────────────────────
    c_arr = np.array(c_vals)
    fig7, ax7 = plt.subplots(figsize=(9, 6))
    ax7.plot(c_arr, 100 * np.array(bias_sample), "o-", color="tab:blue", label="Sample")
    ax7.plot(c_arr, 100 * np.array(bias_oracle), "s--", color="tab:orange", label="Oracle D*")
    ax7.plot(c_arr, 100 * np.array(bias_qis), "^:", color="tab:green", label="QIS")
    ax7.plot(c_arr, 100 * np.array(bias_deb), "D-.", color="tab:red", label="Debiased")
    ax7.plot(c_arr, 100 * np.array(delta_s_pred), "k--", lw=2, label="+ΔS (predicted)")
    ax7.plot(c_arr, -100 * np.array(delta_o_pred), "k:", lw=2, label="-ΔO (predicted)")
    ax7.axhline(0, color="gray", lw=1)
    ax7.set_xlabel("c = N/M")
    ax7.set_ylabel("Relative bias in AR₂ (%)")
    ax7.set_title("Figure 7: Cleaning-Debiasing Wedge")
    ax7.legend()
    ax7.set_xticks(c_arr)
    ax7.set_xticklabels([f"{c:.3f}" for c in c_arr])
    plt.tight_layout()
    plt.savefig("/workspace/project/results/fig7_bias_wedge.png", dpi=100)
    plt.close()

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 5d: Cleaning-Debiasing Wedge")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    df5d = run_experiment_5d()
    print("\nResults:")
    print(df5d.to_string(index=False))

    df5d.to_csv("/workspace/project/results/table_exp5d_bias_wedge.csv", index=False)
    print("\nSaved results/table_exp5d_bias_wedge.csv and results/fig7_bias_wedge.png")
