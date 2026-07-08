"""
Experiment 3: Functional-Risk Separation.

Validates that operator-norm and spectral functional rankings can disagree.
Verifies rotation-equivariance: top-K projector of QIS/LW/OAS equals sample.
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
from src.projector import projector_distance_lemma1, operator_norm_error
from src.simulation import generate_factor_model_returns
from src.functionals import compute_AR_K, compute_f1


N = 115
M = 252
K_f = 5
K = 5       # monitor all factors for this experiment
T_EVAL = 20
N_REPS = 10
SEED = 42
ESTIMATORS = ["Sample", "QIS", "LW", "OAS", "MP-clip"]


def _apply_est(S: np.ndarray, method: str, n: int) -> np.ndarray:
    if method == "Sample":
        return S
    if method == "QIS":
        return qis(S, n)
    if method == "LW":
        return ledoit_wolf(S, n)
    if method == "OAS":
        return oas(S, n)
    if method == "MP-clip":
        return mp_clip(S, n)
    raise ValueError(method)


def run_experiment_3() -> tuple:
    """
    Run Experiment 3. Returns (table3, table4, pairwise_df).
    """
    rng = np.random.default_rng(SEED)
    T_total = M + T_EVAL

    # ── Table 3 accumulators ───────────────────────────────────────────
    op_errs = {e: [] for e in ESTIMATORS}
    ar_errs = {e: [] for e in ESTIMATORS}
    f1_errs = {e: [] for e in ESTIMATORS}
    equiv_diffs = {e: [] for e in ["QIS", "LW", "OAS"]}  # equivariance check

    # Pairwise comparison accumulators
    qis_lt_lw_op = []
    qis_lt_lw_ar = []
    mp_lt_qis_ar = []
    mp_lt_qis_f1 = []

    # Gap stratification
    gap_vals = []
    d_hat_vals = {e: [] for e in ESTIMATORS}

    for _ in range(N_REPS):
        ret, Sigma, _, _ = generate_factor_model_returns(
            N=N, M_total=T_total, K_f=K_f, gap_type="strong",
            seed=int(rng.integers(0, 2**31)),
        )
        pop_evals_full = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
        AR_K_true = compute_AR_K(pop_evals_full, K)
        f1_true = compute_f1(pop_evals_full)

        for t in range(M, T_total):
            window = ret[t - M : t]
            S_t = sample_cov(window)
            est_results = {}
            for est in ESTIMATORS:
                C = _apply_est(S_t, est, M)
                evals, evecs = np.linalg.eigh(C)
                evals = evals[::-1]
                evecs = evecs[:, ::-1]
                est_results[est] = (C, evals, evecs)

            # Gap stratification: eigengap of population
            gap_12 = (pop_evals_full[0] - pop_evals_full[1]) / max(pop_evals_full[0], 1e-12)
            gap_vals.append(gap_12)

            for est in ESTIMATORS:
                C, evals, evecs = est_results[est]
                op_err = operator_norm_error(C, Sigma)
                ar_err = abs(compute_AR_K(evals, K) - AR_K_true)
                f1_err = abs(compute_f1(evals) - f1_true)
                op_errs[est].append(op_err)
                ar_errs[est].append(ar_err)
                f1_errs[est].append(f1_err)

                # D_hat for gap stratification
                U_K = evecs[:, :K]
                d_hat_vals[est].append(0.0)  # placeholder (need previous)

            # Equivariance check: P_K(QIS) == P_K(Sample)?
            _, evals_s, evecs_s = est_results["Sample"]
            U_K_s = evecs_s[:, :K]
            for eqv_est in ["QIS", "LW", "OAS"]:
                _, evals_e, evecs_e = est_results[eqv_est]
                U_K_e = evecs_e[:, :K]
                P_s = U_K_s @ U_K_s.T
                P_e = U_K_e @ U_K_e.T
                diff_f = np.linalg.norm(P_s - P_e, "fro")
                equiv_diffs[eqv_est].append(diff_f)

            # Pairwise comparison
            qis_op = op_errs["QIS"][-1]
            lw_op = op_errs["LW"][-1]
            qis_ar = ar_errs["QIS"][-1]
            lw_ar = ar_errs["LW"][-1]
            mp_ar = ar_errs["MP-clip"][-1]
            mp_f1 = f1_errs["MP-clip"][-1]
            qis_f1 = f1_errs["QIS"][-1]
            qis_lt_lw_op.append(1 if qis_op < lw_op else 0)
            qis_lt_lw_ar.append(1 if qis_ar < lw_ar else 0)
            mp_lt_qis_ar.append(1 if mp_ar < qis_ar else 0)
            mp_lt_qis_f1.append(1 if mp_f1 < qis_f1 else 0)

    # ── Build Table 3 ─────────────────────────────────────────────────
    rows3 = []
    for est in ESTIMATORS:
        rows3.append({
            "Estimator": est,
            "Mean_op_err": np.mean(op_errs[est]),
            "Mean_AR_K_err": np.mean(ar_errs[est]),
            "Mean_f1_err": np.mean(f1_errs[est]),
        })
    table3 = pd.DataFrame(rows3).set_index("Estimator")

    # Equivariance summary
    equi_pass = {e: np.mean(np.array(equiv_diffs[e]) < 1e-8) * 100 for e in equiv_diffs}

    pairwise = pd.DataFrame({
        "QIS_beats_LW_op_pct": [100 * np.mean(qis_lt_lw_op)],
        "QIS_beats_LW_AR_pct": [100 * np.mean(qis_lt_lw_ar)],
        "MPclip_beats_QIS_AR_pct": [100 * np.mean(mp_lt_qis_ar)],
        "MPclip_beats_QIS_f1_pct": [100 * np.mean(mp_lt_qis_f1)],
        "LW_equivariant_pct": [equi_pass["LW"]],
        "QIS_equivariant_pct": [equi_pass["QIS"]],
        "OAS_equivariant_pct": [equi_pass["OAS"]],
    })

    # ── Table 4: robustness by M ──────────────────────────────────────
    M_vals = [460, 230, 115, 77]
    rows4 = []
    rng4 = np.random.default_rng(SEED + 1)
    for M_rob in M_vals:
        c_rob = N / M_rob
        T_total_r = M_rob + 10
        est_op_r = {e: [] for e in ESTIMATORS}
        est_ar_r = {e: [] for e in ESTIMATORS}
        for _ in range(5):
            ret_r, Sigma_r, _, _ = generate_factor_model_returns(
                N=N, M_total=T_total_r, K_f=K_f, gap_type="strong",
                seed=int(rng4.integers(0, 2**31)),
            )
            pop_evals_r = np.sort(np.linalg.eigvalsh(Sigma_r))[::-1]
            AR_K_true_r = compute_AR_K(pop_evals_r, K)
            for t in range(M_rob, T_total_r):
                window_r = ret_r[t - M_rob : t]
                S_r = sample_cov(window_r)
                for est in ESTIMATORS:
                    try:
                        C_r = _apply_est(S_r, est, M_rob)
                        ev_r = np.sort(np.linalg.eigvalsh(C_r))[::-1]
                        est_op_r[est].append(operator_norm_error(C_r, Sigma_r))
                        est_ar_r[est].append(abs(compute_AR_K(ev_r, K) - AR_K_true_r))
                    except Exception:
                        pass
        for est in ESTIMATORS:
            if est_op_r[est]:
                rows4.append({
                    "M": M_rob,
                    "c=N/M": round(c_rob, 3),
                    "Estimator": est,
                    "Mean_op_err": np.mean(est_op_r[est]),
                    "Mean_AR_K_err": np.mean(est_ar_r[est]),
                })
    table4 = pd.DataFrame(rows4)

    # ── Gap stratification figure (Figure 4) ─────────────────────────
    gap_arr = np.array(gap_vals)
    tertiles = np.percentile(gap_arr, [33.3, 66.7])
    tertile_labels = ["Low gap", "Mid gap", "High gap"]

    def _tertile_idx(arr, t):
        if t == 0:
            return arr <= tertiles[0]
        if t == 1:
            return (arr > tertiles[0]) & (arr <= tertiles[1])
        return arr > tertiles[1]

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    bar_width = 0.15
    x = np.arange(3)
    for i, est in enumerate(ESTIMATORS):
        stds = []
        for t in range(3):
            mask = _tertile_idx(gap_arr, t)
            op_arr_t = np.array(op_errs[est])[mask]
            stds.append(np.std(op_arr_t) if len(op_arr_t) > 1 else 0.0)
        ax4.bar(x + i * bar_width, stds, bar_width, label=est)
    ax4.set_xticks(x + 2 * bar_width)
    ax4.set_xticklabels(tertile_labels)
    ax4.set_ylabel("Std of op-norm error")
    ax4.set_title("Figure 4: Op-norm std by gap tertile")
    ax4.legend()
    plt.tight_layout()
    plt.savefig("/workspace/project/results/fig4_gap_tertile.png", dpi=100)
    plt.close()

    # ── Error bar figure (Figure 3) ───────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
    labels3 = ESTIMATORS
    for ax3, (vals_dict, title) in zip(
        axes3,
        [(op_errs, "Op-norm error"), (ar_errs, "AR_K error"), (f1_errs, "f₁ error")],
    ):
        means = [np.mean(vals_dict[e]) for e in labels3]
        ax3.bar(labels3, means, color=["tab:blue","tab:orange","tab:green","tab:red","tab:purple"])
        ax3.set_title(title)
        ax3.set_ylabel("Mean error")
    plt.suptitle("Figure 3: Mean Errors by Estimator")
    plt.tight_layout()
    plt.savefig("/workspace/project/results/fig3_error_bars.png", dpi=100)
    plt.close()

    return table3, table4, pairwise


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 3: Functional-Risk Separation")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table3, table4, pairwise = run_experiment_3()

    print("\nTable 3:")
    print(table3.to_string())
    print("\nTable 4:")
    print(table4.to_string())
    print("\nPairwise:")
    print(pairwise.to_string())

    table3.to_csv("/workspace/project/results/table3_functional_errors.csv")
    table4.to_csv("/workspace/project/results/table4_robustness.csv")
    pairwise.to_csv("/workspace/project/results/table3_pairwise.csv")
    print("\nSaved tables to results/")
