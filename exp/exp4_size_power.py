"""
Experiment 4: Size and Power of the Two-Window Monitor.

Validates analytic null (Proposition 2), bootstrap (Algorithm 2),
detectability frontier, and scale-change scale-invariance.
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
from src.simulation import apply_rotation
from src.monitoring import analytic_null_simulator, bootstrap_projector_movement


# ── Parameters ────────────────────────────────────────────────────────────────
N_DIM = 50
M_WIN = 126
K = 2
S_STRIDE = 31
N_REPS = 25        # paper: 200
N_DRAWS_ANALYTIC = 50    # paper: 400
N_BOOT = 15        # paper: 60
SEED = 42
ALPHA_BAND = 0.75

THETA_GRID = [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]


def _make_sigma1(gap: str) -> np.ndarray:
    """Build Sigma_1 diagonal for strong/weak gap."""
    if gap == "strong":
        diag_vals = [5.0, 2.0] + [0.5] * (N_DIM - 2)
    else:
        diag_vals = [5.0, 4.5] + [0.5] * (N_DIM - 2)
    return np.diag(diag_vals)


def _gen_gaussian(Sigma: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((n, N_DIM))
    return z @ L.T


def _gen_t5(Sigma: np.ndarray, n: int, rng: np.random.Generator, nu: float = 5.0) -> np.ndarray:
    L = np.linalg.cholesky(Sigma)
    xi = rng.chisquare(nu, size=n) / nu
    z = rng.standard_normal((n, N_DIM))
    return (np.sqrt(xi)[:, None]) * (z @ L.T)


def _run_two_window(
    Sigma1: np.ndarray,
    Sigma2: np.ndarray,
    dist: str,
    window: str,
    rng: np.random.Generator,
) -> dict:
    """
    One two-window evaluation: returns dict with D_hat and test flags.
    """
    # Window A: always M observations from Sigma1
    gen = _gen_gaussian if dist == "gaussian" else _gen_t5
    retA = gen(Sigma1, M_WIN, rng)

    # Window B
    if window == "post":
        retB = gen(Sigma2, M_WIN, rng)
    else:  # onset: M-s from Sigma1, s from Sigma2
        retB1 = gen(Sigma1, M_WIN - S_STRIDE, rng)
        retB2 = gen(Sigma2, S_STRIDE, rng)
        retB = np.vstack([retB1, retB2])

    S_A = sample_cov(retA)
    S_B = sample_cov(retB)
    C_A = qis(S_A, M_WIN)
    C_B = qis(S_B, M_WIN)

    # Top-K eigenvectors
    def _top_K(C):
        evals, evecs = np.linalg.eigh(C)
        return evecs[:, ::-1][:, :K]

    U_A = _top_K(C_A)
    U_B = _top_K(C_B)
    D_hat = projector_distance_lemma1(U_B, U_A, K)

    # ── Band test ────────────────────────────────────────────────────
    pop_evals1 = np.sort(np.linalg.eigvalsh(Sigma1))[::-1]
    pop_evals2 = np.sort(np.linalg.eigvalsh(Sigma2))[::-1]
    Delta_K_A = pop_evals1[K - 1] - pop_evals1[K]
    Delta_K_B = pop_evals2[K - 1] - pop_evals2[K]
    eta_A = operator_norm_error(C_A, Sigma1)
    eta_B = operator_norm_error(C_B, Sigma2)
    T_A = compute_band_term(eta_A, C_A, Sigma1, Delta_K_A, K)
    T_B = compute_band_term(eta_B, C_B, Sigma2, Delta_K_B, K)
    tau_star = compute_capped_band(T_B, T_A, K)
    flag_band = 1 if D_hat > ALPHA_BAND * tau_star else 0

    # ── Analytic test ────────────────────────────────────────────────
    evals_A = np.sort(np.linalg.eigvalsh(C_A))[::-1]
    sub_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    null_draws = analytic_null_simulator(evals_A, K, S_STRIDE, M_WIN, N_DRAWS_ANALYTIC, sub_rng)
    q95_analytic = np.quantile(null_draws, 0.95)
    flag_analytic = 1 if D_hat > q95_analytic else 0

    # ── Bootstrap test ───────────────────────────────────────────────
    boot_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    boot_draws = bootstrap_projector_movement(
        retA, C_A, K, S_STRIDE, M_WIN, "qis", N_BOOT, boot_rng
    )
    q95_boot = np.quantile(boot_draws, 0.95) if len(boot_draws) > 0 else np.sqrt(2 * K)
    flag_boot = 1 if D_hat > q95_boot else 0

    return {
        "D_hat": D_hat,
        "tau_star": tau_star,
        "q95_analytic": q95_analytic,
        "q95_boot": q95_boot,
        "flag_band": flag_band,
        "flag_analytic": flag_analytic,
        "flag_boot": flag_boot,
    }


def run_experiment_4() -> pd.DataFrame:
    """Run Experiment 4 and return Table 5 as DataFrame."""
    rng_master = np.random.default_rng(SEED)

    records = []
    for gap in ["strong", "weak"]:
        Sigma1 = _make_sigma1(gap)
        for dist in ["gaussian", "t5"]:
            for window in ["post", "onset"]:
                for theta in THETA_GRID:
                    # Rotation alternative
                    Sigma2 = apply_rotation(Sigma1, theta, plane=(0, K))
                    true_d = np.sqrt(2) * abs(np.sin(theta))

                    band_flags = []
                    analytic_flags = []
                    boot_flags = []

                    for _ in range(N_REPS):
                        rep_rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                        try:
                            result = _run_two_window(Sigma1, Sigma2, dist, window, rep_rng)
                            band_flags.append(result["flag_band"])
                            analytic_flags.append(result["flag_analytic"])
                            boot_flags.append(result["flag_boot"])
                        except Exception as e:
                            # Skip failed reps gracefully
                            pass

                    if band_flags:
                        records.append({
                            "gap": gap,
                            "dist": dist,
                            "window": window,
                            "theta": theta,
                            "true_D": round(true_d, 4),
                            "Band_%": round(100 * np.mean(band_flags), 1),
                            "Analytic_%": round(100 * np.mean(analytic_flags), 1),
                            "Bootstrap_%": round(100 * np.mean(boot_flags), 1),
                            "n_reps": len(band_flags),
                        })
                    print(
                        f"  ({gap},{dist},{window},theta={theta:.2f}): "
                        f"Band={100*np.mean(band_flags) if band_flags else 'N/A':.1f}%  "
                        f"Analytic={100*np.mean(analytic_flags) if analytic_flags else 'N/A':.1f}%",
                        flush=True,
                    )

                # ── Scale-change null ─────────────────────────────
                Sigma2_scale = 1.5 * Sigma1
                sc_band = []
                sc_analytic = []
                sc_boot = []
                for _ in range(N_REPS):
                    rep_rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                    try:
                        result = _run_two_window(Sigma1, Sigma2_scale, dist, window, rep_rng)
                        sc_band.append(result["flag_band"])
                        sc_analytic.append(result["flag_analytic"])
                        sc_boot.append(result["flag_boot"])
                    except Exception:
                        pass
                if sc_band:
                    records.append({
                        "gap": gap,
                        "dist": dist,
                        "window": window,
                        "theta": "scale*1.5",
                        "true_D": 0.0,
                        "Band_%": round(100 * np.mean(sc_band), 1),
                        "Analytic_%": round(100 * np.mean(sc_analytic), 1),
                        "Bootstrap_%": round(100 * np.mean(sc_boot), 1),
                        "n_reps": len(sc_band),
                    })

    table5 = pd.DataFrame(records)
    return table5


def make_figure5(table5: pd.DataFrame) -> None:
    """Power curves: rejection rate vs theta."""
    theta_rows = table5[table5["theta"] != "scale*1.5"].copy()
    theta_rows["theta"] = theta_rows["theta"].astype(float)

    fig5, axes5 = plt.subplots(2, 2, figsize=(12, 9))
    for ax5, (gap, window) in zip(
        axes5.flat,
        [("strong","post"),("strong","onset"),("weak","post"),("weak","onset")]
    ):
        for dist, ls in [("gaussian","-"), ("t5","--")]:
            sub = theta_rows[
                (theta_rows["gap"]==gap) & (theta_rows["window"]==window) &
                (theta_rows["dist"]==dist)
            ].sort_values("theta")
            if len(sub) > 0:
                ax5.plot(sub["theta"], sub["Analytic_%"], marker="o", ls=ls, label=f"{dist}-Analytic")
                ax5.plot(sub["theta"], sub["Bootstrap_%"], marker="s", ls=ls, alpha=0.7, label=f"{dist}-Boot")
        ax5.axhline(5, color="k", lw=1, ls=":")
        ax5.set_title(f"{gap} gap / {window}")
        ax5.set_xlabel("θ (rotation angle)")
        ax5.set_ylabel("Rejection rate (%)")
        ax5.legend(fontsize=7)
    plt.suptitle("Figure 5: Power curves vs rotation angle θ")
    plt.tight_layout()
    plt.savefig("/workspace/project/results/fig5_power_curves.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 4: Size and Power of Two-Window Monitor")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    table5 = run_experiment_4()
    print("\nTable 5:")
    print(table5.to_string())

    table5.to_csv("/workspace/project/results/table5_size_power.csv", index=False)
    print("\nSaved to results/table5_size_power.csv")

    make_figure5(table5)
    print("Saved to results/fig5_power_curves.png")
