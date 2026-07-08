"""
Experiment 6: Computational Cost Benchmarking.

Measures wall-clock time for covariance fitting, monitor step,
and full test procedures (band, analytic, bootstrap).
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, ledoit_wolf, qis, mp_clip
from src.projector import projector_distance_lemma1, compute_band_term, compute_capped_band
from src.monitoring import analytic_null_simulator, bootstrap_projector_movement


N_DIM = 115
M_VALUES = [77, 115, 230, 460]
N_REPS_TIMING = 15   # paper: 30
SEED = 99
K = 2
N_DRAWS_ANALYTIC = 200   # paper: 400


def _time_ms(fn, n_reps: int = N_REPS_TIMING) -> tuple:
    """Time fn() in milliseconds. Returns (mean_ms, std_ms)."""
    times = []
    for i in range(n_reps + 1):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i > 0:  # skip first (warmup)
            times.append((t1 - t0) * 1000.0)
    return np.mean(times), np.std(times)


def run_table9() -> pd.DataFrame:
    """Table 9: covariance fitting and monitor step timing."""
    rng = np.random.default_rng(SEED)
    records = []

    for M in M_VALUES:
        for est_name in ["Sample", "LW", "QIS", "MP-clip"]:
            def make_fn():
                _M = M
                _est = est_name
                def fn():
                    R = rng.standard_normal((_M, N_DIM))
                    R -= R.mean(0)
                    S = sample_cov(R)
                    if _est == "Sample":
                        return S
                    elif _est == "LW":
                        return ledoit_wolf(S, _M)
                    elif _est == "QIS":
                        return qis(S, _M)
                    elif _est == "MP-clip":
                        return mp_clip(S, _M)
                return fn

            mean_fit, std_fit = _time_ms(make_fn())

            # Monitor step: use a pre-computed pair of eigenvectors
            R_prev = rng.standard_normal((M, N_DIM))
            R_curr = rng.standard_normal((M, N_DIM))
            S_prev = sample_cov(R_prev)
            S_curr = sample_cov(R_curr)
            U_prev = np.linalg.eigh(S_prev)[1][:, ::-1][:, :K]
            Sigma_dummy = np.eye(N_DIM)

            def monitor_fn():
                C_curr_ = qis(S_curr, M)
                evals_, evecs_ = np.linalg.eigh(C_curr_)
                U_curr_ = evecs_[:, ::-1][:, :K]
                projector_distance_lemma1(U_curr_, U_prev, K)
                eta_ = np.max(np.abs(np.linalg.eigvalsh(C_curr_ - Sigma_dummy)))
                Delta_K_ = np.sort(np.linalg.eigvalsh(Sigma_dummy))[::-1][K - 1] - \
                           np.sort(np.linalg.eigvalsh(Sigma_dummy))[::-1][K]
                T_ = compute_band_term(eta_, C_curr_, Sigma_dummy, Delta_K_, K)

            mean_mon, std_mon = _time_ms(monitor_fn)

            records.append({
                "M": M,
                "Estimator": est_name,
                "Fit_mean_ms": round(mean_fit, 2),
                "Fit_std_ms": round(std_fit, 2),
                "Monitor_mean_ms": round(mean_mon, 2),
                "Monitor_std_ms": round(std_mon, 2),
            })
            print(f"  M={M} {est_name}: fit={mean_fit:.2f}ms  monitor={mean_mon:.2f}ms", flush=True)

    return pd.DataFrame(records)


def run_table10() -> pd.DataFrame:
    """Table 10: timing of full test procedures."""
    rng = np.random.default_rng(SEED)
    records = []

    for (N, M) in [(50, 126), (115, 252)]:
        R_A = rng.standard_normal((M, N))
        R_A -= R_A.mean(0)
        S_A = sample_cov(R_A)
        C_A = qis(S_A, M)
        evals_A = np.sort(np.linalg.eigvalsh(C_A))[::-1]

        # Band test
        def band_fn():
            Sigma_dum = np.eye(N)
            eta_ = np.max(np.abs(np.linalg.eigvalsh(C_A - Sigma_dum)))
            Delta_K_ = 1.0 - 0.0
            T_ = compute_band_term(eta_, C_A, Sigma_dum, Delta_K_, K)
            T_prev = T_
            compute_capped_band(T_, T_prev, K)

        mean_band, _ = _time_ms(band_fn)
        records.append({"N": N, "M": M, "Procedure": "Band", "mean_ms": round(mean_band, 2)})

        # Analytic test (N_DRAWS_ANALYTIC draws)
        sub_rng = np.random.default_rng(42)

        def analytic_fn():
            analytic_null_simulator(evals_A, K, 31, M, N_DRAWS_ANALYTIC, sub_rng)

        mean_an, _ = _time_ms(analytic_fn, n_reps=5)
        records.append({"N": N, "M": M, "Procedure": f"Analytic ({N_DRAWS_ANALYTIC} draws)", "mean_ms": round(mean_an, 2)})

        # Bootstrap-QIS B=20, B=40
        for B in [20, 40]:
            boot_rng = np.random.default_rng(43)

            def boot_qis_fn():
                bootstrap_projector_movement(R_A, C_A, K, 31, M, "qis", B, boot_rng)

            mean_bq, _ = _time_ms(boot_qis_fn, n_reps=5)
            records.append({"N": N, "M": M, "Procedure": f"Bootstrap-QIS (B={B})", "mean_ms": round(mean_bq, 2)})

        # Bootstrap-sample B=20, B=40
        for B in [20, 40]:
            boot_rng2 = np.random.default_rng(44)

            def boot_sample_fn():
                bootstrap_projector_movement(R_A, C_A, K, 31, M, "sample", B, boot_rng2)

            mean_bs, _ = _time_ms(boot_sample_fn, n_reps=5)
            records.append({"N": N, "M": M, "Procedure": f"Bootstrap-sample (B={B})", "mean_ms": round(mean_bs, 2)})

        for r in records[-6:]:
            print(f"  ({N},{M}) {r['Procedure']}: {r['mean_ms']:.2f}ms", flush=True)

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 6: Computational Benchmarking")
    print("=" * 60)
    os.makedirs("/workspace/project/results", exist_ok=True)

    print("\n-- Table 9: covariance fitting and monitor timing --")
    table9 = run_table9()
    print(table9.to_string(index=False))
    table9.to_csv("/workspace/project/results/table9_timing_fit.csv", index=False)

    print("\n-- Table 10: procedure timing --")
    table10 = run_table10()
    print(table10.to_string(index=False))
    table10.to_csv("/workspace/project/results/table10_timing_proc.csv", index=False)

    print("\nSaved results/table9_timing_fit.csv and results/table10_timing_proc.csv")
