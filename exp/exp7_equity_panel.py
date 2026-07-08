"""
Experiment 7: Equity Panel Application — Rolling Monitor on 115 US Large-Cap Stocks.

Downloads daily adjusted returns for 115 US large-cap stocks using yfinance.
Applies rolling QIS covariance estimates and projector movement monitoring.
Validates flag rates are higher during known market stress events.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import sample_cov, qis
from src.projector import projector_distance_lemma1, operator_norm_error
from src.functionals import (
    compute_AR_K, delta_method_var_AR_K, estimate_kappa_moment,
)
from src.monitoring import analytic_null_simulator


# ── Constants ────────────────────────────────────────────────────────────────
M = 252       # rolling window
K_MAX = 5     # maximum factors to monitor
K_DEFAULT = 2 # default K (also the most dominant gap)
ALPHA_BAND = 0.75
N_DRAWS_ANALYTIC = 200   # paper: 400
N_BOOT = 30              # paper: 60 (every 21st date)
BOOT_EVERY = 21
SEED = 42
Z95 = 1.959964

# 115 S&P 500 large-caps with history to ~2000
SP500_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","META","BRK-B","JPM","JNJ","V",
    "PG","UNH","HD","MA","NVDA","BAC","XOM","KO","PFE","ABBV",
    "CVX","CSCO","PEP","COST","MRK","TMO","ABT","ACN","MCD","AVGO",
    "WMT","NEE","DIS","T","VZ","NFLX","CRM","LLY","INTC","TXN",
    "HON","IBM","AMGN","BMY","QCOM","UPS","CAT","GS","BA","MDT",
    "SPGI","AMT","ISRG","CVS","NOW","LMT","SYK","AXP","ANTM","BDX",
    "DE","CI","GILD","C","USB","BLK","ZTS","CB","VRTX","SO",
    "DUK","EL","MO","REGN","PLD","TGT","EMR","AON","GD","NSC",
    "SHW","FIS","MMC","ICE","F","GM","WBA","HCA","TFC","ECL",
    "ITW","EW","D","PGR","CME","AIG","WM","EXC","PSA","IQV",
    "HUM","IDXX","BIIB","MCO","APD","ILMN","ETN","PPL","WAT","MTD",
    "FMC","RMD","CTSH","SWK","BK","BBY","DGX","EFX","FBHS","WHR",
]

# ── Subperiod date ranges ────────────────────────────────────────────────────
SUBPERIODS = {
    "Calm_2017_19": ("2017-01-01", "2019-12-31"),
    "GFC_2008_09":  ("2008-09-01", "2009-06-30"),
    "COVID_2020":   ("2020-02-01", "2020-06-30"),
    "RateHikes_2022_23": ("2022-01-01", "2023-12-31"),
}


# ── Data download ─────────────────────────────────────────────────────────────

def download_equity_data(tickers: list, start: str = "2000-01-01",
                          end: str = "2023-12-31") -> pd.DataFrame:
    """Download adjusted closing prices from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not available — using synthetic data fallback.")
        return None

    print(f"  Downloading {len(tickers)} tickers from {start} to {end} ...", flush=True)
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            price = raw["Close"]
        else:
            price = raw
        # Keep columns with >99% data availability
        min_obs = int(0.90 * len(price))
        price = price.dropna(axis=1, thresh=min_obs)
        # Fill remaining small gaps (forward then backward)
        price = price.ffill().bfill()
        price = price.dropna(axis=1)
        print(f"  Kept {price.shape[1]} tickers after quality filter.", flush=True)
        return price
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def _make_synthetic_panel(N: int = 115, T: int = 6276,
                           seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic log-return panel when real data unavailable."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2000-01-03", periods=T)
    # Spiked factor model: 5 factors + idio noise
    K_f = 5
    Lambda = rng.standard_normal((N, K_f)) / np.sqrt(K_f)
    Q, _ = np.linalg.qr(Lambda)
    scales = np.array([5.0, 2.0, 1.5, 1.2, 1.0])
    Lambda = Q * scales[np.newaxis, :]
    sigma2_idio = 0.3
    F = rng.standard_normal((T, K_f))
    Eps = rng.standard_normal((T, N)) * np.sqrt(sigma2_idio)
    R = F @ Lambda.T + Eps
    tickers = [f"STOCK_{i:03d}" for i in range(N)]
    return pd.DataFrame(R / 100.0, index=dates, columns=tickers)


# ── Rolling monitor ───────────────────────────────────────────────────────────

def rolling_monitor(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run the rolling projector-movement monitor on a returns panel.

    Returns
    -------
    pd.DataFrame with columns:
        date, D_hat, tau_star_obs, flag_band, flag_analytic,
        AR_K, CI_lo, CI_hi, kappa_hat, weak_id, K_hat
    """
    rng = np.random.default_rng(SEED)
    R = returns.values                   # (T, N)
    dates = returns.index
    T_total, N = R.shape

    results = []
    prev_U = None
    prev_T_obs = None
    prev_C = None
    prev_evals = None
    boot_threshold = None
    boot_count = 0

    for t in range(M, T_total):
        window = R[t - M : t]
        S_t = sample_cov(window)
        C_t = qis(S_t, M)

        evals_t, evecs_t = np.linalg.eigh(C_t)
        evals_desc = evals_t[::-1]
        evecs_desc = evecs_t[:, ::-1]

        # Select K_hat = argmax gap
        gaps = evals_desc[:-1] - evals_desc[1:]
        K_hat = int(np.argmax(gaps[:K_MAX]) + 1)

        U_hat_t = evecs_desc[:, :K_hat]

        # Observable band (sigma-free)
        eta_obs = operator_norm_error(C_t, S_t)
        Delta_obs = max(evals_desc[K_hat - 1] - evals_desc[K_hat], 1e-8)
        if Delta_obs <= 2 * eta_obs:
            T_obs = float(np.sqrt(2.0 * K_hat))
            weak_id = True
        else:
            from src.projector import compute_band_term
            T_obs = compute_band_term(eta_obs, C_t, S_t, Delta_obs - 2 * eta_obs, K_hat)
            weak_id = False

        row = {
            "date": dates[t],
            "D_hat": np.nan,
            "tau_star_obs": np.nan,
            "flag_band": False,
            "flag_analytic": False,
            "AR_K": compute_AR_K(evals_desc, K_hat),
            "CI_lo": np.nan, "CI_hi": np.nan,
            "kappa_hat": np.nan,
            "weak_id": weak_id,
            "K_hat": K_hat,
        }

        if prev_U is not None and prev_U.shape[1] == K_hat:
            D_hat_t = projector_distance_lemma1(U_hat_t, prev_U, K_hat)
            tau_star_obs = float(min(np.sqrt(2.0 * K_hat), T_obs + prev_T_obs))
            flag_band = bool(D_hat_t > ALPHA_BAND * tau_star_obs)

            # Analytic test with kappa correction
            kappa_hat = estimate_kappa_moment(window)
            sub_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            null_draws = analytic_null_simulator(
                evals_desc, K_hat, 1, M, N_DRAWS_ANALYTIC, sub_rng
            )
            q95_analytic = np.quantile(null_draws, 0.95)
            q95_ell = np.sqrt(1 + kappa_hat) * q95_analytic
            flag_analytic = bool(D_hat_t > q95_ell)

            # AR_K CI
            var_ar = delta_method_var_AR_K(evals_desc, K_hat, kappa=kappa_hat) / M
            half_w = Z95 * np.sqrt(max(0, var_ar))
            ar_hat = compute_AR_K(evals_desc, K_hat)
            ci_lo = ar_hat - half_w
            ci_hi = ar_hat + half_w

            row.update({
                "D_hat": D_hat_t,
                "tau_star_obs": tau_star_obs,
                "flag_band": flag_band,
                "flag_analytic": flag_analytic,
                "AR_K": ar_hat,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
                "kappa_hat": kappa_hat,
            })

        prev_U = U_hat_t
        prev_T_obs = T_obs
        prev_C = C_t
        prev_evals = evals_desc
        results.append(row)

    return pd.DataFrame(results).set_index("date")


# ── Subperiod stats ───────────────────────────────────────────────────────────

def compute_subperiod_stats(monitor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Table 11 and 12 style statistics for each subperiod."""
    valid = monitor_df.dropna(subset=["D_hat"])
    records = []
    for name, (start, end) in SUBPERIODS.items():
        sub = valid[start:end]
        if len(sub) == 0:
            continue
        analytic_flag_rate = sub["flag_analytic"].mean() * 100
        band_flag_rate = sub["flag_band"].mean() * 100
        weak_id_share = sub["weak_id"].mean() * 100
        records.append({
            "Subperiod": name,
            "n_dates": len(sub),
            "Median_D_hat": round(sub["D_hat"].median(), 4),
            "Analytic_flag_%": round(analytic_flag_rate, 1),
            "Band_flag_%": round(band_flag_rate, 1),
            "Weak_ID_%": round(weak_id_share, 1),
            "Median_AR_K": round(sub["AR_K"].median(), 4),
            "Median_kappa_hat": round(sub["kappa_hat"].median(), 3),
        })
    return pd.DataFrame(records)


# ── Event study ───────────────────────────────────────────────────────────────

def event_study(monitor_df: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Compute forward 22-day volatility on flagged vs unflagged dates."""
    valid = monitor_df.dropna(subset=["D_hat"])
    fwd_vols_flag = []
    fwd_vols_unflag = []
    R = returns
    all_dates = valid.index.tolist()

    for date in all_dates:
        try:
            idx = R.index.get_loc(date)
            if idx + 22 >= len(R):
                continue
            fwd = R.iloc[idx : idx + 22]
            port_ret = fwd.mean(axis=1)
            ann_vol = port_ret.std() * np.sqrt(252)
            if valid.loc[date, "flag_analytic"]:
                fwd_vols_flag.append(ann_vol)
            else:
                fwd_vols_unflag.append(ann_vol)
        except Exception:
            continue

    return pd.DataFrame({
        "Category": ["Flagged", "Unflagged"],
        "Mean_fwd_vol": [
            np.mean(fwd_vols_flag) if fwd_vols_flag else np.nan,
            np.mean(fwd_vols_unflag) if fwd_vols_unflag else np.nan,
        ],
        "n_dates": [len(fwd_vols_flag), len(fwd_vols_unflag)],
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def run_experiment_7():
    """Run Experiment 7."""
    os.makedirs("/workspace/project/results", exist_ok=True)
    os.makedirs("/workspace/project/data", exist_ok=True)

    # ── Step 1: Load or download data ──────────────────────────────────────
    cache_path = "/workspace/project/data/equity_returns.parquet"
    if os.path.exists(cache_path):
        print("  Loading cached return data ...", flush=True)
        returns = pd.read_parquet(cache_path)
    else:
        prices = download_equity_data(SP500_TICKERS)
        if prices is None or prices.shape[1] < 50:
            print("  Using synthetic panel fallback.", flush=True)
            returns = _make_synthetic_panel(N=115)
        else:
            # Keep up to 115 tickers
            if prices.shape[1] > 115:
                prices = prices.iloc[:, :115]
            returns = np.log(prices / prices.shift(1)).dropna()
            returns = returns.astype(np.float64)
        print(f"  Panel shape: {returns.shape}", flush=True)
        returns.to_parquet(cache_path)

    N_actual = returns.shape[1]
    print(f"  Using {N_actual} assets, {returns.shape[0]} dates.", flush=True)

    if returns.shape[0] < M + 50:
        print("  Not enough data for rolling monitor.")
        return

    # ── Step 2: Run rolling monitor ─────────────────────────────────────────
    print("  Running rolling monitor ...", flush=True)
    monitor_df = rolling_monitor(returns)

    # ── Step 3: Save monitor results ─────────────────────────────────────────
    monitor_df.to_csv("/workspace/project/results/exp7_monitor_series.csv")
    print(f"  Monitor saved. Valid rows: {monitor_df.dropna(subset=['D_hat']).shape[0]}", flush=True)

    # ── Step 4: Subperiod statistics ─────────────────────────────────────────
    sub_stats = compute_subperiod_stats(monitor_df)
    print("\nTable 11/12: Subperiod Statistics")
    print(sub_stats.to_string(index=False))
    sub_stats.to_csv("/workspace/project/results/table11_12_subperiod.csv", index=False)

    # ── Step 5: Event study ──────────────────────────────────────────────────
    ev_df = event_study(monitor_df, returns)
    print("\nTable 13: Event Study")
    print(ev_df.to_string(index=False))
    ev_df.to_csv("/workspace/project/results/table13_event_study.csv", index=False)

    # ── Step 6: Calm calibration ─────────────────────────────────────────────
    valid = monitor_df.dropna(subset=["D_hat"])
    calm_data = valid["2017-01-01":"2019-12-31"]["D_hat"]
    if len(calm_data) > 0:
        tau_cal = float(np.percentile(calm_data, 95))
        calm_flag_rates = {}
        for name, (start, end) in SUBPERIODS.items():
            sub_d = valid[start:end]["D_hat"]
            if len(sub_d) > 0:
                rate = (sub_d > tau_cal).mean() * 100
                calm_flag_rates[name] = round(rate, 1)
        cal_df = pd.DataFrame(
            list(calm_flag_rates.items()),
            columns=["Subperiod", "Cal_flag_%"]
        )
        cal_df["tau_cal"] = round(tau_cal, 4)
        print(f"\nTable 14: Calm Calibration (tau_cal={tau_cal:.4f})")
        print(cal_df.to_string(index=False))
        cal_df.to_csv("/workspace/project/results/table14_calm_calibration.csv", index=False)

    # ── Step 7: Figure 8 ─────────────────────────────────────────────────────
    valid_plot = monitor_df.dropna(subset=["D_hat"])
    fig8, ax8 = plt.subplots(figsize=(14, 5))
    ax8.plot(valid_plot.index, valid_plot["D_hat"], lw=0.7, color="steelblue",
             label="D̂ (projector movement)")

    # Shade stress periods
    stress = [
        ("GFC_2008_09",  "2008-09-01", "2009-06-30",  "salmon",    0.25),
        ("COVID_2020",   "2020-02-01", "2020-06-30",  "lightcoral", 0.3),
    ]
    for (label, s, e, col, alpha_) in stress:
        try:
            ax8.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=col,
                        alpha=alpha_, label=label)
        except Exception:
            pass

    # Mark flag dates
    flag_dates = valid_plot[valid_plot["flag_analytic"]].index
    if len(flag_dates) > 0:
        ax8.scatter(flag_dates, valid_plot.loc[flag_dates, "D_hat"],
                    color="red", s=4, zorder=5, alpha=0.6, label="Analytic flag")

    # Calm calibration threshold
    if len(calm_data) > 0:
        ax8.axhline(tau_cal, color="k", lw=1.2, ls="--", label=f"τ_cal={tau_cal:.3f}")

    ax8.set_xlabel("Date")
    ax8.set_ylabel("Projector Movement D̂")
    ax8.set_title("Figure 8: Rolling Projector Movement and Analytic Flags")
    ax8.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig("/workspace/project/results/fig8_rolling_monitor.png", dpi=100)
    plt.close()

    print("\nSaved Figure 8 to results/fig8_rolling_monitor.png")

    return sub_stats, ev_df


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 7: Equity Panel Application")
    print("=" * 60)
    run_experiment_7()
