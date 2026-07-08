"""Spectral functionals and delta-method variance estimators for the QCA project.

Implements absorption ratio, first-factor share, spectral entropy, trace,
their delta-method asymptotic variances, kurtosis estimators, and the
BBP-based spike-debiasing procedure.
"""

import numpy as np
from scipy import linalg as sp_linalg
from scipy import optimize as sp_optimize
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Point estimators of spectral functionals
# ---------------------------------------------------------------------------

def compute_AR_K(eigenvalues: np.ndarray, K: int) -> float:
    """Compute the K-factor Absorption Ratio.

    AR_K = sum(eigenvalues[:K]) / sum(eigenvalues).

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues sorted in **descending** order.
    K : int
        Number of leading eigenvalues to include.

    Returns
    -------
    float
        Absorption ratio (value in [0, 1]).
    """
    T = float(np.sum(eigenvalues))
    return float(np.sum(eigenvalues[:K])) / T


def compute_f1(eigenvalues: np.ndarray) -> float:
    """Compute the first-factor variance share.

    f1 = eigenvalues[0] / sum(eigenvalues).

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues sorted in **descending** order.

    Returns
    -------
    float
        First-factor share (value in [0, 1]).
    """
    return float(eigenvalues[0]) / float(np.sum(eigenvalues))


def compute_spectral_entropy(eigenvalues: np.ndarray) -> float:
    """Compute the spectral (Shannon) entropy.

    H = -sum_i p_i * log(p_i),   p_i = eigenvalues_i / sum(eigenvalues).

    Probabilities are clipped at 1e-15 before taking the log to avoid
    numerical issues.

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues (need not be sorted).

    Returns
    -------
    float
        Spectral entropy (>= 0).
    """
    T = float(np.sum(eigenvalues))
    p = eigenvalues / T
    p_clipped = np.clip(p, 1e-15, None)
    return float(-np.sum(p_clipped * np.log(p_clipped)))


def compute_trace(eigenvalues: np.ndarray) -> float:
    """Compute the trace (sum of eigenvalues).

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues.

    Returns
    -------
    float
        Sum of all eigenvalues.
    """
    return float(np.sum(eigenvalues))


# ---------------------------------------------------------------------------
# Delta-method asymptotic variance estimators
# ---------------------------------------------------------------------------

def delta_method_var_AR_K(
    eigenvalues: np.ndarray,
    K: int,
    kappa: float = 0.0,
) -> float:
    """Compute the delta-method asymptotic variance for the absorption ratio AR_K.

    sigma2_AR = (2*(1+kappa)/T^4)
                * ((T - S_K)^2 * sum(lambda[:K]^2)
                   + S_K^2     * sum(lambda[K:]^2))

    where T = sum(eigenvalues), S_K = sum(eigenvalues[:K]).

    This is the *M-scaled* variance (i.e., M * Var(AR_K)).

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues sorted in **descending** order.
    K : int
        Number of leading eigenvalues in the absorption ratio.
    kappa : float, optional
        Excess-kurtosis parameter of the elliptical distribution (default 0).

    Returns
    -------
    float
        Asymptotic delta-method variance (M-scaled).
    """
    T = float(np.sum(eigenvalues))
    S_K = float(np.sum(eigenvalues[:K]))
    sum_sq_top = float(np.sum(eigenvalues[:K] ** 2))
    sum_sq_bot = float(np.sum(eigenvalues[K:] ** 2))
    return (2.0 * (1.0 + kappa) / T ** 4) * (
        (T - S_K) ** 2 * sum_sq_top + S_K ** 2 * sum_sq_bot
    )


def delta_method_var_f1(
    eigenvalues: np.ndarray,
    kappa: float = 0.0,
) -> float:
    """Compute the delta-method asymptotic variance for the first-factor share f1.

    sigma2_f1 = (2*(1+kappa)/T^2)
                * ((1 - f1)^2 * lambda[0]^2
                   + f1^2     * sum(lambda[1:]^2))

    where T = sum(eigenvalues), f1 = lambda[0]/T.

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues sorted in **descending** order.
    kappa : float, optional
        Excess-kurtosis parameter (default 0).

    Returns
    -------
    float
        Asymptotic delta-method variance (M-scaled).
    """
    T = float(np.sum(eigenvalues))
    f1 = float(eigenvalues[0]) / T
    sum_sq_rest = float(np.sum(eigenvalues[1:] ** 2))
    return (2.0 * (1.0 + kappa) / T ** 2) * (
        (1.0 - f1) ** 2 * float(eigenvalues[0]) ** 2
        + f1 ** 2 * sum_sq_rest
    )


def delta_method_var_H(
    eigenvalues: np.ndarray,
    kappa: float = 0.0,
) -> float:
    """Compute the delta-method asymptotic variance for the spectral entropy H.

    sigma2_H = (2*(1+kappa)/T^2)
               * sum_i lambda_i^2 * (log(p_i) + H)^2

    where p_i = lambda_i / T and H = -sum_i p_i * log(p_i).

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues (need not be sorted).
    kappa : float, optional
        Excess-kurtosis parameter (default 0).

    Returns
    -------
    float
        Asymptotic delta-method variance (M-scaled).
    """
    T = float(np.sum(eigenvalues))
    p = eigenvalues / T
    p_clipped = np.clip(p, 1e-15, None)
    H = float(-np.sum(p_clipped * np.log(p_clipped)))
    log_p = np.log(p_clipped)
    return (2.0 * (1.0 + kappa) / T ** 2) * float(
        np.sum(eigenvalues ** 2 * (log_p + H) ** 2)
    )


def delta_method_var_trace(
    eigenvalues: np.ndarray,
    kappa: float = 0.0,
) -> float:
    """Compute the delta-method asymptotic variance for the trace.

    sigma2_tr = 2*(1+kappa)*sum(lambda^2) + kappa*(sum(lambda))^2.

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Eigenvalues.
    kappa : float, optional
        Excess-kurtosis parameter (default 0).

    Returns
    -------
    float
        Asymptotic delta-method variance (M-scaled).
    """
    sum_sq = float(np.sum(eigenvalues ** 2))
    total = float(np.sum(eigenvalues))
    return 2.0 * (1.0 + kappa) * sum_sq + kappa * total ** 2


# ---------------------------------------------------------------------------
# Kurtosis (kappa) estimators
# ---------------------------------------------------------------------------

def estimate_kappa_moment(
    returns: np.ndarray,
    S: np.ndarray = None,
) -> float:
    """Estimate the elliptical-distribution kappa via coordinate-wise moments.

    For each asset i, computes the excess kurtosis of the marginal distribution
    and averages:

        kappa_hat = max(0, mean_i [(m4_i / m2_i^2 - 3) / 3])

    where m2_i = mean(r_i^2), m4_i = mean(r_i^4) after demeaning.

    Parameters
    ----------
    returns : np.ndarray of shape (M, N)
        Return observations.
    S : np.ndarray of shape (N, N), optional
        Sample covariance (unused; kept for API uniformity).

    Returns
    -------
    float
        Moment-based kappa estimate (clipped to >= 0).
    """
    R = returns - returns.mean(axis=0)     # demean
    m2 = np.mean(R ** 2, axis=0)           # (N,)
    m4 = np.mean(R ** 4, axis=0)           # (N,)
    # Avoid division by near-zero second moments
    safe_m2 = np.where(m2 > 1e-15, m2, 1e-15)
    excess_kurt_i = m4 / safe_m2 ** 2 - 3.0
    kappa_hat = float(np.mean(excess_kurt_i)) / 3.0
    return max(0.0, kappa_hat)


def estimate_kappa_radial_mle(
    returns: np.ndarray,
    S: np.ndarray,
) -> float:
    """Estimate kappa via radial maximum likelihood (Remark 4).

    Computes the Mahalanobis distances R_tau = r^T S^{-1} r for each return r,
    then fits the degrees-of-freedom parameter nu of an F(d_eff, nu)
    distribution to R_tau / d_eff by MLE, where d_eff = tr(S)^2 / tr(S^2).

    Returns kappa = max(0, 2/(nu_hat - 4)) if nu_hat < 100, else 0.

    Parameters
    ----------
    returns : np.ndarray of shape (M, N)
        Return observations (assumed demeaned or mean-zero).
    S : np.ndarray of shape (N, N)
        Sample covariance matrix.

    Returns
    -------
    float
        MLE-based kappa estimate (>= 0).
    """
    M_obs, N = returns.shape

    # Mahalanobis distances R_tau = r^T S^{-1} r  (shape M,)
    try:
        L = np.linalg.cholesky(S + 1e-10 * np.eye(N))
        sol = np.linalg.solve(L, returns.T)   # (N, M)
        R_tau = np.sum(sol ** 2, axis=0)       # (M,)
    except np.linalg.LinAlgError:
        R_tau = np.array([
            float(r @ sp_linalg.solve(S, r, assume_a='pos'))
            for r in returns
        ])

    # Under t_nu elliptical: R_tau ~ N * F(N, nu)  (approx, large M)
    # so x = R_tau / N  ~  F(N, nu),  with mean nu/(nu-2) -> 1 as nu->inf.
    x_data = np.clip(R_tau / float(N), 1e-12, None)
    dfn = float(N)

    def neg_loglik(nu: float) -> float:
        return -float(np.sum(sp_stats.f.logpdf(x_data, dfn=dfn, dfd=nu)))

    # Coarse grid to locate global minimum before bounded refinement
    nu_grid = np.array([4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0,
                        20.0, 30.0, 50.0, 100.0, 150.0, 200.0])
    nll_grid = np.array([neg_loglik(nu) for nu in nu_grid])
    idx0 = int(np.argmin(nll_grid))
    lo = float(nu_grid[max(0, idx0 - 1)])
    hi = float(nu_grid[min(len(nu_grid) - 1, idx0 + 1)])
    lo = max(4.001, lo)
    hi = min(200.0, hi)

    result = sp_optimize.minimize_scalar(
        neg_loglik, bounds=(lo, hi), method='bounded'
    )
    nu_hat = float(result.x)
    if result.fun > nll_grid[idx0] + 0.5:   # refinement diverged
        nu_hat = float(nu_grid[idx0])

    if nu_hat >= 100.0:
        kappa_hat = 0.0
    else:
        kappa_hat = max(0.0, 2.0 / (nu_hat - 4.0))
    return kappa_hat


# ---------------------------------------------------------------------------
# Spike-debiased absorption ratio
# ---------------------------------------------------------------------------

def _estimate_sigma2_mp(eigenvalues: np.ndarray, c: float) -> float:
    """Estimate noise variance sigma^2 from the MP bulk.

    Iterative algorithm: start from the lower half of the spectrum
    (to avoid spike contamination), then refine using the mean of
    eigenvalues that fall under the current MP upper-edge estimate.

    For the Marchenko–Pastur law MP(sigma^2, c) the bulk eigenvalues
    have mean == sigma^2, so iterating sigma2 <- mean(bulk) converges.
    """
    sqrt_c = np.sqrt(c)
    # Seed: use the global median. For a spiked model with few spikes the
    # global median equals the bulk median ≈ sigma^2. This avoids the
    # systematic downward bias of using only the bottom-half median at small N.
    sigma2 = float(np.median(eigenvalues))
    sigma2 = max(sigma2, 1e-8)

    n = eigenvalues.size
    for _ in range(20):
        lambda_plus = sigma2 * (1.0 + sqrt_c) ** 2
        bulk = eigenvalues[eigenvalues <= lambda_plus]
        if bulk.size < max(3, n // 5):
            break
        new_sigma2 = float(bulk.mean())   # mean of MP(sigma2,c) == sigma2
        if abs(new_sigma2 - sigma2) < 1e-10 * sigma2:
            sigma2 = new_sigma2
            break
        sigma2 = new_sigma2
        sigma2 = max(sigma2, 1e-8)
    return sigma2


def spike_debias_AR_K(
    eigenvalues: np.ndarray,
    K: int,
    N: int,
    M: int,
    sigma2_est: float = None,
    kappa: float = 0.0,
) -> tuple:
    """Compute the spike-debiased absorption ratio AR_K.

    Applies the BBP (Baik-Ben Arous-Péché) inverse formula to correct the
    upward bias in sample spike eigenvalues, preserves the empirical trace,
    and computes the debiased absorption ratio.

    BBP inverse debiasing for each spike lam_hat.
    Solves λ² - (lam_hat + σ²(1-c))λ + lam_hat·σ² = 0:

        B            = lam_hat + sigma2*(1-c)
        discriminant = B^2 - 4*lam_hat*sigma2
        lam_deb      = (B + sqrt(max(0, discriminant))) / 2

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N_dim,)
        Sample eigenvalues sorted in **descending** order.
    K : int
        Number of factors for the absorption ratio.
    N : int
        Number of assets (dimension).
    M : int
        Number of observations (sample size).
    sigma2_est : float, optional
        Noise variance estimate.  If None, estimated from the MP bulk.
    kappa : float, optional
        Excess-kurtosis parameter (currently unused; reserved for future use).

    Returns
    -------
    AR_K_debiased : float
        Debiased absorption ratio.
    eigenvalues_debiased : np.ndarray of shape (N_dim,)
        Full array of debiased eigenvalues (sorted descending).
    """
    c = N / M
    sqrt_c = np.sqrt(c)
    lam = np.array(eigenvalues, dtype=float)

    if sigma2_est is None:
        sigma2_est = _estimate_sigma2_mp(lam, c)

    lambda_plus = sigma2_est * (1.0 + sqrt_c) ** 2
    # Spikes are eigenvalues sufficiently above lambda_plus
    spike_threshold = lambda_plus + 0.01 * sigma2_est
    spike_mask = lam > spike_threshold
    bulk_mask = ~spike_mask

    eigs_deb = lam.copy()
    # BBP inverse: λ² - (lam_hat + σ²(1-c))λ + lam_hat·σ² = 0
    # B = lam_hat + σ²(1-c),  disc = B² - 4·lam_hat·σ²
    for idx in np.where(spike_mask)[0]:
        lam_hat = lam[idx]
        B    = lam_hat + sigma2_est * (1.0 - c)
        disc = B ** 2 - 4.0 * lam_hat * sigma2_est
        lam_deb = (B + np.sqrt(max(0.0, disc))) / 2.0
        eigs_deb[idx] = lam_deb

    # Trace preservation: redistribute spike reduction *additively* to bulk only.
    # This keeps debiased spike eigenvalues at their BBP-inverse values and
    # prevents the proportional re-scaling from partially un-doing the debiasing.
    if bulk_mask.sum() > 0 and spike_mask.sum() > 0:
        total_reduction = float(np.sum(lam[spike_mask])) - float(np.sum(eigs_deb[spike_mask]))
        eigs_deb[bulk_mask] += total_reduction / float(bulk_mask.sum())

    AR_K_deb = float(np.sum(eigs_deb[:K])) / float(np.sum(eigs_deb))
    return AR_K_deb, eigs_deb
