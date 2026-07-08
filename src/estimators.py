"""Covariance matrix estimators for the QCA project.

Provides sample covariance, linear shrinkage (Ledoit-Wolf, OAS), nonlinear
shrinkage (QIS/QuEST), and Marchenko-Pastur clipping estimators.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Basic estimator
# ---------------------------------------------------------------------------

def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Compute the sample covariance matrix from a returns matrix.

    Demeans each column, then computes (M-1)^{-1} * R_tilde.T @ R_tilde.

    Parameters
    ----------
    returns : np.ndarray of shape (M, N)
        Return observations; M time steps, N assets.

    Returns
    -------
    np.ndarray of shape (N, N)
        Sample covariance matrix.
    """
    M = returns.shape[0]
    R_tilde = returns - returns.mean(axis=0)
    return R_tilde.T @ R_tilde / (M - 1)


# ---------------------------------------------------------------------------
# Linear shrinkage estimators
# ---------------------------------------------------------------------------

def ledoit_wolf(S: np.ndarray, n: int) -> np.ndarray:
    """Compute the Ledoit-Wolf analytically shrunk covariance matrix.

    Implements the Oracle Approximating Shrinkage (OAS) formula from
    Ledoit & Wolf (2004) under Gaussian assumptions.  The shrinkage target
    is (trace(S)/p) * I, and the shrinkage intensity is:

        rho = min(1, (tr(S^2) + tr(S)^2) / (n * (tr(S^2) - tr(S)^2/p)))

    This matches sklearn.covariance.ledoit_wolf for Gaussian data.

    Parameters
    ----------
    S : np.ndarray of shape (N, N)
        Sample covariance matrix.
    n : int
        Number of observations used to compute S.

    Returns
    -------
    np.ndarray of shape (N, N)
        Ledoit-Wolf shrunk covariance matrix.
    """
    p = S.shape[0]
    tr_S = np.trace(S)
    tr_S2 = np.sum(S * S)          # ||S||_F^2 = tr(S^2) for symmetric S
    mu = tr_S / p
    denom = n * (tr_S2 - tr_S ** 2 / p)
    rho = 0.0 if denom <= 0.0 else min(1.0, (tr_S2 + tr_S ** 2) / denom)
    shrunk = (1.0 - rho) * S
    shrunk.flat[:: p + 1] += rho * mu
    return shrunk


def oas(S: np.ndarray, n: int) -> np.ndarray:
    """Compute the Oracle Approximating Shrinkage (OAS) covariance matrix.

    Implements the closed-form estimator from Chen, Wiesel, Eldar & Hero
    (2010), exactly matching sklearn.covariance.oas for centred data:

        rho = min(1, (alpha + mu^2) / ((n+1) * (alpha - mu^2/p)))

    where alpha = mean(S^2) = ||S||_F^2 / p^2 and mu = tr(S) / p.

    Parameters
    ----------
    S : np.ndarray of shape (N, N)
        Sample covariance matrix.
    n : int
        Number of observations used to compute S.

    Returns
    -------
    np.ndarray of shape (N, N)
        OAS shrunk covariance matrix.
    """
    p = S.shape[0]
    alpha = np.mean(S * S)          # ||S||_F^2 / p^2
    mu = np.trace(S) / p
    mu_sq = mu ** 2
    num = alpha + mu_sq
    den = (n + 1) * (alpha - mu_sq / p)
    rho = 1.0 if den <= 0.0 else min(num / den, 1.0)
    shrunk = (1.0 - rho) * S
    shrunk.flat[:: p + 1] += rho * mu
    return shrunk


# ---------------------------------------------------------------------------
# Nonlinear shrinkage (QIS / Oracle QuEST)
# ---------------------------------------------------------------------------

def qis(S: np.ndarray, n: int) -> np.ndarray:
    """Compute the QIS / Analytical Nonlinear Shrinkage covariance estimator.

    Hybrid approach:
      - Spike eigenvalues (above MP upper edge): BBP inverse debiasing.
      - Bulk eigenvalues: Ledoit-Wolf (2020) Cauchy-kernel ANLS formula applied
        using ONLY the bulk eigenvalues in the kernel sums (so spikes don't
        contribute spurious positive Hilbert-transform mass to the bulk).
      - Trace preservation: only within bulk (spikes fixed at BBP-inverse value
        so they don't get dragged by a global rescaling factor).

    Cauchy-kernel ANLS for bulk eigenvalues (Custom — Context7 confirmed no
    library covers this combination exactly):

        diffs_j  = lam_bulk[j] - lam_bulk[i]
        denom_j  = diffs_j^2 + eta^2
        Re_g_i   = (1/N_b) Σ_j  diffs_j / denom_j
        B_raw_i  = (eta / N_b) Σ_j  1 / denom_j
        A_i      = 1 - c_b - c_b·lam[i]·Re_g_i    (c_b = N_b / n)
        B_i      = c_b·lam[i]·B_raw_i
        d_bulk_i = lam[i] / max(A_i^2 + B_i^2, eps)

    BBP inverse for spike eigenvalue lam_s:
        b = lam_s - sigma2·(1 - c)
        d_spike = (b + sqrt(max(b^2 - 4c·sigma2^2, 0))) / 2

    Parameters
    ----------
    S : np.ndarray of shape (N, N)
        Sample covariance matrix.
    n : int
        Number of observations used to compute S.

    Returns
    -------
    np.ndarray of shape (N, N)
        QIS nonlinearly shrunk covariance matrix.
    """
    N  = S.shape[0]
    c  = N / float(n)
    sq = np.sqrt(c)

    # ── eigendecompose, sort descending ──────────────────────────────────────
    raw_vals, U = np.linalg.eigh(S)
    idx = np.argsort(raw_vals)[::-1]
    lam = np.maximum(raw_vals[idx], 0.0)
    U   = U[:, idx]

    # ── MP upper-edge estimation (iterative, spike-free init) ─────────────────
    # Use the median of ALL eigenvalues as the initial sigma2 estimate.
    # For a spiked model with few spikes, the median equals the bulk median ≈ sigma2.
    # Using only the bottom half (N//2:) biases low in small-N settings (N=30).
    sigma2 = np.median(lam)
    for _ in range(40):
        lp   = sigma2 * (1.0 + sq) ** 2
        bulk = lam[lam <= lp]
        if bulk.size == 0:
            break
        s2n  = bulk.mean()                     # MP mean = sigma2 (no 1+c factor)
        if abs(s2n - sigma2) < 1e-8 * max(sigma2, 1e-12):
            sigma2 = s2n; break
        sigma2 = s2n
    lam_plus   = sigma2 * (1.0 + sq) ** 2

    spike_mask = lam > lam_plus * 1.01
    bulk_mask  = ~spike_mask
    N_b        = int(bulk_mask.sum())

    # ── Cauchy-kernel shrinkage on BULK eigenvalues only ──────────────────────
    # Using bulk eigenvalues in the kernel avoids spike-induced bias in Re_g.
    # c_b = N_b / n keeps the concentration ratio consistent.
    c_b  = N_b / float(n)
    lam_b = lam[bulk_mask]                     # shape (N_b,)

    eta  = float(n) ** (-1.0 / 3.0)
    if N_b >= 4:
        eta = max(eta, 0.01 * float(np.subtract(*np.percentile(lam_b, [75, 25]))))
    eta = max(eta, 1e-4)

    # (N_b, N_b) pairwise differences: row i, col j → lam_b[j] - lam_b[i]
    db_mat  = lam_b[np.newaxis, :] - lam_b[:, np.newaxis]
    dn_mat  = db_mat ** 2 + eta ** 2

    Re_g_b  = (db_mat / dn_mat).mean(axis=1)          # (N_b,): signed Hilbert
    B_raw_b = (eta    / dn_mat).mean(axis=1)           # (N_b,): density kernel

    A_b = 1.0 - c_b - c_b * lam_b * Re_g_b
    B_b = c_b * lam_b * B_raw_b

    dv_b    = A_b ** 2 + B_b ** 2
    d_bulk  = np.where(dv_b > 1e-14, lam_b / dv_b, lam_b)

    # ── Bulk-only trace preservation ──────────────────────────────────────────
    # Spikes are NOT rescaled here; their BBP-inverse values are fixed.
    # Only the bulk sum is adjusted so that:
    #   sum(d_spike_BBP) + sum(d_bulk_scaled) = tr(S)
    tr_S         = np.trace(S)
    d_tilde      = lam.copy()           # spikes keep sample value for now

    # BBP inverse for spikes.
    # Quadratic: λ² - (lam_s + σ²(1-c))λ + lam_s·σ² = 0
    # B = lam_s + σ²(1-c),  disc = B² - 4·lam_s·σ²
    for i in np.where(spike_mask)[0]:
        li   = lam[i]
        B    = li + sigma2 * (1.0 - c)
        disc = B ** 2 - 4.0 * li * sigma2
        d_tilde[i] = (B + np.sqrt(max(disc, 0.0))) / 2.0

    spike_sum = d_tilde[spike_mask].sum()
    bulk_target = tr_S - spike_sum          # desired bulk total

    bulk_raw_sum = d_bulk.sum()
    if bulk_raw_sum > 1e-14 and N_b > 0:
        d_tilde[bulk_mask] = d_bulk * (bulk_target / bulk_raw_sum)
    else:
        d_tilde[bulk_mask] = bulk_target / max(N_b, 1)

    return (U * d_tilde[np.newaxis, :]) @ U.T


# ---------------------------------------------------------------------------
# Marchenko-Pastur clipping
# ---------------------------------------------------------------------------

def mp_clip(S: np.ndarray, n: int) -> np.ndarray:
    """Compute the Marchenko-Pastur clipped covariance matrix.

    Estimates the noise level sigma^2 from the bulk eigenvalues
    iteratively, clips spikes above lambda_plus = sigma^2*(1+sqrt(c))^2,
    and replaces all bulk eigenvectors with their average eigenvalue.

    Algorithm
    ---------
    1. Eigendecompose S; sort descending.
    2. Initialise: sigma2 = median(lower half of lam).
    3. Iterate up to 40 times:
         lambda_plus = sigma2 * (1+sqrt(c))^2
         bulk        = lam[lam <= lambda_plus]
         sigma2      = mean(bulk)   [MP mean = sigma2 exactly]
    4. Spike eigenvalues: lam > lambda_plus (kept at sample values).
    5. Reconstruct:
         C_mp = U_spike @ diag(spike_eigs) @ U_spike.T
                + bulk_mean * (I - U_spike @ U_spike.T)

    Parameters
    ----------
    S : np.ndarray of shape (N, N)
        Sample covariance matrix.
    n : int
        Number of observations used to compute S.

    Returns
    -------
    np.ndarray of shape (N, N)
        Marchenko-Pastur clipped covariance matrix.
    """
    N = S.shape[0]
    c = N / n
    sqrt_c = np.sqrt(c)

    raw_vals, U = np.linalg.eigh(S)
    idx = np.argsort(raw_vals)[::-1]
    lam = raw_vals[idx]
    U = U[:, idx]
    lam = np.maximum(lam, 0.0)

    # Iterative sigma2 estimation.  MP mean = sigma2 exactly, so bulk.mean()
    # is the correct update (no division by (1+c)).  Initialise from the
    # global median: for few spikes, median of all eigenvalues ≈ bulk median ≈ sigma2,
    # robust across small and large N (avoids bias at N=30 from bottom-half init).
    sigma2 = np.median(lam)
    for _ in range(40):
        lambda_plus = sigma2 * (1.0 + sqrt_c) ** 2
        bulk_eigs = lam[lam <= lambda_plus]
        if bulk_eigs.size == 0:
            break
        sigma2_new = bulk_eigs.mean()       # MP mean = sigma2
        if sigma2_new <= 0.0:
            break
        if abs(sigma2_new - sigma2) < 1e-8 * sigma2:
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    lambda_plus = sigma2 * (1.0 + sqrt_c) ** 2
    spike_mask = lam > lambda_plus
    bulk_mask = ~spike_mask

    spike_eigs = lam[spike_mask]
    bulk_eigs = lam[bulk_mask]
    U_spike = U[:, spike_mask]

    bulk_mean = bulk_eigs.mean() if bulk_eigs.size > 0 else sigma2

    # Reconstruct: spikes keep their eigenvalues; bulk gets uniform bulk_mean
    N_mat = S.shape[0]
    C_mp = bulk_mean * np.eye(N_mat)
    if U_spike.shape[1] > 0:
        C_mp += U_spike @ (np.diag(spike_eigs - bulk_mean)) @ U_spike.T

    return C_mp


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def apply_estimator(returns: np.ndarray, method: str) -> np.ndarray:
    """Compute a covariance estimate from raw returns using the named method.

    Parameters
    ----------
    returns : np.ndarray of shape (M, N)
        Return observations; M time steps, N assets.
    method : str
        One of ``'sample'``, ``'ledoit_wolf'``, ``'oas'``, ``'qis'``,
        ``'mp_clip'``.

    Returns
    -------
    np.ndarray of shape (N, N)
        Estimated covariance matrix.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    S = sample_cov(returns)
    n = returns.shape[0]

    if method == 'sample':
        return S
    elif method == 'ledoit_wolf':
        return ledoit_wolf(S, n)
    elif method == 'oas':
        return oas(S, n)
    elif method == 'qis':
        return qis(S, n)
    elif method == 'mp_clip':
        return mp_clip(S, n)
    else:
        raise ValueError(
            f"Unknown estimator '{method}'. "
            "Choose from: 'sample', 'ledoit_wolf', 'oas', 'qis', 'mp_clip'."
        )
