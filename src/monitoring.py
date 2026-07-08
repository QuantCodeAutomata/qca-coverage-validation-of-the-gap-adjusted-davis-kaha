"""Monitoring procedures for projector-movement detection in the QCA project.

Provides the analytic null simulator (Proposition 2), bootstrap calibration
(Algorithm 2), and multiple-testing corrections (BH and BY).
"""

import numpy as np

from .estimators import apply_estimator
from .projector import projector_distance_lemma1


# ---------------------------------------------------------------------------
# Analytic null simulator (Proposition 2)
# ---------------------------------------------------------------------------

def analytic_null_simulator(
    eigenvalues: np.ndarray,
    K: int,
    s: int,
    M: int,
    n_draws: int = 400,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Simulate the null distribution of the projector-movement statistic.

    Implements Proposition 2.  Under the null of no structural break, the
    limiting projector-movement statistic D_tilde is approximated by:

        D_tilde = sqrt(2 * sum_{i<K, j>=K} (G_ij / (lam_i - lam_j))^2)

    where G_ij = (y_new[i,:] @ y_new[j,:] - y_old[i,:] @ y_old[j,:]) / M,
    y_new[i,:] ~ N(0, lam_i * I_s), y_old[i,:] ~ N(0, lam_i * I_s) i.i.d.

    Parameters
    ----------
    eigenvalues : np.ndarray of shape (N,)
        Population eigenvalues sorted in **descending** order; the first K are
        signal spikes.
    K : int
        Number of signal factors (rank of the projector).
    s : int
        Step size between consecutive windows (number of new observations).
    M : int
        Sample size of each estimation window.
    n_draws : int, optional
        Number of Monte Carlo draws (default 400).
    rng : np.random.Generator, optional
        Random generator; a default_rng(0) is used if None.

    Returns
    -------
    np.ndarray of shape (n_draws,)
        Array of simulated D_tilde values under the null.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(eigenvalues)
    lam = np.asarray(eigenvalues, dtype=float)

    # Pre-compute the gap matrix for i < K, j >= K
    lam_top = lam[:K]           # (K,)
    lam_bot = lam[K:]           # (N-K,)
    gaps = lam_top[:, np.newaxis] - lam_bot[np.newaxis, :]  # (K, N-K)
    # Guard against zero gaps (degenerate eigenvalues)
    gaps = np.where(np.abs(gaps) > 1e-15, gaps, 1e-15)

    sqrt_lam = np.sqrt(np.maximum(lam, 0.0))   # (N,)

    D_tildes = np.empty(n_draws)
    for d in range(n_draws):
        # y_new[i, k] ~ N(0, lam_i):  shape (N, s)
        Y_new = rng.standard_normal((N, s)) * sqrt_lam[:, np.newaxis]
        Y_old = rng.standard_normal((N, s)) * sqrt_lam[:, np.newaxis]

        # G = (Y_new @ Y_new.T - Y_old @ Y_old.T) / M
        # Extract the (K) x (N-K) off-diagonal block
        G_new_cross = Y_new[:K, :] @ Y_new[K:, :].T    # (K, N-K)
        G_old_cross = Y_old[:K, :] @ Y_old[K:, :].T    # (K, N-K)
        G_cross = (G_new_cross - G_old_cross) / M       # (K, N-K)

        D_tildes[d] = float(np.sqrt(2.0 * np.sum((G_cross / gaps) ** 2)))

    return D_tildes


# ---------------------------------------------------------------------------
# Bootstrap calibration (Algorithm 2)
# ---------------------------------------------------------------------------

def _top_k_eigvecs(C: np.ndarray, K: int) -> np.ndarray:
    """Return the top-K eigenvectors of C (descending eigenvalue order)."""
    vals, vecs = np.linalg.eigh(C)
    # eigh returns ascending; take last K
    return vecs[:, -K:]


def bootstrap_projector_movement(
    returns_A: np.ndarray,
    C_A: np.ndarray,
    K: int,
    s: int,
    M: int,
    estimator: str = 'qis',
    n_boot: int = 60,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Calibrate the projector-movement statistic via bootstrap (Algorithm 2).

    Under the null, both windows A and B are drawn from the same distribution
    N(0, C_A).  For each bootstrap replicate:

    1. Sample M + s observations from N(0, C_A).
    2. Form window_A = obs[0 : M],   window_B = obs[s : M + s].
    3. Estimate C_A_boot = apply_estimator(window_A, estimator).
    4. Estimate C_B_boot = apply_estimator(window_B, estimator).
    5. Extract top-K eigenvectors U_A_boot, U_B_boot.
    6. D_hat_boot = projector_distance_lemma1(U_A_boot, U_B_boot, K).

    Parameters
    ----------
    returns_A : np.ndarray of shape (M, N)
        Historical returns for window A (used to determine N; the actual
        estimation uses C_A).
    C_A : np.ndarray of shape (N, N)
        Covariance estimate for the baseline window A (used as the DGP for
        the bootstrap).
    K : int
        Number of signal factors.
    s : int
        Step size between windows.
    M : int
        Window size (number of observations per covariance estimate).
    estimator : str, optional
        Covariance estimator key passed to :func:`apply_estimator`
        (default ``'qis'``).
    n_boot : int, optional
        Number of bootstrap replicates (default 60).
    rng : np.random.Generator, optional
        Random generator; a default_rng(0) is used if None.

    Returns
    -------
    np.ndarray of shape (n_boot,)
        Bootstrap D_hat values under the null of no structural break.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = C_A.shape[0]
    # Cholesky factor of C_A for efficient multivariate normal sampling
    L = np.linalg.cholesky(C_A)

    D_hats = np.empty(n_boot)
    for b in range(n_boot):
        Z = rng.standard_normal((M + s, N))
        X = (L @ Z.T).T                           # (M+s, N) from N(0, C_A)
        window_A = X[:M, :]                        # first M rows
        window_B = X[s: M + s, :]                  # rows s … M+s-1

        C_A_boot = apply_estimator(window_A, estimator)
        C_B_boot = apply_estimator(window_B, estimator)

        U_A = _top_k_eigvecs(C_A_boot, K)
        U_B = _top_k_eigvecs(C_B_boot, K)
        D_hats[b] = projector_distance_lemma1(U_A, U_B, K)

    return D_hats


# ---------------------------------------------------------------------------
# Multiple-testing corrections
# ---------------------------------------------------------------------------

def apply_bh_correction(
    pvalues: np.ndarray,
    q: float = 0.10,
) -> np.ndarray:
    """Apply the Benjamini-Hochberg (BH) false discovery rate correction.

    Finds the largest rank k such that p_{(k)} <= k * q / n, and rejects
    the hypotheses corresponding to the k smallest p-values.

    Parameters
    ----------
    pvalues : np.ndarray of shape (n,)
        Array of p-values (one per test).
    q : float, optional
        Target false discovery rate level (default 0.10).

    Returns
    -------
    np.ndarray of shape (n,) of bool
        Boolean rejection array; True means the null hypothesis is rejected.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_pvals = pvalues[order]

    thresholds = np.arange(1, n + 1) * q / n
    below = sorted_pvals <= thresholds

    # Find the largest k with p_{(k)} <= k*q/n
    if not np.any(below):
        return np.zeros(n, dtype=bool)

    k_star = int(np.max(np.where(below)[0]))   # 0-based index of largest k
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[: k_star + 1] = True

    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    rejected[order] = rejected_sorted
    return rejected


def apply_by_correction(
    pvalues: np.ndarray,
    q: float = 0.10,
) -> np.ndarray:
    """Apply the Benjamini-Yekutieli (BY) false discovery rate correction.

    Like BH but uses q / H_n as the effective level, where H_n = sum_{j=1}^n 1/j
    is the n-th harmonic number.  Valid under arbitrary (including positive)
    dependence structure.

    Parameters
    ----------
    pvalues : np.ndarray of shape (n,)
        Array of p-values (one per test).
    q : float, optional
        Target false discovery rate level (default 0.10).

    Returns
    -------
    np.ndarray of shape (n,) of bool
        Boolean rejection array; True means the null hypothesis is rejected.
    """
    n = len(pvalues)
    H_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    q_by = q / H_n
    return apply_bh_correction(pvalues, q=q_by)
