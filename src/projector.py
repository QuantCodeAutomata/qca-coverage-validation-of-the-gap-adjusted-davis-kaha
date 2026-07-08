"""Projector movement and band-width computations for the QCA project.

Implements Lemma 1 (projector distance), population and empirical band terms,
and the fully observable (Sigma-free) band estimator.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core projector distance
# ---------------------------------------------------------------------------

def projector_distance_lemma1(
    U_K_t: np.ndarray,
    U_K_prev: np.ndarray,
    K: int,
) -> float:
    """Compute the Frobenius distance between rank-K projectors via Lemma 1.

    For two orthonormal bases U_K_t and U_K_prev (each of shape N x K),
    the distance between the corresponding rank-K projectors is:

        D = sqrt(max(0, 2*K - 2 * ||U_K_t.T @ U_K_prev||_F^2))

    Parameters
    ----------
    U_K_t : np.ndarray of shape (N, K)
        Orthonormal columns spanning the top-K subspace at time t.
    U_K_prev : np.ndarray of shape (N, K)
        Orthonormal columns spanning the top-K subspace at time t-1.
    K : int
        Rank of the projector.

    Returns
    -------
    float
        Frobenius distance between the two rank-K projectors.
    """
    M_mat = U_K_t.T @ U_K_prev          # (K, K)
    fro_sq = np.sum(M_mat ** 2)
    return float(np.sqrt(max(0.0, 2.0 * K - 2.0 * fro_sq)))


def compute_population_projector_distance(
    Sigma_t: np.ndarray,
    Sigma_prev: np.ndarray,
    K: int,
) -> float:
    """Compute the exact projector distance between two population covariances.

    Eigendecomposes both matrices, extracts the top-K eigenvectors, and
    applies Lemma 1.

    Parameters
    ----------
    Sigma_t : np.ndarray of shape (N, N)
        Population covariance at time t.
    Sigma_prev : np.ndarray of shape (N, N)
        Population covariance at time t-1.
    K : int
        Rank of the projector (number of factors).

    Returns
    -------
    float
        Frobenius distance between the rank-K population projectors.
    """
    _, U_t = np.linalg.eigh(Sigma_t)
    _, U_prev = np.linalg.eigh(Sigma_prev)
    # eigh returns ascending order; take last K columns for top-K
    U_K_t = U_t[:, -K:]
    U_K_prev = U_prev[:, -K:]
    return projector_distance_lemma1(U_K_t, U_K_prev, K)


# ---------------------------------------------------------------------------
# Error and band-term helpers
# ---------------------------------------------------------------------------

def operator_norm_error(C: np.ndarray, Sigma: np.ndarray) -> float:
    """Compute the operator-norm (spectral-norm) error ||C - Sigma||_op.

    Uses np.linalg.eigvalsh and returns the maximum absolute eigenvalue of
    the difference C - Sigma.

    Parameters
    ----------
    C : np.ndarray of shape (N, N)
        Estimated covariance matrix.
    Sigma : np.ndarray of shape (N, N)
        True (or reference) covariance matrix.

    Returns
    -------
    float
        Maximum absolute eigenvalue of C - Sigma.
    """
    diff_eigs = np.linalg.eigvalsh(C - Sigma)
    return float(np.max(np.abs(diff_eigs)))


def compute_band_term(
    eta: float,
    C: np.ndarray,
    Sigma: np.ndarray,
    Delta_K: float,
    K: int,
) -> float:
    """Compute the theoretical band term T_{K,s} for a known Sigma.

    T_{K,s} = min(2*sqrt(2) * min(sqrt(K)*eta, ||C-Sigma||_F) / Delta_K,
                  sqrt(2*K))

    where eta = ||C - Sigma||_op.  If Delta_K <= 0, returns sqrt(2*K)
    (maximally wide band, indicating weak identification).

    Parameters
    ----------
    eta : float
        Operator-norm error ||C - Sigma||_op.
    C : np.ndarray of shape (N, N)
        Estimated covariance matrix.
    Sigma : np.ndarray of shape (N, N)
        True population covariance matrix.
    Delta_K : float
        Spectral gap lambda_K - lambda_{K+1} of Sigma (must be > 0 for a
        meaningful bound).
    K : int
        Rank of the projector.

    Returns
    -------
    float
        Band term T_{K,s}.
    """
    sqrt_2K = float(np.sqrt(2.0 * K))
    if Delta_K <= 0.0:
        return sqrt_2K
    fro = float(np.linalg.norm(C - Sigma, 'fro'))
    error = min(float(np.sqrt(K)) * eta, fro)
    return float(min(2.0 * np.sqrt(2.0) * error / Delta_K, sqrt_2K))


def compute_capped_band(T_t: float, T_prev: float, K: int) -> float:
    """Compute the capped projector-movement band tau*.

    tau* = min(sqrt(2*K), T_t + T_prev)

    Parameters
    ----------
    T_t : float
        Band term for the current window.
    T_prev : float
        Band term for the previous window.
    K : int
        Rank of the projector.

    Returns
    -------
    float
        Capped band tau*.
    """
    return float(min(np.sqrt(2.0 * K), T_t + T_prev))


# ---------------------------------------------------------------------------
# Fully observable band estimator
# ---------------------------------------------------------------------------

def _top_k_eigenvecs(M: np.ndarray, K: int) -> tuple:
    """Return (eigenvalues_desc, U_K) for the top-K eigenpairs of M."""
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx[:K]]


def _spectral_gap(vals_desc: np.ndarray, K: int) -> float:
    """Return lambda_K - lambda_{K+1} from descending eigenvalue array."""
    if K <= 0 or K >= len(vals_desc):
        return 0.0
    return float(vals_desc[K - 1] - vals_desc[K])


def compute_observable_band(
    C_t: np.ndarray,
    C_prev: np.ndarray,
    K: int,
) -> tuple:
    """Estimate the projector-movement band without knowledge of Sigma.

    Uses the difference C_t - C_prev as a proxy for the estimation errors,
    and eigenvalues of each estimated covariance matrix to approximate the
    spectral gap Delta_K.

    Steps
    -----
    1. eta_hat = ||C_t - C_prev||_op / 2  (proxy for ||C - Sigma||_op)
    2. fro_hat = ||C_t - C_prev||_F / 2   (proxy for ||C - Sigma||_F)
    3. Delta_K_t   = lambda_K(C_t)   - lambda_{K+1}(C_t)
    4. Delta_K_prev= lambda_K(C_prev) - lambda_{K+1}(C_prev)
    5. T_hat_t, T_hat_prev computed via the band-term formula for each window.
    6. tau_hat_star = min(sqrt(2*K), T_hat_t + T_hat_prev)
    7. weak_id_flag = True if min(Delta_K_t, Delta_K_prev) is below 1 % of
       the largest eigenvalue of C_t.

    Parameters
    ----------
    C_t : np.ndarray of shape (N, N)
        Estimated covariance matrix for the current window.
    C_prev : np.ndarray of shape (N, N)
        Estimated covariance matrix for the previous window.
    K : int
        Rank of the projector (number of factors).

    Returns
    -------
    T_hat_t : float
        Observable band term for the current window.
    tau_hat_star : float
        Capped observable band tau_hat*.
    weak_id_flag : bool
        True if the spectral gap is too small to provide a tight bound.
    """
    sqrt_2K = float(np.sqrt(2.0 * K))

    diff = C_t - C_prev
    # Proxy for per-window estimation error (triangle inequality: ||C-Sigma|| + ||Sigma-C'|| >= ||C-C'||)
    diff_op = float(np.max(np.abs(np.linalg.eigvalsh(diff))))
    diff_fro = float(np.linalg.norm(diff, 'fro'))
    eta_hat = diff_op / 2.0
    fro_hat = diff_fro / 2.0

    vals_t, _ = np.linalg.eigh(C_t)
    vals_t = vals_t[::-1]
    vals_prev, _ = np.linalg.eigh(C_prev)
    vals_prev = vals_prev[::-1]

    Delta_K_t = _spectral_gap(vals_t, K)
    Delta_K_prev = _spectral_gap(vals_prev, K)

    error_hat = min(float(np.sqrt(K)) * eta_hat, fro_hat)

    if Delta_K_t > 0.0:
        T_hat_t = min(2.0 * np.sqrt(2.0) * error_hat / Delta_K_t, sqrt_2K)
    else:
        T_hat_t = sqrt_2K

    if Delta_K_prev > 0.0:
        T_hat_prev = min(2.0 * np.sqrt(2.0) * error_hat / Delta_K_prev, sqrt_2K)
    else:
        T_hat_prev = sqrt_2K

    tau_hat_star = float(min(sqrt_2K, T_hat_t + T_hat_prev))
    T_hat_t = float(T_hat_t)

    # Weak identification: gap is less than 1 % of the leading eigenvalue
    leading_eig = float(vals_t[0]) if len(vals_t) > 0 else 1.0
    threshold = 0.01 * max(leading_eig, 1e-8)
    weak_id_flag = bool(min(Delta_K_t, Delta_K_prev) < threshold)

    return T_hat_t, tau_hat_star, weak_id_flag
