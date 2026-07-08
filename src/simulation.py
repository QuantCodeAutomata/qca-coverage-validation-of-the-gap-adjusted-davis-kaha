"""Data-generation utilities for the QCA project.

Provides factor-model returns, spiked covariance matrices, rotation helpers,
and Gaussian/elliptical return generators.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core factor-model generator
# ---------------------------------------------------------------------------

def generate_factor_model_returns(
    N: int,
    M_total: int,
    K_f: int,
    var_explained: float = 0.70,
    gap_type: str = 'strong',
    seed: int = 42,
) -> tuple:
    """Generate returns from a K_f-factor model with a controlled spectral gap.

    Model:  r_t = Lambda @ f_t + epsilon_t,
    where   f_t  ~ N(0, I_{K_f}),
            epsilon_t ~ N(0, sigma2_idio * I_N),
            Lambda_{ij} ~ N(0, 1/K_f), columns normalised then scaled.

    The idiosyncratic variance sigma2_idio is chosen so that:

        trace(Lambda @ Lambda.T) / (trace(Lambda @ Lambda.T) + N*sigma2_idio)
        = var_explained.

    Gap scaling (applied to the normalised columns):
      - 'strong': scale col 0 by sqrt(5), col 1 by 1  → lambda_1/lambda_2 >= 5
      - 'weak':   scale col 0 by sqrt(1.1), col 1 by 1 → lambda_1/lambda_2 ≤ 1.1

    Parameters
    ----------
    N : int
        Number of assets (dimension of the return vector).
    M_total : int
        Total number of return observations.
    K_f : int
        Number of latent factors.
    var_explained : float, optional
        Target fraction of variance explained by the factor component
        (default 0.70).
    gap_type : str, optional
        Spectral-gap regime: ``'strong'`` or ``'weak'`` (default ``'strong'``).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    returns : np.ndarray of shape (M_total, N)
        Simulated return matrix.
    Sigma : np.ndarray of shape (N, N)
        True population covariance Lambda @ Lambda.T + sigma2_idio * I.
    Lambda : np.ndarray of shape (N, K_f)
        Factor loading matrix.
    sigma2_idio : float
        Idiosyncratic variance.

    Raises
    ------
    ValueError
        If *gap_type* is not ``'strong'`` or ``'weak'``.
    """
    if gap_type not in ('strong', 'weak'):
        raise ValueError("gap_type must be 'strong' or 'weak'.")

    rng = np.random.default_rng(seed)

    # Draw loadings Lambda_{ij} ~ N(0, 1/K_f)
    Lambda = rng.standard_normal((N, K_f)) / np.sqrt(K_f)

    # Orthonormalise columns via QR so that eigenvalues of Lambda @ Lambda.T
    # are exactly determined by the column scales we apply below.
    Q, _ = np.linalg.qr(Lambda)        # Q: (N, K_f) with orthonormal columns
    Lambda = Q                          # each col has ||.||=1; cols are orthogonal

    # Solve for the leading scale s0 such that the ratio of the two largest
    # eigenvalues of Sigma = Lambda_scaled @ Lambda_scaled.T + sigma2*I equals
    # exactly the target ratio (5 for strong gap, 1.1 for weak gap).
    #
    # Let scales = [s0, 1, 1, ..., 1] (K_f entries).
    # A = sum of remaining squared scales = K_f - 1
    # tr_LLT = s0^2 + A,  sigma2 = (s0^2 + A) * beta,  beta = (1-ve)/(N*ve).
    # eig_1 = s0^2 + sigma2,  eig_2 = 1 + sigma2.
    # ratio r = eig_1/eig_2  =>  s0^2 = (r + (r-1)*A*beta) / (1 + (1-r)*beta)
    target_ratio = 5.0 if gap_type == 'strong' else 1.1
    A = float(K_f - 1)
    beta = (1.0 - var_explained) / (N * var_explained)
    denom_s = 1.0 + (1.0 - target_ratio) * beta
    if abs(denom_s) < 1e-12:
        # Degenerate case – fall back to simple scaling
        s0_sq = target_ratio
    else:
        s0_sq = (target_ratio + (target_ratio - 1.0) * A * beta) / denom_s
    s0_sq = max(s0_sq, 1e-6)

    scales = np.ones(K_f)
    scales[0] = np.sqrt(s0_sq)
    Lambda *= scales[np.newaxis, :]                     # shape (N, K_f)

    # Compute idiosyncratic variance from variance-explained target.
    # With orthonormal cols scaled by scales_k:
    #   trace(Lambda @ Lambda.T) = sum_k scales_k^2
    tr_LLT = float(np.sum(scales ** 2))
    # var_explained = tr_LLT / (tr_LLT + N * sigma2_idio)
    sigma2_idio = tr_LLT * (1.0 - var_explained) / (N * var_explained)

    # True population covariance
    Sigma = Lambda @ Lambda.T + sigma2_idio * np.eye(N)

    # Generate returns: r_t = Lambda f_t + epsilon_t
    F = rng.standard_normal((M_total, K_f))                  # (M_total, K_f)
    Eps = rng.standard_normal((M_total, N)) * np.sqrt(sigma2_idio)
    returns = F @ Lambda.T + Eps                              # (M_total, N)

    return returns, Sigma, Lambda, sigma2_idio


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def apply_rotation(
    Sigma: np.ndarray,
    theta: float,
    plane: tuple = (0, 2),
) -> np.ndarray:
    """Apply a Givens rotation to a covariance matrix.

    Constructs the N x N rotation matrix R_theta that rotates in the
    coordinate plane (*plane[0]*, *plane[1]*):

        R[i, i]           = 1   for i not in plane
        R[plane[0], plane[0]] = cos(theta)
        R[plane[0], plane[1]] = -sin(theta)
        R[plane[1], plane[0]] = sin(theta)
        R[plane[1], plane[1]] = cos(theta)

    and returns R @ Sigma @ R.T.

    Parameters
    ----------
    Sigma : np.ndarray of shape (N, N)
        Symmetric positive semi-definite covariance matrix.
    theta : float
        Rotation angle in radians.
    plane : tuple of two ints, optional
        Coordinate indices defining the rotation plane (default (0, 2)).

    Returns
    -------
    np.ndarray of shape (N, N)
        Rotated covariance matrix R @ Sigma @ R.T.
    """
    N = Sigma.shape[0]
    R = np.eye(N)
    i, j = plane
    R[i, i] = np.cos(theta)
    R[i, j] = -np.sin(theta)
    R[j, i] = np.sin(theta)
    R[j, j] = np.cos(theta)
    return R @ Sigma @ R.T


# ---------------------------------------------------------------------------
# Spiked covariance constructor
# ---------------------------------------------------------------------------

def generate_spiked_cov(
    N: int,
    spikes: list,
    sigma2: float = 0.5,
) -> np.ndarray:
    """Build a diagonal spiked covariance matrix.

    Returns a diagonal matrix whose first ``len(spikes)`` entries are the
    given spike eigenvalues and the remaining N - len(spikes) entries are
    *sigma2*.

    Parameters
    ----------
    N : int
        Dimension of the covariance matrix.
    spikes : list of float
        Spike eigenvalues (ordered arbitrarily; placed in the first K
        diagonal positions).
    sigma2 : float, optional
        Bulk (noise) variance for the remaining N - K entries (default 0.5).

    Returns
    -------
    np.ndarray of shape (N, N)
        Diagonal spiked covariance matrix.
    """
    K = len(spikes)
    diag_entries = np.full(N, sigma2)
    diag_entries[:K] = spikes
    return np.diag(diag_entries)


# ---------------------------------------------------------------------------
# Return generators
# ---------------------------------------------------------------------------

def generate_elliptical_returns(
    Sigma: np.ndarray,
    M: int,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate elliptical t_nu returns with covariance matrix Sigma.

    Each return r_t is drawn as:

        r_t = sqrt(xi_t) * L @ z_t,

    where xi_t ~ chi^2(nu)/nu, L = cholesky(Sigma), z_t ~ N(0, I_N).

    Parameters
    ----------
    Sigma : np.ndarray of shape (N, N)
        Population covariance matrix (must be positive definite).
    M : int
        Number of return observations to generate.
    nu : float
        Degrees of freedom for the mixing chi-squared variable.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray of shape (M, N)
        Simulated elliptical return matrix.
    """
    N = Sigma.shape[0]
    xi = rng.chisquare(nu, size=M) / nu          # (M,) mixing variable
    L = np.linalg.cholesky(Sigma)                 # (N, N)
    z = rng.standard_normal((M, N))               # (M, N)
    return np.sqrt(xi[:, np.newaxis]) * (L @ z.T).T


def generate_gaussian_returns(
    Sigma: np.ndarray,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Gaussian returns with covariance matrix Sigma.

    Each return r_t is drawn as:

        r_t = L @ z_t,   z_t ~ N(0, I_N),

    where L = cholesky(Sigma).

    Parameters
    ----------
    Sigma : np.ndarray of shape (N, N)
        Population covariance matrix (must be positive definite).
    M : int
        Number of return observations to generate.
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    np.ndarray of shape (M, N)
        Simulated Gaussian return matrix.
    """
    N = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((M, N))
    return (L @ z.T).T
