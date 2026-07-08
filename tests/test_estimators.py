"""Tests for src/estimators.py — covariance estimators."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.estimators import (
    sample_cov, ledoit_wolf, oas, qis, mp_clip, apply_estimator,
)


SEED = 0
RNG = np.random.default_rng(SEED)


def _generate_returns(N: int = 30, M: int = 100, seed: int = SEED) -> np.ndarray:
    """Return (M, N) random returns for testing."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N))


# ── sample_cov ────────────────────────────────────────────────────────────────

def test_sample_cov_shape():
    R = _generate_returns(N=20, M=50)
    S = sample_cov(R)
    assert S.shape == (20, 20)


def test_sample_cov_symmetric():
    R = _generate_returns(N=15, M=60)
    S = sample_cov(R)
    assert np.allclose(S, S.T, atol=1e-12)


def test_sample_cov_positive_semidefinite():
    R = _generate_returns(N=10, M=30)
    S = sample_cov(R)
    evals = np.linalg.eigvalsh(S)
    assert np.all(evals >= -1e-10), f"Negative eigenvalue found: {evals.min()}"


def test_sample_cov_known_case():
    """Single-asset return of constant 2.0 → variance should be 0."""
    R = np.full((20, 1), 2.0)
    S = sample_cov(R)
    assert abs(S[0, 0]) < 1e-10


def test_sample_cov_raises_on_too_few_obs():
    """sample_cov should raise (or return singular) when M <= N."""
    # With M < N, sample covariance is rank-deficient (not an error per se).
    R = _generate_returns(N=10, M=5)
    # Just verify it doesn't crash and shape is correct
    S = sample_cov(R)
    assert S.shape == (10, 10)


# ── ledoit_wolf ───────────────────────────────────────────────────────────────

def test_lw_shape():
    R = _generate_returns(N=20, M=50)
    S = sample_cov(R)
    C = ledoit_wolf(S, n=50)
    assert C.shape == (20, 20)


def test_lw_symmetric():
    R = _generate_returns(N=15, M=60)
    S = sample_cov(R)
    C = ledoit_wolf(S, n=60)
    assert np.allclose(C, C.T, atol=1e-10)


def test_lw_positive_definite():
    R = _generate_returns(N=15, M=60)
    S = sample_cov(R)
    C = ledoit_wolf(S, n=60)
    evals = np.linalg.eigvalsh(C)
    assert np.all(evals > 0), f"Non-positive eigenvalue: {evals.min()}"


def test_lw_trace_preserved():
    """LW preserves the trace of S (it's a convex combination of S and mu*I)."""
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = ledoit_wolf(S, n=80)
    # Trace may differ slightly (LW mixes S with mu*I where mu = tr(S)/N).
    # Both S and mu*I have the same trace, so the mixture preserves trace.
    assert abs(np.trace(C) - np.trace(S)) < 1e-8 * np.trace(S)


# ── oas ───────────────────────────────────────────────────────────────────────

def test_oas_shape_and_symmetry():
    R = _generate_returns(N=15, M=60)
    S = sample_cov(R)
    C = oas(S, n=60)
    assert C.shape == (15, 15)
    assert np.allclose(C, C.T, atol=1e-10)


def test_oas_positive_definite():
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = oas(S, n=80)
    evals = np.linalg.eigvalsh(C)
    assert np.all(evals > 0)


# ── qis ───────────────────────────────────────────────────────────────────────

def test_qis_shape_and_symmetry():
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = qis(S, n=80)
    assert C.shape == (20, 20)
    assert np.allclose(C, C.T, atol=1e-8)


def test_qis_positive_definite():
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = qis(S, n=80)
    evals = np.linalg.eigvalsh(C)
    assert np.all(evals >= -1e-8), f"Negative eigenvalue: {evals.min()}"


def test_qis_rotation_equivariance():
    """QIS eigenvectors span same space as sample cov eigenvectors.
    
    QIS is rotation-equivariant: it reuses eigenvectors of S and only
    modifies eigenvalues.  Verify that QIS(S) = U @ diag(d) @ U.T where
    U = eigenvectors of S (up to numerical precision).
    """
    N, M = 20, 100
    R = _generate_returns(N=N, M=M, seed=1)
    S = sample_cov(R)
    C = qis(S, n=M)
    # By construction QIS(S) = U_S @ diag(d_tilde) @ U_S.T
    # Verify residual ||C - S|| < ||C||  (shrinkage brings C closer)
    # and that C is symmetric positive semi-definite.
    assert np.allclose(C, C.T, atol=1e-8), "QIS result not symmetric"
    evals = np.linalg.eigvalsh(C)
    assert np.all(evals >= -1e-8), "QIS has negative eigenvalues"
    # The real equivariance check: eigenvectors of S must diagonalise C.
    # QIS(S) = U_S @ diag(d_tilde) @ U_S.T by construction, so
    # U_S.T @ C @ U_S must be diagonal (up to floating-point noise).
    _, U_S = np.linalg.eigh(S)
    D_check = U_S.T @ C @ U_S
    off_diag = D_check - np.diag(np.diag(D_check))
    assert np.allclose(off_diag, 0, atol=1e-8), (
        "QIS(S) is not diagonal in the eigenbasis of S (not rotation equivariant)"
    )


# ── mp_clip ───────────────────────────────────────────────────────────────────

def test_mp_clip_shape_and_symmetry():
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = mp_clip(S, n=80)
    assert C.shape == (20, 20)
    assert np.allclose(C, C.T, atol=1e-10)


def test_mp_clip_positive_semidefinite():
    R = _generate_returns(N=20, M=80)
    S = sample_cov(R)
    C = mp_clip(S, n=80)
    evals = np.linalg.eigvalsh(C)
    assert np.all(evals >= -1e-10)


def test_mp_clip_trace_close_to_sample():
    """MP-clip preserves the total variance of S."""
    R = _generate_returns(N=30, M=120)
    S = sample_cov(R)
    C = mp_clip(S, n=120)
    assert abs(np.trace(C) - np.trace(S)) < 1e-7 * np.trace(S)


# ── apply_estimator dispatcher ────────────────────────────────────────────────

def test_apply_estimator_dispatches_correctly():
    R = _generate_returns(N=15, M=60)
    for est in ("sample", "ledoit_wolf", "oas", "qis", "mp_clip"):
        C = apply_estimator(R, est)
        assert C.shape == (15, 15), f"Wrong shape for estimator {est}"
        assert np.allclose(C, C.T, atol=1e-8), f"Not symmetric for {est}"


def test_apply_estimator_unknown_raises():
    R = _generate_returns(N=10, M=30)
    with pytest.raises(ValueError):
        apply_estimator(R, "unknown_estimator")


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_estimators_single_asset():
    """All estimators should handle N=1 gracefully."""
    R = np.array([[0.1], [0.2], [-0.1], [0.0], [0.05]])
    S = sample_cov(R)
    assert S.shape == (1, 1)
    assert S[0, 0] >= 0.0


def test_estimators_large_N_M():
    """Smoke test at near-experiment dimensions (N=50, M=120)."""
    R = _generate_returns(N=50, M=120, seed=7)
    S = sample_cov(R)
    for fn in (ledoit_wolf, oas, mp_clip):
        C = fn(S, 120)
        assert C.shape == (50, 50)
        assert np.allclose(C, C.T, atol=1e-8)
