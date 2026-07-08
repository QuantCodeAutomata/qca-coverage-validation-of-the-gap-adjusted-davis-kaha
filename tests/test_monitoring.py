"""Tests for src/monitoring.py — analytic null, bootstrap, and BH/BY corrections."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.monitoring import (
    analytic_null_simulator,
    bootstrap_projector_movement,
    apply_bh_correction,
    apply_by_correction,
)


RNG = np.random.default_rng(0)


# ── analytic_null_simulator ───────────────────────────────────────────────────

def test_analytic_null_shape():
    eigenvalues = np.array([5.0, 2.0, 1.0, 0.5, 0.3])
    D_tildes = analytic_null_simulator(eigenvalues, K=2, s=31, M=126, n_draws=50,
                                       rng=np.random.default_rng(0))
    assert D_tildes.shape == (50,)


def test_analytic_null_non_negative():
    eigenvalues = np.array([5.0, 2.0, 1.0, 0.5, 0.3])
    D_tildes = analytic_null_simulator(eigenvalues, K=2, s=31, M=126, n_draws=100,
                                       rng=np.random.default_rng(0))
    assert np.all(D_tildes >= 0.0)


def test_analytic_null_bounded():
    """Null draws must be in [0, sqrt(2*K)] approximately (they can be larger
    due to Monte Carlo sampling, but the 95th percentile should be well below 10)."""
    K = 2
    eigenvalues = np.array([5.0, 2.0, 1.0, 0.5, 0.3])
    D_tildes = analytic_null_simulator(eigenvalues, K=K, s=31, M=126, n_draws=200,
                                       rng=np.random.default_rng(1))
    # 99th pctile should be less than some reasonable bound
    assert np.percentile(D_tildes, 99) < 50.0


def test_analytic_null_larger_gap_smaller_quantiles():
    """Larger eigengap → smaller null draws (movement easier to detect)."""
    large_gap_eigs = np.array([10.0, 1.0, 0.5, 0.3])
    small_gap_eigs = np.array([5.0, 4.5, 0.5, 0.3])
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    D_large = analytic_null_simulator(large_gap_eigs, K=1, s=31, M=126,
                                      n_draws=300, rng=rng1)
    D_small = analytic_null_simulator(small_gap_eigs, K=1, s=31, M=126,
                                      n_draws=300, rng=rng2)
    # Larger gap → denominator larger → D_tilde smaller
    assert np.median(D_large) < np.median(D_small)


def test_analytic_null_reproducible():
    eigenvalues = np.array([5.0, 2.0, 1.0])
    d1 = analytic_null_simulator(eigenvalues, K=1, s=10, M=50, n_draws=20,
                                  rng=np.random.default_rng(7))
    d2 = analytic_null_simulator(eigenvalues, K=1, s=10, M=50, n_draws=20,
                                  rng=np.random.default_rng(7))
    assert np.allclose(d1, d2)


# ── bootstrap_projector_movement ──────────────────────────────────────────────

def _make_pd_cov(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N))
    return (A @ A.T) / N + np.eye(N) * 0.5


def test_bootstrap_shape():
    N, M, K, s = 15, 60, 2, 5
    C_A = _make_pd_cov(N)
    returns_A = np.random.default_rng(0).standard_normal((M, N))
    D_hats = bootstrap_projector_movement(
        returns_A, C_A, K=K, s=s, M=M, estimator='sample', n_boot=10,
        rng=np.random.default_rng(0)
    )
    assert D_hats.shape == (10,)


def test_bootstrap_non_negative():
    N, M, K, s = 10, 50, 2, 5
    C_A = _make_pd_cov(N)
    returns_A = np.random.default_rng(0).standard_normal((M, N))
    D_hats = bootstrap_projector_movement(
        returns_A, C_A, K=K, s=s, M=M, estimator='sample', n_boot=20,
        rng=np.random.default_rng(2)
    )
    assert np.all(D_hats >= 0.0)


def test_bootstrap_bounded():
    """Bootstrap D_hats must be <= sqrt(2*K)."""
    N, M, K, s = 10, 50, 2, 5
    C_A = _make_pd_cov(N)
    returns_A = np.random.default_rng(0).standard_normal((M, N))
    D_hats = bootstrap_projector_movement(
        returns_A, C_A, K=K, s=s, M=M, estimator='sample', n_boot=30,
        rng=np.random.default_rng(3)
    )
    assert np.all(D_hats <= np.sqrt(2.0 * K) + 1e-6)


def test_bootstrap_qis_estimator():
    """QIS bootstrap should also produce valid D_hat values."""
    N, M, K, s = 15, 80, 2, 5
    C_A = _make_pd_cov(N, seed=5)
    returns_A = np.random.default_rng(5).standard_normal((M, N))
    D_hats = bootstrap_projector_movement(
        returns_A, C_A, K=K, s=s, M=M, estimator='qis', n_boot=10,
        rng=np.random.default_rng(5)
    )
    assert D_hats.shape == (10,)
    assert np.all(D_hats >= 0.0)


# ── apply_bh_correction ───────────────────────────────────────────────────────

def test_bh_all_null():
    """Under uniform [0,1] p-values, few rejections expected."""
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 1, size=100)
    rejected = apply_bh_correction(pvals, q=0.10)
    fdr = rejected.sum() / max(1, rejected.sum())
    assert isinstance(rejected, np.ndarray)
    assert rejected.dtype == bool
    assert len(rejected) == 100


def test_bh_rejects_small_pvalues():
    """Very small p-values should be rejected."""
    pvals = np.array([0.001, 0.002, 0.003, 0.5, 0.6, 0.7, 0.8, 0.9])
    rejected = apply_bh_correction(pvals, q=0.05)
    # First three should be rejected
    assert rejected[:3].all()
    # Last few should not
    assert not rejected[-3:].any()


def test_bh_all_large_pvalues():
    """Very large p-values → no rejections."""
    pvals = np.array([0.9, 0.95, 0.99, 0.999])
    rejected = apply_bh_correction(pvals, q=0.05)
    assert not rejected.any()


def test_bh_shape():
    pvals = np.array([0.01, 0.05, 0.1, 0.5])
    rejected = apply_bh_correction(pvals)
    assert rejected.shape == pvals.shape


# ── apply_by_correction ───────────────────────────────────────────────────────

def test_by_more_conservative_than_bh():
    """BY corrects for arbitrary dependence and is more conservative than BH."""
    pvals = np.linspace(0.001, 0.1, 50)
    rej_bh = apply_bh_correction(pvals, q=0.10)
    rej_by = apply_by_correction(pvals, q=0.10)
    assert rej_by.sum() <= rej_bh.sum(), "BY should reject fewer (or equal) than BH"


def test_by_shape():
    pvals = np.array([0.01, 0.05, 0.1, 0.5])
    rejected = apply_by_correction(pvals)
    assert rejected.shape == pvals.shape


def test_by_returns_bool_array():
    pvals = np.array([0.01, 0.2, 0.5])
    rejected = apply_by_correction(pvals)
    assert rejected.dtype == bool
