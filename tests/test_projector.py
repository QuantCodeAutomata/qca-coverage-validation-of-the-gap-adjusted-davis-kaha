"""Tests for src/projector.py — projector distance and band computations."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.projector import (
    projector_distance_lemma1,
    compute_population_projector_distance,
    operator_norm_error,
    compute_band_term,
    compute_capped_band,
    compute_observable_band,
)

RNG = np.random.default_rng(0)


def _ortho_basis(N: int, K: int, seed: int = 0) -> np.ndarray:
    """Return a random (N, K) orthonormal matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, K))
    Q, _ = np.linalg.qr(A)
    return Q[:, :K]


# ── projector_distance_lemma1 ─────────────────────────────────────────────────

def test_projector_distance_identical_subspace():
    """Identical subspaces → distance = 0."""
    U = _ortho_basis(20, 3)
    d = projector_distance_lemma1(U, U, 3)
    assert d < 1e-12, f"Expected 0, got {d}"


def test_projector_distance_orthogonal_subspaces():
    """Orthogonal K=1 subspaces → max distance sqrt(2)."""
    N = 4
    U1 = np.array([[1.0], [0.0], [0.0], [0.0]])
    U2 = np.array([[0.0], [1.0], [0.0], [0.0]])
    d = projector_distance_lemma1(U1, U2, 1)
    assert abs(d - np.sqrt(2.0)) < 1e-12


def test_projector_distance_non_negative():
    """Distance must be non-negative."""
    U1 = _ortho_basis(30, 2)
    U2 = _ortho_basis(30, 2, seed=7)
    d = projector_distance_lemma1(U1, U2, 2)
    assert d >= 0.0


def test_projector_distance_bounded_by_sqrt_2K():
    """Distance must be <= sqrt(2*K)."""
    K = 3
    U1 = _ortho_basis(20, K)
    U2 = _ortho_basis(20, K, seed=5)
    d = projector_distance_lemma1(U1, U2, K)
    assert d <= np.sqrt(2.0 * K) + 1e-10


def test_projector_distance_symmetric():
    """d(U1, U2) == d(U2, U1)."""
    U1 = _ortho_basis(20, 2)
    U2 = _ortho_basis(20, 2, seed=3)
    d12 = projector_distance_lemma1(U1, U2, 2)
    d21 = projector_distance_lemma1(U2, U1, 2)
    assert abs(d12 - d21) < 1e-12


# ── compute_population_projector_distance ─────────────────────────────────────

def test_population_projector_distance_identical():
    Sigma = np.diag([5.0, 2.0, 1.0, 0.5])
    d = compute_population_projector_distance(Sigma, Sigma, K=2)
    assert d < 1e-10


def test_population_projector_distance_rotation():
    """Small rotation → small but positive distance."""
    N = 10
    Sigma = np.diag(np.array([5.0, 2.0] + [0.5] * (N - 2)))
    # Rotate Sigma by small angle in (e1, e3) plane
    theta = 0.1
    R = np.eye(N)
    R[0, 0] = np.cos(theta)
    R[0, 2] = -np.sin(theta)
    R[2, 0] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    Sigma2 = R @ Sigma @ R.T
    d = compute_population_projector_distance(Sigma, Sigma2, K=1)
    expected = np.sqrt(2.0) * abs(np.sin(theta))
    assert abs(d - expected) < 1e-6


# ── operator_norm_error ───────────────────────────────────────────────────────

def test_operator_norm_error_zero():
    C = np.eye(5) * 2.0
    assert operator_norm_error(C, C) < 1e-12


def test_operator_norm_error_known():
    C = np.diag([3.0, 1.0])
    Sigma = np.diag([2.0, 1.0])
    err = operator_norm_error(C, Sigma)
    assert abs(err - 1.0) < 1e-12


def test_operator_norm_error_positive():
    C = np.diag([2.0, 1.5, 1.0])
    Sigma = np.diag([2.5, 1.0, 0.8])
    err = operator_norm_error(C, Sigma)
    assert err >= 0.0


# ── compute_band_term ─────────────────────────────────────────────────────────

def test_band_term_zero_error():
    """If estimation error is 0, band term should be 0."""
    Sigma = np.diag([3.0, 1.0, 0.5])
    t = compute_band_term(0.0, Sigma, Sigma, Delta_K=2.0, K=1)
    assert t == 0.0


def test_band_term_capped_at_sqrt_2K():
    """Very large error → band term = sqrt(2K)."""
    K = 2
    N = 10
    C = np.eye(N) * 100.0
    Sigma = np.eye(N)
    t = compute_band_term(99.0, C, Sigma, Delta_K=0.1, K=K)
    assert abs(t - np.sqrt(2.0 * K)) < 1e-10


def test_band_term_zero_delta():
    """Delta_K <= 0 → band term = sqrt(2K) (weak identification)."""
    K = 2
    N = 5
    C = np.eye(N)
    Sigma = np.eye(N)
    t = compute_band_term(0.1, C, Sigma, Delta_K=0.0, K=K)
    assert abs(t - np.sqrt(2.0 * K)) < 1e-10


def test_band_term_non_negative():
    C = np.diag([3.0, 1.0, 0.5])
    Sigma = np.diag([2.5, 1.2, 0.6])
    eta = operator_norm_error(C, Sigma)
    t = compute_band_term(eta, C, Sigma, Delta_K=1.3, K=1)
    assert t >= 0.0


# ── compute_capped_band ───────────────────────────────────────────────────────

def test_capped_band_never_exceeds_sqrt_2K():
    K = 2
    for T1, T2 in [(1.0, 1.0), (2.0, 2.0), (0.0, 0.0), (5.0, 5.0)]:
        tau = compute_capped_band(T1, T2, K)
        assert tau <= np.sqrt(2.0 * K) + 1e-12


def test_capped_band_equals_sum_when_small():
    """When T1+T2 < sqrt(2K), tau* = T1+T2."""
    K = 2
    T1, T2 = 0.3, 0.4
    tau = compute_capped_band(T1, T2, K)
    assert abs(tau - (T1 + T2)) < 1e-12


# ── compute_observable_band ───────────────────────────────────────────────────

def test_observable_band_shape():
    N, K = 20, 2
    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, N))
    C = A @ A.T / N + np.eye(N) * 0.1
    T_hat, tau_hat, weak = compute_observable_band(C, C, K)
    # Same matrix → T_hat should be 0 (or very small)
    assert T_hat >= 0.0
    assert 0.0 <= tau_hat <= np.sqrt(2.0 * K) + 1e-10
    assert isinstance(weak, bool)


def test_observable_band_capped():
    """tau_hat* must be <= sqrt(2*K)."""
    N, K = 15, 3
    rng = np.random.default_rng(2)
    A1 = rng.standard_normal((N, N))
    A2 = rng.standard_normal((N, N))
    C1 = A1 @ A1.T / N
    C2 = A2 @ A2.T / N
    _, tau_hat, _ = compute_observable_band(C1, C2, K)
    assert tau_hat <= np.sqrt(2.0 * K) + 1e-10
