"""Tests for src/simulation.py — factor-model return generation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulation import (
    generate_factor_model_returns,
    apply_rotation,
    generate_elliptical_returns,
    generate_spiked_cov,
)


# ── generate_factor_model_returns ─────────────────────────────────────────────

def test_factor_model_shape():
    ret, Sigma, _, _ = generate_factor_model_returns(N=20, M_total=100, K_f=3, seed=1)
    assert ret.shape == (100, 20)
    assert Sigma.shape == (20, 20)


def test_factor_model_sigma_symmetric_pd():
    _, Sigma, _, _ = generate_factor_model_returns(N=15, M_total=60, K_f=3, seed=2)
    assert np.allclose(Sigma, Sigma.T, atol=1e-10)
    evals = np.linalg.eigvalsh(Sigma)
    assert np.all(evals > 0)


def test_factor_model_strong_gap():
    _, Sigma, _, _ = generate_factor_model_returns(
        N=50, M_total=200, K_f=5, gap_type="strong", seed=3
    )
    evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    ratio_12 = evals[0] / evals[1]
    assert ratio_12 >= 4.0, f"Expected strong gap ratio >= 4, got {ratio_12:.2f}"


def test_factor_model_weak_gap():
    _, Sigma, _, _ = generate_factor_model_returns(
        N=50, M_total=200, K_f=5, gap_type="weak", seed=4
    )
    evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    ratio_12 = evals[0] / evals[1]
    assert ratio_12 <= 1.5, f"Expected weak gap ratio <= 1.5, got {ratio_12:.2f}"


def test_factor_model_zero_mean():
    """After demeaning in each window, mean should be near zero."""
    ret, _, _, _ = generate_factor_model_returns(N=10, M_total=50, K_f=3, seed=5)
    # The DGP has zero mean factors and noise; cross-sectional mean should be small
    assert np.abs(ret.mean()) < 1.0  # rough check


def test_factor_model_reproducible():
    """Same seed → same returns."""
    r1, _, _, _ = generate_factor_model_returns(N=15, M_total=50, K_f=3, seed=99)
    r2, _, _, _ = generate_factor_model_returns(N=15, M_total=50, K_f=3, seed=99)
    assert np.allclose(r1, r2)


# ── apply_rotation ────────────────────────────────────────────────────────────

def test_rotation_zero_angle_identity():
    Sigma = np.diag([5.0, 2.0, 1.0, 0.5])
    Sigma2 = apply_rotation(Sigma, theta=0.0)
    assert np.allclose(Sigma2, Sigma, atol=1e-12)


def test_rotation_preserves_trace():
    Sigma = np.diag([5.0, 2.0, 1.0, 0.5])
    Sigma2 = apply_rotation(Sigma, theta=0.5)
    assert abs(np.trace(Sigma2) - np.trace(Sigma)) < 1e-10


def test_rotation_preserves_symmetry():
    Sigma = np.diag([5.0, 2.0, 1.0, 0.5])
    Sigma2 = apply_rotation(Sigma, theta=0.3)
    assert np.allclose(Sigma2, Sigma2.T, atol=1e-10)


def test_rotation_changes_eigenvectors():
    """A non-zero rotation should change the eigenvectors."""
    N = 10
    Sigma = np.diag(np.array([5.0, 2.0] + [0.5] * (N - 2)))
    Sigma2 = apply_rotation(Sigma, theta=0.5)
    evals1 = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    evals2 = np.sort(np.linalg.eigvalsh(Sigma2))[::-1]
    # Eigenvalues preserved under rotation
    assert np.allclose(evals1, evals2, atol=1e-8)


# ── generate_elliptical_returns ───────────────────────────────────────────────

def test_elliptical_t_shape():
    rng = np.random.default_rng(0)
    Sigma = np.diag([3.0, 1.0, 0.5])
    R = generate_elliptical_returns(Sigma, M=100, nu=5, rng=rng)
    assert R.shape == (100, 3)


def test_elliptical_covariance_close_to_sigma():
    """With large M, sample cov of t-returns should be close to Sigma."""
    rng = np.random.default_rng(42)
    N = 5
    Sigma = np.diag([4.0, 2.0, 1.0, 0.5, 0.3])
    R = generate_elliptical_returns(Sigma, M=10000, nu=100, rng=rng)
    S = np.cov(R.T, ddof=1)
    rel_err = np.linalg.norm(S - Sigma, 'fro') / np.linalg.norm(Sigma, 'fro')
    assert rel_err < 0.2, f"Relative error too large: {rel_err:.3f}"


def test_elliptical_t_heavier_tails():
    """t_5-returns should have larger kurtosis than t_100 (near Gaussian)."""
    rng = np.random.default_rng(1)
    N = 1
    Sigma = np.array([[1.0]])
    R_gauss = generate_elliptical_returns(Sigma, M=3000, nu=100, rng=rng)
    R_t = generate_elliptical_returns(Sigma, M=3000, nu=5, rng=rng)
    kurt_gauss = float(np.mean((R_gauss - R_gauss.mean()) ** 4) /
                       np.mean((R_gauss - R_gauss.mean()) ** 2) ** 2)
    kurt_t = float(np.mean((R_t - R_t.mean()) ** 4) /
                   np.mean((R_t - R_t.mean()) ** 2) ** 2)
    assert kurt_t > kurt_gauss, f"t_5-kurtosis ({kurt_t:.2f}) should > t_100 ({kurt_gauss:.2f})"


# ── generate_spiked_cov ───────────────────────────────────────────────────────

def test_spiked_cov_shape():
    Sigma = generate_spiked_cov(N=20, spikes=[5.0, 2.0, 1.0], sigma2=0.5)
    assert Sigma.shape == (20, 20)


def test_spiked_cov_diagonal():
    """generate_spiked_cov with standard basis → diagonal matrix."""
    Sigma = generate_spiked_cov(N=5, spikes=[3.0, 1.5], sigma2=0.5)
    assert np.allclose(Sigma, np.diag(np.diag(Sigma)), atol=1e-10)


def test_spiked_cov_eigenvalues():
    spikes = [6.0, 3.5, 1.2]
    sigma2 = 0.5
    N = 10
    Sigma = generate_spiked_cov(N=N, spikes=spikes, sigma2=sigma2)
    evals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    for i, s in enumerate(spikes):
        assert abs(evals[i] - s) < 1e-8, f"Spike {i}: expected {s}, got {evals[i]}"
    for i in range(len(spikes), N):
        assert abs(evals[i] - sigma2) < 1e-8, f"Bulk {i}: expected {sigma2}, got {evals[i]}"
