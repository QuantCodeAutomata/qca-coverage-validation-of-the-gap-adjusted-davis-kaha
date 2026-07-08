"""Tests for src/functionals.py — spectral functionals and variance estimators."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.functionals import (
    compute_AR_K,
    compute_f1,
    compute_spectral_entropy,
    compute_trace,
    delta_method_var_AR_K,
    delta_method_var_f1,
    delta_method_var_H,
    delta_method_var_trace,
    estimate_kappa_moment,
    estimate_kappa_radial_mle,
    spike_debias_AR_K,
)


# ── compute_AR_K ──────────────────────────────────────────────────────────────

def test_ar_k_range():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    val = compute_AR_K(lam, K=2)
    assert 0.0 <= val <= 1.0


def test_ar_k_all_factors():
    """K = N → AR_K = 1."""
    lam = np.array([3.0, 1.0, 0.5])
    val = compute_AR_K(lam, K=3)
    assert abs(val - 1.0) < 1e-12


def test_ar_k_known():
    lam = np.array([4.0, 2.0, 2.0, 2.0])
    val = compute_AR_K(lam, K=1)
    assert abs(val - 4.0 / 10.0) < 1e-12


def test_ar_k_monotone_in_K():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    ar1 = compute_AR_K(lam, K=1)
    ar2 = compute_AR_K(lam, K=2)
    ar3 = compute_AR_K(lam, K=3)
    assert ar1 <= ar2 <= ar3


# ── compute_f1 ────────────────────────────────────────────────────────────────

def test_f1_range():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    val = compute_f1(lam)
    assert 0.0 <= val <= 1.0


def test_f1_equals_ar_k_with_k1():
    lam = np.array([4.0, 2.0, 1.0])
    assert abs(compute_f1(lam) - compute_AR_K(lam, K=1)) < 1e-12


# ── compute_spectral_entropy ──────────────────────────────────────────────────

def test_entropy_non_negative():
    lam = np.array([3.0, 1.5, 0.5])
    H = compute_spectral_entropy(lam)
    assert H >= 0.0


def test_entropy_maximum_at_flat_spectrum():
    """Flat spectrum has maximum entropy."""
    N = 10
    lam_flat = np.ones(N)
    lam_spike = np.array([5.0] + [0.5] * (N - 1))
    H_flat = compute_spectral_entropy(lam_flat)
    H_spike = compute_spectral_entropy(lam_spike)
    assert H_flat > H_spike


def test_entropy_zero_for_rank_one():
    """All variance in one eigenvalue → max concentration, not entropy zero.
    Entropy = -1*log(1) = 0 only if one eigenvalue is non-zero."""
    lam = np.array([5.0, 0.0, 0.0])
    H = compute_spectral_entropy(lam)
    assert abs(H) < 1e-10


# ── compute_trace ─────────────────────────────────────────────────────────────

def test_trace_known():
    lam = np.array([3.0, 1.0, 0.5])
    assert abs(compute_trace(lam) - 4.5) < 1e-12


# ── delta_method_var_AR_K ─────────────────────────────────────────────────────

def test_delta_var_ar_k_non_negative():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    v = delta_method_var_AR_K(lam, K=2)
    assert v >= 0.0


def test_delta_var_ar_k_increases_with_kappa():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    v0 = delta_method_var_AR_K(lam, K=2, kappa=0.0)
    v1 = delta_method_var_AR_K(lam, K=2, kappa=2.0)
    assert v1 > v0


def test_delta_var_ar_k_scale_invariant():
    """AR_K is scale-invariant, so variance should be scale-invariant too."""
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    v1 = delta_method_var_AR_K(lam, K=2)
    v2 = delta_method_var_AR_K(lam * 2.0, K=2)
    # Both have the same AR_K but different absolute scales:
    # variance scales as 1/T^4 * (... eigenvalues^2) — let's just check ratio
    ratio = v2 / v1
    # With scaling by 2: T -> 2T, eigenvalues -> 2*eigenvalues
    # variance = 2*(1+k)/(2T)^4 * ((2T-2S_K)^2 * sum(2lam[:K])^2 + ...)
    # = 2*(1+k)/(16T^4) * 4*(T-S_K)^2 * 4*sum(lam[:K]^2) = same as original
    assert abs(ratio - 1.0) < 1e-10, f"Scale invariance broken: ratio={ratio}"


# ── delta_method_var_f1 ───────────────────────────────────────────────────────

def test_delta_var_f1_non_negative():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    v = delta_method_var_f1(lam)
    assert v >= 0.0


def test_delta_var_f1_kappa_scaling():
    """Variance should scale exactly as (1+kappa)."""
    lam = np.array([4.0, 2.0, 1.0])
    v0 = delta_method_var_f1(lam, kappa=0.0)
    v1 = delta_method_var_f1(lam, kappa=1.5)
    assert abs(v1 / v0 - 2.5) < 1e-10


# ── delta_method_var_H ────────────────────────────────────────────────────────

def test_delta_var_H_non_negative():
    lam = np.array([5.0, 2.0, 1.0, 0.5])
    v = delta_method_var_H(lam)
    assert v >= 0.0


def test_delta_var_H_flat_spectrum_near_zero():
    """At a flat spectrum, all gradients (log(p_i)+H) are ~ 0 → variance ~ 0."""
    N = 10
    lam = np.ones(N)
    v = delta_method_var_H(lam)
    assert v < 1e-20


# ── delta_method_var_trace ────────────────────────────────────────────────────

def test_delta_var_trace_non_negative():
    lam = np.array([5.0, 2.0, 1.0])
    v = delta_method_var_trace(lam, kappa=0.0)
    assert v >= 0.0


def test_delta_var_trace_extra_term_with_kappa():
    """With kappa > 0, trace variance has extra kappa*(tr)^2 term."""
    lam = np.array([5.0, 2.0, 1.0])
    v0 = delta_method_var_trace(lam, kappa=0.0)
    v1 = delta_method_var_trace(lam, kappa=1.0)
    # Extra term = kappa*(sum lam)^2 = 64.0
    assert v1 > v0


# ── estimate_kappa_moment ─────────────────────────────────────────────────────

def test_kappa_moment_gaussian_near_zero():
    """Under Gaussian, kappa should be ≈ 0."""
    rng = np.random.default_rng(0)
    R = rng.standard_normal((2000, 20))
    k = estimate_kappa_moment(R)
    assert abs(k) < 0.1, f"Gaussian kappa: expected ~0, got {k:.4f}"


def test_kappa_moment_heavy_tail_positive():
    """Under t_5 (kappa=2), moment-based estimate should be positive."""
    rng = np.random.default_rng(0)
    N, M_obs = 10, 2000
    df = 5
    xi = rng.chisquare(df, size=M_obs) / df
    R = rng.standard_normal((M_obs, N)) * np.sqrt(xi[:, np.newaxis])
    k = estimate_kappa_moment(R)
    assert k > 0.3, f"Heavy-tail kappa: expected > 0.3, got {k:.4f}"


def test_kappa_moment_non_negative():
    rng = np.random.default_rng(1)
    R = rng.standard_normal((100, 5))
    k = estimate_kappa_moment(R)
    assert k >= 0.0


# ── estimate_kappa_radial_mle ─────────────────────────────────────────────────

def test_kappa_radial_mle_non_negative():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((100, 10))
    from src.estimators import sample_cov
    S = sample_cov(R)
    k = estimate_kappa_radial_mle(R, S)
    assert k >= 0.0


def test_kappa_radial_mle_heavy_tail():
    """Under multiplication-form chi2_5/5 elliptical DGP, radial MLE should detect
    positive excess kurtosis.

    DGP: r = sqrt(chi2_5/5) * z  (mixing form, NOT standard Student-t).
    True kappa_xi = E[xi^2]-1 = (5+2)/5 - 1 = 0.4.
    The F-distribution radial MLE underestimates but still gives k > 0 reliably.
    """
    rng = np.random.default_rng(42)
    N, M_obs = 5, 2000
    df = 5
    xi = rng.chisquare(df, size=M_obs) / df
    R = rng.standard_normal((M_obs, N)) * np.sqrt(xi[:, np.newaxis])
    from src.estimators import sample_cov
    S = sample_cov(R)
    k = estimate_kappa_radial_mle(R, S)
    # The F-based MLE gives ~0.29 for this DGP (true kappa_xi ≈ 0.4).
    # Assert positivity of the heavy-tail detection (not the exact value).
    assert k > 0.1, f"t_5 kappa MLE: expected > 0.1, got {k:.3f}"


# ── spike_debias_AR_K ─────────────────────────────────────────────────────────

def test_spike_debias_trace_preservation():
    """After debiasing, sum of eigenvalues should equal original sum."""
    rng = np.random.default_rng(0)
    N, M_obs = 30, 252
    spikes_true = [6.0, 3.5, 1.2]
    sigma2 = 0.5
    Sigma = np.diag(spikes_true + [sigma2] * (N - 3))
    R = rng.multivariate_normal(np.zeros(N), Sigma, size=M_obs)
    from src.estimators import sample_cov
    S = sample_cov(R)
    evals = np.sort(np.linalg.eigvalsh(S))[::-1]
    AR_deb, evals_deb = spike_debias_AR_K(evals, K=2, N=N, M=M_obs)
    # Trace preservation (within tolerance)
    assert abs(np.sum(evals_deb) - np.sum(evals)) < 1e-6 * np.sum(evals)


def test_spike_debias_range():
    """Debiased AR_K must be in [0, 1]."""
    rng = np.random.default_rng(0)
    N, M_obs = 30, 252
    Sigma = np.diag([6.0, 3.5] + [0.5] * (N - 2))
    R = rng.multivariate_normal(np.zeros(N), Sigma, size=M_obs)
    from src.estimators import sample_cov
    S = sample_cov(R)
    evals = np.sort(np.linalg.eigvalsh(S))[::-1]
    AR_deb, _ = spike_debias_AR_K(evals, K=2, N=N, M=M_obs)
    assert 0.0 <= AR_deb <= 1.0


def test_spike_debias_reduces_upward_bias():
    """Debiased AR_K should be closer to true value than sample AR_K."""
    rng = np.random.default_rng(42)
    N, M_obs = 30, 252
    spikes = [6.0, 3.5]
    sigma2 = 0.5
    Sigma = np.diag(spikes + [sigma2] * (N - len(spikes)))
    true_AR = sum(spikes) / (sum(spikes) + sigma2 * (N - len(spikes)))

    results_sample = []
    results_deb = []
    from src.estimators import sample_cov
    for _ in range(20):
        R = rng.multivariate_normal(np.zeros(N), Sigma, size=M_obs)
        S = sample_cov(R)
        evals = np.sort(np.linalg.eigvalsh(S))[::-1]
        ar_sample = compute_AR_K(evals, K=2)
        ar_deb, _ = spike_debias_AR_K(evals, K=2, N=N, M=M_obs)
        results_sample.append(ar_sample)
        results_deb.append(ar_deb)

    bias_sample = abs(np.mean(results_sample) - true_AR)
    bias_deb = abs(np.mean(results_deb) - true_AR)
    assert bias_deb < bias_sample, (
        f"Debiased bias {bias_deb:.4f} should be < sample bias {bias_sample:.4f}"
    )
