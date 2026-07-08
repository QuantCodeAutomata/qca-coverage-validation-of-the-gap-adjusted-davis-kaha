# QCA Project — Agent Memory

## Project Structure

```
src/
  __init__.py      # empty package marker
  estimators.py    # covariance estimators
  projector.py     # projector movement / band terms
  simulation.py    # data generation
  functionals.py   # spectral functionals & variance estimators
  monitoring.py    # monitoring procedures (null sim, bootstrap, FDR)
data/              # raw data
exp/               # experiment scripts
results/           # output artefacts
tests/             # test suite
```

## Module APIs

### estimators.py
- `sample_cov(returns)` → (M-1)^{-1} R̃ᵀR̃ after demeaning
- `ledoit_wolf(S, n)` → LW analytical shrinkage (closed-form OAS coefficient)
- `oas(S, n)` → Chen et al. 2010 OAS shrinkage
- `qis(S, n)` → Oracle QuEST nonlinear shrinkage (Stieltjes + Gaussian KDE)
- `mp_clip(S, n)` → Marchenko-Pastur clipping (iterative sigma² estimation)
- `apply_estimator(returns, method)` → dispatch; method ∈ {sample, ledoit_wolf, oas, qis, mp_clip}

### projector.py
- `projector_distance_lemma1(U_K_t, U_K_prev, K)` → sqrt(max(0, 2K − 2‖Uᵀu‖²_F))
- `compute_population_projector_distance(Sigma_t, Sigma_prev, K)` → exact projector distance
- `operator_norm_error(C, Sigma)` → max|eigenvalue(C−Σ)|
- `compute_band_term(eta, C, Sigma, Delta_K, K)` → T_{K,s}
- `compute_capped_band(T_t, T_prev, K)` → tau* = min(√(2K), T_t + T_prev)
- `compute_observable_band(C_t, C_prev, K)` → (T̂_t, τ̂*, weak_id_flag)

### simulation.py
- `generate_factor_model_returns(N, M_total, K_f, var_explained, gap_type, seed)`
  - Uses QR orthogonalisation of Λ columns + analytic scale to achieve exact spectral gap
  - strong gap: λ₁/λ₂ = 5.0 exactly; weak gap: λ₁/λ₂ = 1.1 exactly
  - Returns (returns, Sigma, Lambda, sigma2_idio)
- `apply_rotation(Sigma, theta, plane)` → R @ Sigma @ Rᵀ
- `generate_spiked_cov(N, spikes, sigma2)` → diagonal matrix
- `generate_elliptical_returns(Sigma, M, nu, rng)` → t_nu via χ²/ν mixing
- `generate_gaussian_returns(Sigma, M, rng)` → L @ z

### functionals.py
- `compute_AR_K`, `compute_f1`, `compute_spectral_entropy`, `compute_trace`
- `delta_method_var_AR_K`, `delta_method_var_f1`, `delta_method_var_H`, `delta_method_var_trace`
  → M-scaled asymptotic variances (actual variance = σ²/M)
- `estimate_kappa_moment(returns)` → excess-kurtosis-based kappa
- `estimate_kappa_radial_mle(returns, S)` → MLE via F(d_eff, nu) fit to Mahalanobis distances
- `spike_debias_AR_K(eigenvalues, K, N, M, sigma2_est, kappa)` → BBP-debiased AR_K

### monitoring.py
- `analytic_null_simulator(eigenvalues, K, s, M, n_draws, rng)` → Proposition 2 null draws
- `bootstrap_projector_movement(returns_A, C_A, K, s, M, estimator, n_boot, rng)` → Algorithm 2
- `apply_bh_correction(pvalues, q)` → Benjamini-Hochberg boolean rejection array
- `apply_by_correction(pvalues, q)` → Benjamini-Yekutieli (divides q by H_n)

## Key Implementation Decisions

### LW / OAS shrinkage (without raw data)
Both `ledoit_wolf` and `oas` operate on S and n directly using closed-form formulas:
- LW:  ρ = min(1, (tr(S²)+tr(S)²) / (n·(tr(S²)−tr(S)²/p)))
- OAS: ρ = (α+μ²) / ((n+1)·(α−μ²/p))  where α = mean(S²), μ = tr(S)/p

### QIS (QuEST nonlinear shrinkage)
- Re(ĝ(λᵢ+iε)) computed with ε=1e-6 (j=i term contributes 0 automatically)
- Spectral density f̂(λᵢ) from scipy.stats.gaussian_kde with Silverman bandwidth
- d̃ᵢ = λᵢ / (A²+B²); rescaled so Σd̃ = tr(S)

### Factor model gap control
Spectral gap condition is achieved exactly by solving for s₀² analytically:
  s₀² = (r + (r−1)·A·β) / (1 + (1−r)·β)
where r is target ratio, A = K_f−1, β = (1−ve)/(N·ve).

### MP clip sigma² estimation
Iterative trimmed mean: sigma₂_new = mean(bulk) / (1+c); 5 iterations.
Bulk replacement = mean(bulk_eigs); reconstructed as spike block + bulk*I residual.

### BBP spike debiasing
discriminant = (λ̂ − σ²(1−c))² − 4cσ⁴;
λ_deb = (λ̂ + σ²(1−c) + √max(0, discriminant)) / 2;
then trace-preserving rescaling of all eigenvalues.

## Dependencies
- numpy, scipy, scikit-learn (sklearn) — no other libraries
