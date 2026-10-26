# Recoverability: a local Fisher-information approximation

This note derives the expected value and sampling distribution of the
held-out ΔLL/read that the recoverability pipeline measures, and from them the
probability that an effect is called *recovered*. It also explains how the
result was validated against the synthetic-power simulations (`06`), and how
it attributes a failure to effect size, bin width, overdispersion or model
dimension. The implementation is `11_recoverability_theory.py`, with outputs
in `results/…/theory/`. Everything here is local: it is a second-order
expansion around the population fit and makes no claim of exactness for large
effects or very low depth.

## 0. Setting and notation

Transcript *t* has positions *j*, observation bins *b* of width *w*, and binned
40S footprint counts *y_tb* with total *N_t*. The model is a conditional
multinomial with a softmax over positions:

    p_tj = exp(x_tj'θ) / Σ_j' exp(x_tj''θ),    q_tb = Σ_{j∈b} p_tj,    LL(θ) = Σ_t Σ_b y_tb log q_tb(θ).

The parameter is θ = (γ, β):
- γ: the positional nuisance (5′- and AUG-anchored distance bins). It makes up M0.
- β: one feature family (SW, ST, AUG, …). M1 = M0 + β.

Fits are ridge-penalised with λ = 1/σ², σ = 1. The statistic is

    Δ = [LL_te(θ̂_1) − LL_te(θ̂_0)] / N_te,

where θ̂ is fitted on the training chromosomes and scored on the held-out
chromosome. Δ is pooled over *K* folds. *Recovered* means two things: the
transcript-bootstrap 95% CI of Δ excludes 0 (in both cross-replicate
directions for real data), and Δ beats a null threshold. That threshold is the
position-matched permutation null for real data, and the truth-A 95th
percentile in the simulations.

## 1. Signal

Let θ_1, θ_0 be the population fits: the fits to the expected counts of the
training transcripts. The population gain is

    G = E[LL_te(θ_1) − LL_te(θ_0)] = Σ_{t∈te} N_t Σ_b q*_tb log(q_1,tb / q_0,tb).

Δ·N_te is linear in the counts, so G depends on the data only through
E[y] = N q*. **Overdispersion therefore does not change the signal.**

For a correctly specified M1 and a small β, θ_0 is the I-projection of the
truth onto M0. Expanding to second order gives

    G / N  ≈  ½ s_w,      s_w = β' Ī_eff(w) β,      Ī_eff = Ī_ββ − Ī_βγ Ī_γγ⁻¹ Ī_γβ        (1)

Here Ī is the per-read Fisher information. The Schur complement removes the
part of the feature effect that position alone can explain, which is what the
position-matched design is meant to do. Bin width enters only through q_tb and
the positional design, so s_w is recomputed for w = 1, 5, 10 and 20 nt.

## 2. Estimation penalty

The fits deviate from their population targets by θ̂ − θ ≈ H z, where:
- H = (A + λI)⁻¹;
- A is the expected Hessian of −LL **under the true distribution**;
- z is the training score, with Var(z) = B (the sandwich).

Expanding LL_te to second order, the expected excess held-out loss of M1
over M0 is

    P = ½ tr(I_te H_1 B H_1) − ½ tr(I_te H_0 B H_0).

For a correctly specified model, homogeneous transcripts, ridge negligible
and Dirichlet-multinomial noise with concentration κ, this reduces to

    B = F̄ A,     F_t = (N_t + κ)/(1 + κ),     F̄ = Σ N_t F_t / Σ N_t,
    E[Δ] ≈ ½ s_w − F̄ d / (2 N_tr)                                                        (2)

with d the effective dimension of β (ridge degrees of freedom of the efficient
information). **Overdispersion multiplies the estimation penalty**, not the
signal.

## 3. Sampling distribution

Δ is, to second order, a Gaussian quadratic form in two independent pieces:
the test-count noise and the training score. Its terms are

    N_te Δ ≈ G + l'z + z'Wz + ξ'Mz + ν,

with the following parts.

| term | what it is |
|---|---|
| ν | test-count noise of the population gain |
| ξ'Mz | test noise × fit noise |
| z'Wz | the penalty term (mean −P) |
| l'z | fit noise × the train/test score shift |

The train/test shift appears because θ_1 is optimal for the training
transcripts, not the test ones. In the leading-order homogeneous case, per read
and pooled over *K* folds (ρ = 1/(K−1)),

    Var(Δ) ≈ F̄ s_w / N + K F̄² d ρ(1 + ρ/2) / N²                                          (3)

The first term is count noise on the signal. The second is training noise,
which is present even when β = 0. Under the null, Δ ~ N(−F̄d/(2N_tr), K F̄² d ρ(1+ρ/2)/N²).

## 4. Detection probability

The transcript bootstrap resamples transcripts with their counts, conditional
on the fit. Its SE therefore contains count noise and between-transcript
heterogeneity *h* of the per-read gain, but not training noise:

    SE² ≈ F̄ (s_w + F̄ d/N_tr) / N + h,     h = Σ_t E[N_t²](g_t − ḡ)² / N²                (4)

With μ, σ from (2)–(3) and c_null the null 95th percentile,

    P_det ≈ 1 − Φ( (max(z_.975·SE, c_null) − μ) / σ ).                                    (5)

In every simulated cell c_null < z·SE, so the CI condition is the binding one.
For real data the two cross-replicate directions use disjoint count sets
(different training and test libraries). To leading order they are therefore
independent, and P_both ≈ P_det².

## 5. Required depth, and the overdispersion ceiling

Write N_eff = N/F̄ for the effective number of reads. Combining (2)–(5) with
SE ≈ σ, 80% power requires

    N_eff ≥ N* = d′² / ( s_w (√(z² + d′) − z)² ),     d′ = dK/(K−1),  z = z_.975 + z_.8 = 2.80   (6)

The limits are:
- d → 0: N* → 4z²/s_w (noise-limited).
- d′ ≫ z²: N* → d′/s_w (dimension-limited).

Under the Dirichlet-multinomial,

    N_eff = (Σ N_t)² / Σ N_t F_t  →  (1 + κ) T_eff   as depth → ∞,   T_eff = (Σ N_t)² / Σ N_t²,

where T_eff is the read-weighted effective number of transcripts.

**Depth saturation.** If N* > (1 + κ)·T_eff, the effect is not recoverable at
any sequencing depth of the same library; only more transcripts or less
overdispersion help. Under multinomial noise (κ → ∞) every s_w > 0 is
recoverable at some depth.

## 6. What the simulations required beyond the textbook expansion

1. **Nuisance misspecification.** `06` plants a 1-nt positional truth, but
   the 10-nt model cannot represent it. Two things follow:
   - The expected Hessian under the truth is not the model's Fisher
     information, so the sandwich has to use the true Hessian.
   - A high-dimensional block can absorb transcript-specific positional
     misfit on the training transcripts, and that fit does not transfer. In
     yeast under the null, SW gains +2.5e-4/read in-sample but loses
     6.6e-5/read on held-out chromosomes, at any depth. This is a
     transcript-level overfitting that depth cannot remove.
2. **Ridge × misfit.** The ridge shrinks that spurious fit in a strongly
   non-quadratic way. The population targets θ_1, θ_0 are therefore refitted
   exactly at each depth (10 anchor depths, 5 to 50,000 reads/leader), never
   corrected by a quadratic step.
3. **Fold overlap.** Each fold's test chromosomes are in every other fold's
   training set. The first-order term l'z is correlated across folds and is
   pooled exactly through a per-fold block decomposition of the scores.

The implementation evaluates (1)–(5) with the exact matrices (population fits,
true-Hessian sandwich, fold structure). The closed forms (2), (3) and (6) are
the citable summary, and are evaluated alongside.

## 7. Validation

_Filled in from `theory/theory_calibration.tsv` after the run._

## 8. Decomposition

_Filled in from `theory/theory_decomposition.tsv` and `theory/theory_real_data.tsv`._
