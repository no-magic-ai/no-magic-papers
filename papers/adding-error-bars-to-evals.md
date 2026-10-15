---
slug: adding-error-bars-to-evals
title: "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"
authors:
  - Evan Miller
venue: arXiv
year: 2024
arxiv_id: "2411.00640"
doi: null
url: https://arxiv.org/abs/2411.00640
discovered_via: maintainer
discovered_date: 2026-10-15
status: implemented
themes:
  primary: evaluation
  secondary: []
tags:
  - error-bars
  - confidence-intervals
  - paired-differences
  - power-analysis
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microerrorbars
  target_path: 01-foundations/microerrorbars.py
  target_tier: 01-foundations
  batch_label: evaluation-seed
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microerrorbars.py
    script_slug: microerrorbars
    commit: null
    release: null
    media_repo: no-magic-viz
    media_status: linked
    scene_path: scenes/scene_microerrorbars.py
    preview_path: previews/microerrorbars.gif
    media_note: null
lesson:
  path: no-magic-papers/lessons/adding-error-bars-to-evals.md
  status: planned
dependencies_on_other_papers: []
---

## TL;DR

An eval score is a sample mean over questions drawn from a super-population, so it carries sampling error and should be reported with a standard error. Miller gives formulas for single-model, unpaired, paired and clustered standard errors, and for the number of questions needed to detect a given score difference.

## Problem

Model evals are usually reported as a bare number, and rankings are read off small gaps. How uncertain is an eval score, how should two models be compared on the same questions, and how many questions does an eval need?

## Contribution

- Treats the questions as a sample from a super-population and derives the CLT standard error of a mean score (Eq. 1–3, Sec. 2.1).
- Shows that questions sharing a passage are not independent and gives a cluster-adjusted standard error (Sec. 2.2).
- Compares two models by unpaired (Eq. 5) and paired (Eq. 7) differences (Sec. 4), and recommends variance-reduction techniques, including resampling answers and next-token probabilities (Sec. 3).
- Gives a sample-size formula (Eq. 9) and its inverse, the minimum detectable effect (Eq. 10), in a power analysis (Sec. 5).

## Method summary

- Single model: SE = √(Var(s)/n), estimated with the sample variance; the interval is the mean ± z·SE (Eq. 1–3). Miller calls the CLT "applicable to any evals having scores with finite variance and a large number of questions", and bootstrapping unnecessary unless a complicated sampling scheme or estimator is used (Sec. 2.1).
- Clustering: when questions come in groups, the naive SE is too small; the clustered SE accounts for within-cluster correlation (Sec. 2.2).
- Two models: the unpaired SE adds the two variances (Eq. 5); the paired analysis uses per-question differences, with variance Var(s_A) + Var(s_B) − 2Cov(s_A, s_B) divided by n, smaller when scores correlate positively (Sec. 4.2, Eq. 7).
- Power: n = (z_{α/2} + z_β)²(ω² + σ_A²/K_A + σ_B²/K_B)/δ² for a paired analysis (Eq. 9); Eq. 10 inverts it for δ given n.

## Key results

These are Miller's own claims, not outputs of the toy program.

- Table 4 reports clustered and naive standard errors on two evals using Anthropic models, described as non-fictional numbers; the clustered standard errors "can be over 3X larger than naive standard errors" (Table 4 caption). Miller concludes that confidence intervals reported for reading-comprehension evals in the cited work are likely anti-conservative (Sec. 2.2).
- Worked example (Sec. 5): with ω² = 1/9, δ = 0.03, α = 0.05 and β = 0.20, Eq. 9 gives n ≈ 969 independent questions, so new evals should contain at least 1,000 questions to signal well. Miller states that these parameters are fictional; his Table 2 numbers are fictional as well.
- The paper recommends computing question-level differences between models and advises against adjusting sampling temperature merely to reduce variance (Sec. 6).

## Relation to existing work

Miller draws on the experiment-design literature, including power analysis ("With Little Power Comes Great Responsibility", 2020), and adapts it to language-model evals. For binomial proportions, Brown, Cai and DasGupta (2001, doi 10.1214/ss/1009213286) recommend the Wilson or equal-tailed Jeffreys interval for small n and the Agresti–Coull interval for larger n; that comparison is not part of Miller's paper.

## Implementation notes

`no-magic/01-foundations/microerrorbars.py` is a toy simulation, not a reproduction of the paper's tables and not a measurement of any real eval. It has three parts at seed 42. Part A draws Bernoulli scores with a known mean and counts how often CLT, percentile-bootstrap and Wilson 95% intervals cover it (one answer per question, K = 1). Part B trains two logistic regressions, one with 8 features and one with 7, scores them on a pool of `POOL_SIZE` items, and compares unpaired (Eq. 5) with paired (Eq. 7) intervals. Part C evaluates Eq. 9 and Eq. 10 and runs a simulated power check.

Extensions beyond Miller: the coverage study and the Wilson reference are this program's additions, not claims of the paper; Miller does not say CLT intervals under-cover at small n. The unpaired minimum-detectable-effect and required-n rows extend Eq. 10, since Eq. 9 assumes a paired analysis. Clustering (Sec. 2.2) is out of scope, so no clustered standard error is computed.

Printed results (one seed, CPython 3.12.8): at μ = 0.95 and n = 100 the CLT interval covered 0.863 and the bootstrap 0.938; the paired interval at n = 30 covered 0.919, below the 0.95 target; the paired-to-unpaired width ratios were printed as 0.738 and 0.744 (n = 30 and n = 500); and the program reproduces Miller's n = 968.997 (unrounded) for ω² = 1/9, δ = 0.03. No result transfers to real LLM evals. `commit` stays `null`.
