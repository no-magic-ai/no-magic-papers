---
slug: ucb1
title: "Finite-time Analysis of the Multiarmed Bandit Problem"
authors:
  - Peter Auer
  - Nicolò Cesa-Bianchi
  - Paul Fischer
venue: Machine Learning
year: 2002
arxiv_id: null
doi: "10.1023/A:1013689704352"
url: https://link.springer.com/article/10.1023/A:1013689704352
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: agents
  secondary:
    - reasoning
tags:
  - bandits
  - exploration
  - regret-bounds
  - ucb
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microbandit
  target_path: 04-agents/microbandit.py
  target_tier: 04-agents
  batch_label: v3-backfill-agents
  review_date: null
implementations:
  - repo: no-magic
    path: 04-agents/microbandit.py
    script_slug: microbandit
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The paper proves that simple index policies — UCB1 chief among them — achieve logarithmic regret on the stochastic multi-armed bandit problem in finite time, not only asymptotically, and gives sharp constants for several variants.

## Problem

The multi-armed bandit problem asks how to allocate pulls between K arms with unknown reward distributions so that cumulative reward is maximized. Lai and Robbins had shown in 1985 that any consistent policy must incur at least logarithmic regret asymptotically, and they constructed an asymptotically optimal index policy whose constants depend on KL divergences between arm distributions. That construction was hard to implement and analyze in finite samples. The field needed an index policy whose performance bound held at every horizon, not only in the limit, and whose construction was elementary enough to apply in practice.

## Contribution

The paper introduces three index-based policies — UCB1, UCB1-Tuned, and UCB2 — and proves finite-time regret bounds for each. UCB1 is the simplest: pull the arm that maximizes the empirical mean plus a confidence radius of sqrt(2 ln n / n_i), where n is the total number of pulls and n_i is the count for arm i. The bound is (8 ln n) / Δ_i summed over suboptimal arms, plus a small constant, where Δ_i is the suboptimality gap. The key innovation over Lai and Robbins is that the bound is uniform in n: it holds at every step, not only asymptotically, with explicit constants that an implementer can use to set hyperparameters.

## Method summary

- For each arm i, maintain n_i (pulls so far) and μ̂_i (sample mean reward).
- At step n, compute the UCB1 index μ̂_i + sqrt(2 ln n / n_i) for every arm; pull the arg-max. Initialize by pulling each arm once.
- UCB1-Tuned replaces the constant 2 with an arm-specific variance-aware term, which is sharper but lacks a clean theoretical bound.
- UCB2 introduces an epoch structure: arm i is pulled in batches whose lengths grow geometrically; the bound replaces the constant 8 with a smaller value at the cost of more bookkeeping.
- Proofs use Hoeffding's inequality on the empirical mean, then bound the expected number of times any suboptimal arm is selected by relating it to the event that its index exceeds the optimal arm's index.

## Key results

UCB1 achieves expected regret O((ln n) Σ 1/Δ_i + Σ Δ_i), where the sums run over suboptimal arms. The constants are explicit. UCB1-Tuned and UCB2 improve constants further; UCB2 gets within a constant factor of the Lai-Robbins lower bound. Empirically, the paper shows that all three outperform ε-greedy with fixed ε on Bernoulli arms and that UCB-Tuned is the practical choice when the variance estimator is reliable. The finite-time guarantees made the bandit framework usable as a primitive in larger systems, including the UCT planner that uses UCB1 inside tree search.

## Relation to existing work

The paper builds on Lai and Robbins (1985) for the asymptotic lower bound, on Agrawal (1995) for an earlier finite-sample index policy, and on Hoeffding's inequality as the concentration tool. The contribution is a finite-time refinement of the Lai-Robbins program that is simple enough to use without modification. UCB1 became a foundational primitive: UCT (Kocsis & Szepesvári 2006) embeds it in tree search, and most contextual-bandit and reinforcement-learning algorithms that use optimism-under-uncertainty trace back to this paper.

## Implementation notes

The pedagogical script benefits from showing UCB1 alongside ε-greedy and Thompson sampling on the same Bernoulli bandits, because the contrast makes the value of the confidence-bound term concrete: ε-greedy explores at a fixed rate forever, UCB1's exploration shrinks as confidence grows, Thompson sampling samples from posteriors instead of using a deterministic bound. Pitfalls: the UCB1 formula needs the total step count n in the numerator, not the per-arm count; the initial round-robin pull is required to avoid division by zero. Plot cumulative regret on a log-x axis to make the logarithmic envelope visible. Reasonable hyperparameters: 10 arms, true means drawn uniformly in [0,1], horizon 10000, ε in {0.01, 0.1} for the baseline, and a single confidence constant for UCB1.

## Open questions

The bounds assume bounded rewards and stationary distributions. Subsequent work extends UCB-style optimism to contextual bandits, linear bandits, and to non-stationary problems with discounting or sliding windows; each of those settings requires its own analysis.

## Further reading

- Lai & Robbins (1985) — the asymptotic lower bound this paper refines.
- Thompson (1933) — the original posterior-sampling rule, often a strong empirical baseline.
- Kocsis & Szepesvári (2006) — UCT, the canonical use of UCB1 inside tree search.
