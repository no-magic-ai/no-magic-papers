---
slug: reinforce
title: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
authors:
  - Ronald J. Williams
venue: Machine Learning
year: 1992
arxiv_id: null
doi: "10.1007/BF00992696"
url: https://link.springer.com/article/10.1007/BF00992696
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: alignment
  secondary: []
tags:
  - reinforce
  - policy-gradient
  - score-function-estimator
  - log-derivative-trick
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microreinforce
  target_path: 02-alignment/microreinforce.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microreinforce.py
    script_slug: microreinforce
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Williams introduces REINFORCE, the score-function policy-gradient estimator: weight the gradient of the log-probability of each sampled action by the resulting return, average over samples, and ascend. The result is a general unbiased estimator of the policy gradient that requires neither a model of the environment nor differentiability of the reward.

## Problem

In the late 1980s, neural networks were trained almost exclusively by backpropagation through differentiable losses. Reinforcement-learning settings violate that assumption: the environment is a black box, the action selection involves a discrete sample, and the reward is observed rather than computed from a differentiable graph. Practitioners needed an estimator that could optimize a stochastic policy parameterized by a neural network using only sampled trajectories and observed rewards.

## Contribution

The paper proves that for any stochastic policy π_θ(a|s), the gradient of the expected return E_π[R] equals E_π[R · ∇_θ log π_θ(a|s)]. This is the score-function (also called log-derivative or REINFORCE) estimator. Sampling actions and computing the per-sample gradient ∇_θ log π_θ(a|s) weighted by the return gives an unbiased Monte-Carlo estimate of the policy gradient. The paper develops the family of associative reinforcement-learning algorithms that follow from this identity for discrete and continuous (Gaussian) policies, gives convergence conditions under suitable step sizes, and analyzes baseline subtraction as a variance-reduction technique that preserves unbiasedness.

## Method summary

- Parameterize the policy π_θ(a|s) so that ∇_θ log π_θ(a|s) is computable; standard choices are softmax over logits for discrete actions and a Gaussian with learned mean (and optionally variance) for continuous actions.
- Sample trajectories by interacting with the environment under π_θ.
- For each sampled action, compute log π_θ(a|s) and back through it (the score).
- Weight each score by the return that followed: full episode return for vanilla REINFORCE, or return-to-go for the causal version that has lower variance.
- Subtract a baseline b(s), often the mean return or a learned state value, to reduce variance without introducing bias.
- Average across samples and apply a gradient ascent step; iterate.

## Key results

The paper proves that the score-function estimator is unbiased for the true policy gradient under the standard sampling distribution. It shows convergence with probability one for stationary problems and decaying step sizes (Robbins-Monro conditions). Empirical demonstrations on simple associative-reward tasks show that REINFORCE solves problems for which standard supervised learning has no signal, including learning to emit specific output patterns conditioned on inputs without labels.

## Relation to existing work

REINFORCE generalizes earlier reinforcement-learning approaches that were tied to specific architectures (e.g. associative reward-penalty networks of Barto and Anandan) and connects them to the broader policy-gradient theory developed in subsequent decades by Sutton, McAllester, and others. Modern algorithms — actor-critic, A2C, A3C, TRPO, PPO, GRPO — all use the score-function identity as their core gradient estimator; they differ in how they compute or approximate the return weight (advantage function, clipped surrogate, group-normalized advantage). The same trick reappears in variational inference (REINFORCE-style gradients for discrete latents) and in any setting where a non-differentiable sampling step sits between parameters and the loss.

## Implementation notes

A pedagogical script can build REINFORCE on a contextual-bandit or short-horizon environment so the variance is manageable. Minimum viable trainer: roll out one trajectory per update (or a small batch), compute log-probabilities of sampled actions, multiply by returns minus baseline, sum and backprop. Pitfalls: forgetting to detach the return weight (it should be a number, not part of the graph); using full-episode return where return-to-go is cheaper and lower variance; omitting the baseline and watching gradient noise dominate. The simplest baseline is a moving average of recent returns. A useful comparison: log per-update gradient norm with and without the baseline; the variance reduction is visible.

## Open questions

REINFORCE's variance grows with trajectory length, which limits its direct use in long-horizon problems and motivates the actor-critic and clipped-surrogate refinements that followed. The paper treats the stationary case; modern practice deals with non-stationarity from policy improvement itself, which is the setting PPO and GRPO target.
