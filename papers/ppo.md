---
slug: ppo
title: "Proximal Policy Optimization Algorithms"
authors:
  - John Schulman
  - Filip Wolski
  - Prafulla Dhariwal
  - Alec Radford
  - Oleg Klimov
venue: arXiv
year: 2017
arxiv_id: "1707.06347"
doi: null
url: https://arxiv.org/abs/1707.06347
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: alignment
  secondary:
    - reasoning
tags:
  - ppo
  - policy-gradient
  - clipped-objective
  - rlhf
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microppo
  target_path: 02-alignment/microppo.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microppo.py
    script_slug: microppo
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: reinforce
---

## TL;DR

PPO replaces TRPO's expensive constrained optimization with a clipped surrogate objective that limits how far the new policy can move from the old per update, giving a first-order policy-gradient method that is simple to implement and stable across a wide range of tasks.

## Problem

Vanilla policy gradients are high-variance and sensitive to step size: too large a step destroys the policy, too small wastes data. Trust-region policy optimization (TRPO, Schulman et al. 2015) addressed this with a hard KL constraint solved by conjugate gradient and a backtracking line search, but the second-order machinery is complex and incompatible with many neural-network frameworks. Practitioners wanted a method with TRPO's stability that runs on a standard first-order optimizer with mini-batch SGD.

## Contribution

PPO introduces two surrogate objectives that approximate TRPO's trust region without solving a constrained problem. The first, PPO-Clip, multiplies the policy-ratio r_t(θ) = π_θ(a|s) / π_old(a|s) by the advantage and clips the ratio to [1-ε, 1+ε]; the loss is the minimum of the clipped and unclipped products. The second, PPO-Penalty, adds an adaptive KL-divergence penalty to the unclipped objective and adjusts the coefficient to hit a target KL. The clipped form is the one most papers and codebases adopt because it has one fewer hyperparameter and works well with default settings. PPO uses generalized advantage estimation (GAE) for advantages and a value-function critic trained jointly.

## Method summary

- Collect a batch of trajectories from the current policy, computing rewards and value estimates.
- Compute advantages with GAE: A_t = Σ (γλ)^k δ_{t+k}, where δ is the TD-error.
- For multiple epochs over the batch, sample minibatches and minimize: L_clip = -E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)] + c1 · L_value - c2 · entropy_bonus.
- Update the policy and value heads with Adam.
- Discard the batch and collect a new one with the updated policy.
- Typical hyperparameters: ε = 0.1 to 0.2, GAE λ = 0.95, γ = 0.99, 4 to 10 epochs per batch, minibatch size 64.

## Key results

On the OpenAI continuous-control benchmark suite (MuJoCo: HalfCheetah, Walker, Ant, Hopper, etc.) and on Atari, PPO matches or exceeds TRPO while being roughly an order of magnitude simpler to implement and faster to train. The clipped form outperforms the adaptive-KL-penalty form on most tasks, which is why it is the default in most modern codebases. The paper also shows that PPO handles parallel actors trivially via simple data aggregation, in contrast to TRPO's batch-coupled second-order step.

## Relation to existing work

PPO sits between vanilla policy gradients (REINFORCE, Williams 1992) and TRPO (Schulman et al. 2015). REINFORCE is simple but unstable; TRPO is stable but complex. PPO inherits TRPO's intuition — limit per-step policy change — and replaces the constraint with a clipped first-order surrogate. PPO became the dominant on-policy RL algorithm and is the policy-optimization stage of standard RLHF pipelines (Christiano et al. 2017; Ouyang et al. 2022, InstructGPT). DPO (Rafailov et al. 2023) and GRPO (Shao et al. 2024) are post-PPO simplifications: DPO removes the RL loop entirely, GRPO removes the value function in favor of group-relative advantages.

## Implementation notes

A pedagogical RLHF-style script needs three pieces: a small policy LM, a tiny reward model (or a hand-coded reward), and a PPO trainer. Minimum viable PPO loop: roll out trajectories under π_old, compute per-token log-probabilities under both π_θ and π_old, form clipped ratios, weight by advantages, minimize. Pitfalls: forgetting to detach π_old's log-probabilities (turning them into part of the graph and contaminating gradients); confusing reward with advantage; using ε too large (clipping fires rarely and the clip provides no benefit). For RLHF specifically, an additional KL penalty against a frozen reference model is conventional and stabilizes training. Compare per-step KL(π_θ‖π_old) and KL(π_θ‖π_ref); both should grow slowly.

## Open questions

PPO's clipping is heuristic; the paper does not prove monotonic improvement, only argues empirically that the clip approximates a trust region. Subsequent work on truly proximal updates and on alternatives (DPO, GRPO, RLOO) revisits the same trade-off space.
