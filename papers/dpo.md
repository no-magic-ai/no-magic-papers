---
slug: dpo
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors:
  - Rafael Rafailov
  - Archit Sharma
  - Eric Mitchell
  - Stefano Ermon
  - Christopher D. Manning
  - Chelsea Finn
venue: NeurIPS
year: 2023
arxiv_id: "2305.18290"
doi: null
url: https://arxiv.org/abs/2305.18290
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: alignment
  secondary: []
tags:
  - dpo
  - preference-optimization
  - rlhf-alternative
  - bradley-terry
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microdpo
  target_path: 02-alignment/microdpo.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microdpo.py
    script_slug: microdpo
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

DPO derives a closed-form relationship between an RLHF-style KL-constrained reward maximizer and the optimal policy, then trains the policy directly on preference pairs with a simple classification-style loss — no separate reward model and no reinforcement learning loop.

## Problem

The standard RLHF pipeline trains a reward model on human preferences, then optimizes a policy against that reward with PPO under a KL penalty against a reference policy. The pipeline is brittle: PPO is sensitive to hyperparameters, the reward model can be exploited, and the two-stage decomposition makes credit assignment between stages difficult. The paper asks whether the same KL-regularized objective can be optimized in one stage, directly from preferences, without an explicit reward model and without on-policy RL.

## Contribution

The paper shows that the optimal policy of the standard KL-regularized RLHF objective has a closed form in terms of the reference policy and an implicit reward function. Inverting this relationship lets one express the implicit reward as a log-ratio of the policy and reference. Substituting that into the Bradley-Terry preference likelihood produces a loss whose only inputs are the policy log-probabilities of the chosen and rejected responses under both the trained and reference models. Optimizing that loss with ordinary gradient descent recovers the same optimum as RLHF + PPO on the same preference data, but without sampling from the policy, without a reward model, and without value or advantage estimation.

## Method summary

- Take a reference policy π_ref (typically the SFT model) and a dataset of (prompt, chosen response, rejected response) triples.
- Define the implicit reward as r(x, y) = β · log(π_θ(y|x) / π_ref(y|x)), where β is the KL coefficient from the original RLHF objective.
- Plug the implicit reward into the Bradley-Terry log-likelihood: L = -log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))).
- Train π_θ by minimizing L with standard supervised optimization; compute the four log-probabilities per batch (chosen and rejected, under θ and under ref).
- The reference is frozen and used only for log-probability scoring.

## Key results

On controlled-sentiment generation, summarization (Reddit TL;DR), and single-turn dialogue (Anthropic-HH), DPO matches or exceeds PPO-based RLHF on both automated and human evaluations while being simpler to implement, more stable, and significantly cheaper to train. The paper reports that DPO's win rate against PPO baselines is at or above 50% across these settings, with much lower variance across seeds. The method is also far less hyperparameter-sensitive than PPO; β alone controls the KL trade-off.

## Relation to existing work

DPO replaces the reward-modeling and RL stages of standard RLHF (Christiano et al. 2017; Ouyang et al. 2022, InstructGPT) with a single supervised step, while keeping the same KL-regularized objective. It is closely related to preference-based reinforcement learning and to contrastive methods. Subsequent work generalizes the same idea: IPO (Azar et al. 2023) replaces the Bradley-Terry assumption with a more conservative loss to avoid overfitting; KTO (Ethayarajh et al. 2024) handles unpaired preferences using prospect-theory utilities; SimPO (Meng et al. 2024) drops the reference model entirely. GRPO sits in a related but distinct branch where group-normalized rewards replace the value function.

## Implementation notes

A pedagogical script needs only a small SFT base, a tiny preference dataset, and a forward pass that scores chosen and rejected responses under both the trainable and reference models. The minimal trainer is: tokenize, compute log-probabilities of chosen and rejected sequences under both models, form the DPO loss, backprop. Pitfalls: the reference model must be in eval mode and frozen; numerical stability requires summing log-probabilities at the token level then differencing, not subtracting raw probabilities; β too large makes the policy collapse onto the reference, β too small lets it drift. A useful comparison is to plot per-step chosen vs rejected log-ratio — both should rise, but chosen should rise faster.

## Open questions

DPO's derivation assumes the Bradley-Terry preference model is correct; when preferences are noisier or non-transitive (multi-rater disagreement), DPO can overfit to the preference data. Followup work (IPO, KTO) addresses this. The paper also leaves open how to combine DPO with online preference collection, which subsequent work has explored.
