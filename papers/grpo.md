---
slug: grpo
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors:
  - Zhihong Shao
  - Peiyi Wang
  - Qihao Zhu
  - Runxin Xu
  - Junxiao Song
  - Xiao Bi
  - Haowei Zhang
  - Mingchuan Zhang
  - Y. K. Li
  - Y. Wu
  - Daya Guo
venue: arXiv
year: 2024
arxiv_id: "2402.03300"
doi: null
url: https://arxiv.org/abs/2402.03300
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: alignment
  secondary:
    - reasoning
tags:
  - grpo
  - rlhf
  - math-reasoning
  - group-relative-advantage
  - critic-free
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microgrpo
  target_path: 02-alignment/microgrpo.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microgrpo.py
    script_slug: microgrpo
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: ppo
---

## TL;DR

GRPO is the reinforcement-learning algorithm introduced in DeepSeekMath: it samples a group of responses per prompt, normalizes their rewards within the group to compute advantages, and removes PPO's value network entirely, cutting memory and compute while matching or exceeding PPO on math reasoning.

## Problem

PPO requires a value-function critic that is roughly the same size as the policy, doubling memory during training and adding its own optimization variance. The critic must be initialized and pretrained, and its accuracy strongly affects PPO's stability. For long-form reasoning tasks where rewards are sparse and arrive only at the end of a trajectory, the critic is asked to estimate values for intermediate states it rarely sees with high confidence. The paper asks whether one can keep PPO's clipped surrogate but eliminate the critic by exploiting structure in how reasoning data is sampled.

## Contribution

GRPO replaces the per-step value-baseline with a per-prompt group baseline. For each prompt, the policy samples G responses; the reward for each response is computed (rule-based or with a reward model); advantages are obtained by standardizing the rewards within the group (subtract the group mean, divide by the group standard deviation). The clipped policy-gradient objective from PPO is then applied with these advantages. Because the baseline now comes from sampling the same prompt G times, no separate critic is needed. The paper combines GRPO with a math-tuned base model and rule-based rewards for verified math answers, producing DeepSeekMath, which became the recipe of choice for subsequent reasoning-RL work, including DeepSeek-R1.

## Method summary

- For each prompt x, sample G complete responses {y_1, ..., y_G} from the current policy π_θ.
- Score each response with a reward r_i (rule-based correctness for math; preference reward model for general tasks).
- Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + ε); these advantages are constant across the tokens of response y_i.
- Apply the PPO clipped surrogate per token: L = -E[min(ρ_t · A_i, clip(ρ_t, 1-ε_clip, 1+ε_clip) · A_i)] + β · KL(π_θ || π_ref).
- The KL penalty against a frozen reference is kept (not the per-step KL approximation; an unbiased estimator is used).
- No value head, no GAE, no separate critic optimizer. Memory drops by roughly a factor of two relative to PPO.

## Key results

DeepSeekMath-7B trained with GRPO + rule-based rewards reaches 51.7% on MATH (competition mathematics), exceeding many larger open models and approaching GPT-4-class accuracy. Ablations show GRPO matches PPO at lower compute cost on the same prompts and rewards. The paper also documents that GRPO scales better than PPO with the number of samples per prompt because the baseline quality improves directly with G.

## Relation to existing work

GRPO sits in the line PPO → RLOO → GRPO. RLOO (Ahmadian et al. 2024) had already proposed a leave-one-out baseline computed from group samples. GRPO's contribution is the standardized within-group advantage and a clean pairing with rule-based reasoning rewards. DeepSeek-R1 builds on GRPO (with rule-based accuracy and format rewards) to elicit long-chain reasoning behaviors from a base model without supervised fine-tuning. DPO and KTO are alternative offline preference methods; GRPO remains in the online-RL family but trades the critic for samples.

## Implementation notes

A pedagogical script needs a tiny policy LM, a reward function (a rule-based one is easiest — for example, a regex check on a target string), a sampler that draws G completions per prompt, and the PPO clipped loss with group-normalized advantages. Pitfalls: forgetting to detach π_old log-probabilities (just like PPO); choosing G too small (G < 4) makes the standardization noisy; choosing G too large is wasteful. KL penalty is not optional — without it the policy degrades quickly. Useful comparison: run the same script with PPO + value head and with GRPO and compare reward curves and memory usage.

## Open questions

GRPO's group-relative advantage assumes the G samples are diverse enough to give a useful baseline; for low-entropy policies the within-group variance collapses and advantages become noisy. The paper does not analyze this regime. DeepSeek-R1 addresses the related collapse problem with cold-start data and multi-stage training.
