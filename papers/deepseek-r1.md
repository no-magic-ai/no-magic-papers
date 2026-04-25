---
slug: deepseek-r1
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors:
  - DeepSeek-AI
venue: Nature
year: 2025
arxiv_id: "2501.12948"
doi: "10.1038/s41586-025-09422-z"
url: https://arxiv.org/abs/2501.12948
discovered_via: seed-papers-proposal
discovered_date: 2026-04-25
status: summarized
themes:
  primary: reasoning
  secondary:
    - alignment
tags:
  - reinforcement-learning
  - grpo
  - reasoning
  - distillation
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micror1
  target_path: 02-alignment/micror1.py
  target_tier: 02-alignment
  batch_label: reasoning-seed
  review_date: null
implementations: []
lesson:
  path: no-magic-papers/lessons/deepseek-r1.md
  status: drafted
dependencies_on_other_papers:
  - slug: grpo
---

## TL;DR

DeepSeek-R1 shows that large-scale rule-reward reinforcement learning can elicit long-chain reasoning behaviors from a base model, then uses cold-start data, rejection sampling, and distillation to make the result usable and transferable.

## Problem

Reasoning models before R1 leaned heavily on supervised traces, process supervision, or closed model recipes. That made it unclear whether reasoning behavior came from human-written demonstrations or from the optimization pressure itself. The field also lacked an open recipe showing how pure RL, supervised cleanup, and distillation interact at reasoning-model scale.

## Contribution

The paper separates two artifacts. DeepSeek-R1-Zero applies GRPO directly to DeepSeek-V3-Base with rule-based accuracy and format rewards, showing that self-verification, reflection, and longer test-time reasoning can emerge without supervised fine-tuning. DeepSeek-R1 adds thousands of cold-start reasoning examples, reasoning RL, rejection-sampled SFT data, broad-domain SFT, and a final RL stage to improve readability, language consistency, helpfulness, and harmlessness. The paper then distills R1 trajectories into smaller Qwen and Llama dense models.

## Method summary

- Start from a base model and sample groups of responses per prompt.
- Score responses with rule-based rewards for verifiable tasks such as math and code, plus a format reward for the required reasoning/answer structure.
- Optimize with GRPO, using group-relative reward normalization instead of a learned value model.
- Observe R1-Zero's self-evolution: longer generated reasoning, reflection, and strategy revision emerge during RL.
- Add cold-start reasoning data to stabilize readability and reduce language mixing.
- Use rejection sampling from the RL checkpoint to build roughly 600k reasoning SFT examples and combine them with about 200k non-reasoning examples.
- Distill the larger R1 reasoning behavior into smaller dense models with supervised fine-tuning.

## Key results

DeepSeek-R1-Zero increases AIME 2024 pass@1 from 15.6% to 71.0%, and reaches 86.7% with majority voting. DeepSeek-R1 reports 79.8% pass@1 on AIME 2024, 97.3% on MATH-500, 65.9% on LiveCodeBench, and a 2029 Codeforces rating, broadly matching OpenAI-o1-1217 on the paper's reasoning benchmarks. The distilled models are also strong: DeepSeek-R1-Distill-Qwen-32B reports 72.6% on AIME 2024, 94.3% on MATH-500, and 57.2% on LiveCodeBench.

## Relation to existing work

R1 builds on GRPO from DeepSeekMath and on the broader RLHF/post-training lineage, but its central contrast is against SFT-first reasoning recipes. It rejects process reward models for this setting because fine-grained step correctness is hard to define and model-based PRMs can introduce reward hacking. It also reports MCTS as difficult to scale for open-ended token generation because the search space and value-model requirements become unwieldy.

## Implementation notes

A `no-magic` implementation should not try to train a frontier model. The tractable core is a tiny policy trained on verifiable synthetic tasks with GRPO: sample several answers, apply deterministic correctness and format rewards, normalize rewards within the group, and update the policy against a frozen reference. `microgrpo.py` already covers the algorithmic backbone; `micror1.py` should add the R1-specific recipe around cold start, rule rewards, rejection sampling, and distillation at toy scale.

## Open questions

The paper exposes the shape of the recipe but not enough detail to reproduce the full production run. Data construction, reward distributions, filtering thresholds, safety RL, and infrastructure remain the major non-public components.
