---
slug: speculative-decoding
title: "Fast Inference from Transformers via Speculative Decoding"
authors:
  - Yaniv Leviathan
  - Matan Kalman
  - Yossi Matias
venue: ICML
year: 2023
arxiv_id: "2211.17192"
doi: null
url: https://arxiv.org/abs/2211.17192
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - speculative-decoding
  - draft-model
  - rejection-sampling
  - inference-speedup
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microspeculative
  target_path: 03-systems/microspeculative.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microspeculative.py
    script_slug: microspeculative
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Speculative decoding accelerates autoregressive inference by using a small fast draft model to propose K future tokens, then verifying them in parallel with one forward pass of the large target model and using a rejection-sampling rule that yields a sample distributed exactly as the target's would have been — giving 2-3× wall-clock speedup with no quality loss.

## Problem

Autoregressive decoding from a large language model is sequential: each token depends on the previous, so generation cost is K forward passes for K tokens. The bottleneck is memory bandwidth, not compute — a forward pass at batch size 1 transfers the entire weight matrix from HBM but uses a tiny fraction of the GPU's compute. Methods that exploit this slack (larger batch sizes, KV-cache reuse) help when many requests are concurrent but not for a single fast request. The paper asks whether one can use the unused compute to verify many draft tokens in parallel and recover the savings as wall-clock speedup, without changing the output distribution.

## Contribution

The paper introduces speculative decoding. A small draft model M_q proposes K candidate tokens autoregressively (cheap, since M_q is small). The large target model M_p then runs one forward pass on the K candidates in parallel, computing P(x_i | prefix) for each candidate position. A rejection-sampling rule decides per token whether to accept the candidate: accept x_i with probability min(1, P(x_i | prefix) / Q(x_i | prefix)); if rejected, sample from the residual distribution P_resid(x) ∝ max(0, P(x) - Q(x)). The first rejected position truncates the speculation; accepted prefix plus one residual sample becomes the next K' ≤ K accepted tokens. The paper proves that under this rule the output distribution exactly matches sampling from M_p alone — the speedup is free in quality terms. The paper demonstrates 2-3× speedup on T5-XXL inference using a small T5-small as the draft.

## Method summary

- Pick a draft model M_q (small, fast) and a target model M_p (large, accurate). M_q must share or be compatible with M_p's tokenizer.
- For each generation step batch:
  - Generate K candidate tokens from M_q autoregressively given the current prefix.
  - Run M_p's forward pass on the prefix + K candidates in parallel; this produces target probabilities at all K+1 positions in one pass.
  - For i = 0 .. K-1, decide: accept candidate_i with probability min(1, P_p(x_i | prefix_i) / P_q(x_i | prefix_i)); reject otherwise.
  - On the first rejection at position j, sample a fresh token from the residual distribution P_resid(x) ∝ max(0, P_p(x | prefix_j) - P_q(x | prefix_j)); truncate the speculation, append accepted_prefix + residual_sample.
  - If all K accepted, also append a sample from M_p's distribution at the next position (free, since the forward pass also computed it).
- Iterate until end of generation. Per outer step, K+1 tokens are generated with one M_p call.
- Speedup depends on K and on draft acceptance rate; the paper analyzes the optimal K for various model-pair sizes.

## Key results

On T5-XXL with a T5-small draft, speculative decoding produces 2-3× end-to-end speedup with zero quality change measured by exact-match output distribution. On other model pairs the speedup ranges from 1.5× to 4× depending on draft accuracy and target latency. The paper also gives a theoretical analysis predicting speedup as a function of expected acceptance length, and shows the analysis matches measured wall-clock improvements.

## Relation to existing work

Two contemporaneous papers introduced speculative decoding independently: Leviathan, Kalman, Matias 2023 (this paper) and Chen et al. 2023 (arXiv:2302.01318, "Accelerating Large Language Model Decoding with Speculative Sampling"). The two papers' algorithms are essentially identical with different presentations of the rejection rule; both papers are cited together in subsequent work. Predecessor ideas — non-autoregressive decoding, blockwise parallel decoding — also targeted parallelism but did not preserve the output distribution. Subsequent extensions: Medusa (Cai et al. 2024) fuses the draft into the target via extra prediction heads; EAGLE refines the draft architecture; tree-based speculation (SpecInfer, Sequoia) speculates over branching trees rather than linear chains; lookahead decoding does not use a draft model at all but parallelizes via pattern matching. Speculative decoding is now standard in vLLM, TGI, TensorRT-LLM, and most production LLM-serving stacks.

## Implementation notes

A pedagogical script can implement speculative decoding with two small Transformers — a draft (e.g. 2-layer) and a target (e.g. 6-layer) — on a tiny task. Minimum viable loop: draft K tokens autoregressively; run target on the prefix + draft tokens in one parallel pass; iterate the rejection rule per token; emit accepted tokens and residual sample. Pitfalls: computing the residual distribution by subtraction without clamping at zero (gives negative probabilities); forgetting that the target's forward also yields a free extra token at position K (skip it and you lose the bonus); using a draft with a different tokenizer than the target (the rejection probability becomes meaningless). Useful diagnostic: measure mean accepted-tokens-per-iteration; high values (close to K) mean the draft is too small to disagree, low values mean the draft and target disagree often and the speedup is limited.

## Open questions

The paper does not analyze how to choose the draft architecture or training objective; subsequent work proposed many variants. Speculative decoding's interaction with batched serving is also non-trivial — across a batch of requests with different acceptance rates, naive batching can lose the speedup, motivating tree-based and dynamic-batching variants.
