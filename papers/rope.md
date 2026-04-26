---
slug: rope
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors:
  - Jianlin Su
  - Yu Lu
  - Shengfeng Pan
  - Bo Wen
  - Yunfeng Liu
venue: arXiv
year: 2021
arxiv_id: "2104.09864"
doi: null
url: https://arxiv.org/abs/2104.09864
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: long-context
  secondary:
    - architecture
tags:
  - rope
  - rotary-position-embedding
  - relative-position
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microrope
  target_path: 03-systems/microrope.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microrope.py
    script_slug: microrope
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: transformer
---

## TL;DR

Rotary Position Embedding (RoPE) encodes absolute position by rotating each query and key vector by an angle proportional to its position; the inner product of two rotated vectors then depends only on their relative position, giving relative-position attention through pure absolute-position math with no additional parameters.

## Problem

Standard Transformers add learned or sinusoidal position embeddings to token embeddings before the first layer. This works but couples the embedding magnitude to the position signal and gives no clean way to extrapolate to longer sequences than seen during training. Earlier relative-position schemes (Shaw et al. 2018, T5 buckets) added a per-head bias term to the attention score, which improved long-range modeling but required new trainable parameters and modified the attention computation. The paper asks whether one can encode position so that the attention score naturally depends only on the relative distance, without adding any parameters or changing the attention formula.

## Contribution

RoPE replaces additive position embeddings with a rotation applied to query and key vectors before the dot product. For position m and a 2D pair of feature dimensions, RoPE multiplies the pair by a 2D rotation matrix R_θm where θ depends on the dimension. Generalizing to a d-dimensional vector, RoPE applies pairwise rotations across d/2 pairs, each with its own frequency. The key algebraic property: ⟨R_θm · q, R_θn · k⟩ = ⟨q, R_θ(n-m) · k⟩ — the inner product depends only on the relative position n - m. The attention score therefore inherits relative-position behavior for free, with no additional parameters and no change to the attention formula. The paper applies RoPE to standard Transformers (RoFormer) and demonstrates improved performance on long-text and Chinese language modeling tasks.

## Method summary

- Define a base frequency θ (the paper uses θ_i = 10000^(-2i/d) for the i-th feature pair, matching the sinusoidal-position-embedding base).
- For position m, construct a block-diagonal rotation matrix R_m built from 2D rotation blocks: each pair of feature dimensions (2i, 2i+1) is rotated by angle m · θ_i.
- In each attention layer, apply R_m to the query at position m and R_n to the key at position n before computing the dot product.
- Implementation shortcut: R_m can be applied via element-wise multiplication with cos and sin tables — no explicit matrix construction needed. The cos and sin tables can be precomputed once for all positions up to the maximum sequence length.
- Standard scaled dot-product attention then proceeds normally; values V are not rotated.

## Key results

RoFormer with RoPE matches or exceeds standard Transformer baselines on multiple language-modeling and machine-translation benchmarks, with particularly strong results on long-document tasks where the relative-position property matters. Subsequent work (especially LLaMA, GPT-NeoX, Mistral, DeepSeek) adopted RoPE as the default position encoding for decoder-only LLMs at scale, and a series of variants — NTK-aware scaling (Bowen Peng 2023), YaRN (Peng et al. 2023), Position Interpolation (Chen et al. 2023) — extend pretrained RoPE-based models to longer contexts at inference time by adjusting the frequency schedule.

## Relation to existing work

RoPE replaces additive position embeddings (sinusoidal in the original Transformer; learned in BERT/GPT-1/GPT-2) with multiplicative rotations. It contrasts with attention-bias-based relative position (Shaw et al. 2018, T5 buckets, ALiBi by Press et al. 2022): those add a per-pair bias to the attention score, while RoPE achieves relative behavior via the inner-product geometry of rotated vectors. ALiBi and RoPE both extrapolate beyond training length better than absolute embeddings, with RoPE being the dominant choice in modern LLMs. The position-interpolation and NTK-aware scaling techniques exploit RoPE's specific frequency structure to extend context length post hoc.

## Implementation notes

A pedagogical script can implement RoPE for a small attention head and demonstrate the relative-position property numerically. Minimum viable implementation: precompute cos and sin tables of shape [max_seq_len, d/2]; in attention forward, apply (q_even, q_odd) → (q_even·cos - q_odd·sin, q_even·sin + q_odd·cos) — the standard 2D rotation. Pitfalls: applying RoPE to V (do not — only Q and K rotate); applying it after the softmax instead of before (it must rotate the inputs to the dot product); using a wrong frequency base for tasks at very different scales (the choice of 10000 is essentially convention from sinusoidal embeddings). Useful demo: compute attention scores with rotated Q at position m=0 and K at position n=k for various k, then with m=5 and n=5+k, and verify the scores are identical for the same relative offset k.

## Open questions

The frequency choice θ_i = 10000^(-2i/d) is heuristic; tasks at different scales may benefit from different bases (this is the territory NTK-aware and YaRN scaling exploits). The paper does not address how RoPE interacts with other position-related mechanisms (sliding-window attention, Mamba-style recurrence); subsequent architectures combine these in various ways.
