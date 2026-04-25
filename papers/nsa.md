---
slug: nsa
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
authors:
  - Jingyang Yuan
  - Huazuo Gao
  - Damai Dai
  - Junyu Luo
  - Liang Zhao
  - Zhengyan Zhang
  - Zhenda Xie
  - Y. X. Wei
  - Lean Wang
  - Zhiping Xiao
  - Yuqing Wang
  - Chong Ruan
  - Ming Zhang
  - Wenfeng Liang
  - Wangding Zeng
venue: arXiv
year: 2025
arxiv_id: "2502.11089"
doi: "10.48550/arXiv.2502.11089"
url: https://arxiv.org/abs/2502.11089
discovered_via: seed-papers-proposal
discovered_date: 2026-04-25
status: summarized
themes:
  primary: architecture
  secondary:
    - long-context
    - efficient-inference
tags:
  - sparse-attention
  - long-context
  - gqa
  - hardware-aligned
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micronsa
  target_path: 03-systems/micronsa.py
  target_tier: 03-systems
  batch_label: sparse-attention-seed
  review_date: null
implementations: []
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

NSA replaces full attention with three trainable sparse branches: compressed global blocks, selected fine-grained blocks, and a sliding local window, designed so sparse attention is efficient during both training and inference.

## Problem

Sparse attention often looks good on paper but fails to deliver practical speedups. Some methods only help decoding or prefilling, some scatter KV-cache reads in hardware-hostile patterns, and many are post-hoc inference tricks that a full-attention pretrained model was never optimized to use. Long-context models need sparsity that is native to training and aligned with accelerator memory access.

## Contribution

The paper proposes Native Sparse Attention, a hierarchical sparse attention architecture built into pretraining. For each query, NSA constructs a smaller set of representation keys and values from three sources: compressed coarse blocks for global context, selected token blocks for high-value fine detail, and a local sliding window. A learned gate combines branch outputs. The selection path is blockwise and shared across GQA/MQA groups so it maps to efficient KV loading rather than random per-head token fetches.

## Method summary

- Compress sequential KV blocks into coarse representations with an MLP and intra-block position encoding.
- Use attention scores over compressed blocks to derive importance scores for selection blocks.
- Select top-ranked contiguous blocks, preserving fine-grained keys and values only where the compressed path predicts they matter.
- Maintain a sliding window branch for local dependencies so compression and selection do not have to learn short-range shortcuts.
- Combine compressed, selected, and sliding outputs through a learned gate.
- Implement sparse selection around GQA groups so query heads sharing KV also share selected blocks, reducing memory traffic.
- Train the sparse attention module end to end rather than applying sparsity only after full-attention pretraining.

## Key results

NSA is pretrained on a 27B-parameter backbone over 260B tokens and compared with a full-attention baseline. Table 1 reports a higher average score for NSA on general benchmarks: 0.456 versus 0.443 for Full Attention. Table 2 reports the highest LongBench average among compared methods: NSA reaches 0.469, above Full Attention at 0.437 and Exact-Top at 0.423. For reasoning SFT, Table 3 reports AIME scores of 0.121 at 8k and 0.146 at 16k for NSA-R, compared with 0.046 and 0.092 for Full Attention-R. The efficiency section reports up to 9.0x forward and 6.0x backward speedup at 64k context, and an expected 11.6x decoding speedup from reduced KV memory access.

## Relation to existing work

NSA contrasts with KV eviction, query-aware selection, hashing, clustering, and exact top-k sparse attention. Its main claim is not only that fewer tokens are attended to, but that the sparse pattern is trainable, blockwise, GQA-aware, and implemented around arithmetic intensity. It complements FlashAttention-style tiling by preserving contiguous block access rather than replacing hardware-aware attention kernels with scattered token gathers.

## Implementation notes

A `no-magic` implementation should model the three-branch data flow rather than GPU kernels. Use a toy causal sequence, divide keys and values into blocks, compute compressed block scores, select top blocks, add a sliding local window, and gate the three outputs. Show the difference between per-head random token selection and group-shared block selection by counting memory reads. The CPU version should explicitly separate algorithmic sparsity from hardware speed; it can demonstrate fewer score computations and cleaner memory access without pretending to reproduce Triton performance.

## Open questions

The paper's strongest speed claims depend on specialized kernels, GQA/MQA deployment, and accelerator arithmetic intensity. A pedagogical implementation can preserve the sparse attention contract but not the production throughput story.
