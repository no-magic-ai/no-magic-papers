---
slug: kv-cache
title: "Efficiently Scaling Transformer Inference"
authors:
  - Reiner Pope
  - Sholto Douglas
  - Aakanksha Chowdhery
  - Jacob Devlin
  - James Bradbury
  - Anselm Levskaya
  - Jonathan Heek
  - Kefan Xiao
  - Shivani Agrawal
  - Jeff Dean
venue: MLSys
year: 2023
arxiv_id: "2211.05102"
doi: null
url: https://arxiv.org/abs/2211.05102
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - kv-cache
  - inference
  - autoregressive-decoding
  - memory-bound
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microkv
  target_path: 03-systems/microkv.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microkv.py
    script_slug: microkv
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: transformer
---

## TL;DR

The paper analyzes the scaling behavior of transformer inference, formalizes the KV-cache as the central memory artifact of autoregressive decoding, and gives partitioning strategies for prefill and decode phases that match the right parallelism scheme to the right phase, achieving near-roofline efficiency on TPU and GPU clusters.

## Problem

Transformer inference has two distinct phases with very different compute-memory profiles. The prefill phase processes the prompt in parallel — compute-bound and well-served by tensor parallelism. The decode phase emits one token at a time, conditioned on the cached keys and values from all previous positions. Decode is dominated by KV-cache reads from HBM, not by compute, and grows linearly with context length. Earlier serving systems treated inference as a single mode, optimizing for one phase and paying for it in the other. The KV-cache itself was treated as an implementation detail rather than a first-class object whose layout and partitioning matter as much as the model weights.

## Contribution

The paper makes the KV-cache central to inference analysis. It formalizes how the KV-cache scales with context length, batch size, and head count, and shows that for typical context lengths (1k-32k tokens) at large model scales the KV-cache rivals or exceeds the model parameters in memory footprint. It then derives partitioning strategies for the two phases. Prefill: tensor parallelism (Megatron-style) is optimal because compute dominates. Decode: a different partitioning that minimizes KV-cache reads per generated token. The paper analyzes weight stationary, activation stationary, and KV-cache stationary placements and shows the optimal choice depends on batch size — small batches favor weight stationary, large batches favor KV-cache stationary. The combined recipe achieves throughput close to the memory-bandwidth roofline at varying batch sizes and context lengths.

## Method summary

- KV-cache: for each transformer layer, store the keys and values of all previous tokens; per-layer cache size is 2 · num_heads · head_dim · seq_len · sizeof(dtype) per request.
- Prefill: process the entire prompt in parallel with standard tensor-parallel attention; populate the KV-cache.
- Decode: per generated token, run attention against the cached K, V (no recomputation) plus the new query.
- Partitioning analysis: model an inference layer as a graph of weight tensors, KV-cache tensors, and activations; choose a partition that minimizes total memory traffic per generated token under the cluster's interconnect topology.
- Three regimes: weight stationary (replicate KV-cache, partition weights — best for very small batches where KV-cache fits in memory), activation stationary (partition KV-cache by head, replicate weights — best for medium batches), KV-cache stationary (partition KV-cache across devices in a way that aligns with attention's data flow — best for large batches).
- Multi-query and grouped-query attention (Shazeer 2019, Ainslie et al. 2023) reduce KV-cache size by sharing keys and values across heads, shifting the regime boundaries.

## Key results

The paper achieves ~70% of theoretical peak FLOP/s for prefill and ~90% of memory-bandwidth roofline for decode on PaLM-540B inference across a range of batch sizes and context lengths. The optimal partitioning depends on the regime: at batch size 1 with context 8k, the KV-cache exceeds the model parameters and dominates partitioning choice; at batch size 256 with short contexts, weights dominate. The paper also documents that multi-query attention reduces decode-phase memory traffic by ~10× at the cost of small quality regression, motivating adoption in many deployed systems.

## Relation to existing work

The paper formalizes inference patterns that earlier work had used informally. It connects to FlashAttention (Dao et al. 2022, this card cites it) which applies similar memory-IO analysis at the kernel level, to PagedAttention (Kwon et al. 2023) which addresses KV-cache memory fragmentation across requests, and to multi-query / grouped-query attention (Shazeer 2019, Ainslie 2023) which shrinks the per-request KV-cache. The KV-cache concept itself is older than this paper — it was used in GPT-2 inference and earlier — but no prior paper had given it the central analytical role. Subsequent work on KV-cache compression (KIVI, MiniCache), eviction (StreamingLLM, H2O), and offloading (FlexGen) all build on the framing this paper provides.

## Implementation notes

A pedagogical script can demonstrate the KV-cache by implementing autoregressive decoding two ways: naive recomputation (re-run attention over the full prefix at each step) and cached (store K and V per step, only compute the new query's attention). Minimum viable cache: a list per layer of (key, value) tensors that grows by one position per generated token; the attention operation reads the whole list to compute the new step's attention scores. Pitfalls: forgetting to detach cached tensors from the gradient graph (turns inference into a training-graph leak); growing the cache as a Python list with O(N) reallocations rather than preallocating to max sequence length; failing to apply RoPE or other position-dependent transforms to cached K (the cache stores K AFTER position transformation, not before). Useful diagnostic: time generation with and without cache at various sequence lengths — the no-cache version's per-token cost grows linearly while the cached version stays constant per token.

## Open questions

KV-cache compression and eviction are active areas; this paper documents the cost but does not propose lossy reduction. The interaction between KV-cache layout, multi-query/grouped-query attention, and PagedAttention block management is non-trivial and is still being optimized in production systems.
