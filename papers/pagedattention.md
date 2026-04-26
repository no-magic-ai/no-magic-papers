---
slug: pagedattention
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
authors:
  - Woosuk Kwon
  - Zhuohan Li
  - Siyuan Zhuang
  - Ying Sheng
  - Lianmin Zheng
  - Cody Hao Yu
  - Joseph E. Gonzalez
  - Hao Zhang
  - Ion Stoica
venue: SOSP
year: 2023
arxiv_id: "2309.06180"
doi: null
url: https://arxiv.org/abs/2309.06180
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary:
    - long-context
tags:
  - vllm
  - kv-cache
  - paging
  - serving
  - memory-management
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micropaged
  target_path: 03-systems/micropaged.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/micropaged.py
    script_slug: micropaged
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: transformer
---

## TL;DR

PagedAttention applies the OS virtual-memory paging idea to KV-cache storage during LLM serving: KV blocks are stored in fixed-size pages, mapped through a per-sequence block table, and shared across requests where prefixes match — eliminating the internal fragmentation that wastes 60-80% of memory in conventional LLM servers.

## Problem

Production LLM servers like Hugging Face TGI and FasterTransformer reserved one contiguous chunk of GPU memory per request, sized for the request's maximum possible sequence length. Most requests finished far below that maximum, so the reserved space sat idle — fragmentation that wasted most of the available memory and forced the server to batch fewer concurrent requests than the GPU could otherwise handle. Beam search and parallel sampling made this worse: each beam needed its own KV-cache copy even though the beams shared a common prefix. The paper asks whether the OS virtual-memory pattern — paging plus page tables — can eliminate this fragmentation in LLM serving.

## Contribution

The paper introduces PagedAttention and the vLLM serving system built around it. KV-cache is stored in fixed-size physical blocks (each holding a constant number of tokens, typically 16). Each request has a block table mapping its logical token positions to physical block IDs; the attention kernel reads the table to find the right blocks. Three benefits follow. First, no internal fragmentation: a request consumes exactly the number of blocks it needs, rounded up to the block size, not the worst-case allocation. Second, copy-on-write block sharing: parallel beams or shared prefixes (system prompt, few-shot exemplars) reference the same physical blocks until one diverges. Third, dynamic resizing: a sequence can grow without preallocation. The paper reports vLLM achieves 2-4× higher throughput than the strongest prior LLM servers at the same latency.

## Method summary

- Allocate GPU memory as a pool of fixed-size KV-cache blocks (each block holds K tokens for one layer, one head dimension, KV pair).
- Per-request block table: an array mapping logical KV positions to physical block IDs.
- Attention kernel: given a query, walk the block table to fetch the relevant key and value blocks; compute attention with on-the-fly block addressing rather than over a contiguous tensor.
- Block allocator: maintain a free-block list; allocate on demand as sequences extend.
- Copy-on-write sharing: when a request forks (beam search, sampling K outputs), its block table is copied but the physical blocks are shared until a write would diverge them; on write, the affected block is duplicated and the table updated.
- Preemption: when memory pressure exceeds threshold, the scheduler can swap out a request's blocks to host memory and resume later.

## Key results

vLLM with PagedAttention achieves 2-4× higher request throughput than the prior state-of-the-art LLM servers (TGI, FasterTransformer) at comparable latency, on workloads from short single-turn chat to long-context generation. On parallel sampling, the prefix-sharing speedup is even larger (up to 7-9× throughput improvement on workloads with long shared prompts). Memory utilization rises from 20-40% in conventional servers to 90%+ with vLLM. The paper also documents that the kernel-level overhead of indirect block lookups is small relative to the gains from higher batching.

## Relation to existing work

PagedAttention applies a classic OS technique (virtual memory paging) to the LLM-serving domain. It contrasts with continuous-batching servers (Orca by Yu et al. 2022) that improved scheduling but kept the contiguous-allocation pattern, and with naive KV-cache compression schemes that traded quality for memory. Subsequent work refined the recipe: SGLang and TensorRT-LLM adopt PagedAttention-style block management; speculative-decoding integrations require careful block-sharing reasoning; PD-disaggregated serving (prefill and decode on separate GPUs) builds atop the same memory model. The block-table primitive is now standard infrastructure in modern LLM-serving stacks.

## Implementation notes

A pedagogical script can implement a CPU-side PagedAttention demo: allocate a block pool as a 3D array, maintain a per-sequence block-table dict, write a small attention loop that walks the table. The toy version does not need GPU kernels — the algorithmic content is the indirection layer and the copy-on-write logic. Pitfalls: forgetting to update the block table when a sequence grows past a block boundary; failing to refcount shared blocks (forking creates a shared block; freeing a sequence must not free blocks still referenced by another); using a block size that does not match the attention kernel's tile size in production introduces unaligned memory accesses. Useful demo: spawn two requests with a shared 1000-token prefix and confirm the block tables share the same physical blocks until both diverge.

## Open questions

The paper does not address quality/throughput trade-offs of approximate KV-cache (quantization, eviction); subsequent work — KV-cache quantization (KIVI, Mixed-Precision KV), eviction policies (StreamingLLM, H2O) — handles this on top of PagedAttention. The block size is a hyperparameter that interacts with hardware tiling and request-length distribution.
