---
slug: flash-attention
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
authors:
  - Tri Dao
  - Daniel Y. Fu
  - Stefano Ermon
  - Atri Rudra
  - Christopher Ré
venue: NeurIPS
year: 2022
arxiv_id: "2205.14135"
doi: null
url: https://arxiv.org/abs/2205.14135
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary:
    - long-context
tags:
  - flash-attention
  - io-aware
  - tiled-softmax
  - kernel-fusion
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microflash
  target_path: 03-systems/microflash.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microflash.py
    script_slug: microflash
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: transformer
---

## TL;DR

FlashAttention computes exact softmax attention by tiling the QK^T matrix into blocks small enough to fit in GPU SRAM, avoiding the O(N²) HBM read/write of the full attention matrix; the result is the same numerically but several times faster and with linear (not quadratic) memory footprint in the sequence length.

## Problem

Standard attention computes the N×N matrix S = QK^T, applies row-wise softmax to get P, then computes the output O = PV. On modern GPUs the bottleneck is not arithmetic but memory bandwidth: the S and P matrices are written to and read from high-bandwidth memory (HBM), and the round trips dominate runtime. Worse, holding S and P in HBM costs O(N²) memory, which is the binding limit on context length. Approximate-attention methods (Performer, Linformer, Reformer) reduce both costs but trade exactness for approximation. The paper asks whether a careful tiling and recomputation strategy can deliver exact attention with sub-quadratic memory and lower wall-clock time.

## Contribution

The paper introduces FlashAttention, an IO-aware attention algorithm. It exploits two facts: (1) GPU SRAM (the on-chip cache) is orders of magnitude faster than HBM, but small (tens of KB per streaming multiprocessor), and (2) softmax can be computed online over blocks of inputs using running max and sum statistics, the same trick used in numerically-stable LogSumExp. FlashAttention tiles Q, K, V into blocks that fit in SRAM, computes attention block by block while maintaining the running softmax statistics, and writes only the final output O back to HBM — never materializing the N×N S or P matrices. The backward pass uses the same tiling plus a recomputation trick: it recomputes S and P from the saved O and softmax statistics rather than storing them, trading additional FLOPs for dramatically less memory.

## Method summary

- Tile Q, K, V row-wise into blocks of B_q queries and B_k keys/values, sized so that each block fits in SRAM with the per-block running statistics.
- Outer loop over Q blocks; for each Q block, iterate over K, V blocks.
- Per K, V block: load into SRAM, compute partial S = QK^T, accumulate into the running softmax (track per-row max and per-row sum) without writing S to HBM.
- After all K, V blocks: scale the accumulated PV by the row-wise softmax denominator and write the final O block to HBM.
- Backward pass: recompute S and P per tile from cached O and the per-row softmax statistics, accumulate gradients into dQ, dK, dV; this trades extra FLOPs for an order-of-magnitude memory reduction.
- Output is bit-identical to standard attention (up to floating-point reassociation): the algorithm is exact, not approximate.

## Key results

On a single A100 GPU, FlashAttention is 2 to 4× faster than the optimized PyTorch attention at sequence length 4096, and 7.6× faster at sequence length 8192. Memory drops from O(N²) to O(N), enabling training and inference at sequence lengths previously infeasible (16K, 32K context). On end-to-end model training (BERT-large, GPT-2), the same speedup translates to 15-25% wall-clock improvements. The paper also shows quality is unchanged — gradients match the standard implementation to numerical precision.

## Relation to existing work

FlashAttention sits in the same family as memory-efficient attention (Rabe & Staats 2021, "Self-Attention Does Not Need O(n²) Memory"), which proved the recomputation idea but did not deliver a fast GPU implementation. Approximate-attention work (Linformer, Performer, Reformer) reduces complexity at the cost of exactness; FlashAttention shows you can have both exact attention and substantial speedup. Subsequent versions — FlashAttention-2 (Dao 2023, parallelism improvements), FlashAttention-3 (Shah et al. 2024, FP8 and Hopper-specific tiling) — refine the implementation further. Mamba and other linear-attention alternatives compete on a different axis: they change the algorithm's complexity rather than its IO pattern.

## Implementation notes

A pedagogical script can implement the FlashAttention forward pass in pure NumPy or pure Python. Minimum viable version: tile Q rows and K, V rows into blocks; loop over blocks while maintaining per-row max and sum; write the final scaled output. The script does not need GPU SRAM to demonstrate the algorithm — the tiling pattern and online softmax are what matter pedagogically. Pitfalls: forgetting to track BOTH per-row max and per-row sum (only one is not enough for a numerically stable rescaling); applying the softmax denominator at the end rather than online during accumulation gives wrong results when blocks span the full key range. A useful comparison: implement standard attention and FlashAttention side by side on a small problem and verify outputs match to floating-point precision.

## Open questions

The paper is engineering-focused; the algorithmic core (online softmax + tiling) is well-understood. Open questions concern hardware co-design: how attention IO patterns interact with newer tensor cores, FP8 formats, and emerging GPU architectures. The FlashAttention-2 and FlashAttention-3 papers address parts of this.
