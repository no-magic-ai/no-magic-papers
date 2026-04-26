---
slug: roofline
title: "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
authors:
  - Samuel Williams
  - Andrew Waterman
  - David Patterson
venue: Communications of the ACM
year: 2009
arxiv_id: null
doi: "10.1145/1498765.1498785"
url: https://dl.acm.org/doi/10.1145/1498765.1498785
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - roofline
  - performance-model
  - arithmetic-intensity
  - memory-bandwidth
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microroofline
  target_path: 03-systems/microroofline.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microroofline.py
    script_slug: microroofline
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The roofline model plots a kernel's achievable performance against its arithmetic intensity (FLOPs per byte of memory traffic): the achievable performance is bounded above by min(peak FLOP/s, bandwidth × intensity), so a kernel either lives under the bandwidth-limited "slope" of the roof or under the compute-limited "ceiling," and the model immediately reveals which.

## Problem

Performance analysis on multicore CPUs and accelerators is dominated by two resources — peak floating-point throughput and memory bandwidth — but raw numbers from hardware specs do not tell a programmer where a particular kernel will land. A kernel that does many FLOPs per byte of data movement is compute-bound; a kernel that does few FLOPs per byte is memory-bound. Without a unifying picture, programmers spent time optimizing the wrong thing — squeezing FLOPs from a memory-bound kernel that the bandwidth ceiling already capped, or trying to hide memory latency on a compute-bound kernel that did not need it.

## Contribution

The paper introduces the roofline model: a single log-log plot whose x-axis is arithmetic intensity (FLOPs per byte of DRAM traffic) and whose y-axis is achievable performance (GFLOP/s). The "roof" is the piecewise function min(peak compute, peak bandwidth × intensity). Any kernel's measured performance is plotted as a point; its position relative to the roof reveals whether it is compute-bound or memory-bound and how close it is to the architectural ceiling. The model also accommodates secondary "ceilings" for specific optimizations not yet applied — without SIMD, without TLP, without ILP — so the gap between observed performance and the next ceiling identifies the highest-payoff optimization. The paper applies the model to several SpMV, stencil, and dense matrix kernels on contemporary multicore CPUs.

## Method summary

- Measure or specify the machine's peak compute throughput P_c (FLOP/s) and peak DRAM bandwidth B (bytes/s).
- For a kernel, compute its arithmetic intensity I = FLOPs / bytes-moved-from-DRAM.
- Achievable performance bound: A(I) = min(P_c, B · I).
- Plot the bound on a log-log graph; the curve is a slope of B up to the knee at I_ridge = P_c / B, then a flat ceiling at P_c.
- Plot measured kernel performance as a point at (I, observed FLOP/s). Distance below the roof reveals the optimization gap.
- Add secondary ceilings for incremental optimizations: e.g. a "no SIMD" ceiling at P_c / SIMD_width, a "no balanced FP units" ceiling, a "no software prefetching" bandwidth ceiling. The gaps between successive ceilings rank the highest-impact optimizations.

## Key results

The paper applies roofline analysis to seven scientific kernels (SpMV, stencils, FFT, dense matmul, etc.) on the AMD Opteron, Intel Clovertown, and Sun Niagara2 architectures. Each kernel-machine pair lands at a specific arithmetic intensity: SpMV is memory-bound (low intensity, near the bandwidth slope), dense matmul is compute-bound (high intensity, near the compute ceiling). The model's prescriptions match measured speedups: applying the optimizations whose ceilings sit just above a kernel's current point delivers the predicted gain, while applying optimizations beyond the binding ceiling produces no benefit.

## Relation to existing work

Earlier performance models — STREAM benchmark for memory bandwidth alone, or peak-FLOP-vs-realized-FLOP plots without intensity — captured one resource at a time. Roofline unifies them and exposes the trade-off. Subsequent work extended it: hierarchical roofline models add ceilings for each level of cache, not just DRAM (Ilic et al. 2014); the cache-aware roofline lets programmers see which level of the memory hierarchy is binding. The roofline framework is now standard in HPC performance analysis (NERSC's CrayPat, Intel's Advisor) and increasingly used for GPU kernel analysis (Yang et al. 2018), including for deep-learning kernels — FlashAttention's design, for instance, can be read as moving an attention kernel from memory-bound to compute-bound by raising arithmetic intensity through tiling.

## Implementation notes

A pedagogical script can compute and plot a roofline for several toy kernels: matrix multiply (high intensity), vector add (very low intensity), elementwise activation (low intensity). Minimum viable implementation: estimate FLOPs and bytes-moved analytically per kernel; measure wall-clock to derive observed FLOP/s; plot the bandwidth slope and compute ceiling on log-log axes; mark each kernel's point. Pitfalls: counting bytes incorrectly — only count bytes that traverse DRAM, not bytes that hit cache; underestimating peak bandwidth by using textbook numbers rather than STREAM measurements; comparing a kernel that does single-precision against a peak-FLOP measured for double-precision. Useful diagnostic: instrument a deep-learning op (e.g. attention vs matmul vs softmax) and place each on the roofline — the plot makes immediately visible which ops are memory-bound and would benefit from tiling vs which are compute-bound and need ILP/SIMD work.

## Open questions

The original model assumes a single bandwidth and compute ceiling; modern systems with HBM, NVLink, and accelerator-specific tensor cores need extended versions. The model also conflates achievable peak with sustained peak; for some workloads the difference is large and a more careful empirical-roof measurement is needed.
