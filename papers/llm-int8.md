---
slug: llm-int8
title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"
authors:
  - Tim Dettmers
  - Mike Lewis
  - Younes Belkada
  - Luke Zettlemoyer
venue: NeurIPS
year: 2022
arxiv_id: "2208.07339"
doi: null
url: https://arxiv.org/abs/2208.07339
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - quantization
  - int8
  - mixed-precision
  - outlier-features
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microquant
  target_path: 03-systems/microquant.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microquant.py
    script_slug: microquant
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

LLM.int8() quantizes most transformer matrix multiplications to 8-bit while keeping a small set of "outlier" feature dimensions in 16-bit, recovering the full quality of 16-bit inference at half the memory cost — the first quantization recipe that survives intact at 6.7-billion-parameter scale.

## Problem

Standard 8-bit quantization (uniform per-tensor or per-channel scaling) had worked for vision and small language models but failed spectacularly on large language models. Transformers above ~2.7 billion parameters develop emergent feature outliers — a small set of feature dimensions whose magnitudes are 6-100× larger than typical features and dominate attention and FFN computations. Naive INT8 quantization clamps or scales these outliers down, which in turn destroys the model's predictions on the systematic patterns those outliers encode. The paper asks whether one can preserve quality at scale while keeping the memory and bandwidth wins of INT8.

## Contribution

The paper introduces LLM.int8(): a mixed-precision matrix-multiplication scheme that quantizes most of the matrix in INT8 but isolates the columns of the input that contain outlier features and computes their contribution in FP16. The decomposition is per-column for the activation tensor: for each column, check the maximum absolute value; if above a threshold (the paper uses 6.0), route that column through an FP16 path and the corresponding rows of the weight matrix through an FP16 path. The remaining columns and rows go through INT8 with vector-wise scaling (one quantization scale per output channel). The two paths' partial sums are added in FP32 to produce the final output. Because outlier features are a tiny fraction of total dimensions (typically < 0.1% even at 175B), the FP16 path adds negligible compute while preserving model quality. The paper demonstrates the recipe on OPT models from 125M up to 175B and shows zero perplexity degradation across the scale.

## Method summary

- For matmul Y = X · W, with X of shape (m, k) and W of shape (k, n).
- Compute per-column maxes of X; mark columns with absmax > threshold T as "outlier" indices.
- Decompose X into X_outlier (FP16, the marked columns) and X_normal (INT8, the rest); decompose W into the corresponding W_outlier (FP16 rows) and W_normal (INT8 rows).
- INT8 path: vector-wise quantize X_normal per row and W_normal per column to INT8; matmul to produce INT32 partial sums; dequantize to FP16 with the saved scales.
- FP16 path: standard X_outlier · W_outlier in FP16.
- Sum the two partial outputs in FP32; cast to FP16 for downstream layers.
- Threshold T = 6.0 in the paper; outlier sparsity < 0.1% at 175B scale; INT8 path covers > 99.9% of total compute.

## Key results

LLM.int8() preserves zero-shot accuracy of OPT models at scales from 125M to 175B parameters with no measurable degradation, while reducing inference memory by approximately 50%. Naive 8-bit quantization without outlier handling drops accuracy by tens of points on OPT-6.7B and larger; with outlier handling, the gap closes to within noise. The paper documents the emergence of outlier features as a function of scale (rare below 1B, ubiquitous above 6.7B) and shows the same dimensions remain outliers across many input sequences — meaning the outlier-column identification can be amortized.

## Relation to existing work

LLM.int8() sits in the post-training-quantization line: GPTQ (Frantar et al. 2022), AWQ (Lin et al. 2023), and SmoothQuant (Xiao et al. 2022) take different approaches to the same outlier problem. GPTQ uses second-order information to push more compute into INT4 with calibration; AWQ scales weight magnitudes per channel; SmoothQuant migrates outlier magnitude from activations to weights via a learned per-channel rescaling. LLM.int8() differs by handling the outlier dimensions at runtime via decomposition rather than via static recipe; this trades a small amount of FP16 compute for a quality guarantee across model scales. Quantization-aware training methods (Jacob et al. 2018, the foundational INT8 paper) and QLoRA (which uses NF4 + LoRA on top of a frozen quantized base) target adjacent points in the trade-off space.

## Implementation notes

A pedagogical script can implement INT8 matmul with vector-wise scaling and demonstrate the outlier failure mode on a tiny transformer. Minimum viable implementation: per-tensor (or per-row/per-column) scale calibration, INT8 quantization, fake-INT8 forward (scale, round, clip, then dequantize for the demo), and a comparison run with and without an outlier path. Pitfalls: forgetting that round-to-nearest-int needs to handle ties consistently; using per-tensor scales when per-channel scales are needed for accuracy; underestimating how much outlier handling matters — the demo will look fine without outliers and degrade catastrophically once a synthetic outlier is injected. Useful diagnostic: plot the per-channel max of activations across a few thousand tokens and observe the heavy-tailed distribution that motivates outlier handling.

## Open questions

The paper does not analyze why outlier features emerge with scale; subsequent work (Bondarenko et al. 2023, others) has investigated mechanistic causes in attention. INT4 quantization at scale was an open problem at the time of the paper and was addressed by GPTQ and successors.
