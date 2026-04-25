---
slug: turboquant
title: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
authors:
  - Amir Zandieh
  - Majid Daliri
  - Majid Hadian
  - Vahab Mirrokni
venue: arXiv
year: 2025
arxiv_id: "2504.19874"
doi: "10.48550/arXiv.2504.19874"
url: https://arxiv.org/abs/2504.19874
discovered_via: maintainer
discovered_date: 2026-04-25
status: implemented
themes:
  primary: efficient-inference
  secondary:
    - retrieval
tags:
  - vector-quantization
  - kv-cache
  - qjl
  - random-rotation
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microturboquant
  target_path: 03-systems/microturboquant.py
  target_tier: 03-systems
  batch_label: efficient-inference-seed
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microturboquant.py
    script_slug: microturboquant
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v2.0.0
lesson:
  path: no-magic-papers/lessons/turboquant.md
  status: published
dependencies_on_other_papers: []
---

## TL;DR

TurboQuant makes vector quantization online and data-oblivious by randomly rotating each vector, scalar-quantizing the rotated coordinates, and adding a one-bit residual sketch when unbiased inner products matter.

## Problem

KV-cache compression and vector search both need low-bit vectors that preserve geometry. Data-dependent quantizers can work well after calibration, but that is a poor fit for streaming KV-cache entries and changing retrieval corpora. Simple scalar quantizers are fast but waste bits on worst-case coordinate structure, and MSE-optimized quantizers can bias inner-product estimates.

## Contribution

The paper gives two related quantizers. The first targets MSE: rotate a unit vector with a shared random orthogonal matrix, quantize each coordinate with a scalar Lloyd-Max codebook tuned to the rotated-coordinate distribution, then rotate back during dequantization. The second targets inner products: run the MSE quantizer with one fewer bit and apply a one-bit Quantized Johnson-Lindenstrauss transform to the residual so the final estimator is unbiased.

## Method summary

- Normalize or store vector norms separately so the main analysis works on unit vectors.
- Sample a random rotation. After rotation, each coordinate has a distribution determined by dimension, not by the original vector direction.
- Precompute scalar quantization centroids for that coordinate distribution and the chosen bit width.
- Encode by rotating, assigning each coordinate to its nearest centroid, and storing the centroid indices plus the norm metadata.
- Decode by replacing indices with centroids and applying the inverse rotation.
- For inner products, encode the residual between the original vector and MSE reconstruction with one-bit QJL signs.
- Estimate inner products by combining the MSE reconstruction term with the unbiased residual estimator.

## Key results

The authors prove distortion upper bounds for both MSE and unbiased inner-product estimation and lower bounds showing the rates are near-optimal up to a small constant factor. Empirically, TurboQuant matches the full-precision Needle-In-A-Haystack score of 0.997 for Llama-3.1-8B-Instruct under the reported compression setting. On LongBench, the paper reports quality neutrality at 3.5 bits per channel and marginal degradation at 2.5 bits per channel. In nearest-neighbor search experiments, TurboQuant's quantization time is effectively zero compared with Product Quantization and RabitQ: for 4-bit quantization, Table 2 reports 0.0007s at d=200, 0.0013s at d=1536, and 0.0021s at d=3072.

## Relation to existing work

TurboQuant sits between simple online scalar quantization and data-dependent product quantization. Like QJL, it is data-oblivious and supports unbiased inner-product estimation; unlike pure one-bit sketches, it also provides a high-quality reconstruction path through scalar centroids. It is closest in spirit to rotation-based quantization work such as QuaRot and PolarQuant, but the paper frames the method as a vector-quantization rate-distortion result rather than only an LLM systems trick.

## Implementation notes

A `no-magic` implementation should not attempt to reproduce the GPU KV-cache benchmark. The pedagogical core is the pipeline: random orthogonal rotation, scalar quantization, inverse rotation, and optional QJL residual signs. Use small dense vectors where Gram-Schmidt is affordable in pure Python. Compare against absmax quantization on anisotropic vectors so the benefit of rotating before quantizing is visible. Keep QJL as a separate residual path; otherwise readers miss why the MSE quantizer and inner-product quantizer are not the same object.

## Open questions

The paper's production story depends on fast structured rotations, outlier-channel handling, and accelerator implementation details. Those are systems decisions around the core algorithm, not necessary for the first educational implementation.
