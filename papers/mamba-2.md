---
slug: mamba-2
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
authors:
  - Tri Dao
  - Albert Gu
venue: ICML
year: 2024
arxiv_id: "2405.21060"
doi: "10.48550/arXiv.2405.21060"
url: https://arxiv.org/abs/2405.21060
discovered_via: seed-papers-proposal
discovered_date: 2026-04-25
status: implemented
themes:
  primary: architecture
  secondary:
    - efficient-inference
    - long-context
tags:
  - ssm
  - mamba
  - state-space-duality
  - semiseparable-matrices
routing:
  decision: reference-only
  target_repo: no-magic
  target_script_slug: microssm
  target_path: 03-systems/microssm.py
  target_tier: 03-systems
  batch_label: architecture-seed
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microssm.py
    script_slug: microssm
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v2.0.0
  - repo: no-magic
    path: 03-systems/microcomplexssm.py
    script_slug: microcomplexssm
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v2.0.0
  - repo: no-magic
    path: 03-systems/microdiscretize.py
    script_slug: microdiscretize
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v2.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Mamba-2 shows that attention and state space models can be expressed through structured semiseparable matrices, then uses that duality to design a faster selective SSM layer and a more hardware-efficient algorithm.

## Problem

Transformers and SSMs have usually been treated as separate sequence-modeling families: attention is expressive but quadratic, while recurrent SSMs are linear-time but constrained by their state update structure. Mamba-1 made selective SSMs practical, but the theoretical relationship to attention and the algorithmic path to larger state dimensions were still incomplete.

## Contribution

The paper introduces Structured State Space Duality. It proves that linear attention and linear SSMs are dual views of the same family of structured semiseparable matrix transformations. From that framing, the authors define SSD, a layer that enlarges the SSM state dimension, improves matrix-multiply utilization, and supports both recurrent and convolutional/parallel forms. Mamba-2 is the resulting architecture built around SSD and a refined block design.

## Method summary

- Represent sequence transformations as structured semiseparable matrices rather than only as recurrences or attention score matrices.
- Show that linear attention and linear SSMs occupy the same structured-matrix family under different parameterizations.
- Replace scalar SSM channels with larger state dimensions so the layer can use dense matrix operations more efficiently.
- Use SSD algorithms that can run in recurrent mode for decoding or parallel/chunked mode for training.
- Simplify the Mamba block around the SSD layer, reducing overhead compared with Mamba-1.
- Evaluate Mamba-2 as both a language model architecture and an algorithmic bridge between attention and SSMs.

## Key results

The paper reports that Mamba-2 is 2x to 8x faster than Mamba-1 while remaining competitive with Transformers and Mamba-1 on language modeling. It also reports stronger hardware utilization from larger state dimensions and structured matrix multiplication. The practical result is not only better speed; the paper gives a common mathematical language for attention-like and SSM-like sequence layers, making hybrid and generalized layers easier to reason about.

## Relation to existing work

Mamba-2 builds directly on Mamba-1, S4-style state space models, and linear attention. Its core move is to stop treating SSM recurrence and attention as unrelated mechanisms. Compared with Mamba-1, it pushes more work into matrix-multiply-friendly structure. Compared with Transformers, it preserves linear-time recurrence for inference while retaining an attention-like structured matrix interpretation.

## Implementation notes

This card is the back-reference exemplar for existing `no-magic` scripts. `microssm.py` teaches the selective SSM recurrence, `microcomplexssm.py` shows how complex eigenvalues produce rotation-like dynamics, and `microdiscretize.py` explains how continuous-time dynamics become discrete updates. A future `micromamba2.py` could add the SSD-specific structured semiseparable view, but the current implementation references already cover the family needed to introduce Mamba-2. The toy version should compare recurrent scan, parallel/chunked form, and an attention-like matrix view over the same small sequence.

## Open questions

The paper's full value depends on optimized kernels and model-scale training. A single-file implementation should focus on the duality and algorithmic equivalence, not on reproducing throughput numbers.
