---
slug: titans-2501
title: "Titans: Learning to Memorize at Test Time"
authors:
  - Ali Behrouz
  - Peilin Zhong
  - Vahab Mirrokni
venue: arXiv
year: 2025
arxiv_id: "2501.00663"
doi: "10.48550/arXiv.2501.00663"
url: https://arxiv.org/abs/2501.00663
discovered_via: seed-papers-proposal
discovered_date: 2026-04-25
status: summarized
themes:
  primary: long-context
  secondary:
    - architecture
tags:
  - neural-memory
  - test-time-learning
  - long-context
  - memory-architecture
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microtitans
  target_path: 01-foundations/microtitans.py
  target_tier: 01-foundations
  batch_label: long-context-seed
  review_date: null
implementations: []
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Titans adds a neural long-term memory module that updates at test time, then combines that memory with short-term attention through context, layer, or gated-branch variants.

## Problem

Transformers model direct dependencies accurately but pay quadratic cost and are limited by a fixed context window. Linear recurrent models scale better, but they compress all history into a fixed hidden state or matrix-valued state, which can overflow on long sequences. The paper argues that long-context models need a memory system that can store useful historical abstractions rather than only append tokens or compress them linearly.

## Contribution

The core contribution is a neural memory module trained as an online meta-learner. At test time, incoming tokens update the memory module's parameters using a surprise signal: inputs that violate the current memory produce larger gradients and should be memorized more strongly. The memory update includes momentum-like accumulated surprise and a data-dependent forgetting mechanism. Around that module, the authors define three Titans variants: Memory as Context (MAC), Memory as Gate (MAG), and Memory as Layer (MAL).

## Method summary

- Treat attention as short-term memory and the learned neural memory module as long-term memory.
- Compute surprise from the gradient of an associative memory loss with respect to the incoming data.
- Accumulate surprise over time so the model can remember spans around surprising events, not only the single most surprising token.
- Apply a forgetting mechanism equivalent to weight decay, allowing memory to clear stale information under fixed capacity.
- Train the neural memory efficiently by tensorizing mini-batch gradient descent so more work is expressed as matrix multiplication.
- Combine memory with attention in three architectures: retrieved memory as context, memory as a gated branch, or memory as a layer.

## Key results

The paper reports that Titans outperform Transformers, modern linear recurrent models, and hybrid attention-recurrent baselines across language modeling, commonsense reasoning, genomics, time series, and recall-heavy tasks. On the RULER S-NIAH benchmark at 16K context, Titans variants keep high scores where baselines collapse: for S-NIAH-W at 16K, Table 2 reports 95.2 for MAC, 88.2 for MAG, and 90.4 for MAL, while TTT, Mamba2, and DeltaNet report 0.0, 0.0, and 0.0 respectively. The paper also reports scaling beyond a 2M-token context window on needle-in-haystack-style tasks.

## Relation to existing work

Titans sits between attention, modern recurrent models, and test-time training. Compared with linear Transformers and Mamba-style recurrent models, it uses a deeper nonlinear memory rather than a fixed linear state. Compared with TTT-style layers, it adds momentum-like surprise accumulation and forgetting. The three architectural variants also make the paper a useful taxonomy for how memory can be connected to attention.

## Implementation notes

A `no-magic` implementation should isolate the neural memory update before building a full sequence model. Use a tiny MLP as memory, a synthetic key-value recall task, and a surprise score derived from the memory loss gradient. Then compare three integrations: append retrieved memory to a short attention context, gate memory output against attention output, and replace a layer with the memory module. The important toy result is not benchmark scale; it is showing that test-time memory parameters change during inference and that forgetting prevents stale facts from dominating.

## Open questions

The paper's implementation details are still large-system choices: memory depth, tensorized training, chunking, and benchmark-specific architectures. The single-file version should preserve the online memory mechanism and leave large-scale throughput claims out of scope.
