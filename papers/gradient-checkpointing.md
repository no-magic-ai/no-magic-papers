---
slug: gradient-checkpointing
title: "Training Deep Nets with Sublinear Memory Cost"
authors:
  - Tianqi Chen
  - Bing Xu
  - Chiyuan Zhang
  - Carlos Guestrin
venue: arXiv
year: 2016
arxiv_id: "1604.06174"
doi: null
url: https://arxiv.org/abs/1604.06174
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: efficient-inference
  secondary: []
tags:
  - gradient-checkpointing
  - activation-recomputation
  - memory-compute-tradeoff
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microcheckpoint
  target_path: 03-systems/microcheckpoint.py
  target_tier: 03-systems
  batch_label: v3-backfill-systems
  review_date: null
implementations:
  - repo: no-magic
    path: 03-systems/microcheckpoint.py
    script_slug: microcheckpoint
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The paper shows that training a deep network of N layers needs only O(sqrt(N)) memory for activations instead of O(N) by checkpointing a small subset of activations and recomputing the rest during the backward pass — at the cost of one extra forward pass.

## Problem

Standard backpropagation requires storing every activation produced during the forward pass so that the backward pass can compute gradients with respect to each layer's parameters. For a deep network with N layers and per-layer activation size A, the memory cost is O(N · A). On a fixed GPU memory budget this caps the depth or width of trainable models, especially for sequence models where the per-step activation footprint is large. Practitioners worked around this with smaller batch sizes, gradient accumulation, or model parallelism — none of which addressed the underlying memory cost. The paper asks whether one can systematically trade additional compute for activation memory.

## Contribution

The paper introduces gradient checkpointing (also called activation recomputation): pick a subset of K layers as checkpoints; during forward pass, store activations only at those K layers; during backward pass, when the gradient needs the activations between two checkpoints, recompute them by running the forward pass from the previous checkpoint. With K = sqrt(N) checkpoints evenly spaced, the activation memory drops from O(N) to O(sqrt(N)), at the cost of approximately one extra forward pass over the network (the recomputation between checkpoints, summed across the backward pass, is roughly equivalent to one full forward). The paper analyzes the optimal placement of checkpoints, gives algorithms for both feedforward and recurrent networks, and demonstrates the technique enables training models several times deeper than would otherwise fit in memory.

## Method summary

- Mark K layers in the network as checkpoints (the paper studies K = sqrt(N) and other choices); checkpoint activations are stored during forward pass.
- Discard non-checkpoint activations after they are used by the next layer.
- Backward pass: when gradients arrive at a non-checkpoint layer, find the previous checkpoint and rerun the forward pass from there to that layer to reconstruct the needed activations.
- For sequence models with T time steps, the analogous technique stores activations at every K-th time step and recomputes the others during BPTT.
- The optimal checkpoint spacing balances the memory savings against the recomputation cost; for N layers and K checkpoints, peak memory is O(N/K + K) and total compute is one forward + one backward + (K-1) extra recomputations of subsegments — minimized at K = sqrt(N) giving O(sqrt(N)) memory and ~1.5× compute.

## Key results

The paper demonstrates training a 1000-layer feedforward network and a recurrent net unrolled for 1000 time steps using fewer than 7 GB of activation memory — configurations that would require tens or hundreds of gigabytes without checkpointing. The compute overhead is consistently around 30-40% per training step, in line with the theoretical "approximately one extra forward pass" prediction. The technique works across architectures and frameworks; the paper provides drop-in patterns for MXNet that translate directly to other deep-learning libraries.

## Relation to existing work

The recompute-instead-of-store idea predates this paper in domains like sparse-recovery and out-of-core scientific computing; the paper's contribution is the formal analysis for deep networks plus a practical recipe. PyTorch's `torch.utils.checkpoint`, JAX's `jax.checkpoint` (rematerialization), and TensorFlow's `tf.recompute_grad` are all direct implementations of this technique. Subsequent work extended the idea: selective checkpointing chooses checkpoints based on per-layer activation cost rather than count alone (relevant for transformers where attention activations dominate); FlashAttention's backward pass uses the same recomputation pattern within a single attention kernel; ZeRO-Offload (Rajbhandari et al. 2021) combines checkpointing with offloading to host memory. Modern training stacks routinely combine all three.

## Implementation notes

A pedagogical script can demonstrate gradient checkpointing on a small feedforward stack. Minimum viable implementation: define a forward function f_segment(x) for each between-checkpoint segment; during backward, when gradient needs activations within a segment, call f_segment(checkpoint_input) again. The Python implementation can wrap this as a custom autograd function that saves only the segment's input on the forward and recomputes on the backward. Pitfalls: forgetting to disable autograd inside the segment's forward (which would needlessly track gradients of recomputation); inadvertently checkpointing through random ops without seeding (recomputation produces different randomness, breaking dropout/sampling layers — fix by saving and restoring the RNG state); double-counting compute when checkpointing nested segments. Useful comparison: train the same model with and without checkpointing on a small problem and verify final loss curves match exactly while peak memory drops.

## Open questions

The paper assumes a feedforward-style call graph; modern transformers with attention have non-trivial dependencies that complicate optimal checkpoint placement. The choice between checkpointing, offloading to CPU memory, and offloading to disk is workload-dependent; subsequent ZeRO-style work generalizes the trade-off space.
