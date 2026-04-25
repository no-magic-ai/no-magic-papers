---
slug: ntm
title: "Neural Turing Machines"
authors:
  - Alex Graves
  - Greg Wayne
  - Ivo Danihelka
venue: arXiv
year: 2014
arxiv_id: "1410.5401"
doi: null
url: https://arxiv.org/abs/1410.5401
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: agents
  secondary:
    - architecture
tags:
  - external-memory
  - differentiable-memory
  - content-addressing
  - location-addressing
  - copy-task
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micromemory
  target_path: 04-agents/micromemory.py
  target_tier: 04-agents
  batch_label: v3-backfill-agents
  review_date: null
implementations:
  - repo: no-magic
    path: 04-agents/micromemory.py
    script_slug: micromemory
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Neural Turing Machines couple a neural network controller with an external memory matrix and fully-differentiable read and write heads, so a model can learn algorithms whose state lives outside its hidden activations.

## Problem

Recurrent networks at the time of writing had to compress all of their long-term state into a fixed-size hidden vector, which made them weak at tasks that require holding many distinct items in memory or operating on them with index-like access. Symbolic computers solve those tasks trivially with random-access memory, but their pointers and writes are not differentiable, so they cannot be optimized end-to-end with gradient descent. The paper asks whether a neural model can be augmented with an explicit memory store while staying differentiable everywhere.

## Contribution

The paper introduces the Neural Turing Machine, an architecture made of three parts: a controller (an LSTM or feedforward network), a memory matrix of size N rows by M columns, and a small set of read and write heads. Each head emits a soft attention distribution over memory rows produced by combining content-based addressing (cosine similarity to a key) with location-based addressing (a learned shift over the previous attention vector). Writes are implemented as an erase vector plus an add vector, both gated by the head's attention. Because every operation is differentiable, the entire system is trained end-to-end by backpropagation through time on input-output examples of small algorithmic tasks.

## Method summary

- Controller emits, at every time step, a key, a key strength, an interpolation gate, a shift distribution, and a sharpening exponent for each head.
- Content addressing: similarity scores between the key and every memory row produce a softmax-weighted address.
- Location addressing: the previous head address is interpolated with the content address, convolved with the shift distribution, and sharpened.
- Reads compute a weighted average of memory rows under the head's address.
- Writes apply an erase vector (multiplicative reduction) followed by an add vector (additive update), both weighted by the address.
- The architecture is trained with backpropagation through time on small synthetic tasks: copy, repeat copy, associative recall, dynamic N-grams, and priority sort.

## Key results

The paper reports that NTMs solve the copy task with near-perfect generalization to sequences much longer than those seen during training, while a plain LSTM degrades sharply once the test length exceeds the training distribution. Similar generalization gains appear on repeat copy and associative recall. On priority sort, the NTM with an LSTM controller learns a sorting algorithm rather than a lookup table. Trace inspections show that the learned read and write addresses behave like algorithmic pointers, including iteration over a list and key-based retrieval.

## Relation to existing work

NTMs sit between classical recurrent neural networks and symbolic systems. They are a direct ancestor of the Differentiable Neural Computer, which adds dynamic memory allocation and temporal linkage; of memory networks (Sukhbaatar et al. 2015), which use a similar attention pattern but a fixed memory; and of attention-based language models, where the soft addressing primitive becomes the dominant computational pattern. They also influenced subsequent work on neural program induction, such as Neural Programmer-Interpreters and Neural GPU.

## Implementation notes

A pedagogical version can fix N and M small (for example 32 by 16) and use a single read head and a single write head with an LSTM controller. The copy task is the most useful first benchmark because correct behavior is easy to spot from per-step outputs. The two implementation pitfalls that bite first are numerical: the sharpening exponent passed through softmax can blow up, so apply a softplus or clip; and the convolutional shift requires circular indexing that is easy to get wrong off-by-one. Training is sensitive to gradient clipping, since the through-time backprop for memory writes accumulates over the full sequence. Useful ablations: drop the location-addressing branch and watch generalization to longer copies fail; reduce memory rows and watch tasks that need many items fail abruptly.

## Open questions

The paper does not address how memory should grow with task complexity; that gap motivated the Differentiable Neural Computer. It also leaves open how to scale the addressing scheme to large memories, since soft attention over every row is linear in N at every step.
