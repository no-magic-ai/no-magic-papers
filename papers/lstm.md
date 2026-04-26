---
slug: lstm
title: "Long Short-Term Memory"
authors:
  - Sepp Hochreiter
  - Jürgen Schmidhuber
venue: Neural Computation
year: 1997
arxiv_id: null
doi: "10.1162/neco.1997.9.8.1735"
url: https://www.bioinf.jku.at/publications/older/2604.pdf
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - lstm
  - recurrent
  - gated-cell
  - vanishing-gradient
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microlstm
  target_path: 01-foundations/microlstm.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microlstm.py
    script_slug: microlstm
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

LSTM introduces a recurrent cell with multiplicative input, output, and (in later refinements) forget gates around a self-connected linear memory unit, allowing gradients to flow over thousands of time steps without vanishing or exploding.

## Problem

Recurrent neural networks of the early 1990s could not learn dependencies that spanned more than a handful of time steps. Hochreiter's earlier analysis showed why: the gradient of a sequence's loss with respect to a hidden state at an earlier step is a product of Jacobians whose spectral radius typically shrinks (vanishing gradients) or grows (exploding gradients) exponentially with depth. Without a mechanism to preserve gradient magnitude, the model could not credit-assign across long contexts, no matter how many parameters were added.

## Contribution

LSTM replaces the simple recurrent state with a constant-error-carousel: a linear self-connection with weight 1 that, by construction, neither shrinks nor grows the gradient over time. The model surrounds this carousel with multiplicative gates that decide when information enters the cell (input gate) and when the cell influences the output (output gate). The forget gate, added later by Gers, Schmidhuber, and Cummins (2000), gives the cell the ability to clear its own state. The combined unit is differentiable, can be trained with backpropagation through time, and learns long-range dependencies on tasks where vanilla RNNs fail outright.

## Method summary

- Cell state c_t is a vector that propagates from step to step via c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t, where ⊙ is element-wise product.
- Input gate i_t = σ(W_i · [x_t, h_{t-1}] + b_i) controls how much of the candidate g_t = tanh(W_g · [x_t, h_{t-1}] + b_g) is written into the cell.
- Forget gate f_t = σ(W_f · [x_t, h_{t-1}] + b_f) controls how much of the previous cell state to retain.
- Output gate o_t = σ(W_o · [x_t, h_{t-1}] + b_o) controls how much of the cell state is exposed as the hidden state h_t = o_t ⊙ tanh(c_t).
- Backpropagation through time computes gradients across the additive cell-state recurrence; the linear path through c_t avoids the multiplicative shrinkage that destroys vanilla RNN gradients.
- A common stabilizing trick is to initialize the forget-gate bias to a positive value (typically 1) so the cell starts out remembering by default.

## Key results

The original 1997 paper showed LSTM could learn synthetic long-context tasks (embedded Reber grammars, the noisy temporal-order task, the addition problem) at sequence lengths where vanilla RNNs failed completely or required orders of magnitude more training data. Subsequent work demonstrated LSTM dominance on speech recognition, handwriting recognition, neural machine translation, and language modeling for nearly two decades, until the transformer largely displaced it. Even after that displacement, LSTM remains the standard recurrent baseline whenever sequence modeling without attention is needed.

## Relation to existing work

LSTM directly addresses the vanishing-gradient analysis from Hochreiter's 1991 thesis and from Bengio, Simard, and Frasconi (1994). It precedes the GRU (Cho et al. 2014), which collapses input and forget gates into one update gate and removes the explicit cell state, trading a small amount of expressiveness for fewer parameters. The transformer (Vaswani et al. 2017) replaces recurrence entirely with attention, removing the sequential dependency that bottlenecks LSTM training but introducing quadratic per-step cost. Modern long-context architectures (state-space models, Mamba, Titans) revisit the recurrent idea with different mechanisms for the constant-error-carousel role.

## Implementation notes

A pedagogical script can implement the four gate equations directly without library help. Minimum viable cell: parameters W_i, W_f, W_g, W_o (each shaped [hidden + input, hidden]) plus biases; forward computes the four gate vectors, updates c_t, exposes h_t. Pitfalls: forgetting that c_t and h_t are different tensors (h_t is the gated, tanh-squashed view of c_t); applying the activation in the wrong order around the cell update (tanh before forget-gate masking is wrong); skipping the forget-bias initialization, which makes early training unstable. Useful diagnostic: train on a copy task at increasing sequence lengths, compare LSTM vs vanilla RNN, and watch the RNN flatline at lengths the LSTM solves.

## Open questions

The paper does not analyze what the gates ultimately learn to do; later interpretability work showed gates often specialize to detect specific input patterns. The original formulation is also computationally heavier than necessary — modern engineering uses fused kernels and sometimes drops the input gate or merges gates further. The empirical question of LSTM-vs-transformer on very long sequences is now moot for most applications, though it remains alive in domains where strict streaming and low memory matter.

## Further reading

- Cho et al. (2014) — GRU, the simpler gated cell.
- Vaswani et al. (2017) — Transformer, the attention-based replacement.
