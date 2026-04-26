---
slug: rnn-elman
title: "Finding Structure in Time"
authors:
  - Jeffrey L. Elman
venue: Cognitive Science
year: 1990
arxiv_id: null
doi: "10.1207/s15516709cog1402_1"
url: https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - rnn
  - simple-recurrent-network
  - context-units
  - sequence-modeling
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: micrornn
  target_path: 01-foundations/micrornn.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/micrornn.py
    script_slug: micrornn
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Elman's Simple Recurrent Network feeds a copy of each step's hidden activations back as input on the next step (via context units), giving a feedforward network a memory of arbitrary past states and letting it discover structure in temporal sequences without explicit time-step encoding.

## Problem

Connectionist models in the late 1980s processed inputs as static vectors and could not naturally handle sequences of variable length or temporal dependencies. The two existing approaches — sliding-window inputs (give the network a fixed lookback as parallel inputs) and time-delayed networks (Waibel et al. 1989, TDNN) — each forced the network to treat time as just another spatial dimension, with a fixed window. Cognitive-science questions about how a network discovers temporal structure (word boundaries in continuous speech, phrase structure in sentences, hierarchical sequence patterns) needed an architecture in which time was an intrinsic dimension of computation rather than a flattened spatial axis.

## Contribution

The paper introduces the Simple Recurrent Network (SRN), often called the Elman network. The architecture extends a standard feedforward network by adding context units that hold a copy of the hidden-layer activations from the previous time step and feed them back as additional inputs at the current step. Concretely: at time t, the hidden layer takes both the new input x_t and the context units c_t (which equal h_{t-1}) and computes h_t; the context units are then updated to c_{t+1} = h_t for the next step. The recurrence is implicit in the weight-sharing across time, but training is done with backpropagation as on a feedforward network — the recurrent connections are treated as fixed identity copies during the backward pass (truncated backpropagation), or unrolled for full backprop through time. The paper applies the architecture to three problems — exclusive-or in time, discovering word boundaries in continuous letter sequences, and learning sentence structure — and shows the network learns useful internal representations that cluster meaningfully when projected to two dimensions.

## Method summary

- Architecture: input units x_t, hidden units h_t, output units y_t, and context units c_t with c_t = h_{t-1}.
- Forward pass at time t: h_t = σ(W_xh · x_t + W_ch · c_t + b_h); y_t = σ(W_hy · h_t + b_y).
- Context update: copy h_t into c_{t+1} as a fixed (non-trained) identity connection.
- Train with backpropagation on the per-step output, treating the context-to-hidden weights as ordinary trainable weights and the hidden-to-context copies as fixed.
- For longer dependencies, unroll the network through time and apply backpropagation through time (Werbos 1990); the paper uses truncated unrolling.
- Applications in the paper: predict the next character in a stream of concatenated words (with no spaces), letting the network discover word-boundary structure; predict the next word in simple grammar-generated sentences; XOR over consecutive bits.

## Key results

The SRN learns to predict the next item in temporal sequences and develops internal representations that capture relevant structure: hidden-state clustering reveals discovered word boundaries in the letter-stream task, and learned word vectors group by syntactic category (noun, verb, animate, inanimate) without explicit labels. The XOR-in-time task — solving XOR by remembering a previous bit — demonstrates that the network can carry information across time steps despite being trained only on per-step targets.

## Relation to existing work

The SRN is the foundational form of the modern recurrent neural network. It precedes and is contrasted with Jordan networks (Jordan 1986), where the recurrent state is a copy of the output rather than the hidden layer; Time-Delay Neural Networks (Waibel et al. 1989), which use fixed-size sliding windows; and earlier symbolic-AI grammars. The SRN's vanishing-gradient limitation, identified later (Hochreiter 1991, Bengio et al. 1994), motivated gated cells: LSTM (Hochreiter & Schmidhuber 1997) and GRU (Cho et al. 2014). Modern attention-based architectures (Transformer, Vaswani et al. 2017) replaced the recurrent state entirely with content-based memory access, but the conceptual frame — a hidden state that summarizes past inputs — persists and has resurfaced in state-space models (Mamba, S4) that revisit recurrence with different stability properties.

## Implementation notes

A pedagogical script can implement the Elman SRN and compare it with a gated GRU on a copy task. Minimum viable cell: parameters W_xh, W_ch, W_hy and biases; forward computes h_t and y_t; backward unrolls T steps. Pitfalls: forgetting to detach the previous hidden state across independent batches; using sigmoid throughout slows training compared with tanh for the hidden state; long sequences without gradient clipping can explode. Diagnostic: train an SRN at lengths 5, 10, 20, 50 — failure above ~20 motivates gated alternatives.

## Open questions

The paper does not address the vanishing-gradient problem, which limits the SRN's effective memory to a handful of steps. Subsequent gated cells (LSTM, GRU) and attention address this.
