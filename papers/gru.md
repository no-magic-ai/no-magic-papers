---
slug: gru
title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
authors:
  - Kyunghyun Cho
  - Bart van Merriënboer
  - Caglar Gulcehre
  - Dzmitry Bahdanau
  - Fethi Bougares
  - Holger Schwenk
  - Yoshua Bengio
venue: EMNLP
year: 2014
arxiv_id: "1406.1078"
doi: null
url: https://arxiv.org/abs/1406.1078
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - gru
  - rnn-encoder-decoder
  - reset-gate
  - update-gate
  - machine-translation
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: rnn_vs_gru_vs_lstm
  target_path: 01-foundations/rnn_vs_gru_vs_lstm.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/rnn_vs_gru_vs_lstm.py
    script_slug: rnn_vs_gru_vs_lstm
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers:
  - slug: rnn-elman
  - slug: lstm
---

## TL;DR

The paper introduces the RNN encoder-decoder framework for machine translation and proposes the Gated Recurrent Unit (GRU): a simpler gated cell than LSTM that merges input and forget gates into a single update gate and removes the separate cell state, achieving competitive sequence-modeling quality with fewer parameters.

## Problem

Statistical machine translation in 2014 relied on phrase-based systems with separate components for alignment, phrase scoring, and language modeling. End-to-end neural sequence-to-sequence models were just emerging but used vanilla RNNs whose vanishing gradients limited them on long sequences. LSTM solved the gradient problem but was complex (three gates plus a separate cell state) and computationally expensive. The paper asks whether a simpler gated cell can preserve LSTM's training stability while reducing parameter count, and whether such cells can drive a clean encoder-decoder framework for translation.

## Contribution

The paper makes two contributions. First, the RNN encoder-decoder framework: an encoder RNN reads a source sentence into a fixed-length context vector; a decoder RNN generates the target sentence one token at a time conditioned on that vector. The whole system is trained end-to-end to maximize the conditional probability of the target given the source. Second, the Gated Recurrent Unit (GRU): a recurrent cell with two gates — an update gate z that interpolates between the previous hidden state and a candidate update, and a reset gate r that controls how much of the previous state contributes to the candidate. There is no separate cell state; the update is applied directly to the hidden state. The paper applies the encoder-decoder with GRU to phrase scoring within an existing SMT system and shows substantial BLEU improvements on English-French translation.

## Method summary

- GRU update at time t: z_t = σ(W_z · [x_t, h_{t-1}]), r_t = σ(W_r · [x_t, h_{t-1}]), h̃_t = tanh(W · [x_t, r_t ⊙ h_{t-1}]), h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t.
- Update gate z_t controls how much of the previous state to keep versus replace.
- Reset gate r_t controls how much of the previous state contributes to the candidate update; when r is near 0 the cell behaves as if processing a fresh input.
- Encoder: read source tokens into the GRU; the final hidden state is the context vector c.
- Decoder: another GRU initialized from c that generates target tokens conditioned on c and the previously emitted token.
- Train end-to-end with cross-entropy on the target sentence.

## Key results

In the paper's setup, augmenting Moses (a phrase-based SMT system) with phrase-pair scores from the RNN encoder-decoder improves BLEU on the WMT 2014 English-French task by roughly 0.5 to 1.5 points over Moses alone. Subsequent work (Bahdanau et al. 2015, Sutskever et al. 2014) showed pure neural translation could match and then exceed phrase-based SMT entirely. The GRU itself was adopted broadly: empirical comparisons (Chung et al. 2014; Greff et al. 2017) found GRU and LSTM perform comparably on most sequence tasks, with GRU slightly faster to train per epoch.

## Relation to existing work

GRU is a simplification of LSTM (Hochreiter & Schmidhuber 1997), removing the separate cell state and merging the input and forget gates. Both contrast with the vanilla Elman SRN (rnn-elman card), which lacks gating and suffers from vanishing gradients on long sequences. The encoder-decoder framing in this paper inspired Sutskever et al. 2014's "Sequence to Sequence Learning" (which used LSTM and reversed input order) and was extended a year later by Bahdanau, Cho, and Bengio (2015) with attention, removing the fixed-length-context-vector bottleneck. That attention extension led directly to the Transformer (Vaswani et al. 2017), which retained encoder-decoder structure but replaced recurrence entirely with self-attention.

## Implementation notes

A pedagogical comparison script can place vanilla RNN, GRU, and LSTM side by side on the same task and compare convergence and final accuracy. Minimum viable GRU cell: parameters W_z, W_r, W (with biases), forward computes the two gates and the candidate, then interpolates. Pitfalls: confusing the update gate's interpretation — h_t = (1 - z) · h_{t-1} + z · h̃_t means z near 1 forgets the past, the opposite of the LSTM forget-gate convention; forgetting the element-wise product of r and h_{t-1} inside the candidate; using sigmoid in place of tanh for the candidate prevents the hidden state from going negative and changes representational capacity. Useful diagnostic: on a long-context copy task, plot loss vs sequence length for vanilla RNN, GRU, and LSTM — vanilla RNN should fail above ~20 steps, GRU and LSTM should both succeed up to several hundred.

## Open questions

The paper does not give a theoretical reason GRU should match LSTM; subsequent work (Greff et al. 2017) confirmed empirical parity but did not produce a definitive analytical comparison. Modern long-context architectures (state-space models, attention) supersede both for most applications, but GRU remains a strong default whenever a small recurrent cell is needed.
