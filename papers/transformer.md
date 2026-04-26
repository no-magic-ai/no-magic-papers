---
slug: transformer
title: "Attention Is All You Need"
authors:
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Łukasz Kaiser
  - Illia Polosukhin
venue: NeurIPS
year: 2017
arxiv_id: "1706.03762"
doi: null
url: https://arxiv.org/abs/1706.03762
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary:
    - long-context
tags:
  - transformer
  - self-attention
  - multi-head-attention
  - positional-encoding
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microattention
  target_path: 03-systems/microattention.py
  target_tier: 03-systems
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/attention_vs_none.py
    script_slug: attention_vs_none
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
  - repo: no-magic
    path: 03-systems/microattention.py
    script_slug: microattention
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

The Transformer replaces recurrence and convolution in sequence-to-sequence models with stacked self-attention and feedforward layers, yielding a parallelizable architecture that beats prior recurrent and convolutional translation systems on quality and trains substantially faster.

## Problem

Sequence-to-sequence translation systems before this paper used recurrent or convolutional encoders and decoders. Recurrent models had a sequential bottleneck — each step depended on the previous hidden state, so training time scaled with sequence length and could not parallelize across positions. Convolutional models (ByteNet, ConvS2S) parallelized but required deep stacks to grow the receptive field, making long-range dependencies expensive. Earlier attention mechanisms (Bahdanau et al. 2015, Luong et al. 2015) helped recurrent decoders attend over the encoder's hidden states but kept the recurrence intact. The paper asks whether attention alone, applied within and between layers, can replace recurrence entirely.

## Contribution

The paper introduces the Transformer architecture: a stack of identical encoder layers and decoder layers in which the only computational primitives are scaled dot-product attention, position-wise feedforward networks, residual connections, and layer normalization. Self-attention lets every position in a sequence attend directly to every other position in O(1) sequential operations and O(n²) total operations, enabling full parallelism within a layer. Multi-head attention runs h attention computations in parallel with different linear projections of the same inputs and concatenates the outputs, letting the model attend to different relationships simultaneously. Sinusoidal positional encodings inject sequence-order information additively into token embeddings since attention itself is permutation-invariant. The encoder uses bidirectional self-attention; the decoder uses masked (causal) self-attention plus encoder-decoder cross-attention.

## Method summary

- Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V, where Q, K, V are query, key, and value matrices and d_k is the key dimension. The sqrt(d_k) scaling keeps softmax gradients well-behaved at high dimension.
- Multi-head attention: project Q, K, V h times with different learned linear projections, run scaled dot-product attention in parallel, concatenate, then project back. The paper uses h = 8.
- Position-wise feedforward: FFN(x) = ReLU(x · W_1 + b_1) · W_2 + b_2, applied independently per position.
- Encoder layer: multi-head self-attention → residual + LayerNorm → FFN → residual + LayerNorm.
- Decoder layer: masked multi-head self-attention → residual + LayerNorm → multi-head cross-attention over encoder output → residual + LayerNorm → FFN → residual + LayerNorm.
- Positional encoding: sinusoidal functions of position with different frequencies per embedding dimension; added to token embeddings before the first layer.
- Training: standard cross-entropy on the target sequence; Adam with a custom warmup schedule.

## Key results

The base Transformer (6 layers, 512 hidden, 8 heads, 65M parameters) matches the previous best system on WMT 2014 English-French translation while training in 12 hours on 8 P100 GPUs — roughly 100x less compute than the recurrent baselines it beats. The big Transformer (1024 hidden, 16 heads, 213M parameters) sets a new state of the art on both English-French (41.8 BLEU) and English-German (28.4 BLEU). The paper also reports strong constituency-parsing results, demonstrating the architecture generalizes beyond translation.

## Relation to existing work

The Transformer builds on attention-augmented sequence-to-sequence models (Bahdanau et al. 2015, Luong et al. 2015), removes the recurrence those models retained, and inherits residual connections and layer normalization from earlier deep-network work. It launches the lineage that includes the encoder-only BERT (Devlin et al. 2018), the decoder-only GPT family (Radford et al. 2018+), the encoder-decoder T5 (Raffel et al. 2019), and Vision Transformers (Dosovitskiy et al. 2020). Subsequent work modifies the attention mechanism — multi-query attention (Shazeer 2019), grouped-query attention (Ainslie et al. 2023), Flash Attention (Dao et al. 2022), and sparse-attention variants — but the architectural skeleton from this paper remains the dominant template for modern large-scale sequence models.

## Implementation notes

A pedagogical script can implement scaled dot-product attention, multi-head attention, and a single transformer block from scratch on a tiny task (character-level prediction, copy task). Two scripts in `no-magic` exercise this paper: `attention_vs_none` is a comparison demonstrating attention's contribution by ablating it, and `microattention` walks through MHA, GQA, MQA, and sliding-window variants side by side. Pitfalls: forgetting the sqrt(d_k) scaling in dot-product attention causes softmax saturation at high d; applying the causal mask incorrectly (masking the wrong triangle, or applying it at inference) breaks generation; tying or untying input/output token embeddings is a small but real choice. Useful diagnostic: visualize attention weights from a trained head on a copy task — the head should concentrate on the position to copy, demonstrating learned positional behavior.

## Open questions

The paper does not characterize what individual attention heads learn; subsequent interpretability work (Voita et al. 2019, Clark et al. 2019) showed heads specialize in syntactic and positional roles. The quadratic cost of self-attention motivates the long-context line of work (linear attention, sparse attention, state-space models).
