---
slug: bert
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors:
  - Jacob Devlin
  - Ming-Wei Chang
  - Kenton Lee
  - Kristina Toutanova
venue: NAACL
year: 2019
arxiv_id: "1810.04805"
doi: null
url: https://arxiv.org/abs/1810.04805
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - bert
  - masked-language-model
  - bidirectional
  - pretraining
  - next-sentence-prediction
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microbert
  target_path: 01-foundations/microbert.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microbert.py
    script_slug: microbert
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

BERT pretrains a transformer encoder bidirectionally with a masked-language-model objective and a next-sentence-prediction task, then fine-tunes the same network with a small task-specific head, outperforming task-specific architectures across the GLUE benchmark.

## Problem

Pretrained word embeddings (Word2Vec, GloVe) gave fixed vectors per word and ignored context. ELMo improved on this with contextual embeddings from a left-to-right and right-to-left language model, but the two directions were trained independently and concatenated only at the top, so deeper layers never saw both contexts together. GPT trained a left-to-right transformer and reached strong fine-tuning results, but its strict autoregressive masking blocked any token from attending to its right context. The field needed a way to pretrain a deeply bidirectional encoder without the trivial-prediction problem that bidirectional next-token modeling causes (every token can see itself).

## Contribution

BERT introduces masked language modeling: 15% of input tokens are replaced with a special [MASK] token (or a random token, or kept the same — the paper's mixing recipe), and the model is trained to predict the original tokens at those positions. Because the input no longer contains the answer at the masked positions, the encoder can attend bidirectionally without the leakage problem. A second pretraining task, next-sentence-prediction, classifies whether a candidate sentence B follows sentence A; this gives the model a sentence-level signal useful for question-answering and natural-language-inference tasks. After pretraining on BookCorpus + English Wikipedia, the same encoder is fine-tuned with a single new output layer per task.

## Method summary

- Architecture: a transformer encoder stack (BERT-Base: 12 layers, 768 hidden, 12 heads, 110M parameters; BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M parameters).
- Input format: WordPiece tokens with [CLS] at position 0 (used as the sequence-level summary) and [SEP] separating segments; learned segment embeddings distinguish A from B.
- Masked LM: pick 15% of WordPiece positions; of those, replace 80% with [MASK], 10% with a random token, 10% leave unchanged. Predict the original token at each picked position with a softmax over the vocabulary.
- Next sentence prediction: 50% of training pairs have B as the actual following sentence, 50% have B drawn at random; predict the binary label from the [CLS] representation.
- Fine-tuning: add a single linear layer over [CLS] (for classification) or over each token (for tagging or span tasks); fine-tune end-to-end with a small learning rate.

## Key results

BERT-Large advances the GLUE score from 72.8 to 80.5 at the time of publication, beating prior task-specific models on every component task. On SQuAD v1.1 it reaches 93.2 F1; on SWAG it improves accuracy by 8.3 points over the previous state of the art. Ablations show the bidirectional masked LM is the critical change relative to GPT, while next-sentence-prediction contributes a smaller but real gain on QA and NLI.

## Relation to existing work

BERT extends the encoder side of the original Transformer (Vaswani et al. 2017). It contrasts with GPT-1, which pretrains a decoder for left-to-right generation, and with ELMo, which uses a frozen bi-LSTM. Subsequent work refined the recipe in many directions: RoBERTa (Liu et al. 2019) drops next-sentence-prediction, trains longer with bigger batches, and shows the original recipe was undertuned; ALBERT (Lan et al. 2020) shares parameters across layers; ELECTRA (Clark et al. 2020) replaces MLM with replaced-token detection for sample efficiency; DeBERTa adds disentangled attention. The encoder-only design remains the dominant choice for embedding and classification tasks; the encoder-decoder line (T5, BART) and decoder-only line (GPT family) extend the pretrain-then-fine-tune template in different directions.

## Implementation notes

A pedagogical script can use a tiny encoder (2-4 layers, 64-128 hidden) trained on a small corpus. Minimum viable trainer: tokenize with a small WordPiece or character tokenizer, sample positions for masking, replace per the 80/10/10 rule, run the encoder, project hidden states at masked positions to vocabulary logits, cross-entropy loss. Pitfalls: forgetting that the masking selection should change per batch (a fixed mask is much weaker); applying the loss to all positions instead of only the masked ones; using a non-shuffled sentence-pair sampler so the next-sentence task degenerates. Useful diagnostic: track masked-LM accuracy on a held-out batch; it should rise from chance (1/vocab) to a meaningful fraction within a few hundred steps even on tiny corpora.

## Open questions

The paper does not analyze how mask rate, mask token distribution, or sequence length interact with downstream performance; subsequent work (RoBERTa, T5) explored these. The encoder-only architecture also limits BERT to representation tasks — generation requires the decoder family it inspired but did not include.
