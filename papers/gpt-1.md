---
slug: gpt-1
title: "Improving Language Understanding by Generative Pre-Training"
authors:
  - Alec Radford
  - Karthik Narasimhan
  - Tim Salimans
  - Ilya Sutskever
venue: OpenAI Technical Report
year: 2018
arxiv_id: null
doi: null
url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: architecture
  secondary: []
tags:
  - gpt
  - autoregressive
  - language-model
  - pretraining
  - decoder-transformer
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microgpt
  target_path: 01-foundations/microgpt.py
  target_tier: 01-foundations
  batch_label: v3-backfill-foundations
  review_date: null
implementations:
  - repo: no-magic
    path: 01-foundations/microgpt.py
    script_slug: microgpt
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

GPT-1 pretrains a decoder-only transformer on a large unlabeled corpus with the standard autoregressive next-token objective, then fine-tunes the same network with a small task-specific head and minimal architectural changes, beating task-specific architectures across nine NLP benchmarks.

## Problem

Most NLP at the time used task-specific architectures trained from scratch on labeled data, with pretrained word embeddings as the only transferred component. Fully labeled data was scarce for many tasks, while unlabeled text was abundant. Earlier transfer-learning attempts (semi-supervised LSTMs, ELMo) used pretrained representations as features but kept the downstream architecture task-specific. The paper asks whether a single language-model pretraining recipe followed by light fine-tuning of the same architecture can match or beat task-specific models across diverse NLP tasks.

## Contribution

GPT-1 introduces a two-stage recipe. Stage one: pretrain a 12-layer decoder-only transformer (117M parameters) with the standard left-to-right language-modeling objective on the BookCorpus dataset. Stage two: for each downstream task, format the input as a sequence the language model can consume (with delimiter tokens for tasks with multiple inputs like NLI or QA), add a single linear classification head on the final hidden state, and fine-tune the entire network with a combined language-modeling and task loss. The architectural commitments — decoder-only, learned position embeddings, byte-pair encoding tokenizer, GELU activations — became the template that GPT-2, GPT-3, and most subsequent autoregressive LLMs inherit.

## Method summary

- Architecture: 12-layer transformer decoder, 768 hidden, 12 heads, 3072 FFN dimension, learned position embeddings up to 512 tokens; ~117M parameters.
- Tokenizer: byte-pair encoding with a 40k vocabulary.
- Pretraining objective: standard autoregressive next-token language modeling — maximize Σ log p(x_t | x_{<t}) — on BookCorpus (~7000 unpublished books, ~1B tokens).
- Optimizer: Adam with linear warmup followed by cosine decay; standard cross-entropy loss; dropout 0.1 throughout.
- Fine-tuning: format each task as a sequence with delimiter tokens between inputs (e.g., premise and hypothesis for NLI, question and answer choices for QA); attach a linear head on the final token's hidden state; fine-tune for a few epochs with a small learning rate; combine task loss with the pretraining LM loss as auxiliary objective for stability.

## Key results

GPT-1 advances state of the art on 9 of 12 evaluated NLP tasks — natural language inference (MultiNLI, SNLI, RTE, SciTail), question answering (Story Cloze, RACE), commonsense reasoning (COPA), semantic similarity (STS-B, MRPC, QQP), and classification (CoLA, SST-2). The improvement is largest on tasks with limited labeled data, where the pretrained features carry the most weight. Ablations show that the auxiliary language-modeling loss during fine-tuning helps stability and final accuracy on most tasks, and that the pretraining provides most of the gain — a randomly-initialized transformer with the same fine-tuning recipe performs much worse.

## Relation to existing work

GPT-1 builds on the original Transformer (Vaswani et al. 2017) and on earlier transfer-learning work in NLP (ULMFiT, ELMo). It contrasts with BERT (Devlin et al. 2018), which appeared a few months later: BERT pretrains an encoder bidirectionally with masked language modeling, while GPT pretrains a decoder left-to-right with autoregressive language modeling. The two papers split NLP architecture into two lineages — encoder-only for representation tasks, decoder-only for generation — that converged again with encoder-decoder models (T5, BART) and then re-converged on decoder-only at scale (GPT-2, GPT-3, modern LLMs). GPT-1's recipe also predicts the scaling-law line: scale up the same architecture and the same objective, and capabilities continue to emerge.

## Implementation notes

A pedagogical script can train a tiny GPT on a single text corpus (Shakespeare, the Linux kernel, a single book). Minimum viable model: token embeddings, learned position embeddings, N transformer blocks (each: pre-norm LayerNorm → causal self-attention → residual → pre-norm LayerNorm → MLP → residual), final LayerNorm, untied or tied LM head. Pitfalls: forgetting to apply a causal mask in self-attention (the model sees future tokens and trivially predicts the next one); using post-norm (the original Transformer convention) is harder to train than pre-norm at depth; using a learning rate without warmup tends to diverge in the first few hundred steps. Useful diagnostic: track per-token loss and sample autoregressively every few epochs; the samples should transition from random characters to plausible morphology to plausible syntax over training.

## Open questions

The paper does not establish scaling laws (those came later, with GPT-2 and the Kaplan et al. 2020 study) and does not test how the recipe behaves at much larger scales. It also does not address instruction following, which would emerge as a focus only after GPT-3 and the InstructGPT line.
