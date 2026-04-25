---
slug: lora
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors:
  - Edward J. Hu
  - Yelong Shen
  - Phillip Wallis
  - Zeyuan Allen-Zhu
  - Yuanzhi Li
  - Shean Wang
  - Lu Wang
  - Weizhu Chen
venue: ICLR
year: 2022
arxiv_id: "2106.09685"
doi: null
url: https://arxiv.org/abs/2106.09685
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: parameter-efficient
  secondary:
    - alignment
tags:
  - lora
  - adapters
  - low-rank
  - fine-tuning
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microlora
  target_path: 02-alignment/microlora.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microlora.py
    script_slug: microlora
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

LoRA freezes a pretrained model's weights and learns a low-rank update for selected weight matrices, cutting trainable parameters by orders of magnitude while matching full fine-tuning quality on downstream tasks.

## Problem

Full fine-tuning of large models stores a full copy of every weight per task, which is expensive in disk, memory, and serving time when many task-specific checkpoints are needed. Earlier parameter-efficient methods — adapters and prefix tuning — added inference latency or required architectural changes that complicated deployment. The paper asks whether one can adapt a frozen base model to a new task with a small number of additional trainable parameters and no additional inference cost once trained.

## Contribution

LoRA assumes that the change in weights during adaptation has low intrinsic rank, and parameterizes the change to a weight matrix W as W + BA, where B is d×r and A is r×k with rank r much smaller than min(d,k). Only A and B are trained; W is frozen. At inference the product BA can be merged into W, so LoRA adds zero latency. The paper applies the trick to attention projection matrices in the transformer (typically W_q and W_v, sometimes W_o and W_k) and shows that very small ranks — r as low as 1 to 8 — are sufficient for full fine-tuning quality on GLUE and several generation tasks.

## Method summary

- Freeze all pretrained parameters of the base model.
- For each adapted weight matrix W ∈ R^(d×k), introduce trainable matrices A ∈ R^(r×k) and B ∈ R^(d×r); initialize A from a Gaussian and B to zero so the initial update is zero.
- Replace W·x with (W + (α/r)·B·A)·x in the forward pass; α is a fixed scaling hyperparameter that decouples the learning rate from the rank.
- Train only A and B with standard optimizers; gradients flow through the low-rank product.
- At deployment, compute W' = W + (α/r)·B·A once and use W' as a drop-in replacement; or keep A and B as small per-task add-ons that can be hot-swapped.

## Key results

On GPT-3 175B, LoRA matches or exceeds full fine-tuning on WikiSQL, MNLI, and SAMSum while training roughly 10,000× fewer parameters. On GPT-2 medium and large, LoRA is competitive with full fine-tuning and stronger than adapter and prefix-tuning baselines under the same trainable-parameter budget. The paper also shows that adapting only the attention projections (W_q, W_v) is enough — adapting all weight matrices does not consistently help — and that small ranks (r = 4 or 8) suffice for these tasks.

## Relation to existing work

LoRA replaces or complements two earlier parameter-efficient families. Adapters (Houlsby et al. 2019) insert small trainable modules between transformer sublayers, which adds inference latency. Prefix tuning (Li & Liang 2021) prepends learned tokens to the input, which consumes context length and can be unstable. LoRA differs by applying its update inside existing weight matrices, so the merged model has the same shape and latency as the base. The intrinsic-low-rank assumption is supported by Aghajanyan et al. (2021) on intrinsic dimensionality of fine-tuning. Subsequent work — DoRA, AdaLoRA, QLoRA, VeRA — generalizes the parameterization (decomposed magnitude/direction, dynamic rank, quantized base, shared random projection); QLoRA in particular pairs LoRA with 4-bit quantization to fit very large models on a single GPU.

## Implementation notes

A pedagogical script can apply LoRA to a tiny transformer or even a small MLP — the technique is independent of architecture. The minimal change to a training loop is to (a) freeze all parameters, (b) wrap each target weight in a forward function that adds α/r · B·A·x, (c) register only A and B as trainable, and (d) initialize B to zero. Hyperparameters that matter: r (rank), α (scaling), and which matrices to adapt. Pitfalls: forgetting to initialize B to zero produces a non-trivial initial output and breaks the "starts from base behavior" property; mismatched α/r scaling leads to a learning rate the optimizer cannot tame. Useful ablation: vary r over {1, 2, 4, 8} and watch downstream metric saturate quickly. Side-by-side compare LoRA-trained parameter count against full fine-tuning to make the parameter savings concrete.

## Open questions

LoRA's low-rank assumption may break for tasks that require large structural changes to the base model, such as continual pretraining on a very different domain. The paper does not give a theoretical bound on what ranks suffice; subsequent work (AdaLoRA) addresses this empirically by allocating rank dynamically.
