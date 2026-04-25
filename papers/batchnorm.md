---
slug: batchnorm
title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
authors:
  - Sergey Ioffe
  - Christian Szegedy
venue: ICML
year: 2015
arxiv_id: "1502.03167"
doi: null
url: https://arxiv.org/abs/1502.03167
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: training-dynamics
  secondary: []
tags:
  - normalization
  - batch-statistics
  - covariate-shift
  - regularization
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microbatchnorm
  target_path: 02-alignment/microbatchnorm.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microbatchnorm.py
    script_slug: microbatchnorm
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Batch Normalization standardizes the activations of each layer using batch statistics during training and running statistics during inference, then re-scales and re-shifts with learned parameters, accelerating training and enabling much higher learning rates.

## Problem

Training deep networks before BatchNorm required careful initialization, small learning rates, and saturating-nonlinearity workarounds. The authors framed the difficulty as internal covariate shift: as the parameters of earlier layers change during training, the distribution of activations entering later layers shifts, forcing those later layers to constantly chase a moving target. Practitioners managed this with extensive hyperparameter tuning, but training depth remained limited.

## Contribution

The paper inserts a normalization layer between linear and nonlinear sublayers of a deep network. For each minibatch, the layer computes the per-feature mean and variance across the batch, standardizes the activations, then applies a learned per-feature scale γ and shift β. At inference, the per-batch statistics are replaced with running averages collected during training, so the network is deterministic. The paper proves that the layer is differentiable end-to-end and shows that adding it allows much larger learning rates, removes the need for careful initialization, and acts as a mild regularizer that often eliminates the need for Dropout in convolutional networks.

## Method summary

- Insert BatchNorm after a linear (dense or convolutional) layer and before its nonlinearity.
- For training, given a minibatch of B examples and feature x_i: compute μ_B = (1/B) Σ x_i, σ²_B = (1/B) Σ (x_i - μ_B)², standardize x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε), then output y_i = γ · x̂_i + β.
- γ and β are learned parameters, one pair per feature; initialized to 1 and 0 so the layer initially passes the standardized activations through unchanged.
- Maintain exponential moving averages of μ and σ² across training batches; use these at inference.
- Backpropagation flows through the normalization, including the dependence on the batch statistics — the paper derives the closed-form gradient.

## Key results

On ImageNet with Inception-v1, BatchNorm allows the learning rate to be increased by a factor of 30 and the model trains 14× faster to the same top-1 accuracy. With small additional tweaks (no Dropout, less careful initialization), the resulting model exceeds the previous state of the art on ImageNet. The paper also shows benefits on smaller-scale tasks (MNIST, CIFAR) and provides the empirical framing — internal covariate shift — that motivated a decade of follow-on normalization work.

## Relation to existing work

BatchNorm replaces ad-hoc normalization heuristics (whitening of inputs only, careful initializations like Glorot or He) with a layer-level normalization that operates throughout the network. It triggered a family of variants designed for different settings: Layer Normalization (Ba et al. 2016) for sequence models where batch statistics are noisy; Instance Normalization for style transfer; Group Normalization (Wu & He 2018) for small batches; RMSNorm (Zhang & Sennrich 2019) for transformers, dropping mean-centering for cheaper compute. Subsequent analysis (Santurkar et al. 2018, "How Does Batch Normalization Help Optimization?") argues the benefit is loss-landscape smoothing rather than literal covariate shift reduction; the empirical recipe is the same regardless.

## Implementation notes

A pedagogical script can insert a BatchNorm-style layer into a small MLP or CNN. Minimum viable implementation: compute per-feature batch mean and variance, normalize, scale, shift. Pitfalls: forgetting to keep separate running statistics for inference (the layer behaves differently in train vs eval mode); applying BatchNorm to a single-example batch makes the variance zero and divides by sqrt(ε), producing degenerate gradients; placing BatchNorm after the nonlinearity instead of before is a common error that hurts performance. A useful comparison: train the same architecture with and without BatchNorm, plot loss curves, and show the higher-learning-rate regime that BatchNorm unlocks.

## Open questions

The paper's "internal covariate shift" framing has been challenged by subsequent analysis; the technique works regardless of whether the proposed mechanism is correct. BatchNorm also interacts poorly with very small batch sizes and with sequence models — issues that motivated LayerNorm and GroupNorm. Modern transformer architectures use LayerNorm or RMSNorm rather than BatchNorm.
