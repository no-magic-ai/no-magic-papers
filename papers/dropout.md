---
slug: dropout
title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
authors:
  - Nitish Srivastava
  - Geoffrey Hinton
  - Alex Krizhevsky
  - Ilya Sutskever
  - Ruslan Salakhutdinov
venue: JMLR
year: 2014
arxiv_id: null
doi: null
url: https://jmlr.org/papers/v15/srivastava14a.html
discovered_via: v3-backfill
discovered_date: 2026-04-26
status: implemented
themes:
  primary: training-dynamics
  secondary: []
tags:
  - dropout
  - regularization
  - overfitting
  - bagging
routing:
  decision: backlog-implement
  target_repo: no-magic
  target_script_slug: microdropout
  target_path: 02-alignment/microdropout.py
  target_tier: 02-alignment
  batch_label: v3-backfill-alignment
  review_date: null
implementations:
  - repo: no-magic
    path: 02-alignment/microdropout.py
    script_slug: microdropout
    commit: 4d43527c7eed48f7306ad56e4d04e5bd43cfd045
    release: v3.0.0
lesson:
  path: null
  status: none
dependencies_on_other_papers: []
---

## TL;DR

Dropout randomly zeros each unit's activation with probability p during training, then rescales remaining activations so the expected output matches inference; it acts as an efficient approximation to averaging over an exponential number of thinned subnetworks and reduces overfitting in deep networks.

## Problem

Deep networks trained on limited data overfit aggressively: they memorize training examples and fail to generalize. Classical regularizers (L2 weight decay, early stopping) only partly mitigate this. Bagging — averaging predictions across many independently-trained models — works but is computationally infeasible for large neural networks. The paper asks whether one can approximate the regularization benefit of model averaging without paying its training cost.

## Contribution

The paper introduces Dropout: at each training step, every hidden unit is independently retained with probability p (typically 0.5 for hidden layers, 0.8 for input layers) or zeroed out otherwise. The retained activations are scaled by 1/p so the expected output is unchanged, allowing the model to be evaluated at inference time without dropout while still respecting the training-time activation magnitude. The paper interprets dropout as efficient bagging over exponentially many subnetworks that share weights, and gives extensive empirical evidence that dropout reduces overfitting on vision, speech, and document classification tasks. It also analyzes the effect on hidden-unit features, showing that dropout produces less co-adapted, more independently-useful features.

## Method summary

- For each forward pass during training, sample a binary mask m of the same shape as the activation; m_i is 1 with probability p, 0 with probability 1-p.
- Compute the layer output as (m ⊙ activation) / p, so the expected output equals the activation.
- During backprop, gradients flow only through retained units (the mask is fixed for that pass).
- At inference, use the activation directly without masking; the 1/p scaling at training time guarantees magnitude consistency.
- A common choice is p = 0.5 for hidden layers and p = 0.8 (i.e. 20% drop) for input units.
- For convolutional layers, the paper recommends dropping entire feature maps (Spatial Dropout) rather than individual pixels, though the original paper focuses on dense layers.

## Key results

On MNIST, dropout reduces test error of a feedforward network from 1.60% to 1.35% with no other change. On CIFAR-10 and CIFAR-100, dropout improves a deep CNN by 1-2 percentage points. On ImageNet (with the AlexNet architecture), dropout in the fully-connected layers contributed substantially to the model's winning result. On the TIMIT phoneme-recognition task and on document classification, dropout consistently improved generalization. The paper also analyzes the learned hidden representations and shows that without dropout, units form tightly co-adapted detectors; with dropout, units become more independently meaningful.

## Relation to existing work

Dropout sits alongside L2 weight decay, max-norm constraints, data augmentation, and early stopping as a regularization technique. It generalizes earlier work on noisy training (denoising autoencoders, Vincent et al. 2008) and connects to model averaging and Bayesian inference: Gal & Ghahramani (2016) showed that dropout can be interpreted as variational inference in Bayesian neural networks. Subsequent variants — DropConnect (drops weights instead of activations), Variational Dropout, Concrete Dropout — modify the noise distribution or the per-unit rate. In modern transformer architectures, dropout remains a default component of attention and feedforward sublayers, often at lower rates (0.1 to 0.2) than the original paper recommended.

## Implementation notes

A pedagogical script can show dropout's effect on a small MLP over a moderate dataset (MNIST or a synthetic regression with limited samples). Minimum viable implementation: sample a binary mask via a Bernoulli draw, multiply, divide by p. Pitfalls: forgetting to scale by 1/p during training (leads to a quiet activation drop at inference); applying dropout during evaluation (turns the model stochastic, which may be desired for uncertainty estimation but is not the standard inference path); using dropout immediately before a batch-normalization layer (the variance shift confuses BN's running statistics). The cleanest demo trains the same model with and without dropout on a small training set and plots train vs. test loss; the gap shrinks substantially with dropout.

## Open questions

The optimal dropout rate is empirical; the paper offers heuristic ranges but no theory predicting it. Modern large transformers often use dropout sparingly (0.1) or skip it entirely in favor of other regularizers (weight decay with AdamW, augmentation, or simply more data). The interaction between dropout and other normalization layers is also subtle.
